"""copula_synth.py

Minimal helpers for fitting a regular vine copula to mixed‑type pandas
DataFrames and generating unconditional or conditional synthetic rows.

Exports: fit_vine, sample_vine, conditional_sample_vine,
          df_to_uniform, uniform_to_df
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torchvinecopulib.vinecop import VineCop

__all__ = [
    "fit_vine",
    "sample_vine",
    "conditional_sample_vine",
    "df_to_uniform",
    "uniform_to_df",
]

# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def _check_missing(df: pd.DataFrame) -> None:
    """Fail fast when data still contains NaNs."""
    if df.isna().any().any():
        raise ValueError("Input contains NaNs; impute first or use an imputer.")


def _get_rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    """Always return a NumPy Generator instance."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


# -----------------------------------------------------------------------------
# Distributional transform helpers
# -----------------------------------------------------------------------------

def _forward_dt(col: pd.Series, rng: np.random.Generator):
    """Discrete → uniform using randomised CDF inversion."""
    cat = pd.Categorical(col, ordered=False)
    codes = np.asarray(cat.codes)
    # empirical PMF and lower CDF edges
    _, counts = np.unique(codes, return_counts=True)
    pmf = counts / counts.sum()
    cdf_lo = np.concatenate(([0.0], pmf.cumsum()[:-1]))
    # jitter each observation inside its CDF slice
    u = cdf_lo[codes] + pmf[codes] * rng.random(len(col))
    return u, {
        "type": "discrete",
        "vals": cat.categories.to_numpy(),
        "cdf_lo": cdf_lo,
        "pmf": pmf,
    }


def _forward_cont(col: pd.Series):
    """Continuous → uniform via rank transformation."""
    ranks = col.rank(method="average").to_numpy()
    return (ranks - 0.5) / len(col), {"type": "continuous", "values": col.to_numpy()}


def _inverse_discrete(u: np.ndarray, meta: dict):
    edges = meta["cdf_lo"] + meta["pmf"]
    idx = np.searchsorted(edges, u, side="right")
    return meta["vals"][np.minimum(idx, len(meta["vals"]) - 1)]


def _inverse_cont(u: np.ndarray, meta: dict):
    """Linear interpolation back to the empirical support."""
    u = np.clip(u, 0.0, 1.0)
    values = np.sort(meta["values"])
    pos = u * (len(values) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.ceil(pos).astype(int)
    return (1 - (pos - lo)) * values[lo] + (pos - lo) * values[hi]


# -----------------------------------------------------------------------------
# Public transform API
# -----------------------------------------------------------------------------

def df_to_uniform(
    df: pd.DataFrame,
    discrete_cols: list[str],
    *,
    rng: int | np.random.Generator | None = None,
):
    """Convert dataframe to uniform scale; return (u_df, meta)."""
    _check_missing(df)
    rng = _get_rng(rng)
    uni, meta = {}, {}
    for c in df.columns:
        u, m = (_forward_dt(df[c], rng) if c in discrete_cols else _forward_cont(df[c]))
        uni[c], meta[c] = u, m
    return pd.DataFrame(uni, copy=False), meta


def uniform_to_df(u_df: pd.DataFrame, meta: dict):
    """Inverse transform back to original scale."""
    cols = {}
    for c in u_df.columns:
        m = meta[c]
        u = u_df[c].to_numpy(float, copy=False)
        cols[c] = _inverse_cont(u, m) if m["type"] == "continuous" else _inverse_discrete(u, m)
    return pd.DataFrame(cols, copy=False)


# -----------------------------------------------------------------------------
# Vine‑copula workflow
# -----------------------------------------------------------------------------

def fit_vine(
    df: pd.DataFrame,
    discrete_cols: list[str],
    *,
    first_tree_vars: list[str] | None = None,
    vine_kwargs: dict | None = None,
    rng: int | np.random.Generator | None = 0,
):
    """Fit a vine copula and return (model, meta)."""
    vine_kwargs = vine_kwargs or {}
    u_df, meta = df_to_uniform(df, discrete_cols, rng=rng)

    data = torch.tensor(u_df.to_numpy(np.float32))
    vc = VineCop(num_dim=data.shape[1], is_cop_scale=True)

    # Force specified variables into first tree for exact conditioning
    if first_tree_vars:
        idx_ft = tuple(df.columns.get_loc(c) for c in first_tree_vars)
        vine_kwargs = {**vine_kwargs, "first_tree_vertex": idx_ft}

    vc.fit(obs=data, **vine_kwargs)
    return vc, meta


# -----------------------------------------------------------------------------
# Conditional‑sampling helpers
# -----------------------------------------------------------------------------

def _build_cond_dict(vc: VineCop, val_dict: dict[int, float], cond_idx: list[int]):
    """Translate fixed coordinates to the (v,S) mapping used by torchvine."""
    order = vc.sample_order
    pos = {v: order.index(v) for v in cond_idx}
    dct = {}
    for v in cond_idx:
        S = {w for w in cond_idx if pos[w] < pos[v]}
        dct[(v, frozenset(S))] = torch.tensor([[float(val_dict[v])]])
    return dct


# -----------------------------------------------------------------------------
# Public sampling API
# -----------------------------------------------------------------------------

def sample_vine(vc: VineCop, meta: dict, n: int, *, seed: int = 0):
    """Draw n unconditional synthetic rows."""
    u = vc.sample(num_sample=n, seed=seed, is_sobol=True)
    return uniform_to_df(pd.DataFrame(u.numpy(), columns=list(meta.keys())), meta)


def conditional_sample_vine(
    vc: VineCop,
    meta: dict,
    cond_df: pd.DataFrame,
    discrete_cols: list[str],
    *,
    seed: int = 0,
    rng: int | np.random.Generator | None = None,
):
    """Draw rows that match cond_df exactly on its columns."""
    meta_cols = list(meta.keys())
    if not set(cond_df.columns).issubset(meta_cols):
        raise ValueError("cond_df columns must be a subset of the fitted columns")

    rng = _get_rng(rng)
    u_cond, _ = df_to_uniform(cond_df, [c for c in discrete_cols if c in cond_df.columns], rng=rng)
    cond_idx = [meta_cols.index(c) for c in cond_df.columns]

    u_rows = []
    for row_vals in u_cond.to_numpy():
        val_dict = {idx: row_vals[i] for i, idx in enumerate(cond_idx)}
        dct = _build_cond_dict(vc, val_dict, cond_idx)
        u_row = vc.sample(num_sample=1, seed=int(rng.integers(1_000_000)), is_sobol=False, dct_v_s_obs=dct)
        u_rows.append(u_row)
    u_all = torch.cat(u_rows, dim=0)

    synth = uniform_to_df(pd.DataFrame(u_all.numpy(), columns=meta_cols), meta)
    synth[cond_df.columns] = cond_df.values
    return synth
