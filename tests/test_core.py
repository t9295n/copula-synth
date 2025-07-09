import numpy as np
import pandas as pd
import pytest

from copula_synth import (
    df_to_uniform,
    uniform_to_df,
    fit_vine,
    sample_vine,
    conditional_sample_vine,
)

# ---------------------------------------------------------------------
# 1. Distributional transform is numerically invertible
# ---------------------------------------------------------------------
def test_distributional_transform_roundtrip():
    n = 1_000
    df = pd.DataFrame(
        {
            "cont": np.random.randn(n) * 3.0 + 10.0,
            "disc": np.random.choice(list("ABC"), n, p=[0.2, 0.5, 0.3]),
        }
    )
    u, meta = df_to_uniform(df, ["disc"])
    df_back = uniform_to_df(u, meta)

    ## rank ordering should be identical
    assert (
        df["cont"].sort_values().index.equals(
            df_back["cont"].sort_values().index
        )
    )

    ## small relative error for continuous column
    mae = (df_back["cont"] - df["cont"]).abs().mean()
    assert mae < df["cont"].std() * 0.05  # ≤5 % of σ

    ## discrete column recovered exactly
    assert (df_back["disc"] == df["disc"]).all()


# ---------------------------------------------------------------------
# 2. Conditional sampler returns plausible conditional distribution
# ---------------------------------------------------------------------
def test_conditional_sampler_exactness():
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 65, n),
            "score": rng.normal(600, 50, n),
            "state": rng.choice(["CA", "TX", "NY"], n, p=[0.3, 0.4, 0.3]),
        }
    )

    vc, meta = fit_vine(df, ["age", "state"], first_tree_vars=["state"])
    cond = pd.DataFrame({"state": ["TX"] * 50})

    synth = conditional_sample_vine(vc, meta, cond, ["age", "state"], seed=123)

    ## conditional columns are fixed
    assert (synth["state"] == "TX").all()

    ## mean score in TX should be close to real TX mean
    ## weak test: we only check that the difference is within 3 standard errors
    real_tx_mean = df.loc[df["state"] == "TX", "score"].mean()
    synth_mean = synth["score"].mean()
    tx_sd = df.loc[df["state"] == "TX", "score"].std()
    se = tx_sd / np.sqrt(len(cond))
    print(f"Real TX mean: {real_tx_mean}, Synth TX mean: {synth_mean}, SE: {se}")
    assert abs(synth_mean - real_tx_mean) < 3 * se