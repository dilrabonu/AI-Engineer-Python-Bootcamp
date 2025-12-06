from __future__ import annotations

import numpy as np

from src.preprocessing.preprocessor import IrisPreprocessor

def test_preprocessor_scales_features_to_zero_mean(iris_small_df):
    X = iris_small_df.drop(columns=["species"])

    pre = IrisPreprocessor()
    X_scaled = pre.fit_transform(X)

    means = X_scaled.mean(axis=0)
    assert np.all(np.abs(means) < 1e-6)

    assert X_scaled.shape == X.shape
