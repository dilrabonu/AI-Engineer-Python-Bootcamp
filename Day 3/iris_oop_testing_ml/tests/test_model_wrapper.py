from __future__ import annotations

import numpy as np


def test_model_wrapper_trains_and_evaluates(model_wrapper):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model_wrapper.fit(X, y)
    metrics = model_wrapper.evaluate(X, y)

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
