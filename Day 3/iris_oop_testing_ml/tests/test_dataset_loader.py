from __future__ import annotations

from src.data.dataset_loader import CSVDatasetLoader


def test_load_train_test_shapes(data_config):
    loader = CSVDatasetLoader(data_config)

    X_train, X_test, y_train, y_test = loader.load_train_test()

    assert len(X_train) + len(X_test) == len(y_train) + len(y_test)
    assert X_train.shape[1] == X_test.shape[1]

    assert set(y_train).issubset({0, 1, 2})