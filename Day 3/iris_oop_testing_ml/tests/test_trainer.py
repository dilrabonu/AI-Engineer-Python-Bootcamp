from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock

from src.training.trainer import Trainer
from src.utils.config import TrainingConfig

def test_trainer_run_call_all_components():
    X_train = np.random.rand(4, 2)
    X_test = np.random.rand(2, 2)
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([0, 1])

    # Mocks
    data_loader = MagicMock()
    data_loader.load_train_test.return_value = (X_train, X_test, y_train, y_test)

    preprocessor = MagicMock()
    preprocessor.fit_transform.return_value = X_train
    preprocessor.transform.return_value = X_test

    model = MagicMock()
    model.evaluate.return_value = {"accuracy": 0.9}

    trainer = Trainer(
        data_loader=data_loader,
        preprocessor=preprocessor,
        model=model,
        config=TrainingConfig(),
    )

    trainer.run()

    data_loader.load_train_test.assert_called_once()
    preprocessor.fit_transform.assert_called_once()
    preprocessor.transform.assert_called_once()
    model.fit.assert_called_once()
    model.evaluate.assert_called_once()