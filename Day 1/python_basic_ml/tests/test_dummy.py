from src.utils import get_logger, load_config

def test_imports_work() -> None:
    """ Basic smoke test to verify imports and config loading"""
    logger = get_logger("test_logger")
    assert logger is not None

    config = load_config("config.yaml")
    assert "app_name" in config