import logging
from configs.logging_config import setup_logging, get_logger

def test_setup_logging_is_idempotent():
    setup_logging("INFO")
    n = len(logging.getLogger().handlers)
    setup_logging("INFO")
    assert len(logging.getLogger().handlers) == n  # no duplicate handlers

def test_get_logger_returns_named_logger():
    log = get_logger("mymod")
    assert log.name == "mymod"
