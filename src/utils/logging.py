from logging import (DEBUG, INFO, FileHandler, Formatter, Logger,
                     StreamHandler, getLogger)
from pathlib import Path

DEFAULT_LOG_PATH = Path("~/.cache/").expanduser() / "aiaa-5047-project.log"
LOGGER_CACHE: dict[str, Logger] = {}


def setup_logger(name: str, log_path: Path | None = None) -> Logger:
    global LOGGER_CACHE

    if name in LOGGER_CACHE:
        logger = LOGGER_CACHE[name]
        logger.info(f"Reuse the same logger named {name}")
        return logger

    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    logger = getLogger(name)
    logger.propagate = False
    logger.setLevel(DEBUG)
    formatter = Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

    file_handler = FileHandler(log_path)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Set up a logger named {name}, log file located at {log_path}")
    LOGGER_CACHE[name] = logger

    return logger
