import logging
from pathlib import Path
from tempfile import gettempdir

DEFAULT_LOG_PATH = Path(gettempdir()) / "aiaa-5047-project.log"


def setup_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.info(f"Set up a logger named {name}, log file located at {log_path}")

    return logger
