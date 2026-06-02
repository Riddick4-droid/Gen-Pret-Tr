import logging
import sys

def get_logger(name: str = "decoder_transformer") -> logging.Logger:
    """
    Returns a configured logger that writes to stdout.
    Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] [name] message
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger