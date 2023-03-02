import logging

from typing import Tuple

def get_handler() -> Tuple[logging.FileHandler, logging.StreamHandler]:
    formatter = logging.Formatter('%(asctime)s — %(pathname)s — %(levelname)s — %(message)s')

    file_handler = logging.FileHandler('logs.log')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    return file_handler, stream_handler

def get_logger(log_level: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(log_level)

    file_handler, stream_handler = get_handler()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger