import logging.config
import logging
from typing import Optional


def setup_logging(log_file: str = "app.log", debug_log_file: str = "debug.log", log_level: Optional[str] = None) -> logging.Logger:
    """
    Configure the logging for the application, including file handlers for INFO and DEBUG logs.

    Args:
        log_file (str): The filename for the INFO and above log file.
        debug_log_file (str): The filename for the DEBUG log file.
        log_level (str): The logging level as a string (e.g., 'DEBUG', 'INFO'). If None, default to INFO.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger()
    
    # Prevent adding multiple handlers if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the root logger level based on log_level
    if log_level:
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logger.setLevel(numeric_level)
    else:
        logger.setLevel(logging.INFO)  # Default log level
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler for INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler for INFO and above
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # File handler for DEBUG and above
    debug_fh = logging.FileHandler(debug_log_file)
    debug_fh.setLevel(logging.DEBUG)
    debug_fh.setFormatter(formatter)
    logger.addHandler(debug_fh)

    return logger