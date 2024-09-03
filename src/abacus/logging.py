import logging
import logging.handlers
from pathlib import Path

# Formatting
FORMAT = "%(levelname)s\t[%(asctime)s]\t[%(filename)s:%(lineno)d]\t%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FORMAT)

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Function to set file handler
def set_log_file_handler(ll: logging.Logger, log_file: Path) -> None:
    # Create new log file every monday
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when="W0", backupCount=8)
    file_handler.setFormatter(formatter)
    ll.addHandler(file_handler)
