import logging
import logging.handlers
from pathlib import Path

# Formatting
FORMAT = "[%(asctime)s]\t%(levelname)s\t%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FORMAT)

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

# Add console handler that only shows INFO and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Function to set file handler
def set_log_file_handler(ll: logging.Logger, log_file: Path) -> None:
    # Create new log file every monday, capturing all log levels
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when="W0", backupCount=8)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    ll.addHandler(file_handler)
