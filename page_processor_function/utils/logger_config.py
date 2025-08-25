# utils/logger_config.py
import logging
import os
from datetime import datetime
import sys

LOG_DIR = "/tmp"  # Lambda's writable dir
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
handler.setLevel(logging.INFO)
handler.stream.reconfigure(encoding="utf-8")


def setup_logging(job_id=None, page_number=None):
    """Setup global logging configuration (called once per Lambda execution)."""

    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(
        LOG_DIR, f"log_{job_id or 'job'}_{page_number or 'NA'}_{timestamp}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,  # ensures only one config across all imports
    )

    return log_file


def get_logger(name=None):
    """Get a logger instance. Safe for use in any file."""
    return logging.getLogger(name)
