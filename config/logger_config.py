import logging
from logging.handlers import RotatingFileHandler
import os
import re

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Regex to strip emojis for console
EMOJI_PATTERN = re.compile(
    "["  
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # extended symbols
    "]+", flags=re.UNICODE
)

class RemoveEmojiFilter(logging.Filter):
    def filter(self, record):
        record.msg = EMOJI_PATTERN.sub("", str(record.msg))
        return True

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # ---- Console Handler (emoji removed) ----
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.addFilter(RemoveEmojiFilter())   # <-- IMPORTANT
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    # ---- File Handler (keeps emojis) ----
    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=4, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    # Suppress noisy third-party libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger
