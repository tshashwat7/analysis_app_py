import os
import sys
import json
import csv
import logging
from datetime import datetime

# Adjust Python path to prioritize the local packages
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config_helpers.logger_config import setup_logger
from services.corporate_actions import build_corp_actions_summary_cache
from config.constants import INDEX_TICKERS
from main import load_or_create_index

# Setup logging specific to the warmer
logger = setup_logger()

OUTPUT_DIR = "cache"
JSON_OUT_PATH = os.path.join(OUTPUT_DIR, "upcoming_corp_actions_list.json")
CSV_OUT_PATH = os.path.join(OUTPUT_DIR, "upcoming_corp_actions_list.csv")

def run_warmer():
    logger.info("Starting Daily Corporate Actions Warmer...")
    start_time = datetime.now()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Collect all tickers from all indices
    all_tickers = set()
    logger.info("Loading all active index tickers...")
    
    # We iterate over all the defined indices in constants.py and load their stocks
    # This matches the startup logic in main.py
    for idx_name in INDEX_TICKERS.keys():
        try:
            stocks = load_or_create_index(idx_name)
            for s in stocks:
                if isinstance(s, (list, tuple)) and len(s) >= 1:
                    all_tickers.add(s[0])
        except Exception as e:
            logger.warning(f"Could not load index {idx_name}: {e}")
            
    all_tickers = list(all_tickers)
    logger.info(f"Loaded {len(all_tickers)} unique tickers.")

    if not all_tickers:
        logger.error("No tickers found! Exiting.")
        return

    # 2. Force rebuild of the summary cache using the existing modular logic
    logger.info("Forcing rebuild of corp actions summary cache...")
    try:
        # build_corp_actions_summary_cache(tickers, force=True) returns {ticker: display_string}
        # containing ONLY stocks that have upcoming actions.
        upcoming_actions_map = build_corp_actions_summary_cache(all_tickers, force=True)
        logger.info(f"Successfully rebuilt summary cache. {len(upcoming_actions_map)} stocks have upcoming actions.")
    except Exception as e:
        logger.error(f"Failed to build corporate actions summary cache: {e}")
        return

    # 3. Save the separate lists (JSON & CSV)
    list_data = [{"ticker": k, "action_display": v} for k, v in upcoming_actions_map.items()]

    # Write JSON
    try:
        with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list_data, f, indent=2)
        logger.info(f"Saved JSON list to {JSON_OUT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save JSON list: {e}")

    # Write CSV
    if list_data:
        try:
            with open(CSV_OUT_PATH, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ticker", "action_display"])
                writer.writeheader()
                writer.writerows(list_data)
            logger.info(f"Saved CSV list to {CSV_OUT_PATH}")
        except Exception as e:
            logger.error(f"Failed to save CSV list: {e}")
    else:
        # If no data, still write an empty CSV with headers
        try:
            with open(CSV_OUT_PATH, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ticker", "action_display"])
        except Exception as e:
            pass

    elapsed = datetime.now() - start_time
    logger.info(f"Warmer completed successfully in {elapsed.total_seconds():.2f} seconds.")

if __name__ == "__main__":
    run_warmer()
