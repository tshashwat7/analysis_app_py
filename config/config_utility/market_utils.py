# config/market_utils.py (NEW FILE)
"""
Market Hours & Timezone Utilities
==================================
Handles conversion between UTC (storage) and Asia/Kolkata (display/business logic).
"""

from datetime import datetime, time, timezone
import pytz

# Timezone Constants
UTC = timezone.utc
IST = pytz.timezone("Asia/Kolkata")

# NSE Market Hours (IST)
MARKET_OPEN_IST = time(9, 15)   # 9:15 AM IST
MARKET_CLOSE_IST = time(15, 30)  # 3:30 PM IST

def ensure_utc(dt: datetime) -> datetime:
    """
    Ensures datetime is timezone-aware UTC.
    
    Handles:
    - Naive datetime → assumes UTC
    - Non-UTC aware → converts to UTC
    """
    if dt is None:
        return None
    
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    
    return dt.astimezone(UTC)

def utc_to_ist(utc_dt: datetime) -> datetime:
    """
    Converts UTC datetime to IST datetime.
    
    Args:
        utc_dt: Timezone-aware UTC datetime
    
    Returns:
        Timezone-aware IST datetime
    """
    if utc_dt.tzinfo is None:
        # Naive datetime → assume UTC
        utc_dt = utc_dt.replace(tzinfo=UTC)
    
    return utc_dt.astimezone(IST)


def ist_to_utc(ist_dt: datetime) -> datetime:
    """
    Converts IST datetime to UTC datetime.
    
    Args:
        ist_dt: Timezone-aware or naive IST datetime
    
    Returns:
        Timezone-aware UTC datetime
    """
    if ist_dt.tzinfo is None:
        # Naive datetime → localize to IST
        ist_dt = IST.localize(ist_dt)
    
    return ist_dt.astimezone(UTC)


def is_market_open(check_time: datetime = None) -> bool:
    """
    Checks if NSE market is currently open.
    
    Args:
        check_time: UTC datetime to check (defaults to now)
    
    Returns:
        True if market is open
    """
    if check_time is None:
        check_time = datetime.now(UTC)
    
    # Convert to IST for business logic
    ist_time = utc_to_ist(check_time)
    
    # Check weekday (0=Monday, 6=Sunday)
    if ist_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check time range
    current_time = ist_time.time()
    return MARKET_OPEN_IST <= current_time <= MARKET_CLOSE_IST


def get_current_utc() -> datetime:
    """Returns current UTC time (standardized)."""
    return datetime.now(UTC)


def get_current_ist() -> datetime:
    """Returns current IST time for display."""
    return utc_to_ist(get_current_utc())


def format_ist_for_display(utc_dt: datetime) -> str:
    """
    Formats UTC datetime as IST string for logs/UI.
    
    Example: "2025-12-14 09:15:30 IST"
    """
    ist_dt = utc_to_ist(utc_dt)
    return ist_dt.strftime("%Y-%m-%d %H:%M:%S IST")
