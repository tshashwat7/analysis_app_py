import time
import threading
from functools import wraps
from typing import Callable, Any, Optional

from collections import OrderedDict

MAX_CACHE_ENTRIES = 1000
_SENTINEL = object()

def cached_result(ttl: int = 3600, key_fn: Optional[Callable[..., str]] = None):
    """
    Thread-safe TTL cache decorator with bounded size (P3-3) and 
    protection against duplicate computations (P2-4/NEW-2).
    """
    cache = OrderedDict()
    pending_events = {}  # Track in-flight computations
    lock = threading.Lock()

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                if key_fn:
                    key = key_fn(*args, **kwargs)
                else:
                    # Fallback key: handle unhashable args via repr
                    key = f"{fn.__name__}:{repr(args)}:{repr(kwargs)}"
                
                now = time.time()
                wait_event = None

                with lock:
                    entry = cache.get(key)
                    # Case 1: Hit and valid
                    if entry and entry["value"] is not _SENTINEL and (now - entry["ts"] < ttl):
                        cache.move_to_end(key)
                        return entry["value"]
                    
                    # Case 2: Miss or expired
                    # Check if another thread is already computing this key
                    if key in pending_events:
                        wait_event = pending_events[key]
                    else:
                        # This thread will compute; mark it
                        ev = threading.Event()
                        pending_events[key] = ev
                        cache[key] = {"ts": now, "value": _SENTINEL}
                        cache.move_to_end(key)
                
                # If another thread is computing, wait for it
                if wait_event:
                    finished = wait_event.wait(timeout=30)
                    with lock:
                        entry = cache.get(key)
                        if entry and entry["value"] is not _SENTINEL:
                            return entry["value"]
                    # If wait timed out or result still missing, fall through to compute
                
                try:
                    # Compute value outside the lock
                    value = fn(*args, **kwargs)
                    with lock:
                        cache[key] = {"ts": time.time(), "value": value}
                        # Enforce cache size limit
                        while len(cache) > MAX_CACHE_ENTRIES:
                            cache.popitem(last=False)
                    return value
                finally:
                    # Signal other threads and cleanup the event
                    with lock:
                        # ✅ Fix 3: Clear _SENTINEL before waking waiters so they see a clean
                        # miss and re-enter the normal deduplication path, rather than all
                        # independently computing (thundering herd on exception).
                        entry = cache.get(key)
                        if entry is not None and entry.get("value") is _SENTINEL:
                            cache.pop(key, None)
                        ev = pending_events.pop(key, None)
                        if ev:
                            ev.set()
                
            except Exception as e:
                # ✅ Fix 4: Log before fallback so caching logic failures are visible in production.
                logger.warning(
                    f"[cached_result] Caching logic failed for '{fn.__name__}', "
                    f"falling back to direct call: {e}"
                )
                # On any caching logic failure, fallback to direct call
                return fn(*args, **kwargs)

        def clear_cache():
            """Mutate in-place to ensure all closure references are cleared (P1-6)."""
            with lock:
                cache.clear()
                pending_events.clear()

        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator