import time
import threading
from functools import wraps
from typing import Callable, Any, Optional

def cached_result(ttl: int = 3600, key_fn: Optional[Callable[..., str]] = None):
    """
    Simple thread-safe TTL cache decorator.
    - ttl: seconds to keep an entry
    - key_fn: function(*args, **kwargs) -> str  (must return a hashable string key)
      If key_fn is None we fall back to a conservative string key from args/kwargs repr.
    The wrapper exposes .clear_cache() to force eviction.
    """
    cache = {}
    lock = threading.Lock()

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                if key_fn:
                    key = key_fn(*args, **kwargs)
                else:
                    key = fn.__name__ + ":" + repr((args, kwargs))
                now = time.time()
                with lock:
                    entry = cache.get(key)
                    if entry and (now - entry["ts"] < ttl):
                        return entry["value"]
                # compute and store
                value = fn(*args, **kwargs)
                with lock:
                    cache[key] = {"ts": time.time(), "value": value}
                return value
            except Exception:
                # on any caching logic failure, fallback to direct call
                return fn(*args, **kwargs)

        def clear_cache():
            nonlocal cache
            with lock:
                cache = {}

        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator