# services/indicator_cache.py
"""
Temporary cache for technical indicators with configurable TTL.
Reduces redundant calculations for frequently accessed symbols.
"""

import time
import hashlib
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from threading import Lock
import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    indicators: Dict[str, Dict[str, Any]]
    patterns: Dict[str, Any]
    timestamp: float
    symbol: str
    horizon: str
    df_hash: Optional[str] = None
    benchmark_hash: Optional[str] = None
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired"""
        return (time.time() - self.timestamp) > ttl_seconds
    
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.timestamp


class IndicatorCache:
    """
    Thread-safe LRU cache for technical indicators.
    
    Features:
    - Configurable TTL per horizon
    - Automatic cleanup of expired entries
    - Cache statistics for monitoring
    - Optional hash validation for data freshness
    """
    
    # Default TTL values (in seconds)
    DEFAULT_TTL = {
        "intraday": 8 * 60 * 60,       # 1 hour
        "short_term": 8 * 60 * 60,  # 4 hours  
        "long_term": 8 * 60 * 60,   # 8 hours
        "multibagger": 24 * 60 * 60 # 24 hours
    }
    
    def __init__(self, ttl_config: Dict[str, int] = None, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            ttl_config: Custom TTL values per horizon (seconds)
            max_size: Maximum number of entries before cleanup
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._ttl = ttl_config or self.DEFAULT_TTL.copy()
        self._max_size = max_size
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"Indicator cache initialized with TTL: {self._ttl}")
    
    def _make_key(self, symbol: str, horizon: str) -> str:
        """Generate cache key"""
        return f"{symbol}:{horizon}"
    
    def get(
        self, 
        symbol: str, 
        horizon: str,
        df_hash: Optional[str] = None,
        benchmark_hash: Optional[str] = None,
        validate_hash: bool = False  # NEW: explicit hash validation flag
    ) -> Optional[Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]]:
        """
        Retrieve cached indicators if valid.
        
        Args:
            symbol: Stock symbol
            horizon: Time horizon
            df_hash: Optional hash to validate data freshness
            benchmark_hash: Optional benchmark data hash
            validate_hash: If True, validate hashes; if False, ignore hash mismatches
            
        Returns:
            Tuple of (indicators, patterns) or None if cache miss
        """
        key = self._make_key(symbol, horizon)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                logger.debug(f"Cache MISS: {key} - Entry not found")
                return None
            
            # Check expiration
            ttl = self._ttl.get(horizon, 1200)  # Default 20 min
            if entry.is_expired(ttl):
                logger.debug(f"Cache EXPIRED: {key} (age: {entry.age_seconds():.1f}s, TTL: {ttl}s)")
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                return None
            
            # FIXED: Only validate hash if explicitly requested AND hashes are provided
            if validate_hash:
                if df_hash and entry.df_hash and df_hash != entry.df_hash:
                    logger.debug(f"Cache INVALID: {key} (data hash mismatch)")
                    del self._cache[key]
                    self._misses += 1
                    return None
                
                if benchmark_hash and entry.benchmark_hash and benchmark_hash != entry.benchmark_hash:
                    logger.debug(f"Cache INVALID: {key} (benchmark hash mismatch)")
                    del self._cache[key]
                    self._misses += 1
                    return None
            
            # Cache HIT
            self._hits += 1
            logger.info(f"Cache HIT: {key} (age: {entry.age_seconds():.1f}s, TTL: {ttl}s)")
            return (entry.indicators, entry.patterns)
    
    def set(
        self,
        symbol: str,
        horizon: str,
        indicators: Dict[str, Dict[str, Any]],
        patterns: Dict[str, Any],
        df_hash: Optional[str] = None,
        benchmark_hash: Optional[str] = None
    ) -> None:
        """
        Store indicators in cache.
        
        Args:
            symbol: Stock symbol
            horizon: Time horizon
            indicators: Computed indicators
            patterns: Detected patterns
            df_hash: Optional data hash for validation
            benchmark_hash: Optional benchmark hash
        """
        key = self._make_key(symbol, horizon)
        
        with self._lock:
            # Cleanup if cache is too large
            if len(self._cache) >= self._max_size:
                self._cleanup_expired()
                
                # If still too large, remove oldest entries
                if len(self._cache) >= self._max_size:
                    self._evict_oldest(count=self._max_size // 10)  # Remove 10%
            
            entry = CacheEntry(
                indicators=indicators,
                patterns=patterns,
                timestamp=time.time(),
                symbol=symbol,
                horizon=horizon,
                df_hash=df_hash,
                benchmark_hash=benchmark_hash
            )
            
            self._cache[key] = entry
            logger.info(f"Cache SET: {key} (hash_validation: {df_hash is not None or benchmark_hash is not None})")
    
    def invalidate(self, symbol: str = None, horizon: str = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            symbol: Invalidate specific symbol (all horizons)
            horizon: Invalidate specific horizon (all symbols)
            
        Returns:
            Number of entries removed
        """
        with self._lock:
            if symbol and horizon:
                # Specific entry
                key = self._make_key(symbol, horizon)
                if key in self._cache:
                    del self._cache[key]
                    logger.debug(f"Cache INVALIDATE: {key}")
                    return 1
                return 0
            
            # Bulk invalidation
            keys_to_remove = []
            
            if symbol:
                keys_to_remove = [k for k in self._cache if k.startswith(f"{symbol}:")]
            elif horizon:
                keys_to_remove = [k for k in self._cache if k.endswith(f":{horizon}")]
            else:
                keys_to_remove = list(self._cache.keys())
            
            for key in keys_to_remove:
                del self._cache[key]
            
            count = len(keys_to_remove)
            logger.info(f"Cache INVALIDATE: {count} entries removed")
            return count
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries (must be called with lock held)"""
        keys_to_remove = []
        
        for key, entry in self._cache.items():
            ttl = self._ttl.get(entry.horizon, 1200)
            if entry.is_expired(ttl):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
            self._evictions += 1
        
        if keys_to_remove:
            logger.debug(f"Cache CLEANUP: {len(keys_to_remove)} expired entries removed")
        
        return len(keys_to_remove)
    
    def _evict_oldest(self, count: int) -> None:
        """Evict oldest entries (must be called with lock held)"""
        if not self._cache:
            return
        
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        for key, _ in sorted_entries[:count]:
            del self._cache[key]
            self._evictions += 1
        
        logger.debug(f"Cache EVICT: {count} oldest entries removed")
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache CLEAR: {count} entries removed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "evictions": self._evictions,
                "total_requests": total_requests,
                "ttl_config": self._ttl
            }
    
    def reset_stats(self) -> None:
        """Reset statistics counters"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.info("Cache statistics reset")


# ========================================
# Global Cache Instance
# ========================================

# Singleton instance
_indicator_cache: Optional[IndicatorCache] = None


def get_cache(
    ttl_config: Dict[str, int] = None,
    max_size: int = 1000
) -> IndicatorCache:
    """
    Get or create global cache instance.
    
    Args:
        ttl_config: Custom TTL values (only used on first call)
        max_size: Maximum cache size (only used on first call)
    """
    global _indicator_cache
    
    if _indicator_cache is None:
        _indicator_cache = IndicatorCache(ttl_config, max_size)
    
    return _indicator_cache


def configure_cache(ttl_config: Dict[str, int] = None, max_size: int = 1000) -> None:
    """
    Configure cache settings (creates new instance).
    
    Args:
        ttl_config: Custom TTL values per horizon (seconds)
        max_size: Maximum cache size
    """
    global _indicator_cache
    _indicator_cache = IndicatorCache(ttl_config, max_size)
    logger.info("Cache reconfigured")


# ========================================
# Utility Functions
# ========================================

def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Compute hash of DataFrame for cache validation.
    Uses last 10 rows and key columns to avoid hashing entire large DataFrames.
    
    Args:
        df: DataFrame to hash
        
    Returns:
        Hash string
    """
    try:
        # Use last 10 rows + shape for efficient hashing
        data_str = f"{df.shape}:{df.tail(10).to_json()}"
        return hashlib.md5(data_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Error computing DataFrame hash: {e}")
        return None


# ========================================
# Cached Wrapper Function
# ========================================

def compute_indicators_cached(
    symbol: str,
    horizon: str = "short_term",
    benchmark_symbol: str = "^NSEI",
    sector: str = None,
    force_refresh: bool = False,
    validate_data_hash: bool = False,  # NEW: explicit control
    df: pd.DataFrame = None,  # NEW: optional DataFrame for hash calculation
    benchmark_df: pd.DataFrame = None  # NEW: optional benchmark DataFrame
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Wrapper for compute_indicators with caching.
    
    Args:
        symbol: Stock symbol
        horizon: Time horizon
        benchmark_symbol: Benchmark symbol
        force_refresh: Skip cache and force recalculation
        validate_data_hash: If True, validate data hasn't changed using hash
        df: Optional DataFrame to compute hash from (for validation)
        benchmark_df: Optional benchmark DataFrame
        
    Returns:
        Tuple of (indicators, patterns)
    """
    # Import here to avoid circular imports
    from services.indicators import compute_indicators
    
    cache = get_cache()
    
    # Compute hashes if validation requested
    df_hash = None
    benchmark_hash = None
    if validate_data_hash:
        if df is not None:
            df_hash = compute_dataframe_hash(df)
        if benchmark_df is not None:
            benchmark_hash = compute_dataframe_hash(benchmark_df)
    
    # Try cache first (unless forced refresh)
    # if not force_refresh:
    #     logger.debug(f"Checking cache for {symbol}:{horizon} (validate_hash={validate_data_hash})")
    #     cached = cache.get(
    #         symbol, 
    #         horizon, 
    #         df_hash=df_hash, 
    #         benchmark_hash=benchmark_hash,
    #         validate_hash=validate_data_hash
    #     )
    #     if cached is not None:
    #         logger.info(f"Using cached indicators for {symbol}:{horizon}")
    #         return cached
    # else:
    #     logger.info(f"Force refresh enabled for {symbol}:{horizon}")
    
    # Cache miss - compute indicators
    logger.info(f"Computing indicators for {symbol}:{horizon}")
    indicators, patterns = compute_indicators(
        symbol=symbol,
        horizon=horizon,
        benchmark_symbol=benchmark_symbol,
        sector=sector,
    )
    
    # Store in cache
    # cache.set(symbol, horizon, indicators, patterns, df_hash, benchmark_hash)
    logger.info(f"Cached new indicators for {symbol}:{horizon}")
    
    return indicators, patterns


# ========================================
# DEBUGGING HELPER
# ========================================

def debug_cache_state(symbol: str = None, horizon: str = None):
    """
    Print detailed cache state for debugging.
    
    Args:
        symbol: Filter by symbol
        horizon: Filter by horizon
    """
    cache = get_cache()
    stats = cache.get_stats()
    
    print("\n" + "="*60)
    print("CACHE DEBUG INFO")
    print("="*60)
    print(f"Cache Size: {stats['size']}/{stats['max_size']}")
    print(f"Hit Rate: {stats['hit_rate']}% ({stats['hits']} hits / {stats['total_requests']} total)")
    print(f"Misses: {stats['misses']}")
    print(f"Evictions: {stats['evictions']}")
    print(f"TTL Config: {stats['ttl_config']}")
    print("\nCached Entries:")
    print("-"*60)
    
    with cache._lock:
        for key, entry in cache._cache.items():
            if symbol and not key.startswith(f"{symbol}:"):
                continue
            if horizon and not key.endswith(f":{horizon}"):
                continue
            
            ttl = cache._ttl.get(entry.horizon, 1200)
            age = entry.age_seconds()
            expires_in = max(0, ttl - age)
            
            print(f"Key: {key}")
            print(f"  Age: {age:.1f}s / TTL: {ttl}s")
            print(f"  Expires in: {expires_in:.1f}s")
            print(f"  Has data hash: {entry.df_hash is not None}")
            print(f"  Has benchmark hash: {entry.benchmark_hash is not None}")
            print()
    
    print("="*60 + "\n")


# ========================================
# USAGE EXAMPLES
# ========================================

"""
# BASIC USAGE (without hash validation - RECOMMENDED for most cases)
from services.indicator_cache import compute_indicators_cached

indicators, patterns = compute_indicators_cached(
    symbol="RELIANCE.NS",
    horizon="short_term"
)

# WITH HASH VALIDATION (only if you need to ensure data freshness)
import pandas as pd
from services.indicator_cache import compute_indicators_cached

df = pd.read_csv("data.csv")  # Your data
indicators, patterns = compute_indicators_cached(
    symbol="RELIANCE.NS",
    horizon="short_term",
    validate_data_hash=True,  # Enable validation
    df=df  # Pass DataFrame for hash calculation
)

# FORCE REFRESH
indicators, patterns = compute_indicators_cached(
    symbol="TCS.NS",
    horizon="long_term",
    force_refresh=True  # Bypass cache
)

# CHECK CACHE STATISTICS
from services.indicator_cache import get_cache, debug_cache_state

cache = get_cache()
stats = cache.get_stats()
print(f"Hit Rate: {stats['hit_rate']}%")

# Detailed debugging
debug_cache_state(symbol="RELIANCE.NS")

# MANUAL CACHE CONTROL
cache.invalidate(symbol="RELIANCE.NS")  # Clear specific symbol
cache.invalidate(horizon="intraday")     # Clear specific horizon
cache.clear()                             # Clear everything

# CUSTOM TTL CONFIGURATION
from services.indicator_cache import configure_cache

configure_cache(ttl_config={
    "intraday": 3 * 60,      # 3 minutes
    "short_term": 15 * 60,   # 15 minutes
    "long_term": 30 * 60,    # 30 minutes
    "multibagger": 60 * 60   # 1 hour
}, max_size=500)
"""
