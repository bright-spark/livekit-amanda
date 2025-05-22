"""
Cache Manager for livekit-amanda.

This module provides a configurable caching system with:
1. In-memory LRU cache
2. Persistent disk cache
3. Configurable TTL (time-to-live)
4. Cache invalidation strategies
5. Cache statistics tracking
"""

import os
import time
import json
import hashlib
import logging
import asyncio
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from functools import lru_cache
import aiofiles
import aiofiles.os
from pathlib import Path
import diskcache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cache_manager")

class CacheManager:
    """Base cache manager with memory and disk caching capabilities."""
    
    def __init__(self, 
                 name: str,
                 cache_dir: Optional[str] = None,
                 memory_cache_size: int = 1000,
                 disk_cache_size_limit: Optional[int] = None,  # In bytes, None for unlimited
                 default_ttl: int = 86400,  # 1 day in seconds
                 enable_memory_cache: bool = True,
                 enable_disk_cache: bool = True):
        """Initialize the cache manager.
        
        Args:
            name: Name of the cache (used for directory naming)
            cache_dir: Directory to store cache files. If None, uses default location.
            memory_cache_size: Size of the in-memory LRU cache
            disk_cache_size_limit: Size limit for disk cache in bytes. None for unlimited.
            default_ttl: Default time-to-live for cache entries in seconds
            enable_memory_cache: Whether to enable in-memory caching
            enable_disk_cache: Whether to enable disk caching
        """
        self.name = name
        self.default_ttl = default_ttl
        self.enable_memory_cache = enable_memory_cache
        self.enable_disk_cache = enable_disk_cache
        
        # Set up cache directory
        if cache_dir is None:
            home_dir = os.path.expanduser("~")
            cache_dir = os.path.join(home_dir, ".cache", "livekit-amanda")
        
        self.cache_dir = os.path.join(cache_dir, name)
        
        # Initialize statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "sets": 0,
            "invalidations": 0,
            "last_reset": time.time()
        }
        
        # Initialize memory cache if enabled
        if self.enable_memory_cache:
            # Create a function with LRU cache decorator
            @lru_cache(maxsize=memory_cache_size)
            def _memory_cache_get(key: str) -> Tuple[Any, float]:
                # This is just a placeholder, actual implementation is handled separately
                return None, 0
            
            self._memory_cache_get = _memory_cache_get
            self._memory_cache = {}  # For direct access when needed
        else:
            self._memory_cache_get = None
            self._memory_cache = None
        
        # Initialize disk cache if enabled
        if self.enable_disk_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._disk_cache = diskcache.Cache(self.cache_dir, size_limit=disk_cache_size_limit)
        else:
            self._disk_cache = None
        
        logger.info(f"Initialized {name} cache manager: memory_cache={enable_memory_cache}, disk_cache={enable_disk_cache}")
    
    def get_cache_key(self, query: str, **kwargs) -> str:
        """Generate a cache key from a query and additional parameters.
        
        Args:
            query: The main query string
            **kwargs: Additional parameters to include in the cache key
            
        Returns:
            A string hash representing the cache key
        """
        # Create a dictionary with the query and all kwargs
        key_dict = {"query": query}
        key_dict.update(kwargs)
        
        # Convert to a sorted string representation for consistent hashing
        key_str = json.dumps(key_dict, sort_keys=True)
        
        # Generate a hash
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return key_hash
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        # Try memory cache first if enabled
        if self.enable_memory_cache:
            try:
                value, expiry = self._memory_cache.get(key, (None, 0))
                if value is not None and time.time() < expiry:
                    self.stats["memory_hits"] += 1
                    return value
                self.stats["memory_misses"] += 1
            except Exception as e:
                logger.warning(f"Error accessing memory cache: {e}")
                self.stats["memory_misses"] += 1
        
        # Try disk cache if enabled
        if self.enable_disk_cache:
            try:
                cache_entry = self._disk_cache.get(key)
                if cache_entry is not None:
                    value, expiry = cache_entry
                    if time.time() < expiry:
                        self.stats["disk_hits"] += 1
                        
                        # Update memory cache with this value if enabled
                        if self.enable_memory_cache:
                            self._memory_cache[key] = (value, expiry)
                        
                        return value
                self.stats["disk_misses"] += 1
            except Exception as e:
                logger.warning(f"Error accessing disk cache: {e}")
                self.stats["disk_misses"] += 1
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        self.stats["sets"] += 1
        
        # Calculate expiry time
        ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl
        
        # Update memory cache if enabled
        if self.enable_memory_cache:
            try:
                self._memory_cache[key] = (value, expiry)
            except Exception as e:
                logger.warning(f"Error setting memory cache: {e}")
        
        # Update disk cache if enabled
        if self.enable_disk_cache:
            try:
                self._disk_cache.set(key, (value, expiry), expire=ttl)
            except Exception as e:
                logger.warning(f"Error setting disk cache: {e}")
    
    async def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        self.stats["invalidations"] += 1
        
        # Remove from memory cache if enabled
        if self.enable_memory_cache:
            try:
                if key in self._memory_cache:
                    del self._memory_cache[key]
            except Exception as e:
                logger.warning(f"Error invalidating memory cache: {e}")
        
        # Remove from disk cache if enabled
        if self.enable_disk_cache:
            try:
                self._disk_cache.delete(key)
            except Exception as e:
                logger.warning(f"Error invalidating disk cache: {e}")
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match against cache keys
            
        Returns:
            Number of invalidated entries
        """
        count = 0
        
        # Invalidate matching memory cache entries if enabled
        if self.enable_memory_cache:
            try:
                keys_to_remove = [k for k in self._memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                    count += 1
            except Exception as e:
                logger.warning(f"Error invalidating memory cache pattern: {e}")
        
        # Invalidate matching disk cache entries if enabled
        if self.enable_disk_cache:
            try:
                # This is more complex for disk cache, we need to iterate all keys
                keys_to_remove = [k for k in self._disk_cache if isinstance(k, str) and pattern in k]
                for key in keys_to_remove:
                    self._disk_cache.delete(key)
                    count += 1
            except Exception as e:
                logger.warning(f"Error invalidating disk cache pattern: {e}")
        
        self.stats["invalidations"] += count
        return count
    
    async def invalidate_all(self) -> int:
        """Invalidate all cache entries.
        
        Returns:
            Number of invalidated entries
        """
        count = 0
        
        # Clear memory cache if enabled
        if self.enable_memory_cache:
            try:
                count += len(self._memory_cache)
                self._memory_cache.clear()
            except Exception as e:
                logger.warning(f"Error clearing memory cache: {e}")
        
        # Clear disk cache if enabled
        if self.enable_disk_cache:
            try:
                count += len(self._disk_cache)
                self._disk_cache.clear()
            except Exception as e:
                logger.warning(f"Error clearing disk cache: {e}")
        
        self.stats["invalidations"] += count
        return count
    
    async def invalidate_expired(self) -> int:
        """Invalidate all expired cache entries.
        
        Returns:
            Number of invalidated entries
        """
        count = 0
        current_time = time.time()
        
        # Clear expired memory cache entries if enabled
        if self.enable_memory_cache:
            try:
                keys_to_remove = [k for k, (_, expiry) in self._memory_cache.items() if current_time >= expiry]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                    count += 1
            except Exception as e:
                logger.warning(f"Error clearing expired memory cache entries: {e}")
        
        # For disk cache, expiry is handled automatically by diskcache
        
        self.stats["invalidations"] += count
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.stats.copy()
        
        # Calculate hit rates
        total_memory = stats["memory_hits"] + stats["memory_misses"]
        total_disk = stats["disk_hits"] + stats["disk_misses"]
        
        stats["memory_hit_rate"] = (stats["memory_hits"] / total_memory * 100) if total_memory > 0 else 0
        stats["disk_hit_rate"] = (stats["disk_hits"] / total_disk * 100) if total_disk > 0 else 0
        
        # Add cache sizes
        if self.enable_memory_cache:
            stats["memory_cache_size"] = len(self._memory_cache)
        else:
            stats["memory_cache_size"] = 0
        
        if self.enable_disk_cache:
            stats["disk_cache_size"] = len(self._disk_cache)
            stats["disk_cache_size_bytes"] = self._disk_cache.size
        else:
            stats["disk_cache_size"] = 0
            stats["disk_cache_size_bytes"] = 0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "sets": 0,
            "invalidations": 0,
            "last_reset": time.time()
        }
    
    async def close(self) -> None:
        """Close the cache manager and release resources."""
        if self.enable_disk_cache and self._disk_cache is not None:
            self._disk_cache.close()
            logger.info(f"Closed {self.name} cache manager")

# Global cache managers
_cache_managers = {}

def get_cache_manager(name: str) -> CacheManager:
    """Get or create a cache manager by name.
    
    Args:
        name: Name of the cache manager
        
    Returns:
        CacheManager instance
    """
    if name not in _cache_managers:
        _cache_managers[name] = CacheManager(name=name)
    
    return _cache_managers[name]

async def close_all_cache_managers() -> None:
    """Close all cache managers."""
    for manager in _cache_managers.values():
        await manager.close()
    
    _cache_managers.clear()
