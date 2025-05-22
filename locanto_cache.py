"""
Locanto Search Cache Manager.

This module provides a specialized cache implementation for Locanto job searches with:
1. Configurable TTL for different types of job searches
2. Automatic invalidation of stale job listings
3. Specialized cache key generation for job search parameters
4. Configurable cache sizes and persistence options
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta

from cache_manager import CacheManager, get_cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("locanto_cache")

# Default configuration
DEFAULT_CONFIG = {
    "enable_cache": True,
    "enable_memory_cache": True,
    "enable_disk_cache": True,
    "memory_cache_size": 500,  # Number of entries
    "disk_cache_size_limit": 100 * 1024 * 1024,  # 100 MB
    "default_ttl": 3600,  # 1 hour in seconds
    "job_listing_ttl": 86400,  # 24 hours for job listings
    "search_results_ttl": 3600,  # 1 hour for search results
    "popular_search_ttl": 21600,  # 6 hours for popular searches
    "auto_invalidation_interval": 86400,  # 24 hours
    "max_age_for_jobs": 30,  # Maximum age in days for job listings before forced invalidation
}

class LocantoCache:
    """Specialized cache manager for Locanto job searches."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Locanto cache manager.
        
        Args:
            config: Configuration dictionary. If None, uses default configuration.
        """
        # Merge provided config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Initialize the base cache manager
        self.cache_manager = CacheManager(
            name="locanto",
            memory_cache_size=self.config["memory_cache_size"],
            disk_cache_size_limit=self.config["disk_cache_size_limit"],
            default_ttl=self.config["default_ttl"],
            enable_memory_cache=self.config["enable_memory_cache"],
            enable_disk_cache=self.config["enable_disk_cache"]
        )
        
        # Set up auto-invalidation if enabled
        if self.config["auto_invalidation_interval"] > 0:
            self._setup_auto_invalidation()
        
        logger.info(f"Initialized Locanto cache with config: {self.config}")
    
    def _setup_auto_invalidation(self):
        """Set up automatic cache invalidation."""
        async def _auto_invalidate():
            while True:
                try:
                    # Sleep first to avoid immediate invalidation on startup
                    await asyncio.sleep(self.config["auto_invalidation_interval"])
                    
                    # Invalidate expired entries
                    count = await self.invalidate_expired()
                    
                    # Invalidate old job listings
                    old_jobs_count = await self.invalidate_old_job_listings()
                    
                    logger.info(f"Auto-invalidation: removed {count} expired entries and {old_jobs_count} old job listings")
                except Exception as e:
                    logger.error(f"Error during auto-invalidation: {e}")
        
        # Start the auto-invalidation task
        asyncio.create_task(_auto_invalidate())
        logger.info(f"Set up auto-invalidation with interval: {self.config['auto_invalidation_interval']} seconds")
    
    def get_cache_key(self, query: str, location: Optional[str] = None, 
                      job_type: Optional[str] = None, category: Optional[str] = None,
                      page: int = 1, **kwargs) -> str:
        """Generate a cache key for Locanto job searches.
        
        Args:
            query: Job search query
            location: Job location
            job_type: Type of job (full-time, part-time, etc.)
            category: Job category
            page: Result page number
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a dictionary with all parameters
        key_dict = {
            "query": query.lower().strip() if query else "",
            "location": location.lower().strip() if location else "",
            "job_type": job_type.lower().strip() if job_type else "",
            "category": category.lower().strip() if category else "",
            "page": page
        }
        
        # Add any additional parameters
        key_dict.update(kwargs)
        
        # Generate the key using the base cache manager
        return self.cache_manager.get_cache_key("locanto_search", **key_dict)
    
    def get_job_cache_key(self, job_id: str) -> str:
        """Generate a cache key for a specific job listing.
        
        Args:
            job_id: Locanto job ID
            
        Returns:
            Cache key string
        """
        return self.cache_manager.get_cache_key("locanto_job", job_id=job_id)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.config["enable_cache"]:
            return None
        
        return await self.cache_manager.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        if not self.config["enable_cache"]:
            return
        
        # Use appropriate TTL based on the key prefix
        if ttl is None:
            if key.startswith("locanto_job"):
                ttl = self.config["job_listing_ttl"]
            elif key.startswith("locanto_search"):
                # Check if it's a popular search
                if value and isinstance(value, dict) and value.get("total_results", 0) > 100:
                    ttl = self.config["popular_search_ttl"]
                else:
                    ttl = self.config["search_results_ttl"]
            else:
                ttl = self.config["default_ttl"]
        
        await self.cache_manager.set(key, value, ttl)
    
    async def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        await self.cache_manager.invalidate(key)
    
    async def invalidate_job(self, job_id: str) -> None:
        """Invalidate a specific job listing.
        
        Args:
            job_id: Locanto job ID
        """
        key = self.get_job_cache_key(job_id)
        await self.invalidate(key)
    
    async def invalidate_search(self, query: str, location: Optional[str] = None, 
                               job_type: Optional[str] = None, category: Optional[str] = None,
                               page: int = 1, **kwargs) -> None:
        """Invalidate a specific search result.
        
        Args:
            query: Job search query
            location: Job location
            job_type: Type of job (full-time, part-time, etc.)
            category: Job category
            page: Result page number
            **kwargs: Additional parameters
        """
        key = self.get_cache_key(query, location, job_type, category, page, **kwargs)
        await self.invalidate(key)
    
    async def invalidate_all_searches(self) -> int:
        """Invalidate all search results.
        
        Returns:
            Number of invalidated entries
        """
        return await self.cache_manager.invalidate_pattern("locanto_search")
    
    async def invalidate_all_jobs(self) -> int:
        """Invalidate all job listings.
        
        Returns:
            Number of invalidated entries
        """
        return await self.cache_manager.invalidate_pattern("locanto_job")
    
    async def invalidate_by_location(self, location: str) -> int:
        """Invalidate all cache entries for a specific location.
        
        Args:
            location: Location to invalidate
            
        Returns:
            Number of invalidated entries
        """
        # This is a simplified approach - in a real implementation, you might need
        # to iterate through all keys and check if they match the location
        location_key = location.lower().strip()
        return await self.cache_manager.invalidate_pattern(location_key)
    
    async def invalidate_by_category(self, category: str) -> int:
        """Invalidate all cache entries for a specific job category.
        
        Args:
            category: Job category to invalidate
            
        Returns:
            Number of invalidated entries
        """
        category_key = category.lower().strip()
        return await self.cache_manager.invalidate_pattern(category_key)
    
    async def invalidate_expired(self) -> int:
        """Invalidate all expired cache entries.
        
        Returns:
            Number of invalidated entries
        """
        return await self.cache_manager.invalidate_expired()
    
    async def invalidate_old_job_listings(self) -> int:
        """Invalidate job listings older than the configured maximum age.
        
        Returns:
            Number of invalidated entries
        """
        # In a real implementation, you would need to check the posting date
        # of each job listing and invalidate those that are too old
        # This is a simplified version that assumes job data includes a 'posted_date' field
        count = 0
        max_age_days = self.config["max_age_for_jobs"]
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # For demonstration purposes, we'll just log this
        logger.info(f"Would invalidate job listings older than {cutoff_date}")
        
        # In a real implementation, you would iterate through cached job listings
        # and check their posting dates
        
        return count
    
    async def invalidate_all(self) -> int:
        """Invalidate all cache entries.
        
        Returns:
            Number of invalidated entries
        """
        return await self.cache_manager.invalidate_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.cache_manager.get_stats()
        stats["config"] = self.config
        return stats
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.cache_manager.reset_stats()
    
    async def close(self) -> None:
        """Close the cache manager and release resources."""
        await self.cache_manager.close()

# Singleton instance
_locanto_cache = None

def get_locanto_cache(config: Optional[Dict[str, Any]] = None) -> LocantoCache:
    """Get or create the Locanto cache manager.
    
    Args:
        config: Configuration dictionary. If None, uses default configuration.
        
    Returns:
        LocantoCache instance
    """
    global _locanto_cache
    
    if _locanto_cache is None:
        _locanto_cache = LocantoCache(config=config)
    
    return _locanto_cache

async def close_locanto_cache() -> None:
    """Close the Locanto cache manager."""
    global _locanto_cache
    
    if _locanto_cache is not None:
        await _locanto_cache.close()
        _locanto_cache = None
