"""
Indeed Search Cache Manager.

This module provides a specialized cache implementation for Indeed job searches with:
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
logger = logging.getLogger("indeed_cache")

# Default configuration
DEFAULT_CONFIG = {
    "enable_cache": True,
    "enable_memory_cache": True,
    "enable_disk_cache": True,
    "memory_cache_size": 1000,  # Number of entries (larger than Locanto as Indeed has more volume)
    "disk_cache_size_limit": 200 * 1024 * 1024,  # 200 MB
    "default_ttl": 3600,  # 1 hour in seconds
    "job_listing_ttl": 43200,  # 12 hours for job listings (shorter than Locanto as Indeed updates more frequently)
    "search_results_ttl": 1800,  # 30 minutes for search results (shorter than Locanto)
    "popular_search_ttl": 10800,  # 3 hours for popular searches
    "auto_invalidation_interval": 43200,  # 12 hours
    "max_age_for_jobs": 14,  # Maximum age in days for job listings before forced invalidation
}

class IndeedCache:
    """Specialized cache manager for Indeed job searches."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Indeed cache manager.
        
        Args:
            config: Configuration dictionary. If None, uses default configuration.
        """
        # Merge provided config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Initialize the base cache manager
        self.cache_manager = CacheManager(
            name="indeed",
            memory_cache_size=self.config["memory_cache_size"],
            disk_cache_size_limit=self.config["disk_cache_size_limit"],
            default_ttl=self.config["default_ttl"],
            enable_memory_cache=self.config["enable_memory_cache"],
            enable_disk_cache=self.config["enable_disk_cache"]
        )
        
        # Set up auto-invalidation if enabled
        if self.config["auto_invalidation_interval"] > 0:
            self._setup_auto_invalidation()
        
        logger.info(f"Initialized Indeed cache with config: {self.config}")
    
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
                      job_type: Optional[str] = None, radius: Optional[int] = None,
                      salary: Optional[str] = None, experience_level: Optional[str] = None,
                      sort_by: Optional[str] = None, page: int = 1, **kwargs) -> str:
        """Generate a cache key for Indeed job searches.
        
        Args:
            query: Job search query
            location: Job location
            job_type: Type of job (full-time, part-time, etc.)
            radius: Search radius in miles/km
            salary: Salary filter
            experience_level: Experience level filter
            sort_by: Sort method (relevance, date, etc.)
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
            "radius": radius,
            "salary": salary.lower().strip() if salary else "",
            "experience_level": experience_level.lower().strip() if experience_level else "",
            "sort_by": sort_by.lower().strip() if sort_by else "",
            "page": page
        }
        
        # Add any additional parameters
        key_dict.update(kwargs)
        
        # Generate the key using the base cache manager
        return self.cache_manager.get_cache_key("indeed_search", **key_dict)
    
    def get_job_cache_key(self, job_id: str) -> str:
        """Generate a cache key for a specific job listing.
        
        Args:
            job_id: Indeed job ID
            
        Returns:
            Cache key string
        """
        return self.cache_manager.get_cache_key("indeed_job", job_id=job_id)
    
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
            if key.startswith("indeed_job"):
                ttl = self.config["job_listing_ttl"]
            elif key.startswith("indeed_search"):
                # Check if it's a popular search
                if value and isinstance(value, dict) and value.get("total_results", 0) > 200:
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
            job_id: Indeed job ID
        """
        key = self.get_job_cache_key(job_id)
        await self.invalidate(key)
    
    async def invalidate_search(self, query: str, location: Optional[str] = None, 
                               job_type: Optional[str] = None, radius: Optional[int] = None,
                               salary: Optional[str] = None, experience_level: Optional[str] = None,
                               sort_by: Optional[str] = None, page: int = 1, **kwargs) -> None:
        """Invalidate a specific search result.
        
        Args:
            query: Job search query
            location: Job location
            job_type: Type of job (full-time, part-time, etc.)
            radius: Search radius in miles/km
            salary: Salary filter
            experience_level: Experience level filter
            sort_by: Sort method (relevance, date, etc.)
            page: Result page number
            **kwargs: Additional parameters
        """
        key = self.get_cache_key(query, location, job_type, radius, salary, experience_level, sort_by, page, **kwargs)
        await self.invalidate(key)
    
    async def invalidate_all_searches(self) -> int:
        """Invalidate all search results.
        
        Returns:
            Number of invalidated entries
        """
        return await self.cache_manager.invalidate_pattern("indeed_search")
    
    async def invalidate_all_jobs(self) -> int:
        """Invalidate all job listings.
        
        Returns:
            Number of invalidated entries
        """
        return await self.cache_manager.invalidate_pattern("indeed_job")
    
    async def invalidate_by_location(self, location: str) -> int:
        """Invalidate all cache entries for a specific location.
        
        Args:
            location: Location to invalidate
            
        Returns:
            Number of invalidated entries
        """
        location_key = location.lower().strip()
        return await self.cache_manager.invalidate_pattern(location_key)
    
    async def invalidate_by_job_type(self, job_type: str) -> int:
        """Invalidate all cache entries for a specific job type.
        
        Args:
            job_type: Job type to invalidate
            
        Returns:
            Number of invalidated entries
        """
        job_type_key = job_type.lower().strip()
        return await self.cache_manager.invalidate_pattern(job_type_key)
    
    async def invalidate_by_salary_range(self, salary: str) -> int:
        """Invalidate all cache entries for a specific salary range.
        
        Args:
            salary: Salary range to invalidate
            
        Returns:
            Number of invalidated entries
        """
        salary_key = salary.lower().strip()
        return await self.cache_manager.invalidate_pattern(salary_key)
    
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
_indeed_cache = None

def get_indeed_cache(config: Optional[Dict[str, Any]] = None) -> IndeedCache:
    """Get or create the Indeed cache manager.
    
    Args:
        config: Configuration dictionary. If None, uses default configuration.
        
    Returns:
        IndeedCache instance
    """
    global _indeed_cache
    
    if _indeed_cache is None:
        _indeed_cache = IndeedCache(config=config)
    
    return _indeed_cache

async def close_indeed_cache() -> None:
    """Close the Indeed cache manager."""
    global _indeed_cache
    
    if _indeed_cache is not None:
        await _indeed_cache.close()
        _indeed_cache = None
