"""
Cache Configuration and Tuning Module.

This module provides a unified interface for configuring and tuning all cache implementations:
1. Global cache settings
2. Per-service cache configuration
3. Cache optimization and tuning
4. Cache statistics and monitoring
5. Cache management utilities
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta

# Import cache implementations
try:
    from locanto_cache import get_locanto_cache, close_locanto_cache
    HAS_LOCANTO_CACHE = True
except ImportError:
    HAS_LOCANTO_CACHE = False
    
    # Define dummy function
    def get_locanto_cache(*args, **kwargs):
        logging.warning("Locanto cache not available")
        return None
    
    async def close_locanto_cache():
        pass

try:
    from indeed_cache import get_indeed_cache, close_indeed_cache
    HAS_INDEED_CACHE = True
except ImportError:
    HAS_INDEED_CACHE = False
    
    # Define dummy function
    def get_indeed_cache(*args, **kwargs):
        logging.warning("Indeed cache not available")
        return None
    
    async def close_indeed_cache():
        pass

try:
    from brave_search_cache import get_brave_search_cache, close_brave_search_cache
    HAS_BRAVE_SEARCH_CACHE = True
except ImportError:
    HAS_BRAVE_SEARCH_CACHE = False
    
    # Define dummy function
    def get_brave_search_cache(*args, **kwargs):
        logging.warning("Brave Search cache not available")
        return None
    
    async def close_brave_search_cache():
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cache_config")

# Default global configuration
DEFAULT_GLOBAL_CONFIG = {
    "enable_all_caches": True,
    "enable_memory_caches": True,
    "enable_disk_caches": True,
    "global_disk_cache_dir": None,  # If None, uses default location
    "global_disk_cache_size_limit": 1024 * 1024 * 1024,  # 1 GB total limit
    "auto_tuning_enabled": True,
    "auto_tuning_interval": 86400,  # 24 hours
    "stats_collection_enabled": True,
    "stats_collection_interval": 3600,  # 1 hour
    "cache_cleanup_interval": 86400,  # 24 hours
    "memory_cache_allocation": {
        "locanto": 0.2,  # 20% of memory cache
        "indeed": 0.3,   # 30% of memory cache
        "brave_web": 0.3,  # 30% of memory cache
        "brave_ai": 0.2   # 20% of memory cache
    },
    "disk_cache_allocation": {
        "locanto": 0.2,  # 20% of disk cache
        "indeed": 0.3,   # 30% of disk cache
        "brave_web": 0.3,  # 30% of disk cache
        "brave_ai": 0.2   # 20% of disk cache
    }
}

# Service-specific configuration overrides
DEFAULT_SERVICE_CONFIGS = {
    "locanto": {
        "enable_cache": True,
        "enable_memory_cache": True,
        "enable_disk_cache": True,
        "memory_cache_size": 500,
        "disk_cache_size_limit": 100 * 1024 * 1024,  # 100 MB
        "default_ttl": 3600,  # 1 hour
        "job_listing_ttl": 86400,  # 24 hours
        "search_results_ttl": 3600,  # 1 hour
        "popular_search_ttl": 21600,  # 6 hours
        "auto_invalidation_interval": 86400,  # 24 hours
        "max_age_for_jobs": 30  # 30 days
    },
    "indeed": {
        "enable_cache": True,
        "enable_memory_cache": True,
        "enable_disk_cache": True,
        "memory_cache_size": 1000,
        "disk_cache_size_limit": 200 * 1024 * 1024,  # 200 MB
        "default_ttl": 3600,  # 1 hour
        "job_listing_ttl": 43200,  # 12 hours
        "search_results_ttl": 1800,  # 30 minutes
        "popular_search_ttl": 10800,  # 3 hours
        "auto_invalidation_interval": 43200,  # 12 hours
        "max_age_for_jobs": 14  # 14 days
    },
    "brave_search": {
        "enable_cache": True,
        "enable_memory_cache": True,
        "enable_disk_cache": True,
        
        # Web search cache configuration
        "web_memory_cache_size": 2000,
        "web_disk_cache_size_limit": 300 * 1024 * 1024,  # 300 MB
        "web_default_ttl": 3600,  # 1 hour
        "web_news_ttl": 1800,  # 30 minutes
        "web_popular_search_ttl": 21600,  # 6 hours
        "web_rare_search_ttl": 604800,  # 1 week
        
        # AI search cache configuration
        "ai_memory_cache_size": 1000,
        "ai_disk_cache_size_limit": 200 * 1024 * 1024,  # 200 MB
        "ai_default_ttl": 86400,  # 24 hours
        "ai_popular_search_ttl": 43200,  # 12 hours
        "ai_factual_search_ttl": 604800,  # 1 week
        
        # Common configuration
        "auto_invalidation_interval": 86400,  # 24 hours
        "max_age_for_searches": 7  # 7 days
    }
}

class CacheConfig:
    """Unified cache configuration and management."""
    
    def __init__(self, global_config: Optional[Dict[str, Any]] = None, 
                 service_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize the cache configuration.
        
        Args:
            global_config: Global cache configuration. If None, uses default configuration.
            service_configs: Service-specific configurations. If None, uses default configurations.
        """
        # Merge provided configs with defaults
        self.global_config = DEFAULT_GLOBAL_CONFIG.copy()
        if global_config:
            self.global_config.update(global_config)
        
        self.service_configs = DEFAULT_SERVICE_CONFIGS.copy()
        if service_configs:
            for service, config in service_configs.items():
                if service in self.service_configs:
                    self.service_configs[service].update(config)
                else:
                    self.service_configs[service] = config
        
        # Initialize cache instances
        self._initialize_caches()
        
        # Set up auto-tuning if enabled
        if self.global_config["auto_tuning_enabled"]:
            self._setup_auto_tuning()
        
        # Set up stats collection if enabled
        if self.global_config["stats_collection_enabled"]:
            self._setup_stats_collection()
        
        # Set up cache cleanup
        self._setup_cache_cleanup()
        
        logger.info("Initialized cache configuration")
    
    def _initialize_caches(self):
        """Initialize all cache instances with the configured settings."""
        # Apply global settings to service configs
        for service, config in self.service_configs.items():
            config["enable_cache"] = config.get("enable_cache", True) and self.global_config["enable_all_caches"]
            config["enable_memory_cache"] = config.get("enable_memory_cache", True) and self.global_config["enable_memory_caches"]
            config["enable_disk_cache"] = config.get("enable_disk_cache", True) and self.global_config["enable_disk_caches"]
            
            if self.global_config["global_disk_cache_dir"]:
                config["cache_dir"] = os.path.join(self.global_config["global_disk_cache_dir"], service)
        
        # Initialize Locanto cache if available
        if HAS_LOCANTO_CACHE:
            self.locanto_cache = get_locanto_cache(config=self.service_configs["locanto"])
            logger.info("Initialized Locanto cache")
        else:
            self.locanto_cache = None
            logger.warning("Locanto cache not available")
        
        # Initialize Indeed cache if available
        if HAS_INDEED_CACHE:
            self.indeed_cache = get_indeed_cache(config=self.service_configs["indeed"])
            logger.info("Initialized Indeed cache")
        else:
            self.indeed_cache = None
            logger.warning("Indeed cache not available")
        
        # Initialize Brave Search cache if available
        if HAS_BRAVE_SEARCH_CACHE:
            self.brave_search_cache = get_brave_search_cache(config=self.service_configs["brave_search"])
            logger.info("Initialized Brave Search cache")
        else:
            self.brave_search_cache = None
            logger.warning("Brave Search cache not available")
    
    def _setup_auto_tuning(self):
        """Set up automatic cache tuning."""
        async def _auto_tune():
            while True:
                try:
                    # Sleep first to collect some usage data
                    await asyncio.sleep(self.global_config["auto_tuning_interval"])
                    
                    # Collect stats from all caches
                    stats = self.get_all_stats()
                    
                    # Perform tuning based on stats
                    await self._tune_caches(stats)
                    
                    logger.info("Performed auto-tuning of caches")
                except Exception as e:
                    logger.error(f"Error during auto-tuning: {e}")
        
        # Start the auto-tuning task
        asyncio.create_task(_auto_tune())
        logger.info(f"Set up auto-tuning with interval: {self.global_config['auto_tuning_interval']} seconds")
    
    async def _tune_caches(self, stats: Dict[str, Any]) -> None:
        """Tune caches based on collected statistics.
        
        Args:
            stats: Cache statistics
        """
        # This is a simplified implementation. In a real system, you would
        # analyze usage patterns and adjust cache sizes, TTLs, etc. accordingly.
        
        # Example: Adjust memory cache sizes based on hit rates
        if self.locanto_cache and "locanto" in stats:
            locanto_stats = stats["locanto"]
            memory_hit_rate = locanto_stats.get("memory_hit_rate", 0)
            
            # If hit rate is low, increase cache size
            if memory_hit_rate < 50 and locanto_stats.get("memory_cache_size", 0) < 1000:
                new_size = min(1000, int(locanto_stats.get("memory_cache_size", 500) * 1.2))
                self.service_configs["locanto"]["memory_cache_size"] = new_size
                logger.info(f"Tuned Locanto memory cache size to {new_size}")
        
        # Similar adjustments for other caches...
        
        # Example: Adjust TTLs based on cache hit patterns
        if self.brave_search_cache and "brave_search" in stats:
            brave_stats = stats["brave_search"]
            if "web" in brave_stats and "ai" in brave_stats:
                web_hit_rate = brave_stats["web"].get("memory_hit_rate", 0)
                ai_hit_rate = brave_stats["ai"].get("memory_hit_rate", 0)
                
                # Adjust TTLs based on hit rates
                if web_hit_rate < 40:
                    new_ttl = min(7200, int(self.service_configs["brave_search"]["web_default_ttl"] * 1.5))
                    self.service_configs["brave_search"]["web_default_ttl"] = new_ttl
                    logger.info(f"Tuned Brave Web Search default TTL to {new_ttl}")
                
                if ai_hit_rate < 40:
                    new_ttl = min(172800, int(self.service_configs["brave_search"]["ai_default_ttl"] * 1.5))
                    self.service_configs["brave_search"]["ai_default_ttl"] = new_ttl
                    logger.info(f"Tuned Brave AI Search default TTL to {new_ttl}")
    
    def _setup_stats_collection(self):
        """Set up periodic statistics collection."""
        async def _collect_stats():
            while True:
                try:
                    # Sleep first
                    await asyncio.sleep(self.global_config["stats_collection_interval"])
                    
                    # Collect and log stats
                    stats = self.get_all_stats()
                    logger.info(f"Cache statistics: {json.dumps(stats, indent=2)}")
                except Exception as e:
                    logger.error(f"Error during stats collection: {e}")
        
        # Start the stats collection task
        asyncio.create_task(_collect_stats())
        logger.info(f"Set up stats collection with interval: {self.global_config['stats_collection_interval']} seconds")
    
    def _setup_cache_cleanup(self):
        """Set up periodic cache cleanup."""
        async def _cleanup_caches():
            while True:
                try:
                    # Sleep first
                    await asyncio.sleep(self.global_config["cache_cleanup_interval"])
                    
                    # Perform cleanup
                    await self.cleanup_all_caches()
                    
                    logger.info("Performed cache cleanup")
                except Exception as e:
                    logger.error(f"Error during cache cleanup: {e}")
        
        # Start the cache cleanup task
        asyncio.create_task(_cleanup_caches())
        logger.info(f"Set up cache cleanup with interval: {self.global_config['cache_cleanup_interval']} seconds")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches.
        
        Returns:
            Dictionary with statistics from all caches
        """
        stats = {
            "global_config": self.global_config,
            "timestamp": time.time()
        }
        
        # Collect Locanto stats
        if self.locanto_cache:
            try:
                stats["locanto"] = self.locanto_cache.get_stats()
            except Exception as e:
                logger.error(f"Error collecting Locanto stats: {e}")
                stats["locanto"] = {"error": str(e)}
        
        # Collect Indeed stats
        if self.indeed_cache:
            try:
                stats["indeed"] = self.indeed_cache.get_stats()
            except Exception as e:
                logger.error(f"Error collecting Indeed stats: {e}")
                stats["indeed"] = {"error": str(e)}
        
        # Collect Brave Search stats
        if self.brave_search_cache:
            try:
                stats["brave_search"] = self.brave_search_cache.get_stats()
            except Exception as e:
                logger.error(f"Error collecting Brave Search stats: {e}")
                stats["brave_search"] = {"error": str(e)}
        
        return stats
    
    async def cleanup_all_caches(self) -> Dict[str, int]:
        """Clean up all caches by invalidating expired entries.
        
        Returns:
            Dictionary with number of invalidated entries per cache
        """
        results = {}
        
        # Clean up Locanto cache
        if self.locanto_cache:
            try:
                count = await self.locanto_cache.invalidate_expired()
                results["locanto"] = count
            except Exception as e:
                logger.error(f"Error cleaning up Locanto cache: {e}")
                results["locanto"] = -1
        
        # Clean up Indeed cache
        if self.indeed_cache:
            try:
                count = await self.indeed_cache.invalidate_expired()
                results["indeed"] = count
            except Exception as e:
                logger.error(f"Error cleaning up Indeed cache: {e}")
                results["indeed"] = -1
        
        # Clean up Brave Search cache
        if self.brave_search_cache:
            try:
                web_count = await self.brave_search_cache.invalidate_expired_web()
                ai_count = await self.brave_search_cache.invalidate_expired_ai()
                results["brave_search_web"] = web_count
                results["brave_search_ai"] = ai_count
            except Exception as e:
                logger.error(f"Error cleaning up Brave Search cache: {e}")
                results["brave_search"] = -1
        
        return results
    
    async def invalidate_all_caches(self) -> Dict[str, int]:
        """Invalidate all entries in all caches.
        
        Returns:
            Dictionary with number of invalidated entries per cache
        """
        results = {}
        
        # Invalidate Locanto cache
        if self.locanto_cache:
            try:
                count = await self.locanto_cache.invalidate_all()
                results["locanto"] = count
            except Exception as e:
                logger.error(f"Error invalidating Locanto cache: {e}")
                results["locanto"] = -1
        
        # Invalidate Indeed cache
        if self.indeed_cache:
            try:
                count = await self.indeed_cache.invalidate_all()
                results["indeed"] = count
            except Exception as e:
                logger.error(f"Error invalidating Indeed cache: {e}")
                results["indeed"] = -1
        
        # Invalidate Brave Search cache
        if self.brave_search_cache:
            try:
                count = await self.brave_search_cache.invalidate_all()
                results["brave_search"] = count
            except Exception as e:
                logger.error(f"Error invalidating Brave Search cache: {e}")
                results["brave_search"] = -1
        
        return results
    
    async def invalidate_by_keyword(self, keyword: str) -> Dict[str, int]:
        """Invalidate all cache entries containing a specific keyword.
        
        Args:
            keyword: Keyword to match
            
        Returns:
            Dictionary with number of invalidated entries per cache
        """
        results = {}
        keyword_lower = keyword.lower().strip()
        
        # Invalidate Locanto cache
        if self.locanto_cache:
            try:
                # For Locanto, we'll try to invalidate by location and category
                loc_count = await self.locanto_cache.invalidate_by_location(keyword_lower)
                cat_count = await self.locanto_cache.invalidate_by_category(keyword_lower)
                results["locanto"] = loc_count + cat_count
            except Exception as e:
                logger.error(f"Error invalidating Locanto cache by keyword: {e}")
                results["locanto"] = -1
        
        # Invalidate Indeed cache
        if self.indeed_cache:
            try:
                # For Indeed, we'll try to invalidate by location, job type, and salary range
                loc_count = await self.indeed_cache.invalidate_by_location(keyword_lower)
                job_type_count = await self.indeed_cache.invalidate_by_job_type(keyword_lower)
                salary_count = await self.indeed_cache.invalidate_by_salary_range(keyword_lower)
                results["indeed"] = loc_count + job_type_count + salary_count
            except Exception as e:
                logger.error(f"Error invalidating Indeed cache by keyword: {e}")
                results["indeed"] = -1
        
        # Invalidate Brave Search cache
        if self.brave_search_cache:
            try:
                count = await self.brave_search_cache.invalidate_by_keyword(keyword_lower)
                results["brave_search"] = count
            except Exception as e:
                logger.error(f"Error invalidating Brave Search cache by keyword: {e}")
                results["brave_search"] = -1
        
        return results
    
    async def close(self) -> None:
        """Close all cache managers and release resources."""
        # Close Locanto cache
        if self.locanto_cache:
            try:
                await close_locanto_cache()
            except Exception as e:
                logger.error(f"Error closing Locanto cache: {e}")
        
        # Close Indeed cache
        if self.indeed_cache:
            try:
                await close_indeed_cache()
            except Exception as e:
                logger.error(f"Error closing Indeed cache: {e}")
        
        # Close Brave Search cache
        if self.brave_search_cache:
            try:
                await close_brave_search_cache()
            except Exception as e:
                logger.error(f"Error closing Brave Search cache: {e}")
        
        logger.info("Closed all cache managers")

# Singleton instance
_cache_config = None

def get_cache_config(global_config: Optional[Dict[str, Any]] = None,
                    service_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> CacheConfig:
    """Get or create the cache configuration.
    
    Args:
        global_config: Global cache configuration. If None, uses default configuration.
        service_configs: Service-specific configurations. If None, uses default configurations.
        
    Returns:
        CacheConfig instance
    """
    global _cache_config
    
    if _cache_config is None:
        _cache_config = CacheConfig(global_config=global_config, service_configs=service_configs)
    
    return _cache_config

async def close_cache_config() -> None:
    """Close the cache configuration and all cache managers."""
    global _cache_config
    
    if _cache_config is not None:
        await _cache_config.close()
        _cache_config = None
