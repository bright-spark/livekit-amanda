"""
Brave Search Cache Manager.

This module provides a specialized cache implementation for Brave Search (AI & Web) with:
1. Separate caching configurations for AI and Web searches
2. Configurable TTL for different types of searches
3. Automatic invalidation of stale search results
4. Specialized cache key generation for search parameters
5. Configurable cache sizes and persistence options
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
logger = logging.getLogger("brave_search_cache")

# Default configuration
DEFAULT_CONFIG = {
    "enable_cache": True,
    "enable_memory_cache": True,
    "enable_disk_cache": True,
    
    # Web search cache configuration
    "web_memory_cache_size": 2000,  # Number of entries
    "web_disk_cache_size_limit": 300 * 1024 * 1024,  # 300 MB
    "web_default_ttl": 3600,  # 1 hour in seconds
    "web_news_ttl": 1800,  # 30 minutes for news searches
    "web_popular_search_ttl": 21600,  # 6 hours for popular searches
    "web_rare_search_ttl": 604800,  # 1 week for rare searches
    
    # AI search cache configuration
    "ai_memory_cache_size": 1000,  # Number of entries
    "ai_disk_cache_size_limit": 200 * 1024 * 1024,  # 200 MB
    "ai_default_ttl": 86400,  # 24 hours in seconds
    "ai_popular_search_ttl": 43200,  # 12 hours for popular searches
    "ai_factual_search_ttl": 604800,  # 1 week for factual searches
    
    # Common configuration
    "auto_invalidation_interval": 86400,  # 24 hours
    "max_age_for_searches": 7,  # Maximum age in days for search results before forced invalidation
    "news_keywords": ["news", "latest", "update", "today", "breaking"],  # Keywords that indicate news searches
    "factual_keywords": ["who", "what", "when", "where", "why", "how", "fact", "history", "define"]  # Keywords that indicate factual searches
}

class BraveSearchCache:
    """Specialized cache manager for Brave Search (AI & Web)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Brave Search cache manager.
        
        Args:
            config: Configuration dictionary. If None, uses default configuration.
        """
        # Merge provided config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Initialize the base cache managers for Web and AI searches
        self.web_cache_manager = CacheManager(
            name="brave_web_search",
            memory_cache_size=self.config["web_memory_cache_size"],
            disk_cache_size_limit=self.config["web_disk_cache_size_limit"],
            default_ttl=self.config["web_default_ttl"],
            enable_memory_cache=self.config["enable_memory_cache"],
            enable_disk_cache=self.config["enable_disk_cache"]
        )
        
        self.ai_cache_manager = CacheManager(
            name="brave_ai_search",
            memory_cache_size=self.config["ai_memory_cache_size"],
            disk_cache_size_limit=self.config["ai_disk_cache_size_limit"],
            default_ttl=self.config["ai_default_ttl"],
            enable_memory_cache=self.config["enable_memory_cache"],
            enable_disk_cache=self.config["enable_disk_cache"]
        )
        
        # Set up auto-invalidation if enabled
        if self.config["auto_invalidation_interval"] > 0:
            self._setup_auto_invalidation()
        
        logger.info(f"Initialized Brave Search cache with config: {self.config}")
    
    def _setup_auto_invalidation(self):
        """Set up automatic cache invalidation."""
        async def _auto_invalidate():
            while True:
                try:
                    # Sleep first to avoid immediate invalidation on startup
                    await asyncio.sleep(self.config["auto_invalidation_interval"])
                    
                    # Invalidate expired entries
                    web_count = await self.invalidate_expired_web()
                    ai_count = await self.invalidate_expired_ai()
                    
                    logger.info(f"Auto-invalidation: removed {web_count} expired web entries and {ai_count} expired AI entries")
                except Exception as e:
                    logger.error(f"Error during auto-invalidation: {e}")
        
        # Start the auto-invalidation task
        asyncio.create_task(_auto_invalidate())
        logger.info(f"Set up auto-invalidation with interval: {self.config['auto_invalidation_interval']} seconds")
    
    def get_web_cache_key(self, query: str, country: str = "us", 
                          search_lang: str = "en", ui_lang: str = "en-US",
                          count: int = 10, offset: int = 0, 
                          safe_search: str = "moderate", **kwargs) -> str:
        """Generate a cache key for Brave Web searches.
        
        Args:
            query: Search query
            country: Country code
            search_lang: Search language
            ui_lang: UI language
            count: Number of results
            offset: Results offset
            safe_search: Safe search level
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a dictionary with all parameters
        key_dict = {
            "query": query.lower().strip(),
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
            "count": count,
            "offset": offset,
            "safe_search": safe_search
        }
        
        # Add any additional parameters
        key_dict.update(kwargs)
        
        # Generate the key using the base cache manager
        return self.web_cache_manager.get_cache_key("brave_web_search", **key_dict)
    
    def get_ai_cache_key(self, query: str, country: str = "us",
                         search_lang: str = "en", ui_lang: str = "en-US", **kwargs) -> str:
        """Generate a cache key for Brave AI searches.
        
        Args:
            query: Search query
            country: Country code
            search_lang: Search language
            ui_lang: UI language
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a dictionary with all parameters
        key_dict = {
            "query": query.lower().strip(),
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang
        }
        
        # Add any additional parameters
        key_dict.update(kwargs)
        
        # Generate the key using the base cache manager
        return self.ai_cache_manager.get_cache_key("brave_ai_search", **key_dict)
    
    def _is_news_query(self, query: str) -> bool:
        """Check if a query is likely a news search.
        
        Args:
            query: Search query
            
        Returns:
            True if the query is likely a news search, False otherwise
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.config["news_keywords"])
    
    def _is_factual_query(self, query: str) -> bool:
        """Check if a query is likely a factual search.
        
        Args:
            query: Search query
            
        Returns:
            True if the query is likely a factual search, False otherwise
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.config["factual_keywords"])
    
    async def get_web(self, key: str) -> Optional[Any]:
        """Get a value from the web search cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.config["enable_cache"]:
            return None
        
        return await self.web_cache_manager.get(key)
    
    async def get_ai(self, key: str) -> Optional[Any]:
        """Get a value from the AI search cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.config["enable_cache"]:
            return None
        
        return await self.ai_cache_manager.get(key)
    
    async def set_web(self, key: str, value: Any, query: Optional[str] = None, ttl: Optional[int] = None) -> None:
        """Set a value in the web search cache.
        
        Args:
            key: Cache key
            value: Value to cache
            query: Original search query (used for TTL determination)
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        if not self.config["enable_cache"]:
            return
        
        # Determine appropriate TTL based on query type and result popularity
        if ttl is None:
            if query and self._is_news_query(query):
                ttl = self.config["web_news_ttl"]
            elif value and isinstance(value, dict):
                # Check if it's a popular search
                total_results = 0
                if "web" in value and "results" in value["web"]:
                    total_results = value["web"].get("total", len(value["web"]["results"]))
                
                if total_results > 1000:
                    ttl = self.config["web_popular_search_ttl"]
                elif total_results < 10:
                    ttl = self.config["web_rare_search_ttl"]
                else:
                    ttl = self.config["web_default_ttl"]
            else:
                ttl = self.config["web_default_ttl"]
        
        await self.web_cache_manager.set(key, value, ttl)
    
    async def set_ai(self, key: str, value: Any, query: Optional[str] = None, ttl: Optional[int] = None) -> None:
        """Set a value in the AI search cache.
        
        Args:
            key: Cache key
            value: Value to cache
            query: Original search query (used for TTL determination)
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        if not self.config["enable_cache"]:
            return
        
        # Determine appropriate TTL based on query type
        if ttl is None:
            if query and self._is_factual_query(query):
                ttl = self.config["ai_factual_search_ttl"]
            elif value and isinstance(value, dict):
                # Check if it's a popular search based on sources or points
                source_count = 0
                if "generated_answer" in value and "sources" in value["generated_answer"]:
                    source_count = len(value["generated_answer"]["sources"])
                
                if source_count > 5:
                    ttl = self.config["ai_popular_search_ttl"]
                else:
                    ttl = self.config["ai_default_ttl"]
            else:
                ttl = self.config["ai_default_ttl"]
        
        await self.ai_cache_manager.set(key, value, ttl)
    
    async def invalidate_web(self, key: str) -> None:
        """Invalidate a specific web search cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        await self.web_cache_manager.invalidate(key)
    
    async def invalidate_ai(self, key: str) -> None:
        """Invalidate a specific AI search cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        await self.ai_cache_manager.invalidate(key)
    
    async def invalidate_web_search(self, query: str, country: str = "us", 
                                   search_lang: str = "en", ui_lang: str = "en-US",
                                   count: int = 10, offset: int = 0, 
                                   safe_search: str = "moderate", **kwargs) -> None:
        """Invalidate a specific web search result.
        
        Args:
            query: Search query
            country: Country code
            search_lang: Search language
            ui_lang: UI language
            count: Number of results
            offset: Results offset
            safe_search: Safe search level
            **kwargs: Additional parameters
        """
        key = self.get_web_cache_key(query, country, search_lang, ui_lang, count, offset, safe_search, **kwargs)
        await self.invalidate_web(key)
    
    async def invalidate_ai_search(self, query: str, country: str = "us",
                                  search_lang: str = "en", ui_lang: str = "en-US", **kwargs) -> None:
        """Invalidate a specific AI search result.
        
        Args:
            query: Search query
            country: Country code
            search_lang: Search language
            ui_lang: UI language
            **kwargs: Additional parameters
        """
        key = self.get_ai_cache_key(query, country, search_lang, ui_lang, **kwargs)
        await self.invalidate_ai(key)
    
    async def invalidate_all_web_searches(self) -> int:
        """Invalidate all web search results.
        
        Returns:
            Number of invalidated entries
        """
        return await self.web_cache_manager.invalidate_pattern("brave_web_search")
    
    async def invalidate_all_ai_searches(self) -> int:
        """Invalidate all AI search results.
        
        Returns:
            Number of invalidated entries
        """
        return await self.ai_cache_manager.invalidate_pattern("brave_ai_search")
    
    async def invalidate_by_keyword(self, keyword: str) -> int:
        """Invalidate all cache entries containing a specific keyword.
        
        Args:
            keyword: Keyword to match
            
        Returns:
            Number of invalidated entries
        """
        keyword_lower = keyword.lower().strip()
        web_count = await self.web_cache_manager.invalidate_pattern(keyword_lower)
        ai_count = await self.ai_cache_manager.invalidate_pattern(keyword_lower)
        return web_count + ai_count
    
    async def invalidate_expired_web(self) -> int:
        """Invalidate all expired web search cache entries.
        
        Returns:
            Number of invalidated entries
        """
        return await self.web_cache_manager.invalidate_expired()
    
    async def invalidate_expired_ai(self) -> int:
        """Invalidate all expired AI search cache entries.
        
        Returns:
            Number of invalidated entries
        """
        return await self.ai_cache_manager.invalidate_expired()
    
    async def invalidate_all_web(self) -> int:
        """Invalidate all web search cache entries.
        
        Returns:
            Number of invalidated entries
        """
        return await self.web_cache_manager.invalidate_all()
    
    async def invalidate_all_ai(self) -> int:
        """Invalidate all AI search cache entries.
        
        Returns:
            Number of invalidated entries
        """
        return await self.ai_cache_manager.invalidate_all()
    
    async def invalidate_all(self) -> int:
        """Invalidate all cache entries (both web and AI).
        
        Returns:
            Number of invalidated entries
        """
        web_count = await self.invalidate_all_web()
        ai_count = await self.invalidate_all_ai()
        return web_count + ai_count
    
    def get_web_stats(self) -> Dict[str, Any]:
        """Get web search cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.web_cache_manager.get_stats()
        stats["config"] = {k: v for k, v in self.config.items() if k.startswith("web_") or not k.startswith(("ai_", "web_"))}
        return stats
    
    def get_ai_stats(self) -> Dict[str, Any]:
        """Get AI search cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.ai_cache_manager.get_stats()
        stats["config"] = {k: v for k, v in self.config.items() if k.startswith("ai_") or not k.startswith(("ai_", "web_"))}
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        web_stats = self.get_web_stats()
        ai_stats = self.get_ai_stats()
        
        combined_stats = {
            "web": web_stats,
            "ai": ai_stats,
            "config": self.config,
            "total_memory_hits": web_stats.get("memory_hits", 0) + ai_stats.get("memory_hits", 0),
            "total_memory_misses": web_stats.get("memory_misses", 0) + ai_stats.get("memory_misses", 0),
            "total_disk_hits": web_stats.get("disk_hits", 0) + ai_stats.get("disk_hits", 0),
            "total_disk_misses": web_stats.get("disk_misses", 0) + ai_stats.get("disk_misses", 0),
            "total_sets": web_stats.get("sets", 0) + ai_stats.get("sets", 0),
            "total_invalidations": web_stats.get("invalidations", 0) + ai_stats.get("invalidations", 0)
        }
        
        # Calculate combined hit rates
        total_memory = combined_stats["total_memory_hits"] + combined_stats["total_memory_misses"]
        total_disk = combined_stats["total_disk_hits"] + combined_stats["total_disk_misses"]
        
        combined_stats["total_memory_hit_rate"] = (combined_stats["total_memory_hits"] / total_memory * 100) if total_memory > 0 else 0
        combined_stats["total_disk_hit_rate"] = (combined_stats["total_disk_hits"] / total_disk * 100) if total_disk > 0 else 0
        
        return combined_stats
    
    def reset_web_stats(self) -> None:
        """Reset web search cache statistics."""
        self.web_cache_manager.reset_stats()
    
    def reset_ai_stats(self) -> None:
        """Reset AI search cache statistics."""
        self.ai_cache_manager.reset_stats()
    
    def reset_stats(self) -> None:
        """Reset all cache statistics."""
        self.reset_web_stats()
        self.reset_ai_stats()
    
    async def close(self) -> None:
        """Close the cache managers and release resources."""
        await self.web_cache_manager.close()
        await self.ai_cache_manager.close()

# Singleton instance
_brave_search_cache = None

def get_brave_search_cache(config: Optional[Dict[str, Any]] = None) -> BraveSearchCache:
    """Get or create the Brave Search cache manager.
    
    Args:
        config: Configuration dictionary. If None, uses default configuration.
        
    Returns:
        BraveSearchCache instance
    """
    global _brave_search_cache
    
    if _brave_search_cache is None:
        _brave_search_cache = BraveSearchCache(config=config)
    
    return _brave_search_cache

async def close_brave_search_cache() -> None:
    """Close the Brave Search cache manager."""
    global _brave_search_cache
    
    if _brave_search_cache is not None:
        await _brave_search_cache.close()
        _brave_search_cache = None

async def invalidate_cache_entry(key: str) -> bool:
    """Invalidate a specific cache entry by its key.
    
    Args:
        key: The cache key to invalidate
        
    Returns:
        True if the entry was found and invalidated, False otherwise
    """
    cache = get_brave_search_cache()
    
    # Try to invalidate in both web and AI caches
    web_result = await cache.web_cache_manager.invalidate(key)
    ai_result = await cache.ai_cache_manager.invalidate(key)
    
    if web_result or ai_result:
        logger.info(f"Invalidated cache entry with key: {key}")
        return True
    else:
        logger.warning(f"Cache entry with key: {key} not found")
        return False

async def invalidate_cache_by_query(query: str) -> bool:
    """Invalidate all cache entries that match a query.
    
    This uses a fuzzy matching approach to find and invalidate all entries
    that might be related to the given query.
    
    Args:
        query: The query to match against
        
    Returns:
        True if any entries were invalidated, False otherwise
    """
    cache = get_brave_search_cache()
    query = query.lower().strip()
    
    # Get all keys from both caches
    web_keys = await cache.web_cache_manager.get_all_keys()
    ai_keys = await cache.ai_cache_manager.get_all_keys()
    
    # Find keys that contain the query
    web_matches = [k for k in web_keys if query in k.lower()]
    ai_matches = [k for k in ai_keys if query in k.lower()]
    
    # Invalidate matching entries
    invalidated = False
    
    for key in web_matches:
        if await cache.web_cache_manager.invalidate(key):
            invalidated = True
            logger.info(f"Invalidated web cache entry with key: {key}")
    
    for key in ai_matches:
        if await cache.ai_cache_manager.invalidate(key):
            invalidated = True
            logger.info(f"Invalidated AI cache entry with key: {key}")
    
    if invalidated:
        logger.info(f"Invalidated {len(web_matches) + len(ai_matches)} cache entries matching query: '{query}'")
    else:
        logger.warning(f"No cache entries found matching query: '{query}'")
    
    return invalidated

async def invalidate_cache_by_url(url: str) -> bool:
    """Invalidate all cache entries that contain a specific URL.
    
    Args:
        url: The URL to match against
        
    Returns:
        True if any entries were invalidated, False otherwise
    """
    cache = get_brave_search_cache()
    url = url.lower().strip()
    
    # Get all entries from both caches
    web_entries = await cache.web_cache_manager.get_all_entries()
    ai_entries = await cache.ai_cache_manager.get_all_entries()
    
    # Find entries that contain the URL in their value
    web_matches = []
    ai_matches = []
    
    for key, entry in web_entries.items():
        value = entry.get('value', '')
        if isinstance(value, dict) or isinstance(value, list):
            value_str = json.dumps(value).lower()
        else:
            value_str = str(value).lower()
        
        if url in value_str:
            web_matches.append(key)
    
    for key, entry in ai_entries.items():
        value = entry.get('value', '')
        if isinstance(value, dict) or isinstance(value, list):
            value_str = json.dumps(value).lower()
        else:
            value_str = str(value).lower()
        
        if url in value_str:
            ai_matches.append(key)
    
    # Invalidate matching entries
    invalidated = False
    
    for key in web_matches:
        if await cache.web_cache_manager.invalidate(key):
            invalidated = True
            logger.info(f"Invalidated web cache entry containing URL '{url}' with key: {key}")
    
    for key in ai_matches:
        if await cache.ai_cache_manager.invalidate(key):
            invalidated = True
            logger.info(f"Invalidated AI cache entry containing URL '{url}' with key: {key}")
    
    if invalidated:
        logger.info(f"Invalidated {len(web_matches) + len(ai_matches)} cache entries containing URL: '{url}'")
    else:
        logger.warning(f"No cache entries found containing URL: '{url}'")
    
    return invalidated
