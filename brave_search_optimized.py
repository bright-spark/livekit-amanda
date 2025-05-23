"""
Optimized Brave Search API client with caching and rate limiting.
This module provides a more efficient way to use the Brave Search API.
"""

import logging
import aiohttp
import json
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlencode

class BraveSearchCache:
    """Cache for Brave Search API results to reduce API calls."""
    
    def __init__(self, cache_ttl: int = 3600):
        """Initialize the cache.
        
        Args:
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.hit_count = 0
        self.miss_count = 0
    
    def get_cache_key(self, query: str, **params) -> str:
        """Generate a cache key from the query and parameters.
        
        Args:
            query: The search query
            params: Additional search parameters
            
        Returns:
            A unique cache key string
        """
        # Create a string representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        
        # Create a hash of the query and parameters
        key = f"{query}:{param_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from the cache if it exists and is not expired.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                self.hit_count += 1
                return entry['data']
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        self.cache[key] = {
            'data': value,
            'expires': time.time() + self.cache_ttl
        }
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class RateLimiter:
    """Simple rate limiter to prevent exceeding API rate limits."""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = []
    
    async def wait_if_needed(self) -> None:
        """Wait if necessary to comply with rate limits.
        
        This method will block until it's safe to make another request.
        """
        current_time = time.time()
        
        # Remove request times that are outside the time window
        self.request_times = [t for t in self.request_times if current_time - t < self.time_window]
        
        # If we've reached the maximum number of requests in the time window, wait
        if len(self.request_times) >= self.max_requests:
            oldest_request = min(self.request_times)
            wait_time = oldest_request + self.time_window - current_time
            if wait_time > 0:
                logging.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add the current request time
        self.request_times.append(time.time())

class OptimizedBraveSearchClient:
    """Optimized client for interacting with Brave Search API with caching and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 3600, 
                 max_requests: int = 10, time_window: int = 60):
        """Initialize the optimized Brave Search client.
        
        Args:
            api_key: Brave Search API key. If not provided, will try to get from environment variable.
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds for rate limiting
        """
        self.api_key = api_key or os.environ.get("BRAVE_WEB_SEARCH_API_KEY")
        if not self.api_key:
            logging.warning("No Brave API key provided. Please set BRAVE_WEB_SEARCH_API_KEY environment variable or pass it to the constructor.")
        
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        # Initialize cache and rate limiter
        self.cache = BraveSearchCache(cache_ttl=cache_ttl)
        self.rate_limiter = RateLimiter(max_requests=max_requests, time_window=time_window)
    
    async def search(self, 
                    query: str, 
                    country: str = "us", 
                    search_lang: str = "en", 
                    ui_lang: str = "en-US", 
                    count: int = 10, 
                    offset: int = 0,
                    safe_search: str = "moderate",
                    use_cache: bool = True) -> Dict[str, Any]:
        """Search the web using Brave Search API with caching and rate limiting.
        
        Args:
            query: Search query
            country: Country code for search results
            search_lang: Language code for search results
            ui_lang: UI language code
            count: Number of results to return (max 20)
            offset: Offset for pagination
            safe_search: Safe search level (strict, moderate, off)
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            Dict containing the search results
        """
        params = {
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
            "count": min(count, 20),  # Brave API has a max of 20 results per request
            "offset": offset,
            "safesearch": safe_search
        }
        
        # Generate cache key
        cache_key = self.cache.get_cache_key(query, **params)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logging.info(f"Cache hit for query: {query}")
                return cached_result
        
        # Wait if necessary to comply with rate limits
        await self.rate_limiter.wait_if_needed()
        
        # Make the API request
        try:
            async with aiohttp.ClientSession() as session:
                full_params = params.copy()
                full_params["q"] = query
                
                async with session.get(self.base_url, params=full_params, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"Brave Search API error: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status}", "details": error_text}
                    
                    result = await response.json()
                    
                    # Cache the result if use_cache is True
                    if use_cache:
                        self.cache.set(cache_key, result)
                    
                    return result
        except Exception as e:
            logging.error(f"Error during Brave search: {str(e)}")
            return {"error": str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        logging.info("Brave Search cache cleared")

# Create a singleton instance
_optimized_brave_search_client = None

def get_optimized_brave_search_client(api_key: Optional[str] = None) -> OptimizedBraveSearchClient:
    """Get or create a singleton instance of the OptimizedBraveSearchClient.
    
    Args:
        api_key: Optional API key for Brave Search
        
    Returns:
        OptimizedBraveSearchClient instance
    """
    global _optimized_brave_search_client
    if _optimized_brave_search_client is None:
        _optimized_brave_search_client = OptimizedBraveSearchClient(api_key=api_key)
    return _optimized_brave_search_client
