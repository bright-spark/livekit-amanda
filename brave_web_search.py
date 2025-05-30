"""
Brave Web Search API implementation.

This module provides web search functionality using the Brave Search API with:
1. Configurable caching
2. Configurable rate limiting
3. Statistics tracking
4. Web-specific response formatting
"""

import logging
import asyncio
import json
import os
import time
import hashlib
import pickle
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from urllib.parse import urlencode
import aiohttp
from dotenv import load_dotenv

# Import statistics tracking module
try:
    from brave_search_stats import record_request, get_stats, get_stats_report
    HAS_STATS_TRACKING = True
    logging.info("Brave Search statistics tracking enabled")
except ImportError:
    HAS_STATS_TRACKING = False
    logging.warning("Brave Search statistics tracking not available")
    
    # Define dummy functions for when stats module is not available
    def record_request(*args, **kwargs):
        pass
        
    def get_stats():
        return None
        
    def get_stats_report():
        return "Statistics tracking not available"

# Load environment variables
load_dotenv()

# Get configuration from environment variables
ENABLE_CACHE = os.environ.get("BRAVE_SEARCH_ENABLE_CACHE", "true").lower() == "true"
ENABLE_PERSISTENCE = os.environ.get("BRAVE_SEARCH_ENABLE_PERSISTENCE", "true").lower() == "true"
# Rate limit: 1 for free tier, 20 for paid tier
WEB_RATE_LIMIT = int(os.environ.get("BRAVE_WEB_SEARCH_RATE_LIMIT", "1").split('#')[0].strip())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class FreeTierCache:
    """Aggressive cache for Brave Search API optimized for free tier usage."""
    
    def __init__(self, 
                 cache_ttl: int = 604800,  # 1 week by default
                 memory_cache_size: int = 1000,
                 disk_cache_dir: Optional[str] = None):
        """Initialize the cache with aggressive settings for free tier.
        
        Args:
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 week)
            memory_cache_size: Maximum number of items to keep in memory cache
            disk_cache_dir: Directory to store persistent cache files
        """
        self.memory_cache = {}
        self.cache_ttl = cache_ttl
        self.memory_cache_size = memory_cache_size
        self.hit_count = 0
        self.miss_count = 0
        self.memory_hit_count = 0
        self.disk_hit_count = 0
        
        # LRU tracking
        self.access_times = {}
        
        # Setup disk cache
        if disk_cache_dir:
            self.disk_cache_dir = Path(disk_cache_dir)
        else:
            self.disk_cache_dir = Path(os.path.expanduser("~")) / ".brave_search_cache"
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        logging.info(f"Disk cache enabled at: {self.disk_cache_dir}")
        
        # Load existing cache stats if available
        self._load_stats()
    
    def _get_normalized_query(self, query: str) -> str:
        """Normalize a query for better cache hit rates.
        
        Args:
            query: Original query string
            
        Returns:
            Normalized query string
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = " ".join(query.split())
        
        # Remove common filler words that don't affect search results
        filler_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
                       "for", "with", "by", "about", "like", "of", "from"}
        
        words = query.split()
        if len(words) > 3:  # Only remove filler words if the query is long enough
            words = [word for word in words if word not in filler_words]
            query = " ".join(words)
        
        return query
    
    def get_cache_key(self, query: str, **params) -> str:
        """Generate a cache key from the query and parameters.
        
        Args:
            query: The search query
            params: Additional search parameters
            
        Returns:
            A unique cache key string
        """
        # Normalize the query for better cache hit rates
        normalized_query = self._get_normalized_query(query)
        
        # Create a string representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        
        # Create a hash of the query and parameters
        key = f"{normalized_query}:{param_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_disk_cache_path(self, key: str) -> Path:
        """Get the path to the disk cache file for a key.
        
        Args:
            key: The cache key
            
        Returns:
            Path to the cache file
        """
        # Use the first few characters of the key for directory sharding
        # This prevents too many files in a single directory
        shard = key[:2]
        shard_dir = self.disk_cache_dir / shard
        os.makedirs(shard_dir, exist_ok=True)
        return shard_dir / f"{key}.gz"
    
    def _save_to_disk(self, key: str, entry: Dict[str, Any]) -> bool:
        """Save a cache entry to disk.
        
        Args:
            key: The cache key
            entry: The cache entry to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_path = self._get_disk_cache_path(key)
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
            return True
        except Exception as e:
            logging.warning(f"Failed to save cache entry to disk: {e}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a cache entry from disk.
        
        Args:
            key: The cache key
            
        Returns:
            The cache entry or None if not found or expired
        """
        try:
            cache_path = self._get_disk_cache_path(key)
            if not cache_path.exists():
                return None
                
            with gzip.open(cache_path, 'rb') as f:
                entry = pickle.load(f)
                
            # Check if expired
            if time.time() >= entry['expires']:
                # Remove expired file
                cache_path.unlink(missing_ok=True)
                return None
                
            return entry
        except Exception as e:
            logging.warning(f"Failed to load cache entry from disk: {e}")
            return None
    
    def _manage_memory_cache_size(self) -> None:
        """Ensure memory cache doesn't exceed the maximum size by removing least recently used items."""
        if len(self.memory_cache) <= self.memory_cache_size:
            return
            
        # Sort by access time (oldest first)
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Remove oldest entries until we're under the limit
        entries_to_remove = len(self.memory_cache) - self.memory_cache_size
        for i in range(entries_to_remove):
            if i < len(sorted_keys):
                key_to_remove = sorted_keys[i][0]
                if key_to_remove in self.memory_cache:
                    # Save to disk before removing from memory
                    self._save_to_disk(key_to_remove, self.memory_cache[key_to_remove])
                    del self.memory_cache[key_to_remove]
                    del self.access_times[key_to_remove]
    
    def _save_stats(self) -> None:
        """Save cache statistics to disk."""
        try:
            stats_path = self.disk_cache_dir / "cache_stats.json"
            stats = {
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "memory_hit_count": self.memory_hit_count,
                "disk_hit_count": self.disk_hit_count,
                "last_updated": time.time()
            }
            with open(stats_path, 'w') as f:
                json.dump(stats, f)
        except Exception as e:
            logging.warning(f"Failed to save cache stats: {e}")
    
    def _load_stats(self) -> None:
        """Load cache statistics from disk."""
        try:
            stats_path = self.disk_cache_dir / "cache_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    self.hit_count = stats.get("hit_count", 0)
                    self.miss_count = stats.get("miss_count", 0)
                    self.memory_hit_count = stats.get("memory_hit_count", 0)
                    self.disk_hit_count = stats.get("disk_hit_count", 0)
                    logging.info(f"Loaded cache stats: {stats}")
        except Exception as e:
            logging.warning(f"Failed to load cache stats: {e}")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from the cache if it exists and is not expired.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        # First check memory cache (fastest)
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() < entry['expires']:
                # Update access time for LRU tracking
                self.access_times[key] = time.time()
                self.hit_count += 1
                self.memory_hit_count += 1
                
                # Periodically save stats (every 10 hits)
                if self.hit_count % 10 == 0:
                    self._save_stats()
                    
                return entry['data']
            else:
                # Remove expired entry
                del self.memory_cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        # If not in memory or expired, check disk cache
        disk_entry = self._load_from_disk(key)
        if disk_entry and time.time() < disk_entry['expires']:
            # Found valid entry in disk cache, add to memory cache
            self.memory_cache[key] = disk_entry
            self.access_times[key] = time.time()
            self._manage_memory_cache_size()
            self.hit_count += 1
            self.disk_hit_count += 1
            
            # Periodically save stats (every 10 hits)
            if self.hit_count % 10 == 0:
                self._save_stats()
                
            return disk_entry['data']
        
        self.miss_count += 1
        
        # Periodically save stats (every 10 misses)
        if self.miss_count % 10 == 0:
            self._save_stats()
            
        return None
    
    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        entry = {
            'data': value,
            'expires': time.time() + self.cache_ttl
        }
        
        # Add to memory cache
        self.memory_cache[key] = entry
        self.access_times[key] = time.time()
        
        # Ensure memory cache doesn't exceed size limit
        self._manage_memory_cache_size()
        
        # Also save to disk cache
        # Use asyncio to avoid blocking
        asyncio.create_task(self._async_save_to_disk(key, entry))
    
    async def _async_save_to_disk(self, key: str, entry: Dict[str, Any]) -> None:
        """Asynchronously save a cache entry to disk."""
        try:
            cache_path = self._get_disk_cache_path(key)
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logging.warning(f"Failed to save cache entry to disk: {e}")
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.memory_cache = {}
        self.access_times = {}
        
        # Clear disk cache
        try:
            import shutil
            shutil.rmtree(self.disk_cache_dir)
            os.makedirs(self.disk_cache_dir, exist_ok=True)
            logging.info(f"Disk cache cleared: {self.disk_cache_dir}")
            
            # Reset stats
            self.hit_count = 0
            self.miss_count = 0
            self.memory_hit_count = 0
            self.disk_hit_count = 0
            self._save_stats()
        except Exception as e:
            logging.error(f"Failed to clear disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        # Count disk cache size
        disk_cache_size = 0
        disk_cache_bytes = 0
        try:
            for root, dirs, files in os.walk(self.disk_cache_dir):
                for file in files:
                    if file.endswith('.gz'):
                        disk_cache_size += 1
                        file_path = os.path.join(root, file)
                        disk_cache_bytes += os.path.getsize(file_path)
        except Exception as e:
            logging.warning(f"Failed to calculate disk cache size: {e}")
        
        stats = {
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_size': disk_cache_size,
            'disk_cache_bytes': disk_cache_bytes,
            'hits': self.hit_count,
            'memory_hits': self.memory_hit_count,
            'disk_hits': self.disk_hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
        
        return stats

class WebSearchRateLimiter:
    """Configurable rate limiter for Brave Web Search API.
    
    Can be configured for free tier (1 request per second) or paid tier (20 requests per second).
    """
    
    def __init__(self, requests_per_second: int = 1):
        """Initialize the rate limiter with configurable rate.
        
        Args:
            requests_per_second: Number of requests allowed per second (1 for free tier, 20 for paid tier)
        """
        self.last_request_time = 0
        self.min_interval = 1.0 / requests_per_second  # Convert to interval
    
    async def wait_if_needed(self) -> None:
        """Wait if necessary to comply with the configured rate limit.
        
        This method will block until it's safe to make another request.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_interval:
            wait_time = self.min_interval - time_since_last_request
            logging.info(f"Rate limiting: waiting for {wait_time:.2f} seconds")
            
            # Record rate limiting statistics
            if HAS_STATS_TRACKING:
                record_request(
                    query="",  # No specific query for this record
                    response_time=0.0,  # Not applicable
                    search_type="web",  # Web search
                    rate_limited=True,
                    delay_time=wait_time
                )
                
            await asyncio.sleep(wait_time)
        
        # Update the last request time after waiting
        self.last_request_time = time.time()

class BraveWebSearch:
    """Brave Web Search API client with configurable optimizations."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 enable_cache: Optional[bool] = None,
                 enable_persistence: Optional[bool] = None,
                 rate_limit: Optional[int] = None):
        """Initialize the Brave Search client with configurable optimizations.
        
        Args:
            api_key: Brave Search API key. If not provided, will try to get from environment variable.
            enable_cache: Whether to enable caching. If None, uses the BRAVE_SEARCH_ENABLE_CACHE env var.
            enable_persistence: Whether to enable persistent disk caching. If None, uses the BRAVE_SEARCH_ENABLE_PERSISTENCE env var.
            rate_limit: Requests per second (1 for free tier, 20 for paid tier). If None, uses the BRAVE_SEARCH_RATE_LIMIT env var.
        """
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get("BRAVE_WEB_SEARCH_API_KEY")
        if not self.api_key:
            logging.warning("Brave Web Search API key not provided and not found in environment variables")
        
        # API configuration
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        # Use provided values or fall back to environment variables
        self.enable_cache = enable_cache if enable_cache is not None else ENABLE_CACHE
        self.enable_persistence = enable_persistence if enable_persistence is not None else ENABLE_PERSISTENCE
        self.rate_limit_value = rate_limit if rate_limit is not None else WEB_RATE_LIMIT
        
        logging.info(f"Brave Web Search API configuration: cache={self.enable_cache}, persistence={self.enable_persistence}, rate_limit={self.rate_limit_value}")
        
        # Initialize cache if enabled
        if self.enable_cache:
            self.cache = FreeTierCache(
                cache_ttl=604800,  # 1 week cache TTL
                memory_cache_size=1000,
                disk_cache_dir=None if not self.enable_persistence else None  # Use default if persistence enabled
            )
            logging.info("Brave Web Search cache enabled")
        else:
            self.cache = None
            logging.info("Brave Web Search cache disabled")
        
        # Initialize rate limiter
        self.rate_limiter = WebSearchRateLimiter(requests_per_second=self.rate_limit_value)
        logging.info(f"Brave Web Search rate limiter configured for {self.rate_limit_value} requests per second")
        
        # Connection pool for reusing connections
        self.session = None
        self.session_lock = asyncio.Lock()
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession.
        
        Returns:
            aiohttp.ClientSession instance
        """
        async with self.session_lock:
            if self.session is None or self.session.closed:
                # Configure the session with connection pooling and keepalive
                conn = aiohttp.TCPConnector(
                    limit=5,  # Small connection pool size for free tier
                    ttl_dns_cache=300,  # DNS cache TTL in seconds
                    keepalive_timeout=60  # Keepalive timeout
                )
                
                timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
                
                self.session = aiohttp.ClientSession(
                    connector=conn,
                    timeout=timeout,
                    headers=self.headers
                )
            
            return self.session
    
    async def search(self, 
                    query: str, 
                    country: str = "us", 
                    search_lang: str = "en", 
                    ui_lang: str = "en-US", 
                    count: int = 10, 
                    offset: int = 0,
                    safe_search: str = "moderate",
                    use_cache: bool = True) -> Dict[str, Any]:
        """Search the web using Brave Web Search API with configurable optimizations.
        
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
        # Initialize statistics tracking variables
        start_time = time.time()
        cache_hit = False
        error = False
        rate_limited = False
        delay_time = 0.0
        status_code = 200
        result_count = 0
        # Normalize the query to improve cache hit rate
        normalized_query = " ".join(query.lower().split())
        
        params = {
            "q": normalized_query,
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
            "count": min(count, 20),  # Brave API has a max of 20 results per request
            "offset": offset,
            "safesearch": safe_search
        }
        
        # Generate cache key
        cache_key = self.cache.get_cache_key(normalized_query, **{k: v for k, v in params.items() if k != "q"})
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                cache_hit = True
                logging.info(f"Cache hit for query: {normalized_query}")
                
                # Record cache hit statistics
                if HAS_STATS_TRACKING:
                    # Count results if available
                    if "web" in cached_result and "results" in cached_result["web"]:
                        result_count = len(cached_result["web"]["results"])
                    
                    response_time = time.time() - start_time
                    record_request(
                        query=normalized_query,
                        response_time=response_time,
                        search_type="web",  # Web search
                        cache_hit=True,
                        result_count=result_count
                    )
                
                return cached_result
        
        # Wait if necessary to comply with rate limits
        rate_limit_start = time.time()
        await self.rate_limiter.wait_if_needed()
        rate_limited = (time.time() - rate_limit_start) > 0.01  # If we waited more than 10ms
        delay_time = time.time() - rate_limit_start if rate_limited else 0
        
        # Make the API request
        try:
            session = await self.get_session()
            
            async with session.get(self.base_url, params=params) as response:
                status_code = response.status
                
                if response.status == 429:
                    # Rate limit hit
                    rate_limited = True
                    error = True
                    error_msg = await response.text()
                    logging.error(f"Rate limit exceeded: {error_msg}")
                    
                    # Record rate limit error statistics
                    if HAS_STATS_TRACKING:
                        response_time = time.time() - start_time
                        record_request(
                            query=normalized_query,
                            response_time=response_time,
                            search_type="web",  # Web search
                            cache_hit=False,
                            error=True,
                            rate_limited=True,
                            delay_time=delay_time,
                            status_code=429,
                            result_count=0
                        )
                    
                    return {"error": "Rate limit exceeded", "details": error_msg}
                elif response.status != 200:
                    error = True
                    error_text = await response.text()
                    logging.error(f"Brave Search API error: {response.status} - {error_text}")
                    
                    # Record API error statistics
                    if HAS_STATS_TRACKING:
                        response_time = time.time() - start_time
                        record_request(
                            query=normalized_query,
                            response_time=response_time,
                            search_type="web",  # Web search
                            cache_hit=False,
                            error=True,
                            rate_limited=rate_limited,
                            delay_time=delay_time,
                            status_code=response.status,
                            result_count=0
                        )
                    
                    return {"error": f"API error: {response.status}", "details": error_text}
                
                result = await response.json()
                
                # Count results if available
                if "web" in result and "results" in result["web"]:
                    result_count = len(result["web"]["results"])
                
                # Record successful API request statistics
                if HAS_STATS_TRACKING:
                    response_time = time.time() - start_time
                    record_request(
                        query=normalized_query,
                        response_time=response_time,
                        search_type="web",  # Web search
                        cache_hit=False,
                        error=False,
                        rate_limited=rate_limited,
                        delay_time=delay_time,
                        status_code=status_code,
                        result_count=result_count
                    )
                
                # Cache the result if use_cache is True
                if use_cache:
                    await self.cache.set(cache_key, result)
                
                return result
        except asyncio.TimeoutError:
            error = True
            logging.error("Request timed out")
            
            # Record timeout error statistics
            if HAS_STATS_TRACKING:
                response_time = time.time() - start_time
                record_request(
                    query=normalized_query,
                    response_time=response_time,
                    search_type="web",  # Web search
                    cache_hit=False,
                    error=True,
                    rate_limited=rate_limited,
                    delay_time=delay_time,
                    status_code=0,  # Timeout
                    result_count=0
                )
            
            return {"error": "Request timed out"}
        except Exception as e:
            error = True
            error_msg = str(e)
            logging.error(f"Error during Brave search: {error_msg}")
            
            # Record general error statistics
            if HAS_STATS_TRACKING:
                response_time = time.time() - start_time
                record_request(
                    query=normalized_query,
                    response_time=response_time,
                    search_type="web",  # Web search
                    cache_hit=False,
                    error=True,
                    rate_limited=rate_limited,
                    delay_time=delay_time,
                    status_code=0,  # General error
                    result_count=0
                )
            
            return {"error": error_msg}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
    
    async def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        logging.info("Brave Search cache cleared")
    
    async def close(self) -> None:
        """Close the client and release resources."""
        async with self.session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

# Create a singleton instance
_brave_web_search = None
_client_lock = asyncio.Lock()

async def get_brave_web_search_client(api_key: Optional[str] = None) -> BraveWebSearch:
    """Get or create a singleton instance of the BraveWebSearch.
    
    Args:
        api_key: Optional API key for Brave Web Search
        
    Returns:
        BraveWebSearch instance
    """
    global _brave_web_search
    
    async with _client_lock:
        if _brave_web_search is None:
            _brave_web_search = BraveWebSearch(api_key=api_key)
    
    return _brave_web_search

def format_search_results(results: Dict[str, Any], num_results: int = 5) -> str:
    """Format search results into a readable string with grounding information.
    
    Args:
        results: Search results from the Brave Search API
        num_results: Maximum number of results to include
        
    Returns:
        Formatted string of search results with timestamp and source information
    """
    if "error" in results:
        return f"Search error: {results['error']}"
    
    # Extract query from results if available
    query = results.get("query", {}).get("query", "your search")
    
    if "web" not in results or "results" not in results["web"] or not results["web"]["results"]:
        return f"No search results found for '{query}'."
    
    # Get current timestamp for grounding
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract search metadata for grounding
    web_results = results["web"]["results"]
    total_count = results.get("web", {}).get("totalCount", 0)
    search_timestamp = results.get("timestamp", current_time)
    search_location = results.get("location", "unknown")
    
    # Create a comprehensive grounding header
    formatted = f"""[SEARCH GROUNDING INFORMATION]
- Query: '{query}'
- Results retrieved: {current_time}
- Total results found: {total_count:,}
- Search API: Brave Search

Here are the top {min(len(web_results), num_results)} results for '{query}':\n\n"""
    
    # Add a warning about time-sensitive information if needed
    formatted += "NOTE: These results may contain time-sensitive information. Consider the publication dates when evaluating currency of information.\n\n"
    
    # Extract and include featured snippets if available (these often contain the most accurate, time-sensitive info)
    if "featured_snippet" in results:
        snippet = results["featured_snippet"]
        if snippet:
            snippet_title = snippet.get("title", "Featured Information")
            snippet_description = snippet.get("description", "")
            snippet_url = snippet.get("url", "")
            snippet_source = snippet.get("source", "")
            
            formatted += f"[FEATURED SNIPPET - Likely most accurate and recent information]\n"
            formatted += f"Title: {snippet_title}\n"
            if snippet_source:
                formatted += f"Source: {snippet_source}\n"
            if snippet_url:
                formatted += f"URL: {snippet_url}\n"
            if snippet_description:
                formatted += f"Information: {snippet_description}\n\n"
    
    # Extract and include knowledge graph information if available
    if "knowledge_graph" in results and results["knowledge_graph"]:
        kg = results["knowledge_graph"]
        kg_title = kg.get("title", "")
        kg_description = kg.get("description", "")
        kg_type = kg.get("type", "")
        
        if kg_title or kg_description:
            formatted += f"[KNOWLEDGE PANEL - Verified Information]\n"
            if kg_title:
                formatted += f"Entity: {kg_title}"
                if kg_type:
                    formatted += f" ({kg_type})"
                formatted += "\n"
            if kg_description:
                formatted += f"Description: {kg_description}\n\n"
            
            # Include entity attributes if available
            if "attributes" in kg and kg["attributes"]:
                formatted += "Key facts:\n"
                for attr in kg["attributes"]:
                    name = attr.get("name", "")
                    value = attr.get("value", "")
                    if name and value:
                        formatted += f"- {name}: {value}\n"
                formatted += "\n"
    
    for idx, result in enumerate(web_results[:num_results], 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        description = result.get("description", "No description")
        
        # Extract and format date information if available
        age = result.get("age", "")
        date_info = ""
        if age:
            date_info = f" [Published: {age}]"
        
        # Extract and format domain information for source credibility
        domain = result.get("domain", "")
        domain_info = f" [Source: {domain}]" if domain else ""
        
        formatted += f"{idx}. {title}{date_info}{domain_info}\n   {url}\n"
        if description:
            formatted += f"   {description}\n"
        formatted += "\n"
    
    # Extract and include news results if available (critical for time-sensitive information)
    if "news" in results and "results" in results["news"] and results["news"]["results"]:
        news_results = results["news"]["results"]
        if news_results:
            formatted += "\n[RECENT NEWS RESULTS - Time-sensitive information]\n"
            for i, news in enumerate(news_results[:3], 1):  # Limit to top 3 news items
                news_title = news.get("title", "")
                news_url = news.get("url", "")
                news_description = news.get("description", "")
                news_age = news.get("age", "")
                news_source = news.get("source", "")
                
                formatted += f"News {i}: {news_title}"
                if news_age:
                    formatted += f" [Published: {news_age}]"
                if news_source:
                    formatted += f" [Source: {news_source}]"
                formatted += f"\n   {news_url}\n"
                if news_description:
                    formatted += f"   {news_description}\n"
                formatted += "\n"
    
    # Add a timestamp footer for grounding
    formatted += f"\n[End of search results. Retrieved at {current_time}]\n"
    formatted += "Remember to consider the recency of information when answering time-sensitive questions."
    
    return formatted
_session_cache = {}

async def web_search(context, query, num_results: int = 5) -> str:
    """Perform a web search using the Brave Search API with configurable optimizations.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Formatted search results as a string
    """
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query)
    start_time = time.time()
    logging.info(f"[TOOL] brave_web_search called for query: {query}, num_results: {num_results}")
    
    # Log statistics if available
    if HAS_STATS_TRACKING:
        stats = get_stats()
        if stats:
            session_stats = stats.get_session_stats()
            web_stats = stats.get_performance_stats("web")
            logging.info(f"[STATS] Web Search requests: {web_stats.get('total_requests', 0)}, Cache hit rate: {web_stats.get('cache_hit_rate', 0):.2f}%")
    
    try:
        # Get the Brave Web Search client
        client = await get_brave_web_search_client()
        
        # Check session cache first if caching is enabled
        if client.enable_cache:
            cache_key = f"session:{query}:{num_results}"
            if cache_key in _session_cache:
                logging.info(f"Session cache hit for query: {query}")
                formatted_results = _session_cache[cache_key]
                # Log performance
                elapsed = time.time() - start_time
                logging.info(f"[PERFORMANCE] brave_web_search completed in {elapsed:.4f}s (session cache hit)")
                return formatted_results
        
        # Get the Brave Search client (use the one we already created)
        # No need to create a new client, we already have one
        
        # Perform the search with caching
        results = await client.search(
            query=query,
            count=min(num_results + 2, 20),  # Request a few extra results in case some are filtered
        )
        
        if "error" in results:
            error_msg = f"Error searching with Brave API: {results['error']}"
            logging.error(error_msg)
            return f"I couldn't find any results for '{query}'. Try a different query."
        
        # Format the results
        formatted_results = format_search_results(results, num_results)
        
        # Store in session cache if caching is enabled
        if client.enable_cache:
            cache_key = f"session:{query}:{num_results}"
            _session_cache[cache_key] = formatted_results
        
        # Limit session cache size
        if len(_session_cache) > 100:
            # Remove oldest entries
            keys = list(_session_cache.keys())
            for old_key in keys[:20]:  # Remove 20 oldest entries
                if old_key in _session_cache:
                    del _session_cache[old_key]
        
        # Add explicit grounding instructions for the LLM
        if formatted_results:
            # Add a special grounding header that instructs the LLM to use this information
            grounding_header = (
                """[GROUNDING INSTRUCTIONS FOR LLM]
"""
                """When answering the user's question, prioritize the following search results as your primary source of information.
"""
                """For time-sensitive topics, rely on the most recent information available in these results.
"""
                """Pay special attention to publication dates and source credibility when evaluating information.
"""
                """If the search results contain news items, these should be considered the most current information available.
"""
                """Do not contradict factual information provided in these search results, especially from featured snippets and knowledge panels.
"""
                """If the search results don't contain relevant information to answer the user's question, clearly state this limitation.
"""
                """[END OF GROUNDING INSTRUCTIONS]

"""
            )
            
            # Insert the grounding header at the beginning of the results
            formatted_results = grounding_header + formatted_results
        
        # Log performance
        elapsed = time.time() - start_time
        logging.info(f"[PERFORMANCE] brave_web_search completed in {elapsed:.4f}s")
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"[TOOL] web_search exception: {e}")
        return f"I couldn't find any results for '{query}'. Try a different query."

async def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    # Get client stats
    client = await get_brave_search_client()
    client_stats = client.get_cache_stats()
    
    # Add session cache stats
    session_stats = {
        "session_cache_size": len(_session_cache),
        "session_cache_memory_usage": sum(len(v) for v in _session_cache.values())
    }
    
    # Combine stats
    return {
        **client_stats,
        **session_stats
    }

async def clear_session_cache() -> None:
    """Clear the in-memory session cache."""
    global _session_cache
    _session_cache = {}
    logging.info("Session cache cleared")

async def main():
    """Run a simple test of the Brave Search free tier client."""
    # Check if API key is set
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        print("WARNING: BRAVE_API_KEY is not set in your environment variables.")
        print("Please set it in your .env file and try again.")
        return
    
    print(f"Using Brave API key: {api_key[:5]}...{api_key[-4:]}")
    
    # Test queries
    queries = [
        "python programming",
        "machine learning",
        "web development",
        "data science",
        "artificial intelligence"
    ]
    
    # Run searches
    for query in queries:
        print(f"\nSearching for: {query}")
        start_time = time.time()
        results = await web_search(query, num_results=3)
        elapsed = time.time() - start_time
        print(f"Search completed in {elapsed:.4f} seconds")
        print(results)
        
        # Wait a bit between queries to demonstrate rate limiting
        await asyncio.sleep(0.5)
    
    # Get cache statistics
    stats = await get_cache_stats()
    print(f"\nCache statistics: {stats}")
    
    # Close client
    client = await get_brave_search_client()
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
