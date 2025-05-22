"""
Enhanced Optimized Brave Search API client with advanced caching and rate limiting.
This module provides a more efficient way to use the Brave Search API with minimal rate limits.
"""

import logging
import aiohttp
import json
import os
import time
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urlencode
import pickle
import gzip
from pathlib import Path

class PersistentBraveSearchCache:
    """Advanced cache for Brave Search API results with disk persistence and tiered strategy."""
    
    def __init__(self, cache_ttl: int = 3600, 
                 memory_cache_size: int = 100,
                 disk_cache_dir: Optional[str] = None,
                 disk_cache_enabled: bool = True):
        """Initialize the enhanced cache.
        
        Args:
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            memory_cache_size: Maximum number of items to keep in memory cache
            disk_cache_dir: Directory to store persistent cache files
            disk_cache_enabled: Whether to use disk caching
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
        
        # Disk cache settings
        self.disk_cache_enabled = disk_cache_enabled
        if disk_cache_enabled:
            if disk_cache_dir:
                self.disk_cache_dir = Path(disk_cache_dir)
            else:
                self.disk_cache_dir = Path(os.path.expanduser("~")) / ".brave_search_cache"
            
            # Create cache directory if it doesn't exist
            os.makedirs(self.disk_cache_dir, exist_ok=True)
            logging.info(f"Disk cache enabled at: {self.disk_cache_dir}")
    
    def get_cache_key(self, query: str, **params) -> str:
        """Generate a cache key from the query and parameters.
        
        Args:
            query: The search query
            params: Additional search parameters
            
        Returns:
            A unique cache key string
        """
        # Normalize the query (lowercase, remove extra spaces)
        normalized_query = " ".join(query.lower().split())
        
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
        if not self.disk_cache_enabled:
            return False
            
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
        if not self.disk_cache_enabled:
            return None
            
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
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from the cache if it exists and is not expired.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        # First check memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() < entry['expires']:
                # Update access time for LRU tracking
                self.access_times[key] = time.time()
                self.hit_count += 1
                self.memory_hit_count += 1
                return entry['data']
            else:
                # Remove expired entry
                del self.memory_cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        # If not in memory or expired, check disk cache
        if self.disk_cache_enabled:
            disk_entry = self._load_from_disk(key)
            if disk_entry and time.time() < disk_entry['expires']:
                # Found valid entry in disk cache, add to memory cache
                self.memory_cache[key] = disk_entry
                self.access_times[key] = time.time()
                self._manage_memory_cache_size()
                self.hit_count += 1
                self.disk_hit_count += 1
                return disk_entry['data']
        
        self.miss_count += 1
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
        
        # Also save to disk cache if enabled
        if self.disk_cache_enabled:
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
        
        # Clear disk cache if enabled
        if self.disk_cache_enabled:
            try:
                import shutil
                shutil.rmtree(self.disk_cache_dir)
                os.makedirs(self.disk_cache_dir, exist_ok=True)
                logging.info(f"Disk cache cleared: {self.disk_cache_dir}")
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
        if self.disk_cache_enabled:
            try:
                for root, dirs, files in os.walk(self.disk_cache_dir):
                    for file in files:
                        if file.endswith('.gz'):
                            disk_cache_size += 1
                            file_path = os.path.join(root, file)
                            disk_cache_bytes += os.path.getsize(file_path)
            except Exception as e:
                logging.warning(f"Failed to calculate disk cache size: {e}")
        
        return {
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

class AdaptiveRateLimiter:
    """Advanced rate limiter with adaptive backoff and request prioritization."""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60, 
                 max_retries: int = 3, initial_backoff: float = 1.0):
        """Initialize the adaptive rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
            max_retries: Maximum number of retries for failed requests
            initial_backoff: Initial backoff time in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = []
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = 2.0  # Exponential backoff multiplier
        
        # Track rate limit responses to adapt
        self.rate_limit_hits = 0
        self.last_rate_limit_time = 0
        self.adaptive_window = time_window
        
        # Request queue for prioritization
        self.request_queue = asyncio.PriorityQueue()
        self.processing = False
        self.request_counter = 0  # Counter to ensure unique ordering
    
    async def wait_if_needed(self) -> None:
        """Wait if necessary to comply with rate limits.
        
        This method will block until it's safe to make another request.
        """
        current_time = time.time()
        
        # Remove request times that are outside the time window
        self.request_times = [t for t in self.request_times if current_time - t < self.adaptive_window]
        
        # For strict rate limiting (especially for free tier with 1 req/sec)
        if self.request_times:  # If we have made any previous requests
            last_request_time = max(self.request_times)  # Get the most recent request time
            time_since_last_request = current_time - last_request_time
            
            # Enforce minimum time between requests based on max_requests per time_window
            min_interval = self.time_window / self.max_requests
            
            if time_since_last_request < min_interval:
                wait_time = min_interval - time_since_last_request
                logging.info(f"Enforcing minimum interval, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Also check if we've reached the maximum number of requests in the time window
        if len(self.request_times) >= self.max_requests:
            oldest_request = min(self.request_times)
            wait_time = oldest_request + self.adaptive_window - current_time
            if wait_time > 0:
                logging.info(f"Rate limit reached, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add the current request time (after waiting)
        self.request_times.append(time.time())
    
    def register_rate_limit(self) -> None:
        """Register a rate limit response to adapt the time window."""
        current_time = time.time()
        self.rate_limit_hits += 1
        self.last_rate_limit_time = current_time
        
        # Increase the adaptive window to be more conservative
        self.adaptive_window = min(self.adaptive_window * 1.5, 300)  # Cap at 5 minutes
        logging.warning(f"Rate limit hit, increasing adaptive window to {self.adaptive_window:.2f} seconds")
    
    def register_success(self) -> None:
        """Register a successful request to potentially reduce the time window."""
        current_time = time.time()
        
        # If it's been a while since the last rate limit, gradually reduce the window
        if self.rate_limit_hits > 0 and (current_time - self.last_rate_limit_time) > (self.adaptive_window * 2):
            self.adaptive_window = max(self.time_window, self.adaptive_window * 0.9)
            logging.info(f"Reducing adaptive window to {self.adaptive_window:.2f} seconds")
    
    async def execute_with_retry(self, func, *args, **kwargs) -> Tuple[Any, bool]:
        """Execute a function with retry logic and exponential backoff.
        
        Args:
            func: The async function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple of (result, success)
        """
        retries = 0
        backoff_time = self.initial_backoff
        
        while retries <= self.max_retries:
            try:
                # Wait if needed to comply with rate limits
                await self.wait_if_needed()
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Check if the result indicates a rate limit
                if isinstance(result, dict) and result.get("error", "").startswith("Rate limit"):
                    self.register_rate_limit()
                    retries += 1
                    if retries > self.max_retries:
                        return result, False
                    
                    # Wait with exponential backoff
                    logging.warning(f"Rate limit hit, retrying in {backoff_time:.2f} seconds (retry {retries}/{self.max_retries})")
                    await asyncio.sleep(backoff_time)
                    backoff_time *= self.backoff_factor
                    continue
                
                # Register successful request
                self.register_success()
                return result, True
                
            except Exception as e:
                logging.error(f"Error executing function: {str(e)}")
                retries += 1
                if retries > self.max_retries:
                    return {"error": f"Max retries exceeded: {str(e)}"}, False
                
                # Wait with exponential backoff
                logging.warning(f"Error, retrying in {backoff_time:.2f} seconds (retry {retries}/{self.max_retries})")
                await asyncio.sleep(backoff_time)
                backoff_time *= self.backoff_factor
        
        return {"error": "Max retries exceeded"}, False
    
    async def add_request(self, priority: int, func, *args, **kwargs) -> Any:
        """Add a request to the priority queue.
        
        Args:
            priority: Priority of the request (lower is higher priority)
            func: The async function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function
        """
        # Create a future to get the result
        future = asyncio.Future()
        
        # Get a unique request ID to break ties in priority comparison
        request_id = self.request_counter
        self.request_counter += 1
        
        # Add to queue with (priority, request_id, payload)
        # This ensures items with same priority are ordered by insertion time
        await self.request_queue.put((priority, request_id, (func, args, kwargs, future)))
        
        # Start processing if not already
        if not self.processing:
            self.processing = True
            asyncio.create_task(self._process_queue())
        
        # Wait for the result
        return await future
    
    async def _process_queue(self) -> None:
        """Process requests from the queue in priority order."""
        try:
            while not self.request_queue.empty():
                # Get the next item (priority, request_id, payload)
                priority, request_id, (func, args, kwargs, future) = await self.request_queue.get()
                
                try:
                    # Execute with retry
                    result, success = await self.execute_with_retry(func, *args, **kwargs)
                    
                    # Set the result
                    if not future.done():
                        future.set_result(result)
                        
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                
                # Mark task as done
                self.request_queue.task_done()
        finally:
            self.processing = False

class EnhancedBraveSearchClient:
    """Enhanced client for interacting with Brave Search API with advanced caching and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 cache_ttl: int = 3600,
                 memory_cache_size: int = 100,
                 disk_cache_dir: Optional[str] = None,
                 disk_cache_enabled: bool = True,
                 max_requests: int = 10, 
                 time_window: int = 60,
                 max_retries: int = 3,
                 initial_backoff: float = 1.0):
        """Initialize the enhanced Brave Search client.
        
        Args:
            api_key: Brave Search API key. If not provided, will try to get from environment variable.
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            memory_cache_size: Maximum number of items to keep in memory cache
            disk_cache_dir: Directory to store persistent cache files
            disk_cache_enabled: Whether to use disk caching
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds for rate limiting
            max_retries: Maximum number of retries for failed requests
            initial_backoff: Initial backoff time in seconds
        """
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            logging.warning("Brave Search API key not provided and not found in environment variables")
        
        # API configuration
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        # Initialize enhanced cache and rate limiter
        self.cache = PersistentBraveSearchCache(
            cache_ttl=cache_ttl,
            memory_cache_size=memory_cache_size,
            disk_cache_dir=disk_cache_dir,
            disk_cache_enabled=disk_cache_enabled
        )
        
        self.rate_limiter = AdaptiveRateLimiter(
            max_requests=max_requests, 
            time_window=time_window,
            max_retries=max_retries,
            initial_backoff=initial_backoff
        )
        
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
                    limit=20,  # Connection pool size
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
    
    async def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to the Brave Search API.
        
        Args:
            url: The URL to request
            params: Query parameters
            
        Returns:
            API response as a dictionary
        """
        session = await self.get_session()
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    # Rate limit hit
                    return {"error": "Rate limit exceeded"}
                elif response.status != 200:
                    error_text = await response.text()
                    logging.error(f"Brave Search API error: {response.status} - {error_text}")
                    return {"error": f"API error: {response.status}", "details": error_text}
                
                return await response.json()
        except asyncio.TimeoutError:
            return {"error": "Request timed out"}
        except Exception as e:
            logging.error(f"Error during Brave search request: {str(e)}")
            return {"error": str(e)}
    
    async def search(self, 
                    query: str, 
                    country: str = "us", 
                    search_lang: str = "en", 
                    ui_lang: str = "en-US", 
                    count: int = 10, 
                    offset: int = 0,
                    safe_search: str = "moderate",
                    use_cache: bool = True,
                    priority: int = 5) -> Dict[str, Any]:
        """Search the web using Brave Search API with enhanced caching and rate limiting.
        
        Args:
            query: Search query
            country: Country code for search results
            search_lang: Language code for search results
            ui_lang: UI language code
            count: Number of results to return (max 20)
            offset: Offset for pagination
            safe_search: Safe search level (strict, moderate, off)
            use_cache: Whether to use the cache (default: True)
            priority: Request priority (lower is higher priority, default: 5)
            
        Returns:
            Dict containing the search results
        """
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
                logging.info(f"Cache hit for query: {normalized_query}")
                return cached_result
        
        # Define the request function
        async def do_request():
            return await self._make_request(self.base_url, params)
        
        # Add to priority queue
        result = await self.rate_limiter.add_request(priority, do_request)
        
        # Cache the result if use_cache is True and there was no error
        if use_cache and "error" not in result:
            await self.cache.set(cache_key, result)
        
        return result
    
    async def batch_search(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Perform multiple searches in parallel with proper rate limiting.
        
        Args:
            queries: List of search queries
            **kwargs: Additional parameters to pass to search()
            
        Returns:
            List of search results
        """
        tasks = []
        for query in queries:
            task = asyncio.create_task(self.search(query, **kwargs))
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
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
_enhanced_brave_search_client = None
_client_lock = asyncio.Lock()

async def get_enhanced_brave_search_client(api_key: Optional[str] = None, 
                                         disk_cache_dir: Optional[str] = None,
                                         tier: str = "free") -> EnhancedBraveSearchClient:
    """Get or create a singleton instance of the EnhancedBraveSearchClient.
    
    Args:
        api_key: Optional API key for Brave Search
        disk_cache_dir: Optional directory for disk cache
        tier: API tier ('free' or 'base'). Free tier allows 1 req/sec, base tier allows 20 req/sec.
        
    Returns:
        EnhancedBraveSearchClient instance
    """
    global _enhanced_brave_search_client
    
    # Set rate limits based on tier
    if tier.lower() == "free":
        # Free tier: 1 request per second
        max_requests = 1
        time_window = 1  # 1 second window
        logging.info("Using Brave Search API free tier rate limits: 1 request per second")
    elif tier.lower() == "base":
        # Base tier: 20 requests per second
        max_requests = 20
        time_window = 1  # 1 second window
        logging.info("Using Brave Search API base tier rate limits: 20 requests per second")
    else:
        # Default to free tier as a safe fallback
        max_requests = 1
        time_window = 1
        logging.warning(f"Unknown tier '{tier}', defaulting to free tier rate limits")
    
    async with _client_lock:
        if _enhanced_brave_search_client is None:
            _enhanced_brave_search_client = EnhancedBraveSearchClient(
                api_key=api_key,
                disk_cache_dir=disk_cache_dir,
                disk_cache_enabled=True,
                memory_cache_size=200,  # Increased memory cache size
                cache_ttl=86400,  # 24 hour cache TTL
                max_requests=max_requests,
                time_window=time_window,
                max_retries=3,
                initial_backoff=1.0
            )
    
    return _enhanced_brave_search_client
