"""
Custom Brave Search API implementation.
This module provides web search functionality using the Brave Search API with:
1. Support for both synchronous and asynchronous requests
2. Caching to minimize API calls
3. Detailed search results including web, news, and video content
4. Direct API integration without external package dependencies
"""

import asyncio
import logging
import os
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
ENABLE_CACHE = os.environ.get("BRAVE_SEARCH_ENABLE_CACHE", "true").lower() == "true"
ENABLE_PERSISTENCE = os.environ.get("BRAVE_SEARCH_ENABLE_PERSISTENCE", "true").lower() == "true"
RATE_LIMIT = int(os.environ.get("BRAVE_SEARCH_RATE_LIMIT", "1"))

# Simple in-memory cache
_cache = {}
_cache_ttl = 604800  # 1 week in seconds
_cache_max_size = 1000

# Session cache for even faster lookups
_session_cache = {}


async def brave_web_search(query: str, num_results: int = 5, context=None) -> str:
    """Search the web using Brave Search API.
    
    Args:
        query: The search query string
        num_results: Maximum number of results to return
        context: Optional run context from the agent (not used but included for compatibility)
        
    Returns:
        A formatted string containing search results
    """
    logger.info(f"Performing Brave web search for: {query}")
    
    # Check cache first
    cache_key = f"web_search:{query}:{num_results}"
    if ENABLE_CACHE and cache_key in _cache:
        logger.info(f"Cache hit for query: {query}")
        return _cache[cache_key]
    
    # Prepare API request
    api_key = os.environ.get("BRAVE_WEB_SEARCH_API_KEY")
    if not api_key:
        logger.warning("BRAVE_WEB_SEARCH_API_KEY environment variable not set")
        return "Brave Search API key not found in environment variables."
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    params = {
        "q": query,
        "count": num_results,
        "search_lang": "en"
    }
    
    try:
        # Respect rate limits
        await asyncio.sleep(1.0 / RATE_LIMIT)
        
        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.search.brave.com/res/v1/web/search", 
                                   headers=headers, params=params) as response:
                if response.status != 200:
                    return f"Error: Brave Search API returned status code {response.status}"
                
                data = await response.json()
                
                # Format results
                results = []
                if "web" in data and "results" in data["web"]:
                    for item in data["web"]["results"][:num_results]:
                        title = item.get("title", "No title")
                        url = item.get("url", "No URL")
                        description = item.get("description", "No description")
                        results.append(f"Title: {title}\nURL: {url}\nDescription: {description}\n")
                
                formatted_results = "\n".join(results) if results else "No results found."
                
                # Cache results
                if ENABLE_CACHE:
                    _cache[cache_key] = formatted_results
                    
                return formatted_results
    except Exception as e:
        logger.error(f"Error in Brave web search: {str(e)}")
        return f"Error performing search: {str(e)}"


class RateLimiter:
    """Rate limiter to prevent exceeding API limits."""
    
    def __init__(self, requests_per_second: int = 1):
        """Initialize the rate limiter.
        
        Args:
            requests_per_second: Maximum number of requests per second
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self) -> None:
        """Wait if necessary to comply with rate limits."""
        async with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.min_interval:
                wait_time = self.min_interval - time_since_last_request
                logger.info(f"Rate limiting: waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            
            # Update the last request time after waiting
            self.last_request_time = time.time()

def _get_cache_key(query: str, **kwargs) -> str:
    """Generate a cache key for the search query and parameters.
    
    Args:
        query: The search query
        kwargs: Additional search parameters
        
    Returns:
        Cache key string
    """
    # Normalize the query to improve cache hit rate
    normalized_query = " ".join(query.lower().split())
    
    # Create a key from the query and relevant parameters
    key_parts = [normalized_query]
    
    # Add other parameters that affect results
    for param, value in sorted(kwargs.items()):
        if value is not None:
            key_parts.append(f"{param}:{value}")
    
    # Join and hash to create a fixed-length key
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def _clean_cache():
    """Remove expired entries from the cache and enforce size limits."""
    global _cache
    
    # Remove expired entries
    current_time = time.time()
    expired_keys = [k for k, v in _cache.items() if current_time > v.get('expires', 0)]
    for key in expired_keys:
        del _cache[key]
    
    # Enforce size limit
    if len(_cache) > _cache_max_size:
        # Sort by access time and remove oldest entries
        sorted_keys = sorted(_cache.keys(), key=lambda k: _cache[k].get('last_access', 0))
        keys_to_remove = sorted_keys[:len(_cache) - _cache_max_size]
        for key in keys_to_remove:
            del _cache[key]

def _get_disk_cache_path(cache_key: str) -> Path:
    """Get the path to the disk cache file for a given key.
    
    Args:
        cache_key: The cache key
        
    Returns:
        Path to the cache file
    """
    cache_dir = Path(os.path.expanduser("~")) / ".brave_search_cache"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir / f"{cache_key}.json"

async def _get_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get a value from the cache if it exists and is not expired.
    
    Args:
        cache_key: The cache key
        
    Returns:
        The cached value or None if not found or expired
    """
    # First check memory cache (fastest)
    current_time = time.time()
    if cache_key in _cache and current_time < _cache[cache_key].get('expires', 0):
        _cache[cache_key]['last_access'] = current_time
        return _cache[cache_key]['result']
    
    # If not in memory cache and persistence is enabled, check disk cache
    if ENABLE_PERSISTENCE:
        cache_file = _get_disk_cache_path(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_entry = json.load(f)
                
                if current_time < cache_entry.get('expires', 0):
                    # Load into memory cache for faster access next time
                    _cache[cache_key] = {
                        'result': cache_entry['result'],
                        'expires': cache_entry['expires'],
                        'last_access': current_time
                    }
                    return cache_entry['result']
            except Exception as e:
                logger.warning(f"Failed to read disk cache: {e}")
    
    return None

async def _save_to_cache(cache_key: str, result: Dict[str, Any], ttl: int = _cache_ttl) -> None:
    """Save a value to the cache.
    
    Args:
        cache_key: The cache key
        result: The result to cache
        ttl: Time-to-live in seconds
    """
    current_time = time.time()
    expires = current_time + ttl
    
    # Save to memory cache
    _cache[cache_key] = {
        'result': result,
        'expires': expires,
        'last_access': current_time
    }
    
    # If persistence is enabled, save to disk cache
    if ENABLE_PERSISTENCE:
        cache_file = _get_disk_cache_path(cache_key)
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'result': result,
                    'expires': expires
                }, f)
        except Exception as e:
            logger.warning(f"Failed to write disk cache: {e}")
    
    # Clean cache periodically
    _clean_cache()

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
    
    # Format web results
    formatted += "WEB RESULTS:\n\n"
    
    for i, result in enumerate(web_results[:num_results], 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        description = result.get("description", "No description")
        
        formatted += f"{i}. {title}\n   URL: {url}\n   {description}\n\n"
    
    # Format news results if available
    if "news" in results and "results" in results["news"] and results["news"]["results"]:
        news_results = results["news"]["results"][:min(3, len(results["news"]["results"]))]
        formatted += "NEWS RESULTS:\n\n"
        
        for i, result in enumerate(news_results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            description = result.get("description", "No description")
            age = result.get("age", "Unknown date")
            
            formatted += f"{i}. {title}\n   URL: {url}\n   Date: {age}\n   {description}\n\n"
    
    # Add explicit grounding instructions for the LLM
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
        """Do not contradict factual information provided in these search results.
"""
        """If the search results don't contain relevant information to answer the user's question, clearly state this limitation.
"""
        """[END OF GROUNDING INSTRUCTIONS]

"""
    )
    
    # Insert the grounding header at the beginning of the results
    formatted = grounding_header + formatted
    
    return formatted

class BraveSearchClient:
    """Brave Search API client with configurable optimizations."""
    
    def __init__(self, 
             api_key: Optional[str] = None,
             enable_cache: Optional[bool] = None,
             rate_limit: Optional[int] = None):
        """Initialize the Brave Search client.
        
        Args:
            api_key: Brave Search API key. If not provided, will try to get from environment variable.
            enable_cache: Whether to enable caching. If None, uses the ENABLE_CACHE env var.
            rate_limit: Requests per second (1 for free tier, 20 for paid tier). If None, uses the BRAVE_SEARCH_RATE_LIMIT env var.
        """
        # Handle API key - if explicitly set to None, don't use environment variables
        if api_key is None:
            self.api_key = None
            logger.warning("Brave Search API key explicitly set to None")
        else:
            # Get API key from environment variable if not provided
            self.api_key = api_key or os.environ.get("BRAVE_WEB_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
            if not self.api_key:
                logger.warning("Brave Search API key not provided and not found in environment variables (BRAVE_WEB_SEARCH_API_KEY or BRAVE_API_KEY)")
                # Set to None explicitly to ensure consistent behavior
                self.api_key = None
        
        # API configuration
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
        # Initialize headers with required values
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip"
        }
        
        # Only add the API key to headers if it exists and is not empty
        if self.api_key:  # This will be False for None, empty string, etc.
            self.headers["X-Subscription-Token"] = self.api_key
        
        # Use provided values or fall back to environment variables
        self.enable_cache = enable_cache if enable_cache is not None else ENABLE_CACHE
        self.rate_limit_value = rate_limit if rate_limit is not None else RATE_LIMIT
        
        logger.info(f"Brave Search API configuration: cache={self.enable_cache}, rate_limit={self.rate_limit_value}")
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(requests_per_second=self.rate_limit_value)
        
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
        """Search the web using Brave Search API.
        
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
        # Check if API key is available
        if not self.api_key:
            logger.warning("Brave Search API key is missing. Please set BRAVE_WEB_SEARCH_API_KEY or BRAVE_API_KEY in your .env file.")
            return {"error": "Brave Search API key is missing", "results": [], "web": {"results": []}, "query": {"query": query}}
            
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
        cache_key = _get_cache_key(normalized_query, **params)
        
        # Try to get from cache if use_cache is True
        if use_cache and self.enable_cache:
            cached_result = await _get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {normalized_query}")
                return cached_result
        
        # Wait if necessary to comply with rate limits
        await self.rate_limiter.wait_if_needed()
        
        # Make the API request
        try:
            # Get or create a session
            session = await self.get_session()
            
            # Ensure all parameters are valid types to avoid errors
            sanitized_params = {}
            for key, value in params.items():
                if value is None:
                    logger.warning(f"Parameter {key} is None, using default value")
                    # Use default values for None parameters
                    if key == 'q':
                        sanitized_params[key] = ""
                    elif key == 'count':
                        sanitized_params[key] = 10
                    elif key == 'offset':
                        sanitized_params[key] = 0
                    elif key == 'safesearch':
                        sanitized_params[key] = "moderate"
                    else:
                        sanitized_params[key] = ""
                elif isinstance(value, bool):
                    # Convert boolean to string
                    logger.warning(f"Parameter {key} is boolean, converting to string")
                    sanitized_params[key] = str(value).lower()
                else:
                    # Use the value as is
                    sanitized_params[key] = value
            
            # Make the request with sanitized parameters
            async with session.get(self.base_url, params=sanitized_params) as response:
                if response.status == 429:
                    # Rate limit hit
                    error_msg = await response.text()
                    logger.error(f"Rate limit exceeded: {error_msg}")
                    return {"error": "Rate limit exceeded", "details": error_msg, "results": [], "web": {"results": []}}
                elif response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Brave Search API error: {response.status} - {error_text}")
                    return {"error": f"API error: {response.status}", "details": error_text, "results": [], "web": {"results": []}}
                
                # Parse the response as JSON
                result = await response.json()
                
                # Cache the result if use_cache is True
                if use_cache and self.enable_cache:
                    await _save_to_cache(cache_key, result)
                
                return result
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return {"error": "Request timed out", "results": [], "web": {"results": []}}
        except Exception as e:
            logger.error(f"Error during Brave search: {str(e)}")
            return {"error": str(e), "results": [], "web": {"results": []}}
    
    async def shutdown(self) -> None:
        """Shutdown the client and release resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.info("Brave Search client session closed")

# Create a singleton instance
_brave_search_client = None
_client_lock = asyncio.Lock()

async def get_brave_search_client(api_key: Optional[str] = None) -> BraveSearchClient:
    """Get or create a singleton instance of the BraveSearchClient.
    
    Args:
        api_key: Optional API key for Brave Search
        
    Returns:
        BraveSearchClient instance
    """
    global _brave_search_client
    
    async with _client_lock:
        if _brave_search_client is None:
            _brave_search_client = BraveSearchClient(api_key=api_key)
    
    return _brave_search_client

async def web_search(query: str, num_results: int = 5) -> str:
    """Perform a web search using the Brave Search API.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Formatted search results as a string
    """
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query)
        
    # Ensure num_results is an integer
    if not isinstance(num_results, int):
        try:
            num_results = int(num_results)
        except (ValueError, TypeError):
            num_results = 5  # Default to 5 results if conversion fails
    
    start_time = time.time()
    logger.info(f"Web search called for query: {query}, num_results: {num_results}")
    
    # Check session cache first
    cache_key = f"session:{query}:{num_results}"
    if cache_key in _session_cache:
        logger.info(f"Session cache hit for query: {query}")
        formatted_results = _session_cache[cache_key]
        # Log performance
        elapsed = time.time() - start_time
        logger.info(f"Web search completed in {elapsed:.4f}s (session cache hit)")
        return formatted_results
    
    try:
        # Get the Brave Search client
        client = await get_brave_search_client()
        
        # Perform the search with caching
        results = await client.search(
            query=query,
            count=min(num_results + 2, 20),  # Request a few extra results in case some are filtered
        )
        
        if "error" in results:
            error_msg = f"Error searching with Brave API: {results['error']}"
            logger.error(error_msg)
            return f"I couldn't find any results for '{query}'. Try a different query."
        
        # Format the results
        formatted_results = format_search_results(results, num_results)
        
        # Store in session cache
        _session_cache[cache_key] = formatted_results
        
        # Limit session cache size
        if len(_session_cache) > 100:
            # Remove oldest entries
            keys = list(_session_cache.keys())
            for old_key in keys[:20]:  # Remove 20 oldest entries
                if old_key in _session_cache:
                    del _session_cache[old_key]
        
        # Log performance
        elapsed = time.time() - start_time
        logger.info(f"Web search completed in {elapsed:.4f}s")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Web search exception: {e}")
        return f"I couldn't find any results for '{query}'. Try a different query."

def get_api_config() -> Dict[str, Any]:
    """Get the current API configuration.
    
    Returns:
        Dictionary with API configuration details
    """
    api_key = os.environ.get("BRAVE_WEB_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
    return {
        "api_key_available": bool(api_key),
        "cache_enabled": ENABLE_CACHE,
        "cache_ttl": _cache_ttl,
        "cache_size": len(_cache),
        "cache_max_size": _cache_max_size,
        "rate_limit": RATE_LIMIT,
        "package": "brave_search_api (custom implementation)"
    }

def clear_cache() -> None:
    """Clear the search cache."""
    global _cache, _session_cache
    _cache = {}
    _session_cache = {}
    logger.info("Brave Search cache cleared")

# For testing
if __name__ == "__main__":
    async def test_search():
        result = await web_search("python programming", 5)
        print(result)
    
    asyncio.run(test_search())
