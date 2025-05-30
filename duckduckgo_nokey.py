"""
DuckDuckGo Search implementation without API key using web scraping.
This module provides web search functionality by scraping the DuckDuckGo website:
1. No API key required
2. Caching to minimize requests
3. Rate limiting to avoid being blocked
4. Detailed search results with web content
"""

import asyncio
import logging
import os
import time
import json
import hashlib
import re
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
import requests
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
ENABLE_CACHE = os.environ.get("DUCKDUCKGO_SEARCH_ENABLE_CACHE", "true").lower() == "true"
ENABLE_PERSISTENCE = os.environ.get("DUCKDUCKGO_SEARCH_ENABLE_PERSISTENCE", "true").lower() == "true"
# Requests per second
RATE_LIMIT = float(os.environ.get("DUCKDUCKGO_SEARCH_RATE_LIMIT", "0.5").split('#')[0].strip())
DEFAULT_REGION = os.environ.get("DUCKDUCKGO_SEARCH_REGION", "wt-wt")
DEFAULT_SAFESEARCH = os.environ.get("DUCKDUCKGO_SEARCH_SAFESEARCH", "moderate")
USER_AGENT = os.environ.get("DUCKDUCKGO_SEARCH_USER_AGENT", 
                           "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# Simple in-memory cache
_cache = {}
_cache_ttl = 604800  # 1 week in seconds
_cache_max_size = 1000

# Session cache for even faster lookups
_session_cache = {}

class RateLimiter:
    """Rate limiter to prevent being blocked by DuckDuckGo."""
    
    def __init__(self, requests_per_second: float = 0.5):
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
    cache_dir = Path(os.path.expanduser("~")) / ".duckduckgo_nokey_cache"
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
        results: Search results from DuckDuckGo scraping
        num_results: Maximum number of results to include
        
    Returns:
        Formatted string of search results with timestamp and source information
    """
    if "error" in results:
        return f"Search error: {results['error']}"
    
    # Extract query from results if available
    query = results.get("query", "your search")
    
    if "web_results" not in results or not results["web_results"]:
        return f"No search results found for '{query}'."
    
    # Get current timestamp for grounding
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a comprehensive grounding header
    formatted = f"""[SEARCH GROUNDING INFORMATION]
- Query: '{query}'
- Results retrieved: {current_time}
- Search API: DuckDuckGo Search (No API Key)

Here are the top {min(len(results["web_results"]), num_results)} results for '{query}':\n\n"""
    
    # Add a warning about time-sensitive information
    formatted += "NOTE: These results may contain time-sensitive information. Consider the publication dates when evaluating currency of information.\n\n"
    
    # Format web results
    web_results = results["web_results"][:num_results]
    formatted += "WEB RESULTS:\n\n"
    
    for i, result in enumerate(web_results, 1):
        title = result.get("title", "No title")
        url = result.get("href", "No URL")
        description = result.get("body", "No description")
        
        formatted += f"{i}. {title}\n   URL: {url}\n   {description}\n\n"
    
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

class DuckDuckGoScraper:
    """DuckDuckGo scraper that doesn't require an API key."""
    
    def __init__(self, 
                enable_cache: Optional[bool] = None, 
                rate_limit: Optional[float] = None,
                region: Optional[str] = None,
                safesearch: Optional[str] = None):
        """Initialize the DuckDuckGo scraper.
        
        Args:
            enable_cache: Whether to enable caching. If None, uses the ENABLE_CACHE env var.
            rate_limit: Requests per second. If None, uses the DUCKDUCKGO_SEARCH_RATE_LIMIT env var.
            region: Region code for search results. If None, uses the DEFAULT_REGION env var.
            safesearch: Safe search level (on, moderate, off). If None, uses the DEFAULT_SAFESEARCH env var.
        """
        # Base URL for DuckDuckGo Search
        self.base_url = "https://html.duckduckgo.com/html/"
        
        # User agent to mimic a real browser
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        # Use provided values or fall back to environment variables
        self.enable_cache = enable_cache if enable_cache is not None else ENABLE_CACHE
        self.rate_limit_value = rate_limit if rate_limit is not None else RATE_LIMIT
        self.region = region if region is not None else DEFAULT_REGION
        self.safesearch = safesearch if safesearch is not None else DEFAULT_SAFESEARCH
        
        logger.info(f"DuckDuckGo scraper configuration: cache={self.enable_cache}, rate_limit={self.rate_limit_value}, region={self.region}, safesearch={self.safesearch}")
        
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
                    limit=5,
                    ttl_dns_cache=300,
                    keepalive_timeout=60
                )
                
                timeout = aiohttp.ClientTimeout(total=30)
                
                self.session = aiohttp.ClientSession(
                    connector=conn,
                    timeout=timeout,
                    headers=self.headers
                )
            
            return self.session
    
    async def search(self, 
                    query: str, 
                    region: Optional[str] = None, 
                    safesearch: Optional[str] = None,
                    timelimit: Optional[str] = None,
                    max_results: int = 10,
                    use_cache: bool = True) -> Dict[str, Any]:
        """Search the web using DuckDuckGo website.
        
        Args:
            query: Search query
            region: Region code for search results (e.g., "us-en", "wt-wt")
            safesearch: Safe search level (on, moderate, off)
            timelimit: Time limit for results (d, w, m, y)
            max_results: Number of results to return
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            Dict containing the search results
        """
        # Normalize the query to improve cache hit rate
        normalized_query = " ".join(query.lower().split())
        
        # Use provided values or fall back to instance defaults
        region = region if region is not None else self.region
        safesearch = safesearch if safesearch is not None else self.safesearch
        
        # Generate cache key
        cache_key = _get_cache_key(
            normalized_query, 
            region=region, 
            safesearch=safesearch, 
            timelimit=timelimit
        )
        
        # Try to get from cache if use_cache is True
        if use_cache and self.enable_cache:
            cached_result = await _get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {normalized_query}")
                return cached_result
        
        # Wait if necessary to comply with rate limits
        await self.rate_limiter.wait_if_needed()
        
        # Prepare form data for the search request
        form_data = {
            "q": normalized_query,
            "kl": region,
            "kp": {"on": 1, "moderate": -1, "off": -2}.get(safesearch, -1),
        }
        
        # Add time limit if specified
        if timelimit:
            time_map = {"d": "d", "w": "w", "m": "m", "y": "y"}
            if timelimit in time_map:
                form_data["df"] = time_map[timelimit]
        
        # Make the request
        try:
            session = await self.get_session()
            
            async with session.post(self.base_url, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"DuckDuckGo search error: {response.status} - {error_text[:100]}...")
                    return {"error": f"Search error: {response.status}", "query": normalized_query}
                
                html_content = await response.text()
                
                # Parse the HTML content
                results = self._parse_search_results(html_content, normalized_query, max_results)
                
                # Cache the result if use_cache is True
                if use_cache and self.enable_cache:
                    await _save_to_cache(cache_key, results)
                
                return results
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return {"error": "Request timed out", "query": normalized_query}
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {str(e)}")
            return {"error": str(e), "query": normalized_query}
    
    def _parse_search_results(self, html_content: str, query: str, max_results: int) -> Dict[str, Any]:
        """Parse search results from HTML content.
        
        Args:
            html_content: HTML content from DuckDuckGo
            query: The search query
            max_results: Number of results to return
            
        Returns:
            Dict containing the parsed search results
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Initialize results
            results = {
                "query": query,
                "web_results": [],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Find search result items
            search_results = soup.select('.result')
            
            # Extract information from each result
            for result in search_results[:max_results]:
                try:
                    # Extract title
                    title_elem = result.select_one('.result__title')
                    title = title_elem.get_text().strip() if title_elem else "No title"
                    
                    # Extract URL
                    url_elem = result.select_one('.result__url')
                    url = url_elem.get_text().strip() if url_elem else "No URL"
                    
                    # Extract description
                    desc_elem = result.select_one('.result__snippet')
                    description = desc_elem.get_text().strip() if desc_elem else "No description"
                    
                    # Extract href
                    href_elem = result.select_one('.result__title a')
                    href = href_elem.get('href') if href_elem else "No URL"
                    
                    # Clean up href if it's a redirect URL
                    if href.startswith('/'):
                        href = f"https://duckduckgo.com{href}"
                    
                    # Add to results
                    results["web_results"].append({
                        "title": title,
                        "href": href,
                        "url": url,
                        "body": description
                    })
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return {"error": f"Error parsing search results: {e}", "query": query}
    
    async def shutdown(self) -> None:
        """Shutdown the client and release resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.info("DuckDuckGo scraper session closed")

# Create a singleton instance
_ddg_scraper = None
_client_lock = asyncio.Lock()

async def get_ddg_scraper() -> DuckDuckGoScraper:
    """Get or create a singleton instance of the DuckDuckGoScraper.
    
    Returns:
        DuckDuckGoScraper instance
    """
    global _ddg_scraper
    
    async with _client_lock:
        if _ddg_scraper is None:
            _ddg_scraper = DuckDuckGoScraper()
    
    return _ddg_scraper

async def text(query: str, 
              region: str = "wt-wt",
              safesearch: str = "moderate",
              timelimit: Optional[str] = None,
              max_results: int = 10) -> List[Dict[str, str]]:
    """Perform a text search using DuckDuckGo without an API key.
    
    Args:
        query: The search query
        region: Region code for search results (e.g., "us-en", "wt-wt")
        safesearch: Safe search level (on, moderate, off)
        timelimit: Time limit for results (d, w, m, y)
        max_results: Number of results to return
        
    Returns:
        List of dictionaries with search results
    """
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query)
    
    start_time = time.time()
    logger.info(f"Text search called for query: {query}, region: {region}, max_results: {max_results}")
    
    try:
        # Get the DuckDuckGo scraper
        scraper = await get_ddg_scraper()
        
        # Perform the search with caching
        results = await scraper.search(
            query=query,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results
        )
        
        if "error" in results:
            error_msg = f"Error searching with DuckDuckGo: {results['error']}"
            logger.error(error_msg)
            return []
        
        # Log performance
        elapsed = time.time() - start_time
        logger.info(f"Text search completed in {elapsed:.4f}s")
        
        return results.get("web_results", [])
        
    except Exception as e:
        logger.error(f"Text search exception: {e}")
        return []

async def web_search(query: str, num_results: int = 5) -> str:
    """Perform a web search using DuckDuckGo without an API key.
    
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
        # Get the DuckDuckGo scraper
        scraper = await get_ddg_scraper()
        
        # Perform the search with caching
        results = await scraper.search(
            query=query,
            max_results=min(num_results + 2, 20),  # Request a few extra results in case some are filtered
        )
        
        if "error" in results:
            error_msg = f"Error searching with DuckDuckGo: {results['error']}"
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
    return {
        "api_key_available": False,  # No API key needed
        "cache_enabled": ENABLE_CACHE,
        "cache_ttl": _cache_ttl,
        "cache_size": len(_cache),
        "cache_max_size": _cache_max_size,
        "rate_limit": RATE_LIMIT,
        "package": "duckduckgo_nokey (web scraping implementation)"
    }

def clear_cache() -> None:
    """Clear the search cache."""
    global _cache, _session_cache
    _cache = {}
    _session_cache = {}
    logger.info("DuckDuckGo search cache cleared")

# For testing
if __name__ == "__main__":
    async def test_search():
        result = await web_search("python programming", 5)
        print(result)
    
    asyncio.run(test_search())
