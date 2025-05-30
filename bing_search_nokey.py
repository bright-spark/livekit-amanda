"""
Bing Search implementation without API key using web scraping.
This module provides web search functionality by scraping the Bing website:
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
import random
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from livekit.agents import RunContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple in-memory cache
_cache = {}
_cache_ttl = 3600  # 1 hour in seconds

async def bing_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """Search the web using Bing Search without requiring an API key.
    
    Args:
        context: The run context from the agent
        query: The search query string
        num_results: Maximum number of results to return
        
    Returns:
        A formatted string containing search results
    """
    logger.info(f"Performing Bing search (nokey) for: {query}")
    
    # Check cache first
    cache_key = f"bing_nokey:{query}:{num_results}"
    if cache_key in _cache:
        logger.info(f"Cache hit for query: {query}")
        return _cache[cache_key]
    
    try:
        # Respect rate limits
        await asyncio.sleep(1.0 + random.random())
        
        # Prepare the search URL
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
        
        # Make the request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    return f"Error: Bing Search returned status code {response.status}"
                
                html = await response.text()
                
                # Parse the HTML
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract search results
                results = []
                search_results = soup.select('.b_algo')
                
                for i, result in enumerate(search_results):
                    if i >= num_results:
                        break
                        
                    title_elem = result.select_one('h2 a')
                    url_elem = result.select_one('h2 a')
                    desc_elem = result.select_one('.b_caption p')
                    
                    title = title_elem.get_text() if title_elem else "No title"
                    url = url_elem['href'] if url_elem and 'href' in url_elem.attrs else "No URL"
                    description = desc_elem.get_text() if desc_elem else "No description"
                    
                    results.append(f"Title: {title}\nURL: {url}\nDescription: {description}\n")
                
                formatted_results = "\n".join(results) if results else "No results found."
                
                # Cache results
                _cache[cache_key] = formatted_results
                
                return formatted_results
    except Exception as e:
        logger.error(f"Error in Bing search (nokey): {str(e)}")
        return f"Error performing search: {str(e)}"

# Rest of the implementation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
ENABLE_CACHE = os.environ.get("BING_SEARCH_ENABLE_CACHE", "true").lower() == "true"
ENABLE_PERSISTENCE = os.environ.get("BING_SEARCH_ENABLE_PERSISTENCE", "true").lower() == "true"
# Requests per second
RATE_LIMIT = float(os.environ.get("BING_SEARCH_RATE_LIMIT", "0.5").split('#')[0].strip())
# Maximum number of retries for Bing Search
MAX_RETRIES = int(os.environ.get("BING_SEARCH_MAX_RETRIES", "3").split('#')[0].strip())

# Simple in-memory cache
_cache = {}
_cache_ttl = 604800  # 1 week in seconds
_cache_max_size = 1000

# Session cache for even faster lookups
_session_cache = {}

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

class RateLimiter:
    """Rate limiter to prevent being blocked by Bing."""
    
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
    cache_dir = Path(os.path.expanduser("~")) / ".bing_nokey_cache"
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
        results: Search results from Bing scraping
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
- Search API: Bing Search (No API Key)

Here are the top {min(len(results["web_results"]), num_results)} results for '{query}':\n\n"""
    
    # Add a warning about time-sensitive information
    formatted += "NOTE: These results may contain time-sensitive information. Consider the publication dates when evaluating currency of information.\n\n"
    
    # Format web results
    web_results = results["web_results"][:num_results]
    formatted += "WEB RESULTS:\n\n"
    
    for i, result in enumerate(web_results, 1):
        title = result.get("title", "No title")
        url = result.get("link", "No URL")
        snippet = result.get("snippet", "No description")
        
        formatted += f"{i}. {title}\n   URL: {url}\n   {snippet}\n\n"
    
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

class BingScraper:
    """Bing scraper that doesn't require an API key."""
    
    def __init__(self, 
                enable_cache: Optional[bool] = None, 
                rate_limit: Optional[float] = None,
                max_retries: Optional[int] = None):
        """Initialize the Bing scraper.
        
        Args:
            enable_cache: Whether to enable caching. If None, uses the ENABLE_CACHE env var.
            rate_limit: Requests per second. If None, uses the BING_SEARCH_RATE_LIMIT env var.
            max_retries: Maximum number of retries. If None, uses the BING_SEARCH_MAX_RETRIES env var.
        """
        # Base URL for Bing Search
        self.base_url = "https://www.bing.com/search"
        
        # Use provided values or fall back to environment variables
        self.enable_cache = enable_cache if enable_cache is not None else ENABLE_CACHE
        self.rate_limit_value = rate_limit if rate_limit is not None else RATE_LIMIT
        self.max_retries = max_retries if max_retries is not None else MAX_RETRIES
        
        logger.info(f"Bing scraper configuration: cache={self.enable_cache}, rate_limit={self.rate_limit_value}, max_retries={self.max_retries}")
        
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
                
                # Rotate user agents
                user_agent = random.choice(USER_AGENTS)
                
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.bing.com/'
                }
                
                self.session = aiohttp.ClientSession(
                    connector=conn,
                    timeout=timeout,
                    headers=headers
                )
            
            return self.session
    
    async def search(self, 
                    query: str, 
                    num_results: int = 10,
                    use_cache: bool = True) -> Dict[str, Any]:
        """Search the web using Bing website.
        
        Args:
            query: Search query
            num_results: Number of results to return
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            Dict containing the search results
        """
        # Normalize the query to improve cache hit rate
        normalized_query = " ".join(query.lower().split())
        
        # Generate cache key
        cache_key = _get_cache_key(normalized_query, num_results=num_results)
        
        # Try to get from cache if use_cache is True
        if use_cache and self.enable_cache:
            cached_result = await _get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {normalized_query}")
                return cached_result
        
        # Wait if necessary to comply with rate limits
        await self.rate_limiter.wait_if_needed()
        
        # Parameters for the search query
        params = {
            'q': normalized_query,  # the search query
            'count': str(min(num_results * 2, 50)),  # request more than needed in case some results are filtered
            'form': 'QBLH',
            'sp': '-1',
            'pq': normalized_query,
            'sc': '8-5',
            'qs': 'n',
            'sk': '',
            'cvid': str(int(time.time()))
        }
        
        # Initialize results
        results = {
            "query": normalized_query,
            "web_results": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Make the request with retries
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                session = await self.get_session()
                
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Bing search error: {response.status} - {error_text[:100]}...")
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            # Exponential backoff
                            wait_time = 2 ** retry_count
                            logger.info(f"Retrying in {wait_time} seconds (attempt {retry_count}/{self.max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return {"error": f"Search error: {response.status}", "query": normalized_query}
                    
                    html_content = await response.text()
                    
                    # Parse the HTML content
                    parsed_results = self._parse_search_results(html_content, normalized_query, num_results)
                    
                    # Check if parsing was successful
                    if not parsed_results.get("web_results"):
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            # Exponential backoff
                            wait_time = 2 ** retry_count
                            logger.info(f"No results found, retrying in {wait_time} seconds (attempt {retry_count}/{self.max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    # Cache the result if use_cache is True
                    if use_cache and self.enable_cache:
                        await _save_to_cache(cache_key, parsed_results)
                    
                    return parsed_results
            except asyncio.TimeoutError:
                logger.error("Request timed out")
                retry_count += 1
                if retry_count <= self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** retry_count
                    logger.info(f"Request timed out, retrying in {wait_time} seconds (attempt {retry_count}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    return {"error": "Request timed out", "query": normalized_query}
            except Exception as e:
                logger.error(f"Error during Bing search: {str(e)}")
                retry_count += 1
                if retry_count <= self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** retry_count
                    logger.info(f"Error occurred, retrying in {wait_time} seconds (attempt {retry_count}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    return {"error": str(e), "query": normalized_query}
        
        # If we get here, all retries failed
        return {"error": "All retry attempts failed", "query": normalized_query}
    
    def _parse_search_results(self, html_content: str, query: str, max_results: int) -> Dict[str, Any]:
        """Parse search results from HTML content.
        
        Args:
            html_content: HTML content from Bing
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
            
            # Find all search results
            result_elements = soup.find_all('li', {'class': 'b_algo'}) or soup.find_all('div', {'class': 'b_algo'})
            
            # Extract information from each result
            for result in result_elements[:max_results]:
                try:
                    title_elem = result.find('h2') or result.find('h3')
                    link_elem = result.find('a', href=True)
                    snippet_elem = result.find('p')
                    
                    if title_elem and link_elem:
                        title = title_elem.text.strip()
                        link = link_elem['href']
                        snippet = snippet_elem.text.strip() if snippet_elem else ""
                        
                        results["web_results"].append({
                            "title": title,
                            "link": link,
                            "snippet": snippet
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
            logger.info("Bing scraper session closed")

# Create a singleton instance
_bing_scraper = None
_client_lock = asyncio.Lock()

async def get_bing_scraper() -> BingScraper:
    """Get or create a singleton instance of the BingScraper.
    
    Returns:
        BingScraper instance
    """
    global _bing_scraper
    
    async with _client_lock:
        if _bing_scraper is None:
            _bing_scraper = BingScraper()
    
    return _bing_scraper

async def web_search(query: str, num_results: int = 5) -> str:
    """Perform a web search using Bing without an API key.
    
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
        # Get the Bing scraper
        scraper = await get_bing_scraper()
        
        # Perform the search with caching
        results = await scraper.search(
            query=query,
            num_results=min(num_results + 2, 20),  # Request a few extra results in case some are filtered
        )
        
        if "error" in results:
            error_msg = f"Error searching with Bing: {results['error']}"
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
        "max_retries": MAX_RETRIES,
        "package": "bing_search_nokey (web scraping implementation)"
    }

def clear_cache() -> None:
    """Clear the search cache."""
    global _cache, _session_cache
    _cache = {}
    _session_cache = {}
    logger.info("Bing search cache cleared")

# For testing
if __name__ == "__main__":
    async def test_search():
        result = await web_search("python programming", 5)
        print(result)
    
    asyncio.run(test_search())
