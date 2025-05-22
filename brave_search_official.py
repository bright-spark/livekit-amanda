"""
Brave Search API implementation using the official brave-search package.
This module provides web search functionality using the Brave Search API with:
1. Support for both synchronous and asynchronous requests
2. Caching to minimize API calls
3. Detailed search results including web, news, and video content
4. Support for Goggles to customize search rankings
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List
import json

# Import the official Brave Search package
try:
    from brave import Brave, AsyncBrave
    HAS_BRAVE_SEARCH = True
except ImportError:
    HAS_BRAVE_SEARCH = False
    logging.warning("Official brave-search package not available. Please install with: pip install brave-search")

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
_cache_ttl = 604800  # 1 week in seconds
_cache_max_size = 1000

def _get_cache_key(query: str, count: int, **kwargs) -> str:
    """Generate a cache key for the search query and parameters.
    
    Args:
        query: The search query
        count: Number of results
        kwargs: Additional search parameters
        
    Returns:
        Cache key string
    """
    # Normalize the query to improve cache hit rate
    normalized_query = " ".join(query.lower().split())
    
    # Create a key from the query and relevant parameters
    key_parts = [normalized_query, str(count)]
    
    # Add other parameters that affect results
    for param in ['country', 'search_lang', 'goggles_id']:
        if param in kwargs and kwargs[param]:
            key_parts.append(f"{param}:{kwargs[param]}")
    
    # Join and hash to create a fixed-length key
    key_string = ":".join(key_parts)
    return key_string

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

def format_search_results(results: Dict[str, Any], num_results: int = 5) -> str:
    """Format search results into a readable string with grounding information.
    
    Args:
        results: Search results from the Brave Search API
        num_results: Maximum number of results to include
        
    Returns:
        Formatted string of search results with timestamp and source information
    """
    if not results or isinstance(results, str):
        return f"No search results found."
    
    # Extract query if available
    query = results.query if hasattr(results, 'query') else "your search"
    
    # Get current timestamp for grounding
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a comprehensive grounding header
    formatted = f"""[SEARCH GROUNDING INFORMATION]
- Query: '{query}'
- Results retrieved: {current_time}
- Search API: Brave Search

Here are the top results for '{query}':\n\n"""
    
    # Add a warning about time-sensitive information
    formatted += "NOTE: These results may contain time-sensitive information. Consider the publication dates when evaluating currency of information.\n\n"
    
    # Format web results
    if hasattr(results, 'web_results') and results.web_results:
        web_results = results.web_results[:num_results]
        formatted += "WEB RESULTS:\n\n"
        
        for i, result in enumerate(web_results, 1):
            title = result.title if hasattr(result, 'title') else "No title"
            url = result.url if hasattr(result, 'url') else "No URL"
            description = result.description if hasattr(result, 'description') else "No description"
            
            formatted += f"{i}. {title}\n   URL: {url}\n   {description}\n\n"
    
    # Format news results if available
    if hasattr(results, 'news_results') and results.news_results:
        news_results = results.news_results[:min(3, len(results.news_results))]
        formatted += "NEWS RESULTS:\n\n"
        
        for i, result in enumerate(news_results, 1):
            title = result.title if hasattr(result, 'title') else "No title"
            url = result.url if hasattr(result, 'url') else "No URL"
            description = result.description if hasattr(result, 'description') else "No description"
            age = result.age if hasattr(result, 'age') else "Unknown date"
            
            formatted += f"{i}. {title}\n   URL: {url}\n   Date: {age}\n   {description}\n\n"
    
    # Format video results if available
    if hasattr(results, 'video_results') and results.video_results:
        video_results = results.video_results[:min(2, len(results.video_results))]
        formatted += "VIDEO RESULTS:\n\n"
        
        for i, result in enumerate(video_results, 1):
            title = result.title if hasattr(result, 'title') else "No title"
            url = result.url if hasattr(result, 'url') else "No URL"
            description = result.description if hasattr(result, 'description') else "No description"
            
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

async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using the official Brave Search API with caching.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Formatted string with search results
    """
    if not HAS_BRAVE_SEARCH:
        return f"Brave Search API is not available. Please install the brave-search package."
    
    try:
        logger.info(f"Using Brave Search API for query: '{query}'")
        
        # Check if API key is available
        api_key = os.environ.get("BRAVE_WEB_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
        if not api_key:
            return f"Brave Search API key is missing. Please set BRAVE_WEB_SEARCH_API_KEY or BRAVE_API_KEY in your .env file."
        
        # Generate cache key
        cache_key = _get_cache_key(query, num_results)
        
        # Check cache first
        current_time = time.time()
        if cache_key in _cache and current_time < _cache[cache_key].get('expires', 0):
            logger.info(f"Cache hit for query: '{query}'")
            _cache[cache_key]['last_access'] = current_time
            return _cache[cache_key]['result']
        
        # Clean cache periodically
        _clean_cache()
        
        # Track start time for performance measurement
        start_time = time.time()
        
        # Initialize the AsyncBrave client
        brave = AsyncBrave(api_key=api_key)
        
        # Perform the search
        search_results = await brave.search(
            q=query,
            count=min(num_results, 20),  # API limit is 20 results per request
        )
        
        # Format the results
        formatted_results = format_search_results(search_results, num_results)
        
        # Cache the formatted results
        _cache[cache_key] = {
            'result': formatted_results,
            'expires': current_time + _cache_ttl,
            'last_access': current_time
        }
        
        # Log performance
        elapsed = time.time() - start_time
        logger.info(f"Brave search completed in {elapsed:.4f}s")
        
        return formatted_results
    except Exception as e:
        logger.error(f"Brave Search API error: {e}")
        return f"Error searching with Brave Search API for '{query}': {str(e)}"

def get_api_config() -> Dict[str, Any]:
    """Get the current API configuration.
    
    Returns:
        Dictionary with API configuration details
    """
    api_key = os.environ.get("BRAVE_WEB_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
    return {
        "api_key_available": bool(api_key),
        "cache_enabled": True,
        "cache_ttl": _cache_ttl,
        "cache_size": len(_cache),
        "cache_max_size": _cache_max_size,
        "package": "brave-search (official)"
    }

def clear_cache() -> None:
    """Clear the search cache."""
    global _cache
    _cache = {}
    logger.info("Brave Search cache cleared")

# For testing
if __name__ == "__main__":
    async def test_search():
        result = await web_search("python programming", 5)
        print(result)
    
    asyncio.run(test_search())
