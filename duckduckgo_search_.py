"""
DuckDuckGo search implementation for LiveKit Amanda.
Provides a fallback search option when Brave Search API is unavailable.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuckDuckGoSearch:
    """DuckDuckGo search implementation with caching and rate limiting."""
    
    def __init__(self):
        """Initialize the DuckDuckGo search client."""
        self.client = DDGS()
        self.cache = {}
        self.last_request_time = 0
        self.rate_limit = 1  # 1 request per second
    
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo for the given query.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries with 'title' and 'link' keys
        """
        # Check cache first
        cache_key = f"{query}:{num_results}"
        if cache_key in self.cache:
            logger.info(f"[DuckDuckGo] Cache hit for query: {query}")
            return self.cache[cache_key]
        
        # Rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last_request)
        
        # Perform the search
        try:
            # Run the search in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.client.search(query)
            )
            
            # Update last request time
            self.last_request_time = asyncio.get_event_loop().time()
            
            # Format the results
            formatted_results = []
            if results and hasattr(results, 'results'):
                for result in results.results[:num_results]:
                    formatted_results.append({
                        'title': result.title,
                        'link': result.url,
                        'snippet': result.description if hasattr(result, 'description') else ''
                    })
            
            # Cache the results
            self.cache[cache_key] = formatted_results
            
            return formatted_results
        except Exception as e:
            logger.error(f"[DuckDuckGo] Search error: {e}")
            return []

# Create a singleton instance
_ddg_instance = None

def get_ddg_client() -> DuckDuckGoSearch:
    """Get the singleton DuckDuckGo search client."""
    global _ddg_instance
    if _ddg_instance is None:
        _ddg_instance = DuckDuckGoSearch()
    return _ddg_instance

async def ddg_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a DuckDuckGo search.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        List of dictionaries with 'title' and 'link' keys
    """
    client = get_ddg_client()
    return await client.search(query, num_results)

async def format_ddg_results(results: List[Dict[str, str]], query: str) -> str:
    """
    Format DuckDuckGo search results into a readable string.
    
    Args:
        results: List of search result dictionaries
        query: The original search query
        
    Returns:
        Formatted string with search results
    """
    if not results:
        return f"No results found for '{query}' on DuckDuckGo."
    
    formatted = f"DuckDuckGo search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        link = result.get('link', 'No link')
        snippet = result.get('snippet', '')
        
        formatted += f"{i}. {title}\n   {link}\n"
        if snippet:
            formatted += f"   {snippet}\n"
        formatted += "\n"
    
    return formatted

async def ddg_web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Formatted string with search results
    """
    results = await ddg_search(query, num_results)
    return await format_ddg_results(results, query)
