"""
Brave Search API integration for web search tools.
This module provides web search functionality using the Brave Search API.
"""

import logging
import aiohttp
import json
import os
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlencode
from livekit.agents import function_tool, RunContext

# Handle utils import with try/except
try:
    from utils import sanitize_for_azure, clean_spoken, handle_tool_results
except ImportError:
    try:
        from .utils import sanitize_for_azure, clean_spoken, handle_tool_results
    except ImportError:
        logging.warning("utils module not available, using fallback definitions")
        # Fallback definitions
        def sanitize_for_azure(text):
            return text
            
        def clean_spoken(text):
            return text
            
        async def handle_tool_results(session, text):
            pass

class BraveSearchClient:
    """Client for interacting with Brave Search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Brave Search client.
        
        Args:
            api_key: Brave Search API key. If not provided, will try to get from environment variable.
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
    
    async def search(self, 
                    query: str, 
                    country: str = "us", 
                    search_lang: str = "en", 
                    ui_lang: str = "en-US", 
                    count: int = 10, 
                    offset: int = 0,
                    safe_search: str = "moderate") -> Dict[str, Any]:
        """Search the web using Brave Search API.
        
        Args:
            query: Search query
            country: Country code for search results
            search_lang: Language code for search results
            ui_lang: UI language code
            count: Number of results to return (max 20)
            offset: Offset for pagination
            safe_search: Safe search level (strict, moderate, off)
            
        Returns:
            Dict containing the search results
        """
        params = {
            "q": query,
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
            "count": min(count, 20),  # Brave API has a max of 20 results per request
            "offset": offset,
            "safesearch": safe_search
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"Brave Search API error: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status}", "details": error_text}
                    
                    return await response.json()
        except Exception as e:
            logging.error(f"Error during Brave search: {str(e)}")
            return {"error": str(e)}
    
    def format_search_results(self, results: Dict[str, Any], query: str, num_results: int = 5) -> str:
        """Format search results into a readable string.
        
        Args:
            results: Search results from the Brave Search API
            query: The original search query
            num_results: Maximum number of results to include
            
        Returns:
            Formatted string of search results
        """
        if "error" in results:
            return f"Search error: {results['error']}"
        
        if "web" not in results or "results" not in results["web"]:
            return f"No search results found for '{query}'."
        
        web_results = results["web"]["results"]
        formatted = f"Here are the top {min(len(web_results), num_results)} results for '{query}':\n\n"
        
        for idx, result in enumerate(web_results[:num_results], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            description = result.get("description", "No description")
            
            formatted += f"{idx}. {title}\n   {url}\n"
            if description:
                formatted += f"   {description}\n"
            formatted += "\n"
        
        return formatted

# Create a singleton instance
_brave_search_client = None

def get_brave_search_client(api_key: Optional[str] = None) -> BraveSearchClient:
    """Get or create a singleton instance of the BraveSearchClient.
    
    Args:
        api_key: Optional API key for Brave Search
        
    Returns:
        BraveSearchClient instance
    """
    global _brave_search_client
    if _brave_search_client is None:
        _brave_search_client = BraveSearchClient(api_key=api_key)
    return _brave_search_client

@function_tool
async def web_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """Search the web for information using Brave Search API.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return (1-10)
        
    Returns:
        str: Formatted search results with titles and URLs
    """
    logging.info(f"[TOOL] web_search called for query: {query}, num_results: {num_results}")
    
    try:
        # Get the Brave Search client
        client = get_brave_search_client()
        
        # Perform the search
        results = await client.search(query=query, count=num_results)
        
        if "error" in results:
            error_msg = f"Error searching with Brave API: {results['error']}"
            logging.error(error_msg)
            error_msg = sanitize_for_azure(f"I couldn't find any results for '{query}'. Try a different query.")
            
            # Handle session output for voice responses
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't find any results for your search."
            return error_msg
        
        # Format the results
        formatted_results = client.format_search_results(results, query, num_results)
        formatted_results = sanitize_for_azure(formatted_results)
        
        logging.info(f"[TOOL] web_search results: {formatted_results}")
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, formatted_results)
            return "I've found some results and will read them to you now."
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"[TOOL] web_search exception: {e}")
        error_msg = sanitize_for_azure(f"I couldn't find any results for '{query}'. Try a different query.")
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I couldn't find any results for your search."
        
        return error_msg

@function_tool
async def fallback_web_search(context: RunContext, query: str, num_results: int = 10) -> str:
    """Search the web for information using Brave Search API as a fallback.
    
    This function is maintained for compatibility with existing code but uses
    the same Brave Search API implementation as web_search.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return (1-10)
        
    Returns:
        str: Formatted search results with titles and URLs
    """
    logging.info(f"[TOOL] fallback_web_search called for query: {query}")
    
    # Just use the regular web_search implementation since we're using Brave API for both
    return await web_search(context, query, num_results)
