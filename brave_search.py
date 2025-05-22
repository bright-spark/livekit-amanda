"""
Brave Search API client for livekit-amanda.
This module provides functionality to search the web using Brave Search API.
"""

import logging
import aiohttp
import json
import os
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlencode

class BraveSearchClient:
    """Client for interacting with Brave Search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Brave Search client.
        
        Args:
            api_key: Brave Search API key. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            logging.warning("No Brave API key provided. Please set BRAVE_API_KEY environment variable or pass it to the constructor.")
        
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
    
    def format_search_results(self, results: Dict[str, Any]) -> str:
        """Format search results into a readable string.
        
        Args:
            results: Search results from the Brave Search API
            
        Returns:
            Formatted string of search results
        """
        if "error" in results:
            return f"Search error: {results['error']}"
        
        if "web" not in results or "results" not in results["web"]:
            return "No search results found."
        
        formatted = f"Found {len(results['web']['results'])} results:\n\n"
        
        for idx, result in enumerate(results["web"]["results"], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            description = result.get("description", "No description")
            
            formatted += f"{idx}. {title}\n"
            formatted += f"URL: {url}\n"
            formatted += f"Description: {description}\n\n"
        
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
