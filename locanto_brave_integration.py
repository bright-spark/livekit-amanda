"""
Integration module to connect Locanto functionality with Brave Search API.
This module provides drop-in replacements for existing Locanto search functions.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, TypedDict
from livekit.agents import function_tool, RunContext

# Import the original Locanto types and utilities
try:
    from locanto import LocantoListing, clean_spoken, sanitize_for_azure, handle_tool_results
except ImportError:
    logging.warning("Could not import from locanto.py, using fallback definitions")
    # Define fallback types if imports fail
    class LocantoListing(TypedDict):
        title: str
        description: str
        location: str
        price: str
        date_posted: str
        url: str
        images: List[str]
        contact_info: Optional[str]
        poster_info: Optional[str]
        full_description: Optional[str]
        category_path: List[str]
        age: Optional[str]
        reply_count: Optional[int]
        ad_id: Optional[str]
    
    def clean_spoken(text):
        return text
    
    def sanitize_for_azure(text):
        return text
    
    async def handle_tool_results(session, text):
        pass

# Import the Brave Search functionality
from brave_search_locanto import get_brave_search_client, brave_search_locanto, brave_search_locanto_by_category

# Function tool replacements that match the original Locanto function signatures
@function_tool
async def basic_search_locanto(context: RunContext, query: str, location: str = "", category: str = "") -> str:
    """Search Locanto listings using Brave Search API.
    
    This is a drop-in replacement for the original basic_search_locanto function.
    
    Args:
        context: The run context for the tool
        query: The search query
        location: Optional location filter
        category: Optional category filter
        
    Returns:
        str: Formatted search results
    """
    return await brave_search_locanto(context, query, location, category)

@function_tool
async def search_locanto(context: RunContext, category_path: str = 'personals/men-seeking-men', 
                        location: str = 'western-cape', max_pages: int = 3, return_url: bool = False) -> str:
    """Search Locanto by category using Brave Search API.
    
    This is a drop-in replacement for the original search_locanto function.
    
    Args:
        context: The run context for the tool
        category_path: Category path string (e.g., 'personals/men-seeking-men')
        location: Location to search in
        max_pages: Maximum number of pages to search
        return_url: Whether to return the first URL instead of formatted results
        
    Returns:
        Formatted search results or first URL if return_url is True
    """
    return await brave_search_locanto_by_category(context, category_path, location, max_pages, return_url)

# Class-based replacement for LocantoClient.locanto_search_by_category
class BraveLocantoClient:
    """Client for Locanto searches using Brave Search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Brave Locanto client.
        
        Args:
            api_key: Optional Brave Search API key
        """
        self.brave_client = get_brave_search_client(api_key)
    
    async def locanto_search_by_category(self, category_path: List[str] = ['personals', 'men-seeking-men'], 
                                        location: str = 'western-cape', max_pages: int = 3) -> List[LocantoListing]:
        """Search Locanto.co.za for listings in a specific category and location using Brave Search API.
        
        This is a drop-in replacement for LocantoClient.locanto_search_by_category.
        
        Args:
            category_path: List of category segments to search in (default: ['personals', 'men-seeking-men'])
            location: The location to search in (default: 'western-cape')
            max_pages: Maximum number of pages to search (default: 3)
            
        Returns:
            List of LocantoListing objects containing the search data
        """
        return await self.brave_client.search_locanto(category_path, location, max_pages)

# Helper function to get a BraveLocantoClient instance
_brave_locanto_client = None

def get_brave_locanto_client(api_key: Optional[str] = None) -> BraveLocantoClient:
    """Get or create a singleton instance of the BraveLocantoClient.
    
    Args:
        api_key: Optional API key for Brave Search
        
    Returns:
        BraveLocantoClient instance
    """
    global _brave_locanto_client
    if _brave_locanto_client is None:
        _brave_locanto_client = BraveLocantoClient(api_key=api_key)
    return _brave_locanto_client
