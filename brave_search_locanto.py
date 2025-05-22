"""
Brave Search API integration for Locanto searches.
This module provides functionality to search Locanto using the Brave Search API.
"""

import logging
import aiohttp
import json
import os
from typing import List, Dict, Any, Optional, Union, TypedDict
from urllib.parse import urlencode, urljoin
import asyncio
from bs4 import BeautifulSoup

# Import from locanto.py
try:
    from locanto import LocantoListing, clean_spoken, sanitize_for_azure, handle_tool_results
except ImportError:
    # Define placeholder types if imports fail
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
                    country: str = "za", 
                    search_lang: str = "en", 
                    ui_lang: str = "en-US", 
                    count: int = 10, 
                    offset: int = 0,
                    safe_search: str = "moderate") -> Dict[str, Any]:
        """Search the web using Brave Search API.
        
        Args:
            query: Search query
            country: Country code for search results (za for South Africa)
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

    async def search_locanto(self, 
                           category_path: List[str] = ['personals', 'men-seeking-men'], 
                           location: str = 'western-cape', 
                           max_pages: int = 3) -> List[LocantoListing]:
        """Search Locanto.co.za for listings in a specific category and location using Brave Search API.
        
        Args:
            category_path: List of category segments to search in (default: ['personals', 'men-seeking-men'])
            location: The location to search in (default: 'western-cape')
            max_pages: Maximum number of pages to search (default: 3)
            
        Returns:
            List of LocantoListing objects containing the search data
        """
        try:
            # Build a more specific query for Brave Search
            category_str = ' '.join(category_path)
            search_query = f"{category_str} {location} site:locanto.co.za"
            
            # Calculate total results to fetch based on max_pages (assuming 10 results per page)
            count = min(max_pages * 10, 20)  # Brave API has a max of 20 results per request
            
            results = await self.search(query=search_query, count=count)
            
            listings: List[LocantoListing] = []
            
            if "error" in results:
                logging.error(f"Brave Search API error: {results['error']}")
                return listings
                
            if "web" not in results or "results" not in results["web"] or not results["web"]["results"]:
                logging.info("No results found from Brave Search API")
                return listings
                
            web_results = results["web"]["results"]
            
            for result in web_results:
                title = result.get("title", "No title")
                url = result.get("url", "")
                description = result.get("description", "No description")
                
                # Create a basic listing with the information we have
                listing: LocantoListing = {
                    'title': title,
                    'description': description,
                    'location': location,  # Use the provided location
                    'price': '',  # Not available from search results
                    'date_posted': '',  # Not available from search results
                    'url': url,
                    'images': [],  # Not available from search results
                    'contact_info': None,
                    'poster_info': None,
                    'full_description': description,
                    'category_path': category_path,
                    'age': None,
                    'reply_count': None,
                    'ad_id': None
                }
                
                listings.append(listing)
            
            return listings
            
        except Exception as e:
            logging.error(f"Error in search_locanto: {str(e)}")
            return []

    async def basic_search_locanto(self, query: str, location: str = "", category: str = "") -> str:
        """Search Locanto listings using Brave Search API.
        
        Args:
            query: The search query
            location: Optional location filter
            category: Optional category filter
            
        Returns:
            Formatted search results as a string
        """
        try:
            # Build a more specific query for Brave Search
            search_query = query
            if location:
                search_query += f" {location}"
            if category:
                search_query += f" {category}"
            
            # Add site restriction to focus on Locanto
            search_query += " site:locanto.com"
            
            # Perform the search
            results = await self.search(query=search_query, count=10)
            
            # Format and return the results
            return self.format_search_results(results)
            
        except Exception as e:
            logging.error(f"Brave Search API error: {e}", exc_info=True)
            return f"I couldn't search using Brave Search API: {str(e)}"

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

# Function tool implementations that can be used directly in your code
from livekit.agents import function_tool, RunContext

@function_tool
async def brave_search_locanto(context: RunContext, query: str, location: str = "", category: str = "") -> str:
    """Search Locanto listings using Brave Search API.
    
    Args:
        context: The run context for the tool
        query: The search query
        location: Optional location filter
        category: Optional category filter
        
    Returns:
        str: Formatted search results
    """
    try:
        brave_client = get_brave_search_client()
        results = await brave_client.basic_search_locanto(query, location, category)
        
        # Store the results in the session for later use if needed
        session = getattr(context, 'session', None)
        if session is not None:
            try:
                # Store URLs in session data if available
                if isinstance(results, dict) and "web" in results and "results" in results["web"]:
                    url_map = {}
                    for idx, result in enumerate(results["web"]["results"], 1):
                        url = result.get("url", "")
                        if url:
                            url_map[idx] = url
                    
                    session.userdata['last_locanto_urls'] = url_map
            except Exception as e:
                logging.warning(f"Could not set session.userdata['last_locanto_urls']: {e}")
        
        # Format results for display
        summary = sanitize_for_azure(results)
        summary = clean_spoken(summary)
        
        if session:
            await handle_tool_results(session, summary)
            return "I've found some results and will read them to you now."
        else:
            return summary
        
    except Exception as e:
        logging.error(f"Brave Search API error: {e}", exc_info=True)
        error_msg = f"I couldn't search using Brave Search API: {str(e)}"
        return sanitize_for_azure(error_msg)

@function_tool
async def brave_search_locanto_by_category(context: RunContext, category_path: str = 'personals/men-seeking-men', location: str = 'western-cape', max_pages: int = 3, return_url: bool = False) -> str:
    """Search Locanto by category using Brave Search API.
    
    Args:
        context: The run context for the tool
        category_path: Category path string (e.g., 'personals/men-seeking-men')
        location: Location to search in
        max_pages: Maximum number of pages to search
        return_url: Whether to return the first URL instead of formatted results
        
    Returns:
        Formatted search results or first URL if return_url is True
    """
    try:
        brave_client = get_brave_search_client()
        
        # Convert category path string to list
        categories = category_path.split('/')
        
        # Search using Brave Search API
        listings = await brave_client.search_locanto(categories, location, max_pages)
        
        if not listings:
            summary = "No listings found matching your criteria."
        else:
            first_url = None
            url_map = {}
            summary = f"Found {len(listings)} listings on Locanto:\n\n"
            
            for idx, listing in enumerate(listings, 1):
                title = listing.get('title', 'No title')
                title = clean_spoken(title)
                summary += f"{idx}. {title}\n"
                
                url = listing.get('url')
                if url:
                    url_map[idx] = url
                    if not first_url:
                        first_url = url
                    summary += f"URL: {url}\n"
                
                description = listing.get('description')
                if description:
                    desc = description[:200] + '...' if len(description) > 200 else description
                    summary += f"Description: {clean_spoken(desc)}\n"
                
                # Add location information
                location_val = listing.get('location', location)
                summary += f"Location: {clean_spoken(location_val)}\n"
                
                summary += "\n"
            
            # Store mapping in session.userdata for later use
            session = getattr(context, 'session', None)
            if session is not None:
                try:
                    session.userdata['last_locanto_urls'] = url_map
                except Exception as e:
                    logging.warning(f"Could not set session.userdata['last_locanto_urls']: {e}")
            
            if first_url and return_url:
                return first_url
                
            if first_url:
                summary += f"Would you like to open the first listing in your browser?"
        
        summary = sanitize_for_azure(summary)
        summary = clean_spoken(summary)
        logging.info(f"[TOOL] brave_search_locanto_by_category summary: {summary}")
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, summary)
            return "I've found some results and will read them to you now."
        else:
            return summary
            
    except Exception as e:
        logging.error(f"[TOOL] brave_search_locanto_by_category exception: {e}")
        return sanitize_for_azure(f"Sorry, there was a problem searching Locanto: {e}")
