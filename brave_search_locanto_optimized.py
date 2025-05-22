"""
Optimized Brave Search API integration for Locanto searches.
This module provides functionality to search Locanto using the Brave Search API with caching and rate limiting.
"""

import logging
from typing import List, Dict, Any, Optional, Union, TypedDict
from livekit.agents import function_tool, RunContext

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

# Import the optimized Brave Search client
from brave_search_optimized import get_optimized_brave_search_client

class OptimizedBraveSearchLocantoClient:
    """Optimized client for Locanto searches using Brave Search API with caching and rate limiting."""
    
    def __init__(self):
        """Initialize the optimized Brave Search client for Locanto searches."""
        self.brave_client = get_optimized_brave_search_client()
    
    async def search(self, 
                    query: str, 
                    location: str = "", 
                    category: str = "",
                    count: int = 10) -> Dict[str, Any]:
        """Search Locanto using Brave Search API with caching and rate limiting.
        
        Args:
            query: Search query
            location: Optional location filter
            category: Optional category filter
            count: Number of results to return
            
        Returns:
            Dict containing the search results
        """
        # Build a more specific query for Brave Search
        search_query = query
        if location:
            search_query += f" {location}"
        if category:
            search_query += f" {category}"
        
        # Add site restriction to focus on Locanto
        search_query += " site:locanto.com"
        
        # Use the optimized client with caching
        return await self.brave_client.search(query=search_query, count=count)
    
    async def search_locanto(self, 
                           category_path: List[str] = ['personals', 'men-seeking-men'], 
                           location: str = 'western-cape', 
                           max_pages: int = 3) -> List[LocantoListing]:
        """Search Locanto.co.za for listings in a specific category and location using Brave Search API with caching.
        
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
            
            # Use the optimized client with caching
            results = await self.brave_client.search(query=search_query, count=count)
            
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
_optimized_brave_locanto_client = None

def get_optimized_brave_locanto_client() -> OptimizedBraveSearchLocantoClient:
    """Get or create a singleton instance of the OptimizedBraveSearchLocantoClient.
    
    Returns:
        OptimizedBraveSearchLocantoClient instance
    """
    global _optimized_brave_locanto_client
    if _optimized_brave_locanto_client is None:
        _optimized_brave_locanto_client = OptimizedBraveSearchLocantoClient()
    return _optimized_brave_locanto_client

@function_tool
async def basic_search_locanto(context: RunContext, query: str, location: str = "", category: str = "") -> str:
    """Search Locanto listings using Brave Search API with caching and rate limiting.
    
    Args:
        context: The run context for the tool
        query: The search query
        location: Optional location filter
        category: Optional category filter
        
    Returns:
        str: Formatted search results
    """
    try:
        # Get the optimized Brave Search client
        client = get_optimized_brave_locanto_client()
        
        # Check if this is a repeated query from the same session
        session = getattr(context, 'session', None)
        repeated_query = False
        
        if session and hasattr(session, 'userdata') and 'last_locanto_query' in session.userdata:
            last_query = session.userdata.get('last_locanto_query')
            if last_query == f"{query}_{location}_{category}":
                logging.info(f"Repeated Locanto query detected: {query}")
                repeated_query = True
        
        # Store the current query in session data
        if session and hasattr(session, 'userdata'):
            try:
                session.userdata['last_locanto_query'] = f"{query}_{location}_{category}"
            except Exception as e:
                logging.warning(f"Could not set session.userdata['last_locanto_query']: {e}")
        
        # Search with caching (use_cache=True is the default)
        results = await client.search(query=query, location=location, category=category, count=10)
        
        # Format and return the results
        formatted_results = client.format_search_results(results)
        
        # Store the results in the session for later use if needed
        if session is not None and "web" in results and "results" in results["web"]:
            url_map = {}
            for idx, result in enumerate(results["web"]["results"], 1):
                url = result.get("url", "")
                if url:
                    url_map[idx] = url
            
            try:
                session.userdata['last_locanto_urls'] = url_map
            except Exception as e:
                logging.warning(f"Could not set session.userdata['last_locanto_urls']: {e}")
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"Brave Search API error: {e}", exc_info=True)
        return f"I couldn't search using Brave Search API: {str(e)}"

@function_tool
async def search_locanto(context: RunContext, category_path: str = 'personals/men-seeking-men', 
                        location: str = 'western-cape', max_pages: int = 3, return_url: bool = False) -> str:
    """Search Locanto by category using Brave Search API with caching and rate limiting.
    
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
        # Get the optimized Brave Search client
        client = get_optimized_brave_locanto_client()
        
        # Check if this is a repeated query from the same session
        session = getattr(context, 'session', None)
        repeated_query = False
        
        if session and hasattr(session, 'userdata') and 'last_locanto_category_query' in session.userdata:
            last_query = session.userdata.get('last_locanto_category_query')
            if last_query == f"{category_path}_{location}_{max_pages}":
                logging.info(f"Repeated Locanto category query detected: {category_path}")
                repeated_query = True
        
        # Store the current query in session data
        if session and hasattr(session, 'userdata'):
            try:
                session.userdata['last_locanto_category_query'] = f"{category_path}_{location}_{max_pages}"
            except Exception as e:
                logging.warning(f"Could not set session.userdata['last_locanto_category_query']: {e}")
        
        # Convert category path string to list
        categories = category_path.split('/')
        
        # Search using Brave Search API with caching
        listings = await client.search_locanto(categories, location, max_pages)
        
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
        logging.info(f"[TOOL] search_locanto summary: {summary}")
        
        if session:
            await handle_tool_results(session, summary)
            return "I've found some results and will read them to you now."
        else:
            return summary
            
    except Exception as e:
        logging.error(f"[TOOL] search_locanto exception: {e}")
        return sanitize_for_azure(f"Sorry, there was a problem searching Locanto: {e}")
