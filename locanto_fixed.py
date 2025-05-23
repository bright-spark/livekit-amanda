"""
Fixed version of the Locanto search functionality that handles the case when VoiceAgent userdata is not set.
"""

import logging
from typing import Optional
from livekit.agents import function_tool, RunContext

# Import the original locanto module
import locanto

@function_tool
async def search_locanto_fixed(context: RunContext, category_path: str = 'personals/men-seeking-men', location: str = 'western-cape', max_pages: int = 3, return_url: bool = False) -> str:
    """Search for listings on Locanto with improved error handling."""
    try:
        # Import the correct Brave Search client
        try:
            from brave_web_search import get_brave_web_search_client
            # Build a more specific query for Brave Search
            search_query = category_path.replace('/', ' ')
            if location:
                search_query += f" {location}"
            
            # Add site restriction to focus on Locanto
            search_query += " site:locanto.co.za"
            
            # Get the Brave Web Search client and perform the search
            brave_client = await get_brave_web_search_client()
            logging.info("Using Brave Web Search client for Locanto search")
        except ImportError:
            # Fallback to the unified client if web search module is not available
            from brave_search_api import get_brave_search_client
            
            # Build a more specific query for Brave Search
            search_query = category_path.replace('/', ' ')
            if location:
                search_query += f" {location}"
            
            # Add site restriction to focus on Locanto
            search_query += " site:locanto.co.za"
            
            # Get the Brave Search client and perform the search
            brave_client = await get_brave_search_client()
            logging.info("Using fallback Brave Search client for Locanto search")
        
        # Calculate total results to fetch based on max_pages (assuming 10 results per page)
        count = min(max_pages * 10, 20)  # Brave API has a max of 20 results per request
        
        results = await brave_client.search(query=search_query, count=count)
        
        if "error" in results:
            summary = f"Search error: {results['error']}"
        elif "web" not in results or "results" not in results["web"] or not results["web"]["results"]:
            summary = "No listings found matching your criteria."
        else:
            web_results = results["web"]["results"]
            first_url = None
            url_map = {}
            summary = f"Found {len(web_results)} listings on Locanto:\n\n"
            
            for idx, result in enumerate(web_results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                description = result.get("description", "No description")
                
                title = locanto.clean_spoken(title)
                summary += f"{idx}. {title}\n"
                
                if url:
                    url_map[idx] = url
                    if not first_url:
                        first_url = url
                    summary += f"URL: {url}\n"
                
                if description:
                    desc = description[:200] + '...' if len(description) > 200 else description
                    summary += f"Description: {locanto.clean_spoken(desc)}\n"
                
                # Extract location from description or title if possible
                location_match = None
                if location:
                    location_match = location
                summary += f"Location: {locanto.clean_spoken(location_match or 'Not specified')}\n"
                
                summary += "\n"
            
            # Try to store mapping in session.userdata for later use, but don't fail if it's not available
            try:
                session = getattr(context, 'session', None)
                if session is not None and hasattr(session, 'userdata'):
                    session.userdata['last_locanto_urls'] = url_map
            except Exception as e:
                logging.warning(f"Could not set session.userdata['last_locanto_urls']: {e}")
            
            if first_url and return_url:
                return first_url
                
            if first_url:
                summary += f"Would you like to open the first listing in your browser?"
        
        summary = locanto.sanitize_for_azure(summary)
        summary = locanto.clean_spoken(summary)
        logging.info(f"[TOOL] search_locanto summary: {summary}")
        
        # Try to handle tool results, but don't fail if it's not available
        try:
            session = getattr(context, 'session', None)
            if session is not None:
                try:
                    await locanto.handle_tool_results(session, summary)
                    return "I've found some results and will read them to you now."
                except Exception as e:
                    logging.warning(f"Error in handle_tool_results: {e}")
                    return summary
            else:
                return summary
        except Exception as e:
            logging.warning(f"Error handling session: {e}")
            return summary
            
    except Exception as e:
        logging.error(f"[TOOL] search_locanto exception: {e}")
        return locanto.sanitize_for_azure(f"Sorry, there was a problem searching Locanto: {e}")
