"""
Optimized Brave Search API integration for web search tools.
This module provides web search functionality using the Brave Search API with caching and rate limiting.
"""

import logging
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlencode
from livekit.agents import function_tool, RunContext

# Import the optimized Brave Search client
from brave_search_optimized import get_optimized_brave_search_client

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

def format_search_results(results: Dict[str, Any], query: str, num_results: int = 5) -> str:
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

@function_tool
async def web_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """Search the web for information using Brave Search API with caching and rate limiting.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return (1-10)
        
    Returns:
        str: Formatted search results with titles and URLs
    """
    logging.info(f"[TOOL] web_search called for query: {query}, num_results: {num_results}")
    
    try:
        # Get the optimized Brave Search client
        client = get_optimized_brave_search_client()
        
        # Check if this is a repeated query from the same session
        session = getattr(context, 'session', None)
        repeated_query = False
        
        if session and hasattr(session, 'userdata') and 'last_search_query' in session.userdata:
            last_query = session.userdata.get('last_search_query')
            if last_query == query:
                logging.info(f"Repeated query detected: {query}")
                repeated_query = True
        
        # Store the current query in session data
        if session and hasattr(session, 'userdata'):
            try:
                session.userdata['last_search_query'] = query
            except Exception as e:
                logging.warning(f"Could not set session.userdata['last_search_query']: {e}")
        
        # Perform the search with caching (use_cache=True is the default)
        # If it's a repeated query, we'll still use the cache but log it
        results = await client.search(query=query, count=num_results)
        
        if "error" in results:
            error_msg = f"Error searching with Brave API: {results['error']}"
            logging.error(error_msg)
            error_msg = sanitize_for_azure(f"I couldn't find any results for '{query}'. Try a different query.")
            
            # Handle session output for voice responses
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't find any results for your search."
            return error_msg
        
        # Format the results
        formatted_results = format_search_results(results, query, num_results)
        formatted_results = sanitize_for_azure(formatted_results)
        
        # Log cache statistics periodically
        if random.random() < 0.1:  # Log stats roughly 10% of the time
            stats = client.get_cache_stats()
            logging.info(f"Brave Search cache stats: {stats}")
        
        logging.info(f"[TOOL] web_search results: {formatted_results}")
        
        # Handle session output for voice responses
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

# Import missing modules
import random
