"""
Fallback Search System for LiveKit Amanda.

This module implements a multi-level fallback search system with the following hierarchy:
1. Brave Search API (if API key exists)
2. DuckDuckGo Search (primary fallback)
3. Bing Search (secondary fallback)
4. Google Search (tertiary fallback)
5. Other search engines (final fallback)

The system tries each search method in order until one succeeds.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Brave Search
HAS_BRAVE_SEARCH = False
try:
    from brave_search_free_tier import web_search as brave_web_search
    HAS_BRAVE_SEARCH = True
    logger.info("Brave Search API available")
except ImportError:
    logger.warning("Brave Search API not available")

# Try to import DuckDuckGo Search
HAS_DUCKDUCKGO = False
try:
    from duckduckgo_search import ddg_web_search
    HAS_DUCKDUCKGO = True
    logger.info("DuckDuckGo Search available")
except ImportError:
    logger.warning("DuckDuckGo Search not available")

# Try to import optimized Bing Search for speed
HAS_BING_FAST = False
try:
    from bing_search import scrape_bing as bing_scrape_fast
    from bing_search import async_scrape_bing as bing_async_scrape_fast
    from bing_search import bing_search as bing_search_fast
    from bing_search import format_bing_results as format_bing_results_fast
    HAS_BING_FAST = True
    logger.info("Fast Bing Search available from bing_search.py")
except ImportError:
    logger.warning("Fast Bing Search not available")

# Try to import enriched Bing Search for quality
HAS_BING_QUALITY = False
try:
    from bing_extended import scrape_bing as bing_scrape_quality
    from bing_extended import bing_search as bing_search_quality
    HAS_BING_QUALITY = True
    logger.info("Quality Bing Search available from bing_extended.py")
except ImportError:
    logger.warning("Quality Bing Search not available")

# Try to import Google Search
HAS_GOOGLE = False
try:
    from googlesearch import search as google_search_func
    HAS_GOOGLE = True
    logger.info("Google Search available")
except ImportError:
    logger.warning("Google Search not available")

async def format_bing_results(results: List[Dict[str, Any]], query: str) -> str:
    """Format Bing search results into a readable string."""
    if not results:
        return f"No results found for '{query}' on Bing."
    
    formatted = f"Bing search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        link = result.get('link', 'No link')
        snippet = result.get('snippet', '')
        
        formatted += f"{i}. {title}\n   {link}\n"
        if snippet:
            formatted += f"   {snippet}\n"
        formatted += "\n"
    
    return formatted

async def bing_web_search(query: str, num_results: int = 5, context=None) -> str:
    """Search the web using Bing and return formatted results.
    
    Args:
        query: The search query
        num_results: Number of results to return
        context: Optional run context for the tool
        
    Returns:
        Formatted string with search results
    """
    # First try fast Bing search from bing_search.py (optimized for speed)
    if HAS_BING_FAST:
        try:
            logger.info(f"Using fast Bing search from bing_search.py for query: '{query}'")
            if context:
                # If we have a context, use the function_tool version
                results = await bing_search_fast(context, query, num_results)
                return results
            else:
                # Otherwise use the direct scraper
                results = await bing_async_scrape_fast(query, num_results)
                return format_bing_results_fast(results, query)
        except Exception as e:
            logger.error(f"Fast Bing search error: {e}")
            logger.info(f"Falling back to quality Bing search for query: '{query}'")
    
    # Fall back to quality Bing search from bing_extended.py
    if HAS_BING_QUALITY:
        try:
            logger.info(f"Using quality Bing search from bing_extended.py for query: '{query}'")
            if context:
                # If we have a context, use the function_tool version
                results = await bing_search_quality(context, query, num_results)
                return results
            else:
                # Otherwise use the direct scraper
                results = await bing_scrape_quality(query, num_results=num_results)
                return await format_bing_results(results, query)
        except Exception as e:
            logger.error(f"Quality Bing search error: {e}")
    
    return f"Bing Search is not available or failed for query: '{query}'"

async def google_web_search(query: str, num_results: int = 5) -> str:
    """Search the web using Google and return formatted results."""
    if not HAS_GOOGLE:
        return f"Google Search is not available."
    
    try:
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(google_search_func(query, num_results=num_results))
        )
        
        if not results:
            return f"No results found for '{query}' on Google."
        
        formatted = f"Google search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result}\n\n"
        
        return formatted
    except Exception as e:
        logger.error(f"Google search error: {e}")
        return f"Error searching Google for '{query}': {str(e)}"

async def fallback_search(query: str, num_results: int = 5, context=None) -> str:
    """
    Perform a search using available search engines in fallback order.
    
    Args:
        query: The search query
        num_results: Number of results to return
        context: Optional run context for the tool
        
    Returns:
        Formatted string with search results
    """
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query)
    
    # Ensure num_results is an integer
    if not isinstance(num_results, int):
        try:
            num_results = int(num_results)
        except (ValueError, TypeError):
            num_results = 5
    
    # Limit number of results to a reasonable range
    num_results = max(1, min(num_results, 20))
    
    logger.info(f"Fallback search called for query: '{query}', num_results: {num_results}")
    
    # Try Brave Search first (if available)
    if HAS_BRAVE_SEARCH and os.environ.get("BRAVE_API_KEY"):
        try:
            logger.info(f"Trying Brave Search for query: '{query}'")
            brave_results = await brave_web_search(query, num_results)
            if brave_results and "I couldn't find any results" not in brave_results:
                logger.info(f"Brave Search succeeded for query: '{query}'")
                return brave_results
            logger.warning(f"Brave Search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"Brave Search error: {e}")
    
    # Try Fast Bing Search as first fallback (optimized for speed)
    if HAS_BING_FAST:
        try:
            logger.info(f"Trying Fast Bing Search for query: '{query}'")
            if context:
                bing_results = await bing_search_fast(context, query, num_results)
            else:
                results = await bing_async_scrape_fast(query, num_results)
                bing_results = format_bing_results_fast(results, query)
                
            if bing_results and "No results found" not in bing_results and "An error occurred" not in bing_results:
                logger.info(f"Fast Bing Search succeeded for query: '{query}'")
                return bing_results
            logger.warning(f"Fast Bing Search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"Fast Bing Search error: {e}")
    
    # Try DuckDuckGo as second fallback
    if HAS_DUCKDUCKGO:
        try:
            logger.info(f"Trying DuckDuckGo Search for query: '{query}'")
            ddg_results = await ddg_web_search(query, num_results)
            if ddg_results and "No results found" not in ddg_results:
                logger.info(f"DuckDuckGo Search succeeded for query: '{query}'")
                return ddg_results
            logger.warning(f"DuckDuckGo Search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"DuckDuckGo Search error: {e}")
    
    # Try Quality Bing Search as third fallback (for more comprehensive results)
    if HAS_BING_QUALITY:
        try:
            logger.info(f"Trying Quality Bing Search for query: '{query}'")
            if context:
                bing_results = await bing_search_quality(context, query, num_results)
            else:
                results = await bing_scrape_quality(query, num_results=num_results)
                bing_results = await format_bing_results(results, query)
                
            if bing_results and "No results found" not in bing_results and "An error occurred" not in bing_results:
                logger.info(f"Quality Bing Search succeeded for query: '{query}'")
                return bing_results
            logger.warning(f"Quality Bing Search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"Quality Bing Search error: {e}")
    
    # Try Google as final fallback
    if HAS_GOOGLE:
        try:
            logger.info(f"Trying Google Search for query: '{query}'")
            google_results = await google_web_search(query, num_results)
            if google_results and "No results found" not in google_results:
                logger.info(f"Google Search succeeded for query: '{query}'")
                return google_results
            logger.warning(f"Google Search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"Google Search error: {e}")
    
    # All search methods failed
    return f"I couldn't find any results for '{query}' using any available search engines. Please try a different query or approach."

async def unified_web_search(context, query: str, num_results: int = 5) -> str:
    """
    Unified web search function that uses the fallback search system.
    This function can be registered as a tool with the agent.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Formatted string with search results
    """
    try:
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)
        
        results = await fallback_search(query, num_results)
        
        # Handle session output for voice responses if available
        session = getattr(context, 'session', None)
        if session and hasattr(session, 'add_message'):
            await session.add_message(role="assistant", content=results)
            return "I've found some results and will read them to you now."
        
        return results
    except Exception as e:
        logger.error(f"Unified web search error: {e}")
        error_msg = f"I couldn't find any results for '{query}'. Try a different query."
        
        # Handle session output for voice responses if available
        session = getattr(context, 'session', None)
        if session and hasattr(session, 'add_message'):
            await session.add_message(role="assistant", content=error_msg)
            return "I couldn't find any results for your search."
        
        return error_msg
