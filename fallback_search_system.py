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
import sys
from typing import List, Dict, Any, Optional, Union

# Import debug_search_result function from agent.py
try:
    from agent import debug_search_result, SEARCH_RESULT_MAX_CHARS
    HAS_DEBUG_FUNCTION = True
except ImportError:
    HAS_DEBUG_FUNCTION = False
    # Default max chars if we can't import from agent.py
    SEARCH_RESULT_MAX_CHARS = 1000
    # Fallback debug function if agent.py can't be imported
    def debug_search_result(search_engine: str, query: str, results: any) -> None:
        """Fallback debug output function for search results."""
        # Convert results to string if not already
        if not isinstance(results, str):
            try:
                results_str = str(results)
            except Exception as e:
                results_str = f"[Error converting results to string: {e}]"
        else:
            results_str = results
            
        separator = "=" * 80
        print(separator)
        print(f"[DEBUG] {search_engine} RESULTS FOR: '{query}'")
        print(separator)
        
        # Always print results, but truncate if they're too long
        if len(results_str) > SEARCH_RESULT_MAX_CHARS:
            print(f"{results_str[:SEARCH_RESULT_MAX_CHARS]}...")
            print(f"[TRUNCATED - {len(results_str)} total characters]")
        else:
            print(results_str)
        
        print(separator)
        sys.stdout.flush()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if search engines are enabled via environment variables
BRAVE_SEARCH_ENABLED = os.environ.get("BRAVE_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
DUCKDUCKGO_SEARCH_ENABLED = os.environ.get("DUCKDUCKGO_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
BING_SEARCH_ENABLED = os.environ.get("BING_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
GOOGLE_SEARCH_ENABLED = os.environ.get("GOOGLE_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
WIKIPEDIA_ENABLED = os.environ.get("WIKIPEDIA_ENABLE", "true").lower() in ("true", "1", "yes")

# Try to import Brave Search
HAS_BRAVE_SEARCH = False
if BRAVE_SEARCH_ENABLED:
    try:
        from brave_search_free_tier import web_search as brave_search_api
        HAS_BRAVE_SEARCH = True
        logger.info("Brave Search API available and enabled")
    except ImportError:
        try:
            # Try alternative import path
            from brave_search import web_search as brave_search_api
            HAS_BRAVE_SEARCH = True
            logger.info("Brave Search API available and enabled (alternative import)")
        except ImportError:
            logger.warning("Brave Search API not available")
else:
    logger.info("Brave Search API disabled via environment variable")
    
# Define our own brave_web_search function to add debug output
async def brave_web_search(query: str, num_results: int = 5) -> str:
    """Search the web using Brave Search API and return formatted results.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Formatted string with search results
    """
    if not HAS_BRAVE_SEARCH:
        return f"Brave Search API is not available."
    
    try:
        logger.info(f"Using Brave Search API for query: '{query}'")
        results = await brave_search_api(query, num_results)
        
        # Print debug output
        debug_search_result("BRAVE SEARCH API", query, results)
        
        return results
    except Exception as e:
        logger.error(f"Brave Search API error: {e}")
        return f"Error searching with Brave Search API for '{query}': {str(e)}"

# Try to import DuckDuckGo Search
HAS_DUCKDUCKGO = False
if DUCKDUCKGO_SEARCH_ENABLED:
    try:
        from duckduckgo_search import ddg_web_search as ddg_search_api
        HAS_DUCKDUCKGO = True
        logger.info("DuckDuckGo Search available and enabled")
    except ImportError:
        logger.warning("DuckDuckGo Search not available")
else:
    logger.info("DuckDuckGo Search disabled via environment variable")
    
# Define our own ddg_web_search function to add debug output
async def ddg_web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Formatted string with search results
    """
    if not HAS_DUCKDUCKGO:
        return f"DuckDuckGo Search is not available."
    
    try:
        logger.info(f"Using DuckDuckGo Search for query: '{query}'")
        results = await ddg_search_api(query, num_results)
        
        # Print debug output
        debug_search_result("DUCKDUCKGO SEARCH", query, results)
        
        return results
    except Exception as e:
        logger.error(f"DuckDuckGo Search error: {e}")
        return f"Error searching with DuckDuckGo for '{query}': {str(e)}"

# Try to import optimized Bing Search for speed
HAS_BING_FAST = False
if BING_SEARCH_ENABLED:
    try:
        from bing_search import scrape_bing as bing_scrape_fast
        from bing_search import async_scrape_bing as bing_async_scrape_fast
        from bing_search import bing_search as bing_search_fast
        from bing_search import format_bing_results as format_bing_results_fast
        HAS_BING_FAST = True
        logger.info("Fast Bing Search available and enabled from bing_search.py")
    except ImportError:
        logger.warning("Fast Bing Search not available")

    # Try to import enriched Bing Search for quality
    HAS_BING_QUALITY = False
    try:
        from bing_extended import scrape_bing as bing_scrape_quality
        from bing_extended import bing_search as bing_search_quality
        HAS_BING_QUALITY = True
        logger.info("Quality Bing Search available and enabled from bing_extended.py")
    except ImportError:
        logger.warning("Quality Bing Search not available")
else:
    logger.info("Bing Search disabled via environment variable")
    HAS_BING_QUALITY = False

# Try to import Google Search
HAS_GOOGLE = False
if GOOGLE_SEARCH_ENABLED:
    try:
        from googlesearch import search as google_search_func
        HAS_GOOGLE = True
        logger.info("Google Search available and enabled")
    except ImportError:
        logger.warning("Google Search not available")
else:
    logger.info("Google Search disabled via environment variable")

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
                
                # Print debug output
                debug_search_result("BING SEARCH (FAST)", query, results)
                
                return results
            else:
                # Otherwise use the direct scraper
                results = await bing_async_scrape_fast(query, num_results)
                formatted_results = format_bing_results_fast(results, query)
                
                # Print debug output
                debug_search_result("BING SEARCH (FAST)", query, formatted_results)
                
                return formatted_results
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
                
                # Print debug output
                debug_search_result("BING SEARCH (QUALITY)", query, results)
                
                return results
            else:
                # Otherwise use the direct scraper
                results = await bing_scrape_quality(query, num_results=num_results)
                formatted_results = await format_bing_results(results, query)
                
                # Print debug output
                debug_search_result("BING SEARCH (QUALITY)", query, formatted_results)
                
                return formatted_results
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
        
        # Print debug output
        debug_search_result("GOOGLE SEARCH", query, formatted)
        
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
    
    # Try Brave Search first if available and enabled
    if BRAVE_SEARCH_ENABLED and HAS_BRAVE_SEARCH:
        try:
            logger.info(f"Trying Brave Search API for query: '{query}'")
            brave_results = await brave_web_search(query, num_results)
            
            # Print debug output
            debug_search_result("FALLBACK BRAVE SEARCH", query, brave_results)
            
            if brave_results and "I couldn't find any results" not in brave_results:
                logger.info(f"Brave Search succeeded for query: '{query}'")
                return brave_results
            logger.warning(f"Brave Search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"Brave Search error: {e}")
    
    # Try Bing as second fallback if enabled
    # First try fast Bing search (optimized for speed)
    if BING_SEARCH_ENABLED and HAS_BING_FAST:
        try:
            logger.info(f"Trying Fast Bing Search for query: '{query}'")
            if context:
                # If we have a context, use the function_tool version
                bing_results = await bing_search_fast(context, query, num_results)
                
                # Print debug output
                debug_search_result("FALLBACK BING SEARCH (FAST)", query, bing_results)
                
                if bing_results and "No results found" not in bing_results:
                    logger.info(f"Fast Bing search succeeded for query: '{query}'")
                    return bing_results
            else:
                # Otherwise use the direct scraper
                bing_results = await bing_async_scrape_fast(query, num_results)
                if bing_results:
                    logger.info(f"Fast Bing search succeeded for query: '{query}'")
                    formatted_results = format_bing_results_fast(bing_results, query)
                    
                    # Print debug output
                    debug_search_result("FALLBACK BING SEARCH (FAST)", query, formatted_results)
                    
                    return formatted_results
            logger.warning(f"Fast Bing search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"Fast Bing Search error: {e}")
    
    # If fast Bing search fails, try quality Bing search if enabled
    if BING_SEARCH_ENABLED and HAS_BING_QUALITY:
        try:
            logger.info(f"Trying Quality Bing Search for query: '{query}'")
            if context:
                # If we have a context, use the function_tool version
                bing_results = await bing_search_quality(context, query, num_results)
                
                # Print debug output
                debug_search_result("FALLBACK BING SEARCH (QUALITY)", query, bing_results)
                
                if bing_results and "No results found" not in bing_results:
                    logger.info(f"Quality Bing search succeeded for query: '{query}'")
                    return bing_results
            else:
                # Otherwise use the direct scraper
                bing_results = await bing_scrape_quality(query, num_results)
                if bing_results:
                    logger.info(f"Quality Bing search succeeded for query: '{query}'")
                    formatted_results = await format_bing_results(bing_results, query)
                    return formatted_results
            logger.warning(f"Quality Bing search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"Quality Bing Search error: {e}")
    
    # Try DuckDuckGo as first fallback if enabled
    if DUCKDUCKGO_SEARCH_ENABLED and HAS_DUCKDUCKGO:
        try:
            logger.info(f"Trying DuckDuckGo Search for query: '{query}'")
            ddg_results = await ddg_web_search(query, num_results)
            
            # Print debug output
            debug_search_result("FALLBACK DUCKDUCKGO SEARCH", query, ddg_results)
            
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
                
                # Print debug output
                debug_search_result("FALLBACK QUALITY BING SEARCH", query, bing_results)
            else:
                results = await bing_scrape_quality(query, num_results=num_results)
                bing_results = await format_bing_results(results, query)
                
                # Print debug output
                debug_search_result("FALLBACK QUALITY BING SEARCH", query, bing_results)
                
            if bing_results and "No results found" not in bing_results and "An error occurred" not in bing_results:
                logger.info(f"Quality Bing Search succeeded for query: '{query}'")
                return bing_results
            logger.warning(f"Quality Bing Search returned no results for query: '{query}'")
        except Exception as e:
            logger.error(f"Quality Bing Search error: {e}")
    
    # Try Google as final fallback if enabled
    if GOOGLE_SEARCH_ENABLED and HAS_GOOGLE:
        try:
            logger.info(f"Trying Google Search for query: '{query}'")
            google_results = await google_web_search(query, num_results)
            
            # Print debug output
            debug_search_result("FALLBACK GOOGLE SEARCH", query, google_results)
            
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
        
        # Ensure num_results is an integer
        if not isinstance(num_results, int):
            try:
                num_results = int(num_results)
            except (ValueError, TypeError):
                num_results = 5
        
        # Limit number of results to a reasonable range
        num_results = max(1, min(num_results, 20))
        
        # ALWAYS try Brave Search first if enabled and available
        if BRAVE_SEARCH_ENABLED and HAS_BRAVE_SEARCH:
            try:
                logger.info(f"Unified web search using Brave Search API for query: '{query}'")
                brave_results = await brave_web_search(query, num_results)
                
                if brave_results and "I couldn't find any results" not in brave_results:
                    logger.info(f"Brave Search succeeded for query: '{query}'")
                    
                    # Handle session output for voice responses if available
                    session = getattr(context, 'session', None)
                    if session and hasattr(session, 'add_message'):
                        await session.add_message(role="assistant", content=brave_results)
                        return "I've found some results using Brave Search and will read them to you now."
                    
                    return brave_results
                
                logger.warning(f"Brave Search returned no results for query: '{query}', falling back to other search methods")
            except Exception as e:
                logger.error(f"Brave Search error in unified_web_search: {e}")
        else:
            if not BRAVE_SEARCH_ENABLED:
                logger.info("Brave Search API disabled via environment variable, using fallback search methods")
            else:
                logger.info("Brave Search API not available, using fallback search methods")
        
        # If Brave Search fails or is not available, use the fallback search system
        results = await fallback_search(query, num_results, context)
        
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
