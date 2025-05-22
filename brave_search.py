"""
Brave Search API unified interface for livekit-amanda.

This module provides a unified interface for both Web Search and AI Search 
using the Brave Search API with separate API keys and rate limits for each.
"""

import logging
import aiohttp
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlencode

# Import specialized search modules
try:
    from brave_web_search import web_search as brave_web_search, get_brave_web_search_client
    HAS_WEB_SEARCH = True
    logging.info("Brave Web Search module loaded successfully")
except ImportError:
    HAS_WEB_SEARCH = False
    logging.warning("Brave Web Search module not available")
    
    # Define dummy function for when web search is not available
    async def brave_web_search(*args, **kwargs):
        return "Web search functionality not available"

try:
    from brave_ai_search import ai_search as brave_ai_search, get_brave_ai_search_client
    HAS_AI_SEARCH = True
    logging.info("Brave AI Search module loaded successfully")
except ImportError:
    HAS_AI_SEARCH = False
    logging.warning("Brave AI Search module not available")
    
    # Define dummy function for when AI search is not available
    async def brave_ai_search(*args, **kwargs):
        return "AI search functionality not available"

# Import statistics tracking if available
try:
    from brave_search_stats import get_stats, get_stats_report
    HAS_STATS_TRACKING = True
    logging.info("Brave Search statistics tracking enabled")
except ImportError:
    HAS_STATS_TRACKING = False
    logging.warning("Brave Search statistics tracking not available")
    
    # Define dummy functions for when stats module is not available
    def get_stats():
        return None
        
    def get_stats_report():
        return "Statistics tracking not available"

async def search(context, query, search_type: str = "web", num_results: int = 5) -> str:
    """Unified search function that routes to the appropriate search type.
    
    Args:
        context: The run context for the tool
        query: The search query
        search_type: Type of search to perform ("web" or "ai")
        num_results: Number of results to return (for web search only)
        
    Returns:
        Formatted search results as a string
    """
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query)
    logging.info(f"[TOOL] brave_search called for query: {query}, type: {search_type}")
    
    # Log statistics if available
    if HAS_STATS_TRACKING:
        stats = get_stats()
        if stats:
            session_stats = stats.get_session_stats()
            web_stats = stats.get_performance_stats("web")
            ai_stats = stats.get_performance_stats("ai")
            logging.info(f"[STATS] Web Search requests: {web_stats.get('total_requests', 0)}, AI Search requests: {ai_stats.get('total_requests', 0)}")
    
    # Route to the appropriate search function based on search_type
    if search_type.lower() == "ai":
        if not HAS_AI_SEARCH:
            return "AI search functionality is not available. Please try web search instead."
        
        try:
            return await brave_ai_search(context, query)
        except Exception as e:
            logging.error(f"Error during AI search: {str(e)}")
            return f"An error occurred during AI search: {str(e)}"
    else:  # Default to web search
        if not HAS_WEB_SEARCH:
            return "Web search functionality is not available."
        
        try:
            return await brave_web_search(context, query, num_results=num_results)
        except Exception as e:
            logging.error(f"Error during web search: {str(e)}")
            return f"An error occurred during web search: {str(e)}"

async def web_search(context, query: str, num_results: int = 5) -> str:
    """Perform a web search using the Brave Search API.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Formatted search results as a string
    """
    logging.info(f"[TOOL] brave_web_search called for query: {query}")
    
    if not HAS_WEB_SEARCH:
        return "Web search functionality is not available."
    
    return await brave_web_search(context, query, num_results=num_results)

async def ai_search(context, query: str) -> str:
    """Perform an AI search using the Brave Search API.
    
    Args:
        context: The run context for the tool
        query: The search query
        
    Returns:
        Formatted AI search results as a string
    """
    logging.info(f"[TOOL] brave_ai_search called for query: {query}")
    
    if not HAS_AI_SEARCH:
        return "AI search functionality is not available."
    
    return await brave_ai_search(context, query)

async def get_search_stats() -> str:
    """Get statistics about Brave Search API usage.
    
    Returns:
        Formatted statistics report as a string
    """
    if not HAS_STATS_TRACKING:
        return "Statistics tracking is not available."
    
    return get_stats_report()
