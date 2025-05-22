"""
Master integration module for Brave Search API implementations.
This module provides a centralized way to import all optimized Brave Search functions.
"""

import logging
from typing import Dict, Any

# Import all optimized implementations
from brave_search_tools_optimized import web_search, fallback_web_search
from brave_search_indeed_optimized import indeed_job_search, search_indeed_jobs
from brave_search_locanto_optimized import basic_search_locanto, search_locanto
from brave_search_optimized import get_optimized_brave_search_client

# Export all functions
__all__ = [
    'web_search',
    'fallback_web_search',
    'indeed_job_search', 
    'search_indeed_jobs',
    'basic_search_locanto',
    'search_locanto',
    'get_cache_stats',
    'clear_cache'
]

# Log that we're using the optimized Brave Search API implementation
logging.info("Using optimized Brave Search API implementation with caching and rate limiting")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for the Brave Search API.
    
    Returns:
        Dictionary with cache statistics
    """
    client = get_optimized_brave_search_client()
    return client.get_cache_stats()

def clear_cache() -> None:
    """Clear the Brave Search API cache."""
    client = get_optimized_brave_search_client()
    client.clear_cache()
    logging.info("Brave Search cache cleared")
