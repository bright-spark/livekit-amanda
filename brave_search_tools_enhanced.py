"""
Enhanced Brave Search API integration for web search tools.
This module provides optimized web search functionality using the Brave Search API
with advanced caching, rate limiting, and performance optimizations.
"""

import logging
import asyncio
import json
import os
import time
import random
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from urllib.parse import urlencode
from livekit.agents import function_tool, RunContext

# Import the enhanced Brave Search client
from brave_search_optimized_enhanced import get_enhanced_brave_search_client

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

# Query preprocessing to improve cache hit rates
def preprocess_query(query: str) -> str:
    """Preprocess the query to improve cache hit rates.
    
    Args:
        query: The original search query
        
    Returns:
        Preprocessed query
    """
    # Convert to lowercase
    query = query.lower()
    
    # Remove extra whitespace
    query = " ".join(query.split())
    
    # Remove common filler words that don't affect search results
    filler_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "like"}
    words = query.split()
    if len(words) > 3:  # Only remove filler words if the query is long enough
        words = [word for word in words if word not in filler_words]
        query = " ".join(words)
    
    return query

def extract_key_terms(query: str, max_terms: int = 5) -> str:
    """Extract key terms from a query to improve cache hit rates.
    
    Args:
        query: The original search query
        max_terms: Maximum number of key terms to extract
        
    Returns:
        String of key terms
    """
    # Simple implementation - split by spaces and take the first max_terms words
    # In a production environment, you might use NLP techniques for better extraction
    words = query.lower().split()
    
    # Filter out common stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "like",
                 "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
                 "can", "could", "will", "would", "shall", "should", "may", "might", "must", "of", "from"}
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # If we filtered out too many words, fall back to original words
    if len(filtered_words) < 2 and len(words) >= 2:
        filtered_words = words
    
    # Take up to max_terms words
    key_terms = filtered_words[:max_terms]
    
    return " ".join(key_terms)

def format_search_results(results: Dict[str, Any], query: str, num_results: int = 5) -> str:
    """Format search results into a readable string with enhanced information.
    
    Args:
        results: Search results from the Brave Search API
        query: The original search query
        num_results: Maximum number of results to include
        
    Returns:
        Formatted string of search results
    """
    if "error" in results:
        return f"Search error: {results['error']}"
    
    if "web" not in results or "results" not in results["web"] or not results["web"]["results"]:
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
        
        # Add extra information if available
        if "extra" in result:
            extra = result["extra"]
            if "age" in extra:
                formatted += f"   Published: {extra['age']}\n"
        
        formatted += "\n"
    
    # Add search metadata if available
    if "meta" in results and "query" in results["meta"]:
        meta = results["meta"]
        if "total" in meta:
            formatted += f"Found approximately {meta['total']} results in total.\n"
    
    return formatted

# Session cache to avoid redundant API calls within the same session
_session_cache = {}

@function_tool
async def web_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """Search the web for information using enhanced Brave Search API with advanced caching and rate limiting.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return (1-20)
        
    Returns:
        str: Formatted search results with titles and URLs
    """
    start_time = time.time()
    logging.info(f"[TOOL] web_search called for query: {query}, num_results: {num_results}")
    
    try:
        # Preprocess the query to improve cache hit rates
        processed_query = preprocess_query(query)
        
        # Check session cache first (fastest)
        session = getattr(context, 'session', None)
        session_id = id(session) if session else "no_session"
        cache_key = f"{session_id}:{processed_query}:{num_results}"
        
        if cache_key in _session_cache:
            logging.info(f"Session cache hit for query: {query}")
            formatted_results = _session_cache[cache_key]
            
            # Log performance
            elapsed = time.time() - start_time
            logging.info(f"[PERFORMANCE] web_search completed in {elapsed:.4f}s (session cache hit)")
            
            return formatted_results
        
        # Get the enhanced Brave Search client
        client = await get_enhanced_brave_search_client()
        
        # Check if this is a repeated query from the same session
        repeated_query = False
        
        if session and hasattr(session, 'userdata') and 'last_search_query' in session.userdata:
            last_query = session.userdata.get('last_search_query')
            if last_query == processed_query:
                logging.info(f"Repeated query detected: {processed_query}")
                repeated_query = True
        
        # Store the current query in session data
        if session and hasattr(session, 'userdata'):
            try:
                session.userdata['last_search_query'] = processed_query
            except Exception as e:
                logging.warning(f"Could not set session.userdata['last_search_query']: {e}")
        
        # Determine request priority (lower is higher priority)
        # Prioritize shorter queries as they're more likely to be important
        priority = min(len(processed_query.split()), 10)
        
        # Perform the search with caching
        results = await client.search(
            query=processed_query,
            count=min(num_results + 2, 20),  # Request a few extra results in case some are filtered
            use_cache=True,
            priority=priority
        )
        
        if "error" in results:
            error_msg = f"Error searching with Brave API: {results['error']}"
            logging.error(error_msg)
            error_msg = sanitize_for_azure(f"I couldn't find any results for '{query}'. Try a different query.")
            
            # Handle session output for voice responses
            if session:
                await handle_tool_results(session, error_msg)
                
                # Log performance
                elapsed = time.time() - start_time
                logging.info(f"[PERFORMANCE] web_search completed in {elapsed:.4f}s (error)")
                
                return "I couldn't find any results for your search."
            
            # Log performance
            elapsed = time.time() - start_time
            logging.info(f"[PERFORMANCE] web_search completed in {elapsed:.4f}s (error)")
            
            return error_msg
        
        # Format the results
        formatted_results = format_search_results(results, query, num_results)
        formatted_results = sanitize_for_azure(formatted_results)
        
        # Store in session cache
        _session_cache[cache_key] = formatted_results
        
        # Limit session cache size
        if len(_session_cache) > 100:
            # Remove oldest entries
            keys = list(_session_cache.keys())
            for old_key in keys[:20]:  # Remove 20 oldest entries
                if old_key in _session_cache:
                    del _session_cache[old_key]
        
        # Log cache statistics periodically
        if random.random() < 0.1:  # Log stats roughly 10% of the time
            stats = client.get_cache_stats()
            logging.info(f"Brave Search cache stats: {stats}")
        
        # Handle session output for voice responses
        if session:
            await handle_tool_results(session, formatted_results)
            
            # Log performance
            elapsed = time.time() - start_time
            logging.info(f"[PERFORMANCE] web_search completed in {elapsed:.4f}s (with session)")
            
            return "I've found some results and will read them to you now."
        
        # Log performance
        elapsed = time.time() - start_time
        logging.info(f"[PERFORMANCE] web_search completed in {elapsed:.4f}s")
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"[TOOL] web_search exception: {e}")
        error_msg = sanitize_for_azure(f"I couldn't find any results for '{query}'. Try a different query.")
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            
            # Log performance
            elapsed = time.time() - start_time
            logging.info(f"[PERFORMANCE] web_search completed in {elapsed:.4f}s (exception)")
            
            return "I couldn't find any results for your search."
        
        # Log performance
        elapsed = time.time() - start_time
        logging.info(f"[PERFORMANCE] web_search completed in {elapsed:.4f}s (exception)")
        
        return error_msg

@function_tool
async def fallback_web_search(context: RunContext, query: str, num_results: int = 10) -> str:
    """Search the web for information using enhanced Brave Search API as a fallback.
    
    This function is maintained for compatibility with existing code but uses
    the enhanced Brave Search API implementation.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return (1-20)
        
    Returns:
        str: Formatted search results with titles and URLs
    """
    logging.info(f"[TOOL] fallback_web_search called for query: {query}")
    
    # Just use the regular web_search implementation since we're using Brave API for both
    return await web_search(context, query, num_results)

@function_tool
async def batch_web_search(context: RunContext, queries: List[str], num_results: int = 5) -> Dict[str, str]:
    """Perform multiple web searches in parallel with optimized performance.
    
    Args:
        context: The run context for the tool
        queries: List of search queries
        num_results: Number of results to return per query (1-20)
        
    Returns:
        Dict mapping queries to their formatted search results
    """
    logging.info(f"[TOOL] batch_web_search called for {len(queries)} queries")
    start_time = time.time()
    
    try:
        # Get the enhanced Brave Search client
        client = await get_enhanced_brave_search_client()
        
        # Preprocess queries
        processed_queries = [preprocess_query(q) for q in queries]
        
        # Check which queries are already in session cache
        session = getattr(context, 'session', None)
        session_id = id(session) if session else "no_session"
        
        results_dict = {}
        queries_to_search = []
        query_indices = []
        
        # First check session cache
        for i, (original_query, processed_query) in enumerate(zip(queries, processed_queries)):
            cache_key = f"{session_id}:{processed_query}:{num_results}"
            
            if cache_key in _session_cache:
                logging.info(f"Session cache hit for query: {original_query}")
                results_dict[original_query] = _session_cache[cache_key]
            else:
                queries_to_search.append(processed_query)
                query_indices.append(i)
        
        # If there are queries not in session cache, search them individually
        # (safer than batch search with the current implementation)
        if queries_to_search:
            search_results = []
            
            # Process each query individually to avoid priority queue issues
            for i, processed_query in enumerate(queries_to_search):
                # Determine priority based on query length
                priority = min(len(processed_query.split()), 10)
                
                # Perform search with priority
                result = await client.search(
                    query=processed_query,
                    count=min(num_results + 2, 20),
                    use_cache=True,
                    priority=priority
                )
                
                search_results.append(result)
            
            # Process results
            for i, (processed_query, result) in enumerate(zip(queries_to_search, search_results)):
                original_query = queries[query_indices[i]]
                
                if "error" in result:
                    error_msg = sanitize_for_azure(f"I couldn't find any results for '{original_query}'.")
                    results_dict[original_query] = error_msg
                else:
                    # Format the results
                    formatted_result = format_search_results(result, original_query, num_results)
                    formatted_result = sanitize_for_azure(formatted_result)
                    results_dict[original_query] = formatted_result
                    
                    # Store in session cache
                    cache_key = f"{session_id}:{processed_query}:{num_results}"
                    _session_cache[cache_key] = formatted_result
        
        # Log performance
        elapsed = time.time() - start_time
        logging.info(f"[PERFORMANCE] batch_web_search completed in {elapsed:.4f}s for {len(queries)} queries")
        
        return results_dict
        
    except Exception as e:
        logging.error(f"[TOOL] batch_web_search exception: {e}")
        
        # Return error messages for all queries
        return {q: sanitize_for_azure(f"I couldn't find any results for '{q}'. Try a different query.") for q in queries}

async def clear_session_cache() -> None:
    """Clear the in-memory session cache."""
    global _session_cache
    _session_cache = {}
    logging.info("Session cache cleared")

async def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    # Get client stats
    client = await get_enhanced_brave_search_client()
    client_stats = client.get_cache_stats()
    
    # Add session cache stats
    session_stats = {
        "session_cache_size": len(_session_cache),
        "session_cache_memory_usage": sum(len(v) for v in _session_cache.values())
    }
    
    # Combine stats
    return {
        **client_stats,
        **session_stats
    }
