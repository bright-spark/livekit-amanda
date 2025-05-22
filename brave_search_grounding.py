"""
Brave Search Grounding API.

This module provides specialized functionality for grounding queries to Brave Search
with a dedicated API key and rate limiting specifically for grounding purposes.
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from asyncio import Semaphore

import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brave_search_grounding")

# Get API keys for grounding (supports multiple keys for parallel processing)
def get_grounding_api_keys() -> List[str]:
    """Get all available grounding API keys.
    
    Returns:
        List of API keys for grounding
    """
    # Primary grounding API key
    primary_key = os.getenv("BRAVE_GROUNDING_API_KEY")
    keys = []
    
    # Check for multiple keys with indexed environment variables
    # Format: BRAVE_GROUNDING_API_KEY_1, BRAVE_GROUNDING_API_KEY_2, etc.
    index = 1
    while True:
        key = os.getenv(f"BRAVE_GROUNDING_API_KEY_{index}")
        if not key:
            break
        keys.append(key)
        index += 1
    
    # Add primary key if available
    if primary_key:
        keys.append(primary_key)
    
    # Fallback to web search API keys if no grounding API keys are set
    if not keys:
        web_key = os.getenv("BRAVE_WEB_SEARCH_API_KEY")
        if web_key:
            logger.warning("No BRAVE_GROUNDING_API_KEY found, using BRAVE_WEB_SEARCH_API_KEY as fallback")
            keys.append(web_key)
            
        # Check for multiple web search keys
        index = 1
        while True:
            key = os.getenv(f"BRAVE_WEB_SEARCH_API_KEY_{index}")
            if not key:
                break
            keys.append(key)
            index += 1
    
    if not keys:
        logger.warning("No API keys found for grounding. Some features may not work properly.")
    else:
        logger.info(f"Found {len(keys)} API keys for grounding")
    
    return keys

# Get all available API keys
GROUNDING_API_KEYS = get_grounding_api_keys()

# Check if any API keys are available
HAS_API_KEY = bool(GROUNDING_API_KEYS)

# Get rate limit for grounding API
try:
    GROUNDING_RATE_LIMIT = int(os.getenv("BRAVE_GROUNDING_RATE_LIMIT", "1"))
    if GROUNDING_RATE_LIMIT <= 0:
        GROUNDING_RATE_LIMIT = 1
        logger.warning("Invalid BRAVE_GROUNDING_RATE_LIMIT value. Using default: 1")
    logger.info(f"Grounding rate limit set to: {GROUNDING_RATE_LIMIT} requests per second")
except (ValueError, TypeError):
    GROUNDING_RATE_LIMIT = 1
    logger.warning("Invalid BRAVE_GROUNDING_RATE_LIMIT value. Using default: 1")

class BraveSearchGrounding:
    """Specialized class for grounding queries to Brave Search with elastic parallel processing."""
    
    def __init__(self):
        """Initialize the grounding service with support for multiple API keys."""
        self.sessions = {}
        self.api_keys = GROUNDING_API_KEYS
        self.has_api_key = HAS_API_KEY
        self.current_key_index = 0
        
        # Set up rate limiting - one semaphore per API key for parallel processing
        self.rate_limit = GROUNDING_RATE_LIMIT
        self.key_semaphores = {}
        self.last_request_times = {}
        
        # Initialize semaphores and request times for each key
        for i, key in enumerate(self.api_keys):
            # Each key gets its own semaphore with limit of 1 request at a time
            self.key_semaphores[key] = Semaphore(1)
            self.last_request_times[key] = 0
        
        logger.info(f"Initialized BraveSearchGrounding with {len(self.api_keys)} API keys")
    
    async def _get_session(self, api_key: str) -> aiohttp.ClientSession:
        """Get or create an aiohttp client session for a specific API key.
        
        Args:
            api_key: The API key to get a session for
            
        Returns:
            aiohttp.ClientSession: The client session
        """
        if api_key not in self.sessions or self.sessions[api_key].closed:
            self.sessions[api_key] = aiohttp.ClientSession()
        return self.sessions[api_key]
        
    def _get_next_api_key(self) -> str:
        """Get the next available API key using round-robin rotation.
        
        Returns:
            The next API key to use
        """
        if not self.api_keys:
            raise ValueError("No API keys available for grounding")
            
        # Rotate to the next key
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API request to the Brave Search API using the next available API key.
        
        This method implements a round-robin key rotation strategy with per-key rate limiting,
        allowing for parallel processing across multiple API keys.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response as a dictionary
        """
        if not self.has_api_key:
            raise ValueError("No API keys available for grounding")
        
        # Get the next available API key
        api_key = self._get_next_api_key()
        
        # Base URL for Brave Search API
        base_url = "https://api.search.brave.com"
        url = f"{base_url}/{endpoint}"
        
        # Add API key to headers
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        
        # Get or create session for this API key
        session = await self._get_session(api_key)
        
        # Apply rate limiting for this specific API key
        semaphore = self.key_semaphores[api_key]
        
        async with semaphore:
            # Check if we need to wait before making the request with this key
            current_time = time.time()
            elapsed = current_time - self.last_request_times[api_key]
            wait_time = max(0, 1.0 / self.rate_limit - elapsed)
            
            if wait_time > 0:
                logger.debug(f"Rate limiting for key {api_key[:5]}...: waiting {wait_time:.2f}s before making request")
                await asyncio.sleep(wait_time)
            
            # Update last request time for this key
            self.last_request_times[api_key] = time.time()
            
            # Make the request
            try:
                logger.debug(f"Making request with API key {api_key[:5]}...")
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"Request error: {response.status}, message={error_text!r}, url={response.url}"
                        logger.error(f"API request failed with key {api_key[:5]}...: {error_msg}")
                        raise ValueError(error_msg)
                    
                    return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Client error in API request with key {api_key[:5]}...: {e}")
                raise ValueError(f"Client error: {str(e)}")
            except Exception as e:
                logger.error(f"Error making API request: {e}")
                return {"error": f"Request error: {str(e)}"}
    
    async def ground_web_query(self, query: str, count: int = 5, **kwargs) -> Dict[str, Any]:
        """Ground a query using web search.
        
        Args:
            query: Search query
            count: Number of results to return
            **kwargs: Additional parameters for the search
            
        Returns:
            Grounding search results
        """
        # Prepare parameters
        params = {
            "q": query,
            "count": count,
            **kwargs
        }
        
        # Make the API request - use the correct endpoint for web search
        return await self._make_api_request("res/v1/web/search", params)
    
    async def ground_ai_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Ground a query using AI search.
        
        Args:
            query: Search query
            **kwargs: Additional parameters for the search
            
        Returns:
            Grounding AI search results
        """
        # Prepare parameters
        params = {
            "q": query,
            **kwargs
        }
        
        # Make the API request - use the correct endpoint for AI search
        return await self._make_api_request("res/v1/ai/search", params)
    
    async def ground_query(self, query: str, search_type: str = "web", count: int = 5, **kwargs) -> Dict[str, Any]:
        """Ground a query using the specified search type.
        
        Args:
            query: Search query
            search_type: Type of search ("web" or "ai")
            count: Number of results for web search
            **kwargs: Additional parameters for the search
            
        Returns:
            Grounding search results
        """
        logger.info(f"Grounding query: '{query}' using {search_type} search")
        
        # Always use web search for now since it's more reliable
        # Just add a note if AI was requested but we're using web search instead
        try:
            result = await self.ground_web_query(query, count, **kwargs)
            
            # If AI was requested, add a note about using web search instead
            if search_type == "ai" and "error" not in result:
                if "web" in result and "results" in result["web"]:
                    result["_fallback_note"] = "Using web search results for grounding instead of AI search."
                    # Keep the original search type for formatting purposes
                    result["_requested_search_type"] = "ai"
            
            return result
        except Exception as e:
            logger.error(f"Error in ground_query: {e}")
            return {"error": f"Grounding error: {str(e)}"}
    
    async def format_grounding_results(self, results: Dict[str, Any], search_type: str, query: str) -> str:
        """Format grounding results into a readable string.
        
        Args:
            results: Grounding search results
            search_type: Type of search ("web" or "ai")
            query: Original search query
            
        Returns:
            Formatted string of grounding results
        """
        if "error" in results:
            return f"Grounding error: {results['error']}"
        
        # Get current timestamp for grounding
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Start with a grounding header
        # Use the requested search type if available
        display_search_type = results.get("_requested_search_type", search_type)
        
        formatted = f"""[GROUNDING INFORMATION FROM BRAVE SEARCH]
- Query: '{query}'
- Search type: {display_search_type}
- Retrieved at: {current_time}
- Source: Brave Search API (Grounding)
"""
        
        # Add fallback note if present
        if "_fallback_note" in results:
            formatted += f"- Note: {results['_fallback_note']}\n"
        
        formatted += "\n"
        
        # Handle web search results
        if (search_type == "web" or "_fallback_note" in results) and "web" in results and "results" in results["web"]:
            web_results = results["web"]["results"]
            formatted += f"Found {len(web_results)} results:\n\n"
            
            # Format each result
            for idx, result in enumerate(web_results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                description = result.get("description", "")
                
                # Extract age if available
                age = result.get("age", "")
                date_info = f" [Published: {age}]" if age else ""
                
                # Extract and format domain information
                domain = result.get("domain", "")
                domain_info = f" [Source: {domain}]" if domain else ""
                
                formatted += f"{idx}. {title}{date_info}{domain_info}\n   {url}\n"
                if description:
                    formatted += f"   {description}\n"
                formatted += "\n"
        # Handle AI search results
        elif search_type == "ai" and "ai" in results and "results" in results["ai"] and results["ai"]["results"]:
            ai_results = results["ai"]["results"][0]
            
            if "answer" in ai_results:
                formatted += f"AI Answer:\n{ai_results['answer']}\n\n"
            
            # Add supporting points if available
            if "points" in ai_results and ai_results["points"]:
                formatted += "Supporting points:\n"
                for i, point in enumerate(ai_results["points"], 1):
                    formatted += f"{i}. {point}\n"
                formatted += "\n"
            
            # Add sources if available
            if "sources" in ai_results and ai_results["sources"]:
                formatted += "Sources:\n"
                for i, source in enumerate(ai_results["sources"], 1):
                    title = source.get("title", "No title")
                    url = source.get("url", "")
                    formatted += f"{i}. {title}\n   {url}\n"
                formatted += "\n"
        else:
            formatted += "No search results found.\n\n"
        
        # Add a footer
        formatted += f"[End of grounding information. Retrieved at {current_time}]\n"
        
        return formatted
    
    async def close(self) -> None:
        """Close the grounding service and release resources."""
        # Close all API client sessions
        for api_key, session in list(self.sessions.items()):
            if session is not None and not session.closed:
                await session.close()
                logger.info(f"Closed session for API key {api_key[:5]}...")
        
        # Clear the sessions dictionary
        self.sessions = {}
        logger.info(f"Closed all {len(self.api_keys)} BraveSearchGrounding sessions")

# Singleton instance
_grounding_instance = None

async def get_grounding_service() -> BraveSearchGrounding:
    """Get the singleton instance of the grounding service.
    
    Returns:
        BraveSearchGrounding: The grounding service instance
    """
    global _grounding_instance
    if _grounding_instance is None:
        _grounding_instance = BraveSearchGrounding()
    return _grounding_instance

async def close_grounding_service() -> None:
    """Close the grounding service and release resources."""
    global _grounding_instance
    if _grounding_instance is not None:
        await _grounding_instance.close()
        _grounding_instance = None
        logger.info("Closed grounding service")

async def ground_query(query: str, search_type: str = "web", count: int = 5, **kwargs) -> str:
    """Ground a query using the specified search type and return formatted results.
    
    Args:
        query: Search query
        search_type: Type of search ("web" or "ai")
        count: Number of results for web search
        **kwargs: Additional parameters for the search
        
    Returns:
        Formatted string of grounding results
    """
    try:
        # Get the grounding service
        grounding = await get_grounding_service()
        
        # Ground the query
        results = await grounding.ground_query(query, search_type, count, **kwargs)
        
        # Format the results
        return await grounding.format_grounding_results(results, search_type, query)
    except Exception as e:
        logger.error(f"Error in ground_query: {e}")
        return f"Error grounding query '{query}': {str(e)}"
