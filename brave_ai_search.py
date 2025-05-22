"""
Brave AI Search API implementation.

This module provides AI-specific search functionality using the Brave Search API with:
1. Configurable caching
2. Configurable rate limiting
3. Statistics tracking
4. AI-specific response formatting
"""

import logging
import asyncio
import json
import os
import time
import hashlib
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode
import aiohttp
from dotenv import load_dotenv

# Import statistics tracking module
try:
    from brave_search_stats import record_request, get_stats, get_stats_report
    HAS_STATS_TRACKING = True
    logging.info("Brave AI Search statistics tracking enabled")
except ImportError:
    HAS_STATS_TRACKING = False
    logging.warning("Brave AI Search statistics tracking not available")
    
    # Define dummy functions for when stats module is not available
    def record_request(*args, **kwargs):
        pass
        
    def get_stats():
        return None
        
    def get_stats_report():
        return "Statistics tracking not available"

# Import cache from brave_search_free_tier if available
try:
    from brave_search_free_tier import FreeTierCache
    HAS_CACHE = True
    logging.info("Using existing cache implementation for Brave AI Search")
except ImportError:
    HAS_CACHE = False
    logging.warning("Cache implementation not available for Brave AI Search")

# Load environment variables
load_dotenv()

# Get configuration from environment variables
ENABLE_CACHE = os.environ.get("BRAVE_SEARCH_ENABLE_CACHE", "true").lower() == "true"
ENABLE_PERSISTENCE = os.environ.get("BRAVE_SEARCH_ENABLE_PERSISTENCE", "true").lower() == "true"
AI_RATE_LIMIT = int(os.environ.get("BRAVE_AI_SEARCH_RATE_LIMIT", "1"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class AISearchRateLimiter:
    """Configurable rate limiter for Brave AI Search API."""
    
    def __init__(self, requests_per_second: int = 1):
        """Initialize the rate limiter with configurable rate.
        
        Args:
            requests_per_second: Number of requests allowed per second (1 for free tier, 20 for paid tier)
        """
        self.last_request_time = 0
        self.min_interval = 1.0 / requests_per_second  # Convert to interval
    
    async def wait_if_needed(self) -> None:
        """Wait if necessary to comply with the configured rate limit.
        
        This method will block until it's safe to make another request.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_interval:
            wait_time = self.min_interval - time_since_last_request
            logging.info(f"AI Search rate limiting: waiting for {wait_time:.2f} seconds")
            
            # Record rate limiting statistics
            if HAS_STATS_TRACKING:
                record_request(
                    query="",  # No specific query for this record
                    response_time=0.0,  # Not applicable
                    search_type="ai",  # AI search
                    rate_limited=True,
                    delay_time=wait_time
                )
                
            await asyncio.sleep(wait_time)
        
        # Update the last request time after waiting
        self.last_request_time = time.time()

class BraveAISearch:
    """Brave AI Search API client with configurable optimizations."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 enable_cache: Optional[bool] = None,
                 enable_persistence: Optional[bool] = None,
                 rate_limit: Optional[int] = None):
        """Initialize the Brave AI Search client with configurable optimizations.
        
        Args:
            api_key: Brave AI Search API key. If not provided, will try to get from environment variable.
            enable_cache: Whether to enable caching. If None, uses the BRAVE_SEARCH_ENABLE_CACHE env var.
            enable_persistence: Whether to enable persistent disk caching. If None, uses the BRAVE_SEARCH_ENABLE_PERSISTENCE env var.
            rate_limit: Requests per second (1 for free tier, 20 for paid tier). If None, uses the BRAVE_AI_SEARCH_RATE_LIMIT env var.
        """
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get("BRAVE_AI_SEARCH_API_KEY")
        if not self.api_key:
            logging.warning("Brave AI Search API key not provided and not found in environment variables")
        
        # Base URL for AI Search API
        self.base_url = "https://api.search.brave.com/ai/search"
        
        # Headers for API requests
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        # Use provided values or fall back to environment variables
        self.enable_cache = enable_cache if enable_cache is not None else ENABLE_CACHE
        self.enable_persistence = enable_persistence if enable_persistence is not None else ENABLE_PERSISTENCE
        self.rate_limit_value = rate_limit if rate_limit is not None else AI_RATE_LIMIT
        
        logging.info(f"Brave AI Search API configuration: cache={self.enable_cache}, persistence={self.enable_persistence}, rate_limit={self.rate_limit_value}")
        
        # Initialize cache if enabled and available
        if self.enable_cache and HAS_CACHE:
            self.cache = FreeTierCache(
                cache_ttl=604800,  # 1 week cache TTL
                memory_cache_size=1000,
                disk_cache_dir=None if not self.enable_persistence else None  # Use default if persistence enabled
            )
            logging.info("Brave AI Search cache enabled")
        else:
            self.cache = None
            if not HAS_CACHE:
                logging.warning("Cache implementation not available")
            else:
                logging.info("Brave AI Search cache disabled")
        
        # Initialize rate limiter
        self.rate_limiter = AISearchRateLimiter(requests_per_second=self.rate_limit_value)
        logging.info(f"Brave AI Search rate limiter configured for {self.rate_limit_value} requests per second")
        
        # Connection pool for reusing connections
        self.session = None
        self.session_lock = asyncio.Lock()
    
    async def _ensure_session(self):
        """Ensure that we have an active aiohttp session."""
        if self.session is None or self.session.closed:
            async with self.session_lock:
                if self.session is None or self.session.closed:
                    # Configure the session with connection pooling and keepalive
                    conn = aiohttp.TCPConnector(
                        limit=5,  # Small connection pool size
                        ttl_dns_cache=300,  # DNS cache TTL in seconds
                        keepalive_timeout=60  # Keepalive timeout
                    )
                    
                    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
                    
                    self.session = aiohttp.ClientSession(
                        connector=conn,
                        timeout=timeout,
                        headers=self.headers
                    )
            
            return self.session
    
    async def ai_search(self, 
                      query: str,
                      country: str = "us",
                      search_lang: str = "en",
                      ui_lang: str = "en-US",
                      use_cache: bool = True) -> Dict[str, Any]:
        """Perform an AI search using Brave Search API.
        
        Args:
            query: Search query
            country: Country code for search results
            search_lang: Language code for search results
            ui_lang: UI language code
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            Dict containing the AI search results
        """
        # Initialize statistics tracking variables
        start_time = time.time()
        cache_hit = False
        error = False
        rate_limited = False
        delay_time = 0.0
        status_code = 200
        result_count = 0
        
        # Generate cache key if caching is enabled
        cache_key = None
        if self.enable_cache and self.cache and use_cache:
            # Normalize the query to improve cache hit rate
            normalized_query = " ".join(query.lower().split())
            
            # Generate cache key
            cache_params = {
                "q": normalized_query,
                "country": country,
                "search_lang": search_lang,
                "ui_lang": ui_lang,
            }
            cache_key = self.cache.get_cache_key(normalized_query, **{k: v for k, v in cache_params.items() if k != "q"})
            
            # Try to get from cache
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                cache_hit = True
                logging.info(f"AI Search cache hit for query: {normalized_query}")
                
                # Record cache hit statistics
                if HAS_STATS_TRACKING:
                    # For AI search, count the number of generated points as result count
                    if "generated_answer" in cached_result and "points" in cached_result["generated_answer"]:
                        result_count = len(cached_result["generated_answer"]["points"])
                    
                    response_time = time.time() - start_time
                    record_request(
                        query=normalized_query,
                        response_time=response_time,
                        search_type="ai",  # AI search
                        cache_hit=True,
                        result_count=result_count
                    )
                
                return cached_result
        
        # Wait if necessary to comply with rate limits
        rate_limit_start = time.time()
        await self.rate_limiter.wait_if_needed()
        rate_limited = (time.time() - rate_limit_start) > 0.01  # If we waited more than 10ms
        delay_time = time.time() - rate_limit_start if rate_limited else 0
        
        # Make the API request
        try:
            # Ensure we have a session
            await self._ensure_session()
            
            # Prepare parameters
            params = {
                "q": query,
                "country": country,
                "search_lang": search_lang,
                "ui_lang": ui_lang
            }
            
            # Make the request
            async with self.session.get(self.base_url, params=params) as response:
                status_code = response.status
                
                if response.status == 429:
                    # Rate limit hit
                    rate_limited = True
                    error = True
                    error_msg = await response.text()
                    logging.error(f"AI Search rate limit exceeded: {error_msg}")
                    
                    # Record rate limit error statistics
                    if HAS_STATS_TRACKING:
                        response_time = time.time() - start_time
                        record_request(
                            query=query,
                            response_time=response_time,
                            search_type="ai",  # AI search
                            cache_hit=False,
                            error=True,
                            rate_limited=True,
                            delay_time=delay_time,
                            status_code=429,
                            result_count=0
                        )
                    
                    return {"error": "Rate limit exceeded", "details": error_msg}
                elif response.status != 200:
                    error = True
                    error_text = await response.text()
                    logging.error(f"Brave AI Search API error: {response.status} - {error_text}")
                    
                    # Record API error statistics
                    if HAS_STATS_TRACKING:
                        response_time = time.time() - start_time
                        record_request(
                            query=query,
                            response_time=response_time,
                            search_type="ai",  # AI search
                            cache_hit=False,
                            error=True,
                            rate_limited=rate_limited,
                            delay_time=delay_time,
                            status_code=response.status,
                            result_count=0
                        )
                    
                    return {"error": f"API error: {response.status}", "details": error_text}
                
                result = await response.json()
                
                # Count results if available (for AI search, count the number of generated points)
                if "generated_answer" in result and "points" in result["generated_answer"]:
                    result_count = len(result["generated_answer"]["points"])
                
                # Record successful API request statistics
                if HAS_STATS_TRACKING:
                    response_time = time.time() - start_time
                    record_request(
                        query=query,
                        response_time=response_time,
                        search_type="ai",  # AI search
                        cache_hit=False,
                        error=False,
                        rate_limited=rate_limited,
                        delay_time=delay_time,
                        status_code=status_code,
                        result_count=result_count
                    )
                
                # Cache the result if caching is enabled
                if self.enable_cache and self.cache and use_cache and cache_key:
                    await self.cache.set(cache_key, result)
                
                return result
        except asyncio.TimeoutError:
            error = True
            logging.error("AI Search request timed out")
            
            # Record timeout error statistics
            if HAS_STATS_TRACKING:
                response_time = time.time() - start_time
                record_request(
                    query=query,
                    response_time=response_time,
                    search_type="ai",  # AI search
                    cache_hit=False,
                    error=True,
                    rate_limited=rate_limited,
                    delay_time=delay_time,
                    status_code=0,  # Timeout
                    result_count=0
                )
            
            return {"error": "Request timed out"}
        except Exception as e:
            error = True
            error_msg = str(e)
            logging.error(f"Error during Brave AI search: {error_msg}")
            
            # Record general error statistics
            if HAS_STATS_TRACKING:
                response_time = time.time() - start_time
                record_request(
                    query=query,
                    response_time=response_time,
                    search_type="ai",  # AI search
                    cache_hit=False,
                    error=True,
                    rate_limited=rate_limited,
                    delay_time=delay_time,
                    status_code=0,  # General error
                    result_count=0
                )
            
            return {"error": error_msg}
    
    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()

# Singleton instance
_brave_ai_search = None
_client_lock = asyncio.Lock()

async def get_brave_ai_search_client(api_key: Optional[str] = None) -> BraveAISearch:
    """Get or create a singleton instance of the BraveAISearch.
    
    Args:
        api_key: Optional API key for Brave AI Search
        
    Returns:
        BraveAISearch instance
    """
    global _brave_ai_search
    
    async with _client_lock:
        if _brave_ai_search is None:
            _brave_ai_search = BraveAISearch(api_key=api_key)
    
    return _brave_ai_search

def format_ai_search_results(results: Dict[str, Any]) -> str:
    """Format AI search results into a readable string.
    
    Args:
        results: AI search results from the Brave Search API
        
    Returns:
        Formatted string of AI search results
    """
    if "error" in results:
        return f"AI Search error: {results['error']}"
    
    # Get current timestamp for grounding
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract the query
    query = results.get("query", {}).get("query", "your query")
    
    # Start with a grounding header
    formatted = f"""[AI SEARCH GROUNDING INFORMATION]
- Query: '{query}'
- Results retrieved: {current_time}
- Search API: Brave AI Search

"""
    
    # Extract the generated answer
    if "generated_answer" in results:
        gen_answer = results["generated_answer"]
        
        # Add the main answer
        if "answer" in gen_answer:
            formatted += f"AI ANSWER:\n{gen_answer['answer']}\n\n"
        
        # Add supporting points if available
        if "points" in gen_answer and gen_answer["points"]:
            formatted += "SUPPORTING POINTS:\n"
            for i, point in enumerate(gen_answer["points"], 1):
                formatted += f"{i}. {point}\n"
            formatted += "\n"
        
        # Add sources if available
        if "sources" in gen_answer and gen_answer["sources"]:
            formatted += "SOURCES:\n"
            for i, source in enumerate(gen_answer["sources"], 1):
                title = source.get("title", "No title")
                url = source.get("url", "")
                formatted += f"{i}. {title}\n   {url}\n"
            formatted += "\n"
    else:
        formatted += "No AI-generated answer available for this query.\n\n"
    
    # Add a footer
    formatted += f"[End of AI search results. Retrieved at {current_time}]\n"
    formatted += "This is an AI-generated response based on the latest information available to Brave Search."
    
    return formatted

async def ai_search(context, query: str) -> str:
    """Perform an AI search using the Brave Search API.
    
    Args:
        context: The run context for the tool
        query: The search query
        
    Returns:
        Formatted AI search results as a string
    """
    start_time = time.time()
    logging.info(f"[TOOL] ai_search called for query: {query}")
    
    # Log statistics if available
    if HAS_STATS_TRACKING:
        stats = get_stats()
        if stats:
            session_stats = stats.get_session_stats()
            ai_stats = stats.get_performance_stats("ai")
            logging.info(f"[STATS] AI Search requests: {ai_stats.get('total_requests', 0)}, Cache hit rate: {ai_stats.get('cache_hit_rate', 0):.2f}%")
    
    try:
        # Create a new client for each search to avoid session issues
        client = await get_brave_ai_search_client()
        
        # Perform the AI search
        results = await client.ai_search(query=query)
        
        if "error" in results:
            error_msg = f"Error searching with Brave AI API: {results['error']}"
            logging.error(error_msg)
            return f"I couldn't find an AI-generated answer for '{query}'. Try a different query."
        
        # Format the results
        formatted_results = format_ai_search_results(results)
        
        # Add explicit grounding instructions for the LLM
        grounding_header = (
            """[GROUNDING INSTRUCTIONS FOR LLM]
"""
            """When answering the user's question, use the following AI-generated response as your primary source of information.
"""
            """This AI-generated response is based on the latest information available to Brave Search.
"""
            """Do not contradict the factual information provided in this AI-generated response.
"""
            """If the AI-generated response doesn't contain relevant information to answer the user's question, clearly state this limitation.
"""
            """[END OF GROUNDING INSTRUCTIONS]

"""
        )
        
        # Insert the grounding header at the beginning of the results
        formatted_results = grounding_header + formatted_results
        
        # Log performance
        elapsed = time.time() - start_time
        logging.info(f"[PERFORMANCE] ai_search completed in {elapsed:.4f}s")
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"[TOOL] ai_search exception: {e}")
        return f"I couldn't find an AI-generated answer for '{query}'. Try a different query."

# For testing
if __name__ == "__main__":
    import argparse
    import sys
    
    async def main():
        parser = argparse.ArgumentParser(description="Brave AI Search")
        parser.add_argument("query", help="Search query")
        args = parser.parse_args()
        
        result = await ai_search(None, args.query)
        print(result)
    
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
