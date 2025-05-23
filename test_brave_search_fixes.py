#!/usr/bin/env python3
"""
Comprehensive test script for Brave Search API fixes.
This script tests:
1. API key handling
2. Session cleanup
3. Circular import resolution
4. Cache functionality
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_api_key_handling():
    """Test the API key handling in BraveSearchClient."""
    logger.info("\n=== Testing API key handling ===")
    
    # Import the module and clear any existing cache
    from brave_search_api import BraveSearchClient, clear_cache
    clear_cache()
    
    # Test with no API key
    logger.info("Testing with no API key...")
    client_no_key = BraveSearchClient(api_key=None)
    
    # Verify headers don't contain X-Subscription-Token when API key is missing
    if "X-Subscription-Token" not in client_no_key.headers:
        logger.info("✅ Success: No API key header set when API key is missing")
    else:
        value = client_no_key.headers.get("X-Subscription-Token")
        logger.error(f"❌ Error: API key header is set even when API key is missing. Value: {value!r}")
    
    # Test search with no API key
    result = await client_no_key.search("test query")
    if "error" in result and "API key is missing" in result["error"]:
        logger.info("✅ Success: Search correctly returns error when API key is missing")
    else:
        logger.error("❌ Error: Search doesn't properly handle missing API key")
    
    # Test with valid API key
    api_key = os.environ.get("BRAVE_WEB_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
    if api_key:
        logger.info("Testing with valid API key...")
        client_with_key = BraveSearchClient(api_key=api_key)
        
        # Verify headers contain API key
        if "X-Subscription-Token" in client_with_key.headers and client_with_key.headers["X-Subscription-Token"] == api_key:
            logger.info("✅ Success: API key header correctly set")
        else:
            logger.error("❌ Error: API key header not set correctly")
        
        # Test search with API key
        try:
            result = await client_with_key.search("test query", count=1)
            if "error" not in result:
                logger.info("✅ Success: Search works with valid API key")
            else:
                logger.error(f"❌ Error: Search failed with valid API key: {result['error']}")
        except Exception as e:
            logger.error(f"❌ Error during search with valid API key: {e}")
    else:
        logger.warning("Skipping tests with valid API key - no API key found in environment variables")
    
    await client_no_key.shutdown()
    if api_key:
        await client_with_key.shutdown()

async def test_session_cleanup():
    """Test that session cleanup works correctly."""
    logger.info("\n=== Testing session cleanup ===")
    
    import session_cleanup
    import aiohttp
    
    # First, ensure all existing sessions are cleaned up
    await session_cleanup.cleanup_all_sessions()
    
    # Create a session using the ManagedClientSession
    async with session_cleanup.get_managed_session() as session:
        logger.info(f"Created managed session: {id(session)}")
        
        # Verify it's registered for cleanup
        if session in session_cleanup._active_sessions:
            logger.info("✅ Success: Managed session registered for cleanup")
        else:
            logger.error("❌ Error: Managed session not registered for cleanup")
    
    # Verify session is closed after context exit
    if session.closed:
        logger.info("✅ Success: Managed session properly closed after context exit")
    else:
        logger.error("❌ Error: Managed session not closed after context exit")
        
    # Create a regular session
    session = aiohttp.ClientSession()
    logger.info(f"Created regular session: {id(session)}")
    
    # Explicitly register it
    session_cleanup.register_session(session)
    
    # Verify it's registered
    if session in session_cleanup._active_sessions:
        logger.info("✅ Success: Regular session registered for cleanup")
    else:
        logger.error("❌ Error: Regular session not registered for cleanup")
    
    # Test cleanup
    await session_cleanup.cleanup_all_sessions()
    
    # Verify session is closed
    if session.closed:
        logger.info("✅ Success: Regular session properly closed during cleanup")
    else:
        logger.error("❌ Error: Regular session not closed during cleanup")

async def test_brave_search_functionality():
    """Test the Brave Search functionality with different search engines."""
    logger.info("\n=== Testing Brave Search Functionality ===")
    
    # Test the main web_search function
    from brave_search_api import web_search, clear_cache
    
    # Clear cache first
    clear_cache()
    
    # Test with API key
    api_key = os.environ.get("BRAVE_WEB_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
    if api_key:
        logger.info("Testing Brave Search API with key...")
        start_time = time.time()
        results = await web_search("python programming", 3)
        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.4f}s")
        
        # Check if results contain expected fields
        if "python" in results.lower() and "programming" in results.lower():
            logger.info("✅ Success: Search results contain relevant content")
        else:
            logger.error("❌ Error: Search results don't contain expected content")
        
        # Test caching
        logger.info("Testing cache functionality...")
        start_time = time.time()
        cached_results = await web_search("python programming", 3)
        elapsed = time.time() - start_time
        logger.info(f"Cached search completed in {elapsed:.4f}s")
        
        if elapsed < 0.1:
            logger.info("✅ Success: Cache is working (fast response time)")
        else:
            logger.warning("⚠️ Warning: Cache might not be working optimally")
    else:
        logger.warning("Skipping Brave Search API tests - no API key found in environment variables")
    
    # Test the free tier implementation
    try:
        import brave_search_free_tier
        logger.info("Testing Brave Search Free Tier implementation...")
        
        # Test the search function with a proper query
        query = "artificial intelligence"
        start_time = time.time()
        
        # Create a mock context since the function expects it
        class MockContext:
            pass
        context = MockContext()
        
        # Call with correct parameter order: context, query, num_results
        results = await brave_search_free_tier.web_search(context, query, 3)
        elapsed = time.time() - start_time
        logger.info(f"Free tier search completed in {elapsed:.4f}s")
        
        # Check if results contain expected fields
        if "artificial" in results.lower() and "intelligence" in results.lower():
            logger.info("✅ Success: Free tier search results contain relevant content")
        else:
            logger.warning("⚠️ Warning: Free tier search results don't contain expected content. This might be due to the search API returning different results.")
            # Log the first 200 characters of the results for debugging
            logger.info(f"Result preview: {results[:200]}...")
        
        # Get cache stats
        stats = await brave_search_free_tier.get_cache_stats()
        logger.info(f"Cache statistics: {stats}")
        
        # Clean up resources
        await brave_search_free_tier.cleanup_resources()
        logger.info("Free tier resources cleaned up")
    except ImportError:
        logger.warning("Brave Search Free Tier module not available, skipping tests")

async def test_multiple_search_engines():
    """Test multiple search engines to ensure they work correctly."""
    logger.info("\n=== Testing Multiple Search Engines ===")
    
    search_engines = []
    
    # Try to import each search engine
    try:
        import brave_search_api
        search_engines.append(("Brave Search API", brave_search_api.web_search))
    except ImportError:
        logger.warning("Brave Search API not available")
    
    try:
        import brave_search_free_tier
        search_engines.append(("Brave Search Free Tier", brave_search_free_tier.web_search))
    except ImportError:
        logger.warning("Brave Search Free Tier not available")
    
    try:
        import duckduckgo_search_nokey
        # Check if the module has the expected function
        if hasattr(duckduckgo_search_nokey, 'web_search'):
            search_engines.append(("DuckDuckGo NoKey", duckduckgo_search_nokey.web_search))
        elif hasattr(duckduckgo_search_nokey, 'duckduckgo_search'):
            # Wrapper function to match our expected interface
            async def ddg_search_wrapper(query, num_results=5):
                # Create a mock context since the function expects it
                class MockContext:
                    pass
                context = MockContext()
                return await duckduckgo_search_nokey.duckduckgo_search(context, query, num_results)
            search_engines.append(("DuckDuckGo NoKey", ddg_search_wrapper))
        else:
            # Try to import DDGS directly if available
            try:
                from duckduckgo_search import DDGS
                # Create a custom wrapper
                async def ddg_direct_wrapper(query, num_results=5):
                    with DDGS() as ddgs:
                        results = ddgs.text(query, max_results=num_results)
                        # Format results
                        formatted = []
                        for r in results:
                            formatted.append(f"Title: {r.get('title', 'No title')}\nURL: {r.get('href', 'No URL')}\nDescription: {r.get('body', 'No description')}\n")
                        return "\n".join(formatted)
                search_engines.append(("DuckDuckGo Direct", ddg_direct_wrapper))
                logger.info("Added DuckDuckGo Direct search using duckduckgo_search package")
            except ImportError:
                logger.warning("DuckDuckGo NoKey available but has unexpected interface")
    except ImportError:
        logger.warning("DuckDuckGo NoKey not available")
    
    try:
        import google_search_nokey
        # Check if the module has the expected function
        if hasattr(google_search_nokey, 'web_search'):
            search_engines.append(("Google Search NoKey", google_search_nokey.web_search))
        elif hasattr(google_search_nokey, 'google_search'):
            # Wrapper function to match our expected interface
            async def google_search_wrapper(query, num_results=5):
                # Create a mock context since the function expects it
                class MockContext:
                    pass
                context = MockContext()
                return await google_search_nokey.google_search(context, query, num_results)
            search_engines.append(("Google Search NoKey", google_search_wrapper))
        else:
            logger.warning("Google Search NoKey available but has unexpected interface")
    except ImportError:
        logger.warning("Google Search NoKey not available")
    
    try:
        import bing_search_nokey
        # Check if the module has the expected function
        if hasattr(bing_search_nokey, 'web_search'):
            search_engines.append(("Bing Search NoKey", bing_search_nokey.web_search))
        elif hasattr(bing_search_nokey, 'bing_search'):
            # Wrapper function to match our expected interface
            async def bing_search_wrapper(query, num_results=5):
                # Create a mock context since the function expects it
                class MockContext:
                    pass
                context = MockContext()
                return await bing_search_nokey.bing_search(context, query, num_results)
            search_engines.append(("Bing Search NoKey", bing_search_wrapper))
        else:
            logger.warning("Bing Search NoKey available but has unexpected interface")
    except ImportError:
        logger.warning("Bing Search NoKey not available")
    
    # Test each available search engine
    for name, search_func in search_engines:
        logger.info(f"Testing {name}...")
        try:
            start_time = time.time()
            query = "climate change"
            
            # Special handling for Brave Search Free Tier
            if name == "Brave Search Free Tier":
                # Create a mock context
                class MockContext:
                    pass
                context = MockContext()
                results = await brave_search_free_tier.web_search(context, query, 3)
            else:
                results = await search_func(query, 3)
                
            elapsed = time.time() - start_time
            logger.info(f"{name} search completed in {elapsed:.4f}s")
            
            # Check if results contain expected fields
            if "climate" in results.lower():
                logger.info(f"✅ Success: {name} results contain relevant content")
            else:
                logger.error(f"❌ Error: {name} results don't contain expected content")
        except Exception as e:
            logger.error(f"❌ Error during {name} search: {e}")
    
    # Clean up resources
    try:
        import brave_search_free_tier
        await brave_search_free_tier.cleanup_resources()
    except ImportError:
        pass

async def main():
    """Run all tests."""
    logger.info("Starting comprehensive tests for Brave Search API fixes")
    
    # Test API key handling
    await test_api_key_handling()
    
    # Test session cleanup
    await test_session_cleanup()
    
    # Test Brave Search functionality
    await test_brave_search_functionality()
    
    # Test multiple search engines
    await test_multiple_search_engines()
    
    logger.info("\n=== All tests completed ===")

if __name__ == "__main__":
    asyncio.run(main())
