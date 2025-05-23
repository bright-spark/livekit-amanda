#!/usr/bin/env python3
"""
Final verification script for Brave Search API fixes.
This script verifies:
1. API key handling in BraveSearchClient
2. Session cleanup functionality
3. Circular import resolution
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def verify_api_key_handling():
    """Verify the API key handling in BraveSearchClient."""
    logger.info("\n=== Verifying API key handling ===")
    
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

async def verify_session_cleanup():
    """Verify that session cleanup works correctly."""
    logger.info("\n=== Verifying session cleanup ===")
    
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

async def verify_circular_import():
    """Verify that the circular import issue is resolved."""
    logger.info("\n=== Verifying circular import resolution ===")
    
    # Test importing session_cleanup first
    try:
        import session_cleanup
        logger.info("✅ Success: Imported session_cleanup")
    except ImportError as e:
        logger.error(f"❌ Error importing session_cleanup: {e}")
        return
    
    # Test importing brave_search_free_tier after session_cleanup
    try:
        import brave_search_free_tier
        logger.info("✅ Success: Imported brave_search_free_tier after session_cleanup")
    except ImportError as e:
        logger.error(f"❌ Error importing brave_search_free_tier after session_cleanup: {e}")
        return
    
    # Test importing in reverse order
    try:
        # Force reload of modules
        import importlib
        importlib.reload(session_cleanup)
        importlib.reload(brave_search_free_tier)
        
        # Now try importing in reverse order in a fresh namespace
        import sys
        if "brave_search_free_tier" in sys.modules:
            del sys.modules["brave_search_free_tier"]
        if "session_cleanup" in sys.modules:
            del sys.modules["session_cleanup"]
        
        # Import in reverse order
        import brave_search_free_tier
        logger.info("✅ Success: Imported brave_search_free_tier")
        import session_cleanup
        logger.info("✅ Success: Imported session_cleanup after brave_search_free_tier")
    except ImportError as e:
        logger.error(f"❌ Error during reverse order import test: {e}")
        return
    
    logger.info("Circular import tests completed successfully")

async def verify_brave_search_free_tier():
    """Verify that the Brave Search Free Tier implementation works correctly."""
    logger.info("\n=== Verifying Brave Search Free Tier ===")
    
    try:
        import brave_search_free_tier
        
        # Create a mock context
        class MockContext:
            pass
        context = MockContext()
        
        # Test with a simple query
        query = "python programming"
        logger.info(f"Testing Brave Search Free Tier with query: {query}")
        
        start_time = time.time()
        results = await brave_search_free_tier.web_search(context, query, 3)
        elapsed = time.time() - start_time
        
        logger.info(f"Search completed in {elapsed:.4f}s")
        
        # Check if results contain expected content
        if "python" in results.lower():
            logger.info("✅ Success: Results contain relevant content")
        else:
            logger.warning("⚠️ Warning: Results don't contain expected content. This might be due to caching or API behavior.")
            
        # Get cache stats
        stats = await brave_search_free_tier.get_cache_stats()
        logger.info(f"Cache statistics: {stats}")
        
        # Clean up resources
        await brave_search_free_tier.cleanup_resources()
        logger.info("Resources cleaned up successfully")
        
    except ImportError:
        logger.error("❌ Error: brave_search_free_tier module not available")
    except Exception as e:
        logger.error(f"❌ Error during Brave Search Free Tier test: {e}")

async def main():
    """Run all verification tests."""
    logger.info("Starting final verification of Brave Search API fixes")
    
    # Verify API key handling
    await verify_api_key_handling()
    
    # Verify session cleanup
    await verify_session_cleanup()
    
    # Verify circular import resolution
    await verify_circular_import()
    
    # Verify Brave Search Free Tier
    await verify_brave_search_free_tier()
    
    logger.info("\n=== All verifications completed ===")

if __name__ == "__main__":
    asyncio.run(main())
