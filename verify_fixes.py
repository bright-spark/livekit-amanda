#!/usr/bin/env python3
"""
Final verification script to confirm that both the session cleanup and persistent cache fixes
are working correctly in the Brave Search implementation.
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("fix-verifier")

# Load environment variables
load_dotenv()

# Import our session cleanup module
try:
    from session_cleanup import register_session, _active_sessions, cleanup_all_sessions
    HAS_SESSION_CLEANUP = True
    logger.info("Session cleanup module loaded successfully")
except ImportError:
    HAS_SESSION_CLEANUP = False
    logger.warning("Session cleanup module not available")

# Import the brave search modules
try:
    from brave_search_persistent_cache import BraveSearchPersistentCache
    from brave_search_api import BraveSearchClient
    HAS_BRAVE_SEARCH = True
    logger.info("Brave Search modules loaded successfully")
except ImportError:
    HAS_BRAVE_SEARCH = False
    logger.warning("Brave Search modules not available")

async def verify_brave_search():
    """Verify that the Brave Search API is working correctly with our fixes."""
    if not HAS_BRAVE_SEARCH:
        logger.error("Brave Search modules not available, cannot verify")
        return False
    
    # Get the API key from environment variables
    api_key = os.getenv("BRAVE_WEB_SEARCH_API_KEY")
    if not api_key:
        logger.error("BRAVE_WEB_SEARCH_API_KEY not set in environment variables")
        return False
    
    # Initialize the Brave Search client
    client = BraveSearchClient(api_key=api_key)
    
    try:
        # Perform a search
        logger.info("Performing Brave Search...")
        test_query = "test query for verification " + str(int(time.time()))
        results = await client.search(test_query, count=3, use_cache=True)
        
        if not results:
            logger.error("No results returned from search")
            return False
        
        logger.info(f"Search returned {len(results)} results")
        
        # Perform the same search again to test caching
        logger.info("Performing the same search again to test caching...")
        start_time = time.time()
        cached_results = await client.search(test_query, count=3, use_cache=True)
        end_time = time.time()
        
        if not cached_results:
            logger.error("No results returned from cached search")
            return False
        
        logger.info(f"Cached search returned {len(cached_results)} results in {(end_time - start_time):.4f} seconds")
        
        # Check if the second search was faster (indicating it was cached)
        if end_time - start_time < 0.1:
            logger.info("Second search was very fast, likely served from cache")
        
        return True
    except Exception as e:
        logger.error(f"Error during Brave Search verification: {e}")
        return False
    finally:
        # Close the client session if it exists
        if hasattr(client, 'session') and client.session:
            if not client.session.closed:
                await client.session.close()
                logger.info("Closed Brave Search client session")

async def verify_session_cleanup():
    """Verify that the session cleanup mechanism is working correctly."""
    if not HAS_SESSION_CLEANUP:
        logger.error("Session cleanup module not available, cannot verify")
        return False
    
    import aiohttp
    
    # Create some test sessions
    logger.info("Creating test sessions...")
    sessions = []
    
    for i in range(3):
        session = aiohttp.ClientSession()
        register_session(session)
        sessions.append(session)
    
    # Verify that the sessions are tracked
    logger.info(f"Created {len(sessions)} test sessions")
    logger.info(f"Active sessions tracked: {len(_active_sessions)}")
    
    if len(_active_sessions) >= len(sessions):
        logger.info("Session tracking verification successful!")
    else:
        logger.warning("Session tracking verification failed! Not all sessions are being tracked")
    
    # Clean up the sessions
    logger.info("Cleaning up sessions...")
    await cleanup_all_sessions()
    
    # Verify that the sessions are closed
    all_closed = True
    for i, session in enumerate(sessions):
        if not session.closed:
            all_closed = False
            logger.error(f"Session {i} is not closed!")
    
    if all_closed:
        logger.info("Session cleanup verification successful! All sessions are closed")
        return True
    else:
        logger.error("Session cleanup verification failed! Not all sessions are closed")
        return False

async def verify_persistent_cache():
    """Verify that the persistent cache is working correctly with our fixes."""
    if not HAS_BRAVE_SEARCH:
        logger.error("Brave Search modules not available, cannot verify cache")
        return False
    
    # Initialize the persistent cache with the lower quality threshold
    cache_config = {
        "quality_threshold": 0.1  # Lower threshold to allow more entries to be cached
    }
    
    try:
        # Create the cache instance with our custom config
        cache = BraveSearchPersistentCache(config=cache_config)
        
        # Test storing and retrieving a value
        test_key = "test_key_" + str(int(time.time()))
        test_value = {"test": "data", "quality": 0.2}  # Quality above our threshold but below default
        
        # Store the value
        logger.info(f"Storing test value with key: {test_key}")
        await cache.set(test_key, test_value, search_type="web")
        
        # Retrieve the value
        logger.info(f"Retrieving test value with key: {test_key}")
        retrieved_value = await cache.get(test_key)
        
        if retrieved_value:
            logger.info(f"Successfully retrieved test value: {retrieved_value}")
            return True
        else:
            logger.error(f"Failed to retrieve test value for key: {test_key}")
            return False
    except Exception as e:
        logger.error(f"Error during persistent cache verification: {e}")
        return False
    finally:
        # Clean up resources
        if hasattr(cache, 'close'):
            await cache.close()
            logger.info("Closed cache connection")

async def main():
    """Main function to run all verifications."""
    logger.info("Starting verification of all fixes...")
    
    # Verify the session cleanup
    if HAS_SESSION_CLEANUP:
        logger.info("Verifying session cleanup...")
        session_cleanup_ok = await verify_session_cleanup()
        logger.info(f"Session cleanup verification: {'SUCCESS' if session_cleanup_ok else 'FAILED'}")
    else:
        logger.warning("Session cleanup module not available, skipping verification")
    
    # Verify the persistent cache
    if HAS_BRAVE_SEARCH:
        logger.info("Verifying persistent cache...")
        cache_ok = await verify_persistent_cache()
        logger.info(f"Persistent cache verification: {'SUCCESS' if cache_ok else 'FAILED'}")
    else:
        logger.warning("Brave Search modules not available, skipping verification")
    
    # Verify the Brave Search API with our fixes
    if HAS_BRAVE_SEARCH:
        logger.info("Verifying Brave Search API...")
        search_ok = await verify_brave_search()
        logger.info(f"Brave Search verification: {'SUCCESS' if search_ok else 'FAILED'}")
    else:
        logger.warning("Brave Search modules not available, skipping verification")
    
    logger.info("All verifications complete!")

if __name__ == "__main__":
    asyncio.run(main())
