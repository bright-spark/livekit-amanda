#!/usr/bin/env python3
"""
Script to verify that the persistent cache is working correctly and that client sessions
are being properly managed and cleaned up.
"""

import os
import sys
import asyncio
import logging
import sqlite3
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

logger = logging.getLogger("cache-session-verifier")

# Load environment variables
load_dotenv()

# Import our session cleanup module
try:
    from session_cleanup import (
        register_session, 
        _active_sessions, 
        cleanup_all_sessions
    )
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

async def verify_brave_search_cache():
    """Verify that the Brave Search persistent cache is working correctly."""
    if not HAS_BRAVE_SEARCH:
        logger.error("Brave Search modules not available, cannot verify cache")
        return False
    
    # Get the API key from environment variables
    api_key = os.getenv("BRAVE_WEB_SEARCH_API_KEY")
    if not api_key:
        logger.error("BRAVE_WEB_SEARCH_API_KEY not set in environment variables")
        return False
    
    # Initialize the persistent cache with a lower quality threshold (0.1)
    # Note: BraveSearchPersistentCache initializes automatically when created
    cache_config = {
        "quality_threshold": 0.1  # Lower threshold to allow more entries to be cached
    }
    cache = BraveSearchPersistentCache(config=cache_config)
    
    # Initialize the Brave Search client
    client = BraveSearchClient(api_key=api_key)
    
    try:
        # Perform a search to populate the cache
        logger.info("Performing search to populate cache...")
        test_query = "test query for cache verification"
        results = await client.search(test_query, count=3, use_cache=True)
        
        if not results:
            logger.error("No results returned from search")
            return False
        
        logger.info(f"Search returned {len(results)} results")
        
        # Verify that the results are cached
        logger.info("Verifying cache...")
        cache_key = f"brave_search:{test_query}:3"
        cached_value = await cache.get(cache_key)
        
        if cached_value:
            logger.info("Cache verification successful! Entry found in cache")
            
            # Check the SQLite database directly
            db_path = os.path.expanduser("~/.brave_search_cache.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Query the database
                cursor.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]
                logger.info(f"SQLite database contains {count} entries")
                
                # Get the specific entry
                cursor.execute("SELECT key, value FROM cache WHERE key = ?", (cache_key,))
                row = cursor.fetchone()
                
                if row:
                    logger.info(f"Found entry in SQLite database for key: {cache_key}")
                else:
                    logger.warning(f"Entry not found in SQLite database for key: {cache_key}")
                
                conn.close()
            else:
                logger.warning(f"SQLite database file not found at {db_path}")
            
            return True
        else:
            logger.error("Cache verification failed! Entry not found in cache")
            return False
    except Exception as e:
        logger.error(f"Error during cache verification: {e}")
        return False
    finally:
        # Clean up resources
        if hasattr(cache, 'close'):
            await cache.close()
        elif hasattr(cache, 'cleanup'):
            await cache.cleanup()
        
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

async def main():
    """Main function to run the verification."""
    logger.info("Starting cache and session verification...")
    
    # Verify the session cleanup
    if HAS_SESSION_CLEANUP:
        logger.info("Verifying session cleanup...")
        session_cleanup_ok = await verify_session_cleanup()
        logger.info(f"Session cleanup verification: {'SUCCESS' if session_cleanup_ok else 'FAILED'}")
    else:
        logger.warning("Session cleanup module not available, skipping verification")
    
    # Verify the Brave Search cache
    if HAS_BRAVE_SEARCH:
        logger.info("Verifying Brave Search cache...")
        cache_ok = await verify_brave_search_cache()
        logger.info(f"Cache verification: {'SUCCESS' if cache_ok else 'FAILED'}")
    else:
        logger.warning("Brave Search modules not available, skipping verification")
    
    logger.info("Verification complete!")

if __name__ == "__main__":
    asyncio.run(main())
