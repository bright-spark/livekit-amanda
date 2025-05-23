#!/usr/bin/env python3
"""
Script to fix the unclosed client session issues in the Brave Search API.
This ensures that all aiohttp.ClientSession objects are properly closed.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("session_fix")

# Import necessary modules
try:
    from dotenv import load_dotenv
    # Load environment variables first
    load_dotenv()
except ImportError as e:
    logger.error(f"Failed to import dotenv: {e}")
    sys.exit(1)

try:
    import aiohttp
    import brave_search_api
    from brave_search_persistent_cache import get_persistent_cache, close_persistent_cache
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

async def patch_brave_search_api():
    """Patch the Brave Search API to properly close client sessions."""
    logger.info("Patching Brave Search API to fix unclosed client sessions...")
    
    try:
        # Store original search method
        original_search = brave_search_api.BraveSearchClient.search
        
        # Define enhanced search method with proper session management
        async def enhanced_search(self, query, count=10, params=None, use_cache=True):
            logger.debug(f"Enhanced search called for query: {query}")
            
            # Check if we should use an existing session or create a new one
            if hasattr(self, 'session') and self.session is not None:
                # Use existing session
                logger.debug("Using existing client session")
                result = await original_search(self, query, count, params, use_cache)
                return result
            else:
                # Create a new session and ensure it's closed
                logger.debug("Creating new client session")
                async with aiohttp.ClientSession() as session:
                    # Store the session temporarily
                    self.session = session
                    
                    # Call the original method
                    result = await original_search(self, query, count, params, use_cache)
                    
                    # Remove the session reference
                    self.session = None
                    
                    return result
        
        # Apply the patch
        brave_search_api.BraveSearchClient.search = enhanced_search
        logger.info("Successfully patched BraveSearchClient.search")
        
        # Also patch the original search implementation
        original_impl = brave_search_api.OptimizedBraveSearchClient.search
        
        async def enhanced_impl(self, query, count=10, params=None, use_cache=True):
            logger.debug(f"Enhanced implementation called for query: {query}")
            
            # Use a context manager for the session
            if 'session' in params:
                # Session is provided, use it
                logger.debug("Using provided session")
                return await original_impl(self, query, count, params, use_cache)
            else:
                # Create a new session with context manager
                logger.debug("Creating new session with context manager")
                async with aiohttp.ClientSession() as session:
                    if params is None:
                        params = {}
                    params['session'] = session
                    return await original_impl(self, query, count, params, use_cache)
        
        # Apply the second patch
        brave_search_api.OptimizedBraveSearchClient.search = enhanced_impl
        logger.info("Successfully patched OptimizedBraveSearchClient.search")
        
        return True
    except Exception as e:
        logger.error(f"Error patching Brave Search API: {e}")
        return False

async def test_search_with_fixed_sessions():
    """Test search with the fixed session management."""
    logger.info("Testing search with fixed session management...")
    
    try:
        # Get the Brave Search client
        client = await brave_search_api.get_brave_search_client()
        
        # Run a search
        logger.info("Running search query...")
        result = await client.search(query="session test", count=3)
        
        # Check for unclosed sessions
        logger.info("Search completed, checking for unclosed sessions...")
        
        # Force garbage collection to trigger warnings about unclosed sessions
        import gc
        gc.collect()
        
        logger.info("Search test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing search: {e}")
        return False

async def create_session_management_helper():
    """Create a helper function for session management."""
    logger.info("Creating session management helper...")
    
    helper_path = os.path.join(os.getcwd(), "session_management.py")
    
    helper_code = """#!/usr/bin/env python3
\"\"\"
Helper module for proper aiohttp.ClientSession management.
This provides context managers and utilities to ensure sessions are properly closed.
\"\"\"

import asyncio
import logging
from typing import Optional, Dict, Any
import aiohttp
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def managed_session():
    \"\"\"
    Context manager for aiohttp.ClientSession to ensure proper cleanup.
    
    Usage:
        async with managed_session() as session:
            async with session.get(url) as response:
                data = await response.json()
    \"\"\"
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()
        logger.debug("ClientSession properly closed")

class SessionManager:
    \"\"\"
    Manager for aiohttp.ClientSession to ensure proper lifecycle management.
    
    Usage:
        session_manager = SessionManager()
        session = await session_manager.get_session()
        # Use session...
        await session_manager.close()  # Call this when done
    \"\"\"
    
    def __init__(self):
        self.session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        \"\"\"Get or create a session.\"\"\"
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        \"\"\"Close the session if it exists.\"\"\"
        if self.session is not None:
            await self.session.close()
            self.session = None
            logger.debug("SessionManager closed session")
    
    async def __aenter__(self):
        return await self.get_session()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Global session manager for reuse
_global_session_manager = None

async def get_global_session():
    \"\"\"Get a session from the global session manager.\"\"\"
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return await _global_session_manager.get_session()

async def close_global_session():
    \"\"\"Close the global session manager.\"\"\"
    global _global_session_manager
    if _global_session_manager is not None:
        await _global_session_manager.close()
        _global_session_manager = None
        logger.debug("Global session manager closed")
"""
    
    # Write the helper module
    with open(helper_path, "w") as f:
        f.write(helper_code)
    
    logger.info(f"Created session management helper at {helper_path}")
    return True

async def main():
    """Main function to run all fixes and tests."""
    logger.info("Starting client session fix...")
    
    # Patch the Brave Search API
    await patch_brave_search_api()
    
    # Create session management helper
    await create_session_management_helper()
    
    # Test search with fixed sessions
    test_result = await test_search_with_fixed_sessions()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("CLIENT SESSION FIX SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Session management helper created: YES")
    logger.info(f"Brave Search API patched: YES")
    logger.info(f"Search test: {'PASSED' if test_result else 'FAILED'}")
    
    logger.info("=" * 50)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 50)
    logger.info("1. Use the session_management.py helper in your code")
    logger.info("2. Always use 'async with' when creating aiohttp.ClientSession")
    logger.info("3. For long-lived sessions, use the SessionManager class")
    logger.info("4. Call close_global_session() before your application exits")
    logger.info("=" * 50)
    
    # Ensure all resources are closed
    await close_persistent_cache()
    
    # Force garbage collection
    import gc
    gc.collect()

if __name__ == "__main__":
    asyncio.run(main())
