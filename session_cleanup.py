#!/usr/bin/env python3
"""
Session cleanup utilities for properly managing aiohttp client sessions.
This module provides functions to ensure all client sessions are properly closed
when the application shuts down.
"""

import asyncio
import logging
import atexit
import sys
from typing import List, Dict, Any, Optional, Set
import weakref

# Track all active sessions
_active_sessions = weakref.WeakSet()
_cleanup_lock = asyncio.Lock()
_is_shutting_down = False

def register_session(session):
    """Register an aiohttp ClientSession for cleanup."""
    if session is not None:
        _active_sessions.add(session)
        logging.debug(f"Registered session {id(session)} for cleanup (total: {len(_active_sessions)})")
    return session

async def close_session(session):
    """Close a single session safely."""
    try:
        if hasattr(session, 'closed'):
            if not session.closed:
                await session.close()
                logging.info(f"Closed session {id(session)}")
            else:
                logging.debug(f"Session {id(session)} already closed")
        else:
            logging.warning(f"Session {id(session)} has no 'closed' attribute")
    except Exception as e:
        logging.error(f"Error closing session {id(session)}: {e}")

async def cleanup_all_sessions():
    """Close all tracked sessions."""
    global _is_shutting_down
    
    async with _cleanup_lock:
        if _is_shutting_down:
            logging.debug("Already shutting down, skipping duplicate cleanup")
            return
        
        _is_shutting_down = True
        
        if not _active_sessions:
            logging.info("No active sessions to clean up")
            _is_shutting_down = False
            return
        
        logging.info(f"Cleaning up {len(_active_sessions)} active sessions")
        
        try:
            # Create a copy of the set to avoid modification during iteration
            sessions = list(_active_sessions)
            
            # Close all sessions concurrently
            close_tasks = [close_session(session) for session in sessions]
            await asyncio.gather(*close_tasks, return_exceptions=True)
            
            # Clear the set
            _active_sessions.clear()
            
            logging.info("All sessions cleaned up")
        except Exception as e:
            logging.error(f"Error during session cleanup: {e}")
        finally:
            # Reset the flag to allow future cleanup operations
            _is_shutting_down = False

def sync_cleanup_sessions():
    """Synchronous wrapper for cleanup_all_sessions."""
    try:
        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if one doesn't exist
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the cleanup
        if loop.is_running():
            # If the loop is running, schedule the cleanup
            future = asyncio.run_coroutine_threadsafe(cleanup_all_sessions(), loop)
            # Wait for a short time to allow cleanup to complete
            try:
                future.result(timeout=5)
            except asyncio.TimeoutError:
                logging.warning("Session cleanup timed out")
            except Exception as e:
                logging.error(f"Error during session cleanup: {e}")
        else:
            # If the loop is not running, run the cleanup to completion
            loop.run_until_complete(cleanup_all_sessions())
    except Exception as e:
        logging.error(f"Error during sync session cleanup: {e}")

# Register the cleanup handler with atexit
atexit.register(sync_cleanup_sessions)

# Patch aiohttp.ClientSession to automatically register sessions
try:
    import aiohttp
    
    # Store the original init function
    if not hasattr(aiohttp.ClientSession, '_original_init'):
        aiohttp.ClientSession._original_init = aiohttp.ClientSession.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call the original init using the stored reference
            aiohttp.ClientSession._original_init(self, *args, **kwargs)
            register_session(self)
        
        aiohttp.ClientSession.__init__ = patched_init
        logging.info("Patched aiohttp.ClientSession to auto-register for cleanup")
except ImportError:
    logging.warning("aiohttp not available, session auto-registration disabled")

# Create a context manager for client sessions
class ManagedClientSession:
    """Context manager for aiohttp.ClientSession that ensures proper cleanup."""
    
    def __init__(self, *args, **kwargs):
        self.session = None
        self.args = args
        self.kwargs = kwargs
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession(*self.args, **self.kwargs)
        # Explicitly register the session
        _active_sessions.add(self.session)
        logging.debug(f"ManagedClientSession registered session {id(self.session)} for cleanup")
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()

# Function to get a managed client session
def get_managed_session(*args, **kwargs):
    """Get a managed aiohttp.ClientSession that will be properly cleaned up."""
    return ManagedClientSession(*args, **kwargs)

# Patch the brave_search modules to use managed sessions
def patch_brave_search_modules():
    """Patch the brave_search modules to use managed sessions."""
    # Use a flag to track if we've already patched to avoid circular imports
    global _patched_modules
    if getattr(patch_brave_search_modules, '_patched_modules', False):
        return
    
    patch_brave_search_modules._patched_modules = True
    
    try:
        # Try to import and patch brave_search_api
        try:
            import brave_search_api
            if hasattr(brave_search_api, 'BraveSearchClient'):
                original_search = brave_search_api.BraveSearchClient.search
                
                async def patched_search(self, query, count=10, params=None, use_cache=True):
                    # Use existing session if available
                    if hasattr(self, 'session') and self.session and not self.session.closed:
                        return await original_search(self, query, count, params, use_cache)
                    
                    # Create a new managed session
                    async with get_managed_session() as session:
                        self.session = session
                        result = await original_search(self, query, count, params, use_cache)
                        self.session = None
                        return result
                
                brave_search_api.BraveSearchClient.search = patched_search
                logging.info("Patched brave_search_api.BraveSearchClient.search for proper session management")
        except ImportError as e:
            logging.warning(f"Could not patch brave_search_api: {e}")
        
        # Try to patch brave_search_free_tier - use deferred import to avoid circular imports
        def patch_brave_search_free_tier():
            try:
                import brave_search_free_tier
                
                # Only patch if not already patched
                if not hasattr(brave_search_free_tier, '_session_cleanup_patched') or not brave_search_free_tier._session_cleanup_patched:
                    original_cleanup = brave_search_free_tier.cleanup_resources
                    
                    async def enhanced_cleanup():
                        """Enhanced cleanup that ensures all sessions are closed."""
                        await original_cleanup()
                        await cleanup_all_sessions()
                    
                    brave_search_free_tier.cleanup_resources = enhanced_cleanup
                    
                    # Update the sync_cleanup function
                    original_sync_cleanup = brave_search_free_tier.sync_cleanup
                    
                    def enhanced_sync_cleanup():
                        """Enhanced sync cleanup that ensures all sessions are closed."""
                        original_sync_cleanup()
                        sync_cleanup_sessions()
                    
                    brave_search_free_tier.sync_cleanup = enhanced_sync_cleanup
                    
                    # Mark as patched
                    brave_search_free_tier._session_cleanup_patched = True
                    
                    logging.info("Enhanced brave_search_free_tier cleanup functions to include session cleanup")
            except ImportError as e:
                logging.warning(f"Could not patch brave_search_free_tier: {e}")
        
        # Schedule the patching to happen after this module is fully loaded
        # This breaks the circular import chain
        asyncio.get_event_loop().call_soon(patch_brave_search_free_tier)
        
    except Exception as e:
        logging.warning(f"Error during module patching: {e}")
        import traceback
        traceback.print_exc()

# Initialize the module
def init():
    """Initialize the session cleanup module."""
    logging.info("Initializing session cleanup module")
    patch_brave_search_modules()

# Auto-initialize when imported
init()
