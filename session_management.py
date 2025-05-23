#!/usr/bin/env python3
"""
Helper module for proper aiohttp.ClientSession management.
This provides context managers and utilities to ensure sessions are properly closed.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import aiohttp
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def managed_session():
    """
    Context manager for aiohttp.ClientSession to ensure proper cleanup.
    
    Usage:
        async with managed_session() as session:
            async with session.get(url) as response:
                data = await response.json()
    """
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()
        logger.debug("ClientSession properly closed")

class SessionManager:
    """
    Manager for aiohttp.ClientSession to ensure proper lifecycle management.
    
    Usage:
        session_manager = SessionManager()
        session = await session_manager.get_session()
        # Use session...
        await session_manager.close()  # Call this when done
    """
    
    def __init__(self):
        self.session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create a session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the session if it exists."""
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
    """Get a session from the global session manager."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return await _global_session_manager.get_session()

async def close_global_session():
    """Close the global session manager."""
    global _global_session_manager
    if _global_session_manager is not None:
        await _global_session_manager.close()
        _global_session_manager = None
        logger.debug("Global session manager closed")
