#!/usr/bin/env python3
"""
Debug script for the Brave Search API client to identify the exact cause of the error:
'Invalid variable type: value should be str, int or float, got None of type <class 'NoneType'>'
"""

import os
import sys
import asyncio
import logging
import traceback
import inspect
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("brave-api-debugger")

# Load environment variables
load_dotenv()

# Import the Brave Search API client
try:
    from brave_search_api import BraveSearchClient
    HAS_BRAVE_SEARCH = True
    logger.info("Brave Search API module loaded successfully")
except ImportError:
    HAS_BRAVE_SEARCH = False
    logger.error("Brave Search API module not available")
    sys.exit(1)

# Monkey patch aiohttp.ClientSession to add debug logging
try:
    import aiohttp
    
    original_get = aiohttp.ClientSession.get
    
    async def debug_get(self, url, **kwargs):
        """Debug wrapper for aiohttp.ClientSession.get"""
        logger.debug(f"aiohttp.ClientSession.get called with url: {url}")
        logger.debug(f"Headers: {self._default_headers}")
        logger.debug(f"Kwargs: {kwargs}")
        
        # Log params if present
        if 'params' in kwargs:
            logger.debug(f"Params: {kwargs['params']}")
            # Check for None values in params
            for key, value in kwargs['params'].items():
                if value is None:
                    logger.error(f"None value detected in params for key: {key}")
        
        try:
            return await original_get(self, url, **kwargs)
        except Exception as e:
            logger.error(f"Error in aiohttp.ClientSession.get: {e}")
            logger.error(traceback.format_exc())
            raise
    
    aiohttp.ClientSession.get = debug_get
    logger.info("Monkey patched aiohttp.ClientSession.get for debugging")
except ImportError:
    logger.warning("aiohttp not available, cannot patch for debugging")

async def debug_brave_search():
    """Debug the Brave Search API client."""
    if not HAS_BRAVE_SEARCH:
        logger.error("Brave Search API module not available")
        return
    
    # Get the API key from environment variables
    api_key = os.getenv("BRAVE_WEB_SEARCH_API_KEY")
    if not api_key:
        logger.warning("BRAVE_WEB_SEARCH_API_KEY not set in environment variables")
    
    # Initialize the Brave Search client
    logger.info("Initializing BraveSearchClient")
    client = BraveSearchClient(api_key=api_key)
    
    # Log the client configuration
    logger.info(f"Client API key: {'Set' if client.api_key else 'Not set'}")
    logger.info(f"Client headers: {client.headers}")
    logger.info(f"Client enable_cache: {client.enable_cache}")
    logger.info(f"Client rate_limit_value: {client.rate_limit_value}")
    
    try:
        # Perform a search with detailed parameter logging
        logger.info("Performing test search...")
        test_query = "test query for debugging"
        
        # Log all parameters being passed to the search method
        logger.debug(f"Search parameters:")
        logger.debug(f"  query: {test_query}")
        logger.debug(f"  country: us")
        logger.debug(f"  search_lang: en")
        logger.debug(f"  ui_lang: en-US")
        logger.debug(f"  count: 3")
        logger.debug(f"  offset: 0")
        logger.debug(f"  safe_search: moderate")
        logger.debug(f"  use_cache: True")
        
        # Perform the search with explicit parameters to avoid None values
        results = await client.search(
            query=test_query,
            country="us",
            search_lang="en",
            ui_lang="en-US",
            count=3,
            offset=0,
            safe_search="moderate",
            use_cache=True
        )
        
        # Log the results
        if "error" in results:
            logger.error(f"Search error: {results['error']}")
            if "details" in results:
                logger.error(f"Error details: {results['details']}")
        else:
            logger.info(f"Search successful, returned {len(results.get('web', {}).get('results', []))} results")
        
        return results
    except Exception as e:
        logger.error(f"Error during Brave Search debugging: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}
    finally:
        # Close the client session if it exists
        if hasattr(client, 'session') and client.session:
            if not client.session.closed:
                await client.session.close()
                logger.info("Closed Brave Search client session")

async def main():
    """Main function to run the debugging."""
    logger.info("Starting Brave Search API debugging...")
    
    # Debug the Brave Search API client
    results = await debug_brave_search()
    
    logger.info("Brave Search API debugging complete!")

if __name__ == "__main__":
    asyncio.run(main())
