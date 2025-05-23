#!/usr/bin/env python3
"""
Debug script to verify if the Brave Search persistent cache is working correctly.
This script will:
1. Check if the persistent cache is enabled
2. Verify if the cache files exist
3. Run the same search query twice to check for cache hits
4. Display cache statistics
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cache_debug")

# Import necessary modules
try:
    from brave_search_persistent_cache import get_persistent_cache, close_persistent_cache
    from brave_search_api import get_brave_search_client
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

async def check_cache_files():
    """Check if the persistent cache files exist."""
    logger.info("Checking persistent cache files...")
    
    # Get cache directory from environment variable
    cache_dir = os.environ.get("BRAVE_PERSISTENT_CACHE_DISK_CACHE_DIR", "/tmp/brave_persistent_cache")
    db_path = os.path.join(cache_dir, "brave_search_cache.db")
    
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Database path: {db_path}")
    
    # Check if directory exists
    if not os.path.exists(cache_dir):
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            return False
    else:
        logger.info(f"Cache directory exists: {cache_dir}")
    
    # Check if database file exists
    if os.path.exists(db_path):
        logger.info(f"Database file exists: {db_path}")
        logger.info(f"Database file size: {os.path.getsize(db_path)} bytes")
        return True
    else:
        logger.info(f"Database file does not exist yet: {db_path}")
        return False

async def check_cache_config():
    """Check the persistent cache configuration."""
    logger.info("Checking persistent cache configuration...")
    
    # Check environment variables
    cache_enabled = os.environ.get("BRAVE_PERSISTENT_CACHE_ENABLE", "false").lower() == "true"
    persistence_enabled = os.environ.get("BRAVE_PERSISTENT_CACHE_ENABLE_PERSISTENCE", "false").lower() == "true"
    cache_ttl = int(os.environ.get("BRAVE_PERSISTENT_CACHE_CACHE_TTL", "604800"))
    rate_limit = int(os.environ.get("BRAVE_PERSISTENT_CACHE_RATE_LIMIT", "1"))
    
    logger.info(f"BRAVE_PERSISTENT_CACHE_ENABLE: {cache_enabled}")
    logger.info(f"BRAVE_PERSISTENT_CACHE_ENABLE_PERSISTENCE: {persistence_enabled}")
    logger.info(f"BRAVE_PERSISTENT_CACHE_CACHE_TTL: {cache_ttl}")
    logger.info(f"BRAVE_PERSISTENT_CACHE_RATE_LIMIT: {rate_limit}")
    
    return {
        "cache_enabled": cache_enabled,
        "persistence_enabled": persistence_enabled,
        "cache_ttl": cache_ttl,
        "rate_limit": rate_limit
    }

async def test_cache_with_search():
    """Test the persistent cache by running the same search query twice."""
    logger.info("Testing persistent cache with search queries...")
    
    try:
        # Get the Brave Search client
        client = await get_brave_search_client()
        
        # First search query
        logger.info("Running first search query...")
        start_time = time.time()
        results1 = await client.search(query="persistent cache test", count=3)
        end_time = time.time()
        logger.info(f"First search completed in {end_time - start_time:.2f} seconds")
        
        # Wait a moment
        await asyncio.sleep(1)
        
        # Second search query (same query)
        logger.info("Running second search query (should hit cache)...")
        start_time = time.time()
        results2 = await client.search(query="persistent cache test", count=3)
        end_time = time.time()
        logger.info(f"Second search completed in {end_time - start_time:.2f} seconds")
        
        # Check if results are identical (indicating cache hit)
        results_match = results1 == results2
        logger.info(f"Results match: {results_match}")
        
        return {
            "results_match": results_match,
            "first_query_time": end_time - start_time
        }
    except Exception as e:
        logger.error(f"Error testing cache with search: {e}")
        return {
            "results_match": False,
            "error": str(e)
        }

async def get_cache_stats():
    """Get statistics from the persistent cache."""
    logger.info("Getting persistent cache statistics...")
    
    try:
        # Get the persistent cache
        cache = get_persistent_cache()
        
        # Get statistics
        stats = await cache.get_stats()
        logger.info(f"Cache statistics: {stats}")
        
        return stats
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        return {"error": str(e)}

async def main():
    """Main function to run all checks."""
    logger.info("Starting persistent cache debug...")
    
    # Check cache configuration
    config = await check_cache_config()
    
    # If cache is not enabled, exit
    if not config["cache_enabled"]:
        logger.warning("Persistent cache is not enabled. Enable it by setting BRAVE_PERSISTENT_CACHE_ENABLE=true")
        return
    
    # Check cache files
    files_exist = await check_cache_files()
    
    # Test cache with search
    search_results = await test_cache_with_search()
    
    # Get cache statistics
    stats = await get_cache_stats()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("PERSISTENT CACHE DEBUG SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Cache enabled: {config['cache_enabled']}")
    logger.info(f"Persistence enabled: {config['persistence_enabled']}")
    logger.info(f"Cache files exist: {files_exist}")
    logger.info(f"Search results match: {search_results.get('results_match', False)}")
    logger.info(f"Cache statistics: {json.dumps(stats, indent=2)}")
    logger.info("=" * 50)
    
    # Close the persistent cache
    await close_persistent_cache()

if __name__ == "__main__":
    asyncio.run(main())
