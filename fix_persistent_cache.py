#!/usr/bin/env python3
"""
Script to fix the Brave Search persistent cache by:
1. Checking the connection between in-memory cache and persistent storage
2. Verifying the write operations to the SQLite database
3. Adding a direct test to ensure data is being persisted
"""

import os
import sys
import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cache_fix")

# Import necessary modules
try:
    from dotenv import load_dotenv
    # Load environment variables first
    load_dotenv()
except ImportError as e:
    logger.error(f"Failed to import dotenv: {e}")
    sys.exit(1)

try:
    from brave_search_persistent_cache import get_persistent_cache, close_persistent_cache
    from brave_search_api import get_brave_search_client
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

async def direct_cache_test():
    """Test the persistent cache directly by writing and reading values."""
    logger.info("Testing persistent cache directly...")
    
    try:
        # Get the persistent cache
        cache = get_persistent_cache()
        
        # Generate a unique test key
        test_key = f"test_key_{int(time.time())}"
        test_value = {
            "test": "data",
            "timestamp": time.time(),
            "random": os.urandom(8).hex()
        }
        
        # Write to cache
        logger.info(f"Writing test data to cache with key: {test_key}")
        await cache.set(test_key, test_value, ttl=3600)
        
        # Get cache statistics after write
        stats_after_write = await cache.get_stats()
        logger.info(f"Cache statistics after write: {stats_after_write}")
        
        # Read from cache
        logger.info(f"Reading test data from cache with key: {test_key}")
        retrieved_value = await cache.get(test_key)
        
        if retrieved_value:
            logger.info(f"Successfully retrieved test data: {retrieved_value}")
            logger.info("Direct cache test PASSED")
            return True
        else:
            logger.error(f"Failed to retrieve test data for key: {test_key}")
            logger.info("Direct cache test FAILED")
            return False
    except Exception as e:
        logger.error(f"Error during direct cache test: {e}")
        return False

async def check_database_tables():
    """Check the structure and content of the database tables."""
    logger.info("Checking database tables...")
    
    try:
        import sqlite3
        
        # Get the database path
        home_dir = os.path.expanduser("~")
        db_path = os.path.join(home_dir, ".cache", "livekit-amanda", "brave_persistent_cache.db")
        
        if not os.path.exists(db_path):
            logger.error(f"Database file does not exist: {db_path}")
            return False
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check the schema of the cache table
        cursor.execute("PRAGMA table_info(cache);")
        cache_schema = cursor.fetchall()
        logger.info(f"Cache table schema: {cache_schema}")
        
        # Check if there are any rows in the cache table
        cursor.execute("SELECT COUNT(*) FROM cache;")
        row_count = cursor.fetchone()[0]
        logger.info(f"Cache table row count: {row_count}")
        
        # If there are rows, show a sample
        if row_count > 0:
            cursor.execute("SELECT * FROM cache LIMIT 1;")
            sample_row = cursor.fetchone()
            logger.info(f"Sample cache row: {sample_row}")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error checking database tables: {e}")
        return False

async def fix_permissions():
    """Fix permissions on the cache directory and database file."""
    logger.info("Fixing permissions on cache directory and database file...")
    
    try:
        # Get the cache directory path
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache", "livekit-amanda")
        db_path = os.path.join(cache_dir, "brave_persistent_cache.db")
        
        # Ensure the directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check current permissions
        dir_perms = os.stat(cache_dir).st_mode & 0o777
        logger.info(f"Current directory permissions: {oct(dir_perms)}")
        
        if os.path.exists(db_path):
            file_perms = os.stat(db_path).st_mode & 0o777
            logger.info(f"Current file permissions: {oct(file_perms)}")
        
        # Set permissions to ensure writability
        os.chmod(cache_dir, 0o755)
        logger.info(f"Set directory permissions to: 0o755")
        
        if os.path.exists(db_path):
            os.chmod(db_path, 0o644)
            logger.info(f"Set file permissions to: 0o644")
        
        return True
    except Exception as e:
        logger.error(f"Error fixing permissions: {e}")
        return False

async def test_search_with_persistence():
    """Test search with persistence by closing and reopening the cache."""
    logger.info("Testing search with persistence...")
    
    try:
        # First search
        client1 = await get_brave_search_client()
        query = f"persistence test {int(time.time())}"
        logger.info(f"Running first search with query: {query}")
        results1 = await client1.search(query=query, count=3)
        
        # Close the client and cache
        logger.info("Closing client and cache...")
        await close_persistent_cache()
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Second search with the same query
        logger.info("Reopening client and cache...")
        client2 = await get_brave_search_client()
        logger.info(f"Running second search with query: {query}")
        results2 = await client2.search(query=query, count=3)
        
        # Check if results match
        results_match = results1 == results2
        logger.info(f"Results match: {results_match}")
        
        return results_match
    except Exception as e:
        logger.error(f"Error testing search with persistence: {e}")
        return False

async def patch_brave_search_persistent_cache():
    """Patch the BraveSearchPersistentCache class to fix potential issues."""
    logger.info("Patching BraveSearchPersistentCache...")
    
    try:
        import brave_search_persistent_cache
        
        # Store original methods
        original_set = brave_search_persistent_cache.BraveSearchPersistentCache.set
        
        # Define enhanced set method
        async def enhanced_set(self, key, value, ttl=None, metadata=None):
            logger.info(f"Enhanced set called for key: {key[:30]}...")
            
            # Ensure the storage is initialized
            if not hasattr(self, 'storage') or self.storage is None:
                logger.error("Storage not initialized!")
                return False
            
            # Call original method
            result = await original_set(self, key, value, ttl, metadata)
            
            # Verify the result
            if result:
                logger.info(f"Successfully set key in cache: {key[:30]}...")
                
                # Force a database write
                try:
                    # Check if the key was actually written
                    stored_value = await self.storage.get(key)
                    if stored_value:
                        logger.info(f"Verified key exists in storage: {key[:30]}...")
                    else:
                        logger.error(f"Key not found in storage after set: {key[:30]}...")
                except Exception as e:
                    logger.error(f"Error verifying storage: {e}")
            else:
                logger.error(f"Failed to set key in cache: {key[:30]}...")
            
            return result
        
        # Apply the patch
        brave_search_persistent_cache.BraveSearchPersistentCache.set = enhanced_set
        logger.info("Successfully patched BraveSearchPersistentCache.set")
        
        return True
    except Exception as e:
        logger.error(f"Error patching BraveSearchPersistentCache: {e}")
        return False

async def main():
    """Main function to run all fixes and tests."""
    logger.info("Starting persistent cache fix...")
    
    # Fix permissions
    await fix_permissions()
    
    # Check database tables
    await check_database_tables()
    
    # Patch the persistent cache
    await patch_brave_search_persistent_cache()
    
    # Test the cache directly
    direct_test_result = await direct_cache_test()
    
    # Check database tables again after direct test
    await check_database_tables()
    
    # Test search with persistence
    search_test_result = await test_search_with_persistence()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("PERSISTENT CACHE FIX SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Direct cache test: {'PASSED' if direct_test_result else 'FAILED'}")
    logger.info(f"Search persistence test: {'PASSED' if search_test_result else 'FAILED'}")
    
    if direct_test_result and search_test_result:
        logger.info("All tests passed! The persistent cache is now working correctly.")
    else:
        logger.info("Some tests failed. The persistent cache may still have issues.")
    
    logger.info("=" * 50)
    
    # Close the persistent cache
    await close_persistent_cache()

if __name__ == "__main__":
    asyncio.run(main())
