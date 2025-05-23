#!/usr/bin/env python3
"""
Enhanced debug script to verify if the Brave Search persistent cache is working correctly.
This script adds more detailed debugging and fixes potential issues with the persistent cache.
"""

import os
import sys
import asyncio
import logging
import time
import json
import shutil
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cache_debug")

# Import necessary modules
try:
    from dotenv import load_dotenv
    # Load environment variables first
    load_dotenv()
except ImportError as e:
    logger.error(f"Failed to import dotenv: {e}")
    sys.exit(1)

# Patch environment variables if needed
def patch_environment():
    """Patch environment variables to fix potential issues."""
    # Get the home directory
    home_dir = os.path.expanduser("~")
    
    # Default cache directory
    default_cache_dir = os.path.join(home_dir, ".cache", "livekit-amanda")
    
    # Check and fix BRAVE_PERSISTENT_CACHE_DISK_CACHE_DIR
    cache_dir = os.environ.get("BRAVE_PERSISTENT_CACHE_DISK_CACHE_DIR", "/tmp/brave_persistent_cache")
    actual_cache_dir = default_cache_dir
    
    logger.info(f"Current BRAVE_PERSISTENT_CACHE_DISK_CACHE_DIR: {cache_dir}")
    logger.info(f"Actual cache directory being used: {actual_cache_dir}")
    
    # Update environment variable to match actual path
    os.environ["BRAVE_PERSISTENT_CACHE_DISK_CACHE_DIR"] = actual_cache_dir
    logger.info(f"Updated BRAVE_PERSISTENT_CACHE_DISK_CACHE_DIR to: {actual_cache_dir}")
    
    # Ensure the directory exists
    os.makedirs(actual_cache_dir, exist_ok=True)
    logger.info(f"Ensured cache directory exists: {actual_cache_dir}")
    
    return actual_cache_dir

# Now import the rest of the modules after patching environment
try:
    from brave_search_persistent_cache import get_persistent_cache, close_persistent_cache
    from brave_search_api import get_brave_search_client
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Monkey patch the persistent cache to add more detailed logging
def monkey_patch_persistent_cache():
    """Monkey patch the persistent cache to add more detailed logging."""
    import brave_search_persistent_cache
    
    # Store the original methods
    original_set = brave_search_persistent_cache.BraveSearchPersistentCache.set
    original_get = brave_search_persistent_cache.BraveSearchPersistentCache.get
    
    # Define enhanced methods with more logging
    async def enhanced_set(self, key, value, ttl=None, metadata=None):
        logger.debug(f"PersistentCache.set called with key: {key[:20]}..., ttl: {ttl}")
        result = await original_set(self, key, value, ttl, metadata)
        logger.debug(f"PersistentCache.set completed for key: {key[:20]}...")
        return result
    
    async def enhanced_get(self, key):
        logger.debug(f"PersistentCache.get called with key: {key[:20]}...")
        result = await original_get(self, key)
        hit_miss = "HIT" if result else "MISS"
        logger.debug(f"PersistentCache.get {hit_miss} for key: {key[:20]}...")
        return result
    
    # Apply the monkey patches
    brave_search_persistent_cache.BraveSearchPersistentCache.set = enhanced_set
    brave_search_persistent_cache.BraveSearchPersistentCache.get = enhanced_get
    
    logger.info("Monkey patched persistent cache methods for enhanced logging")

async def check_cache_files(cache_dir):
    """Check if the persistent cache files exist."""
    logger.info("Checking persistent cache files...")
    
    # Expected database paths
    expected_paths = [
        os.path.join(cache_dir, "brave_persistent_cache.db"),
        os.path.join(cache_dir, "brave_search_cache.db")
    ]
    
    found_files = []
    
    # Check all possible database files
    for db_path in expected_paths:
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            logger.info(f"Database file exists: {db_path} (Size: {file_size} bytes)")
            found_files.append({"path": db_path, "size": file_size})
        else:
            logger.info(f"Database file does not exist: {db_path}")
    
    # Check for any SQLite files in the cache directory
    for file in os.listdir(cache_dir):
        if file.endswith(".db") and os.path.join(cache_dir, file) not in expected_paths:
            file_path = os.path.join(cache_dir, file)
            file_size = os.path.getsize(file_path)
            logger.info(f"Found additional database file: {file_path} (Size: {file_size} bytes)")
            found_files.append({"path": file_path, "size": file_size})
    
    return found_files

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

async def test_persistence_across_restarts():
    """Test if the cache persists across simulated restarts."""
    logger.info("Testing persistence across restarts...")
    
    try:
        # First run - populate cache
        logger.info("First run - populating cache...")
        client1 = await get_brave_search_client()
        results1 = await client1.search(query="persistence test across restarts", count=3)
        
        # Get cache stats after first run
        cache1 = get_persistent_cache()
        stats1 = await cache1.get_stats()
        logger.info(f"Cache statistics after first run: {stats1}")
        
        # Close the cache to simulate application restart
        logger.info("Simulating application restart...")
        await close_persistent_cache()
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Second run - should hit persistent cache
        logger.info("Second run - should hit persistent cache...")
        client2 = await get_brave_search_client()
        start_time = time.time()
        results2 = await client2.search(query="persistence test across restarts", count=3)
        end_time = time.time()
        logger.info(f"Second run completed in {end_time - start_time:.2f} seconds")
        
        # Get cache stats after second run
        cache2 = get_persistent_cache()
        stats2 = await cache2.get_stats()
        logger.info(f"Cache statistics after second run: {stats2}")
        
        # Check if results are identical
        results_match = results1 == results2
        logger.info(f"Results match across restarts: {results_match}")
        
        return {
            "results_match": results_match,
            "second_query_time": end_time - start_time,
            "stats_before": stats1,
            "stats_after": stats2
        }
    except Exception as e:
        logger.error(f"Error testing persistence across restarts: {e}")
        return {
            "results_match": False,
            "error": str(e)
        }

async def examine_database_contents(db_files):
    """Examine the contents of the SQLite database files."""
    logger.info("Examining database contents...")
    
    try:
        import sqlite3
        
        db_contents = {}
        
        for db_file in db_files:
            db_path = db_file["path"]
            logger.info(f"Examining database: {db_path}")
            
            try:
                # Connect to the database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get list of tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                table_data = {}
                for table in tables:
                    table_name = table[0]
                    logger.info(f"Found table: {table_name}")
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    row_count = cursor.fetchone()[0]
                    
                    # Get column names
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    # Get sample data (first 5 rows)
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                    sample_rows = cursor.fetchall()
                    
                    table_data[table_name] = {
                        "row_count": row_count,
                        "columns": columns,
                        "sample_rows": [[str(cell)[:50] + "..." if isinstance(cell, str) and len(str(cell)) > 50 else cell for cell in row] for row in sample_rows]
                    }
                
                conn.close()
                db_contents[db_path] = table_data
                
            except Exception as e:
                logger.error(f"Error examining database {db_path}: {e}")
                db_contents[db_path] = {"error": str(e)}
        
        return db_contents
    except ImportError:
        logger.error("SQLite3 module not available")
        return {"error": "SQLite3 module not available"}
    except Exception as e:
        logger.error(f"Error examining database contents: {e}")
        return {"error": str(e)}

async def main():
    """Main function to run all checks."""
    logger.info("Starting enhanced persistent cache debug...")
    
    # Patch environment variables
    cache_dir = patch_environment()
    
    # Monkey patch for enhanced logging
    monkey_patch_persistent_cache()
    
    # Check cache configuration
    config = await check_cache_config()
    
    # If cache is not enabled, exit
    if not config["cache_enabled"]:
        logger.warning("Persistent cache is not enabled. Enable it by setting BRAVE_PERSISTENT_CACHE_ENABLE=true")
        return
    
    # Check cache files
    db_files = await check_cache_files(cache_dir)
    
    # Test cache with search
    search_results = await test_cache_with_search()
    
    # Get cache statistics
    stats = await get_cache_stats()
    
    # Test persistence across restarts
    persistence_results = await test_persistence_across_restarts()
    
    # Examine database contents
    db_contents = await examine_database_contents(db_files)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("ENHANCED PERSISTENT CACHE DEBUG SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Cache enabled: {config['cache_enabled']}")
    logger.info(f"Persistence enabled: {config['persistence_enabled']}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Database files found: {len(db_files)}")
    for db_file in db_files:
        logger.info(f"  - {db_file['path']} ({db_file['size']} bytes)")
    logger.info(f"Search results match: {search_results.get('results_match', False)}")
    logger.info(f"Persistence across restarts: {persistence_results.get('results_match', False)}")
    
    # Log database contents summary
    logger.info("Database contents summary:")
    for db_path, tables in db_contents.items():
        if "error" in tables:
            logger.info(f"  {db_path}: Error - {tables['error']}")
        else:
            logger.info(f"  {db_path}:")
            for table_name, table_data in tables.items():
                logger.info(f"    {table_name}: {table_data['row_count']} rows")
    
    logger.info("=" * 50)
    
    # Close the persistent cache
    await close_persistent_cache()

if __name__ == "__main__":
    asyncio.run(main())
