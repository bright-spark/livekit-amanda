#!/usr/bin/env python3
"""
Script to update the persistent cache configuration to use a lower quality threshold
and ensure that data is properly stored in the SQLite database.
"""

import os
import sys
import asyncio
import logging
import sqlite3
import json
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

logger = logging.getLogger("cache-config-updater")

# Load environment variables
load_dotenv()

# Import the brave search modules
try:
    from brave_search_persistent_cache import BraveSearchPersistentCache
    HAS_BRAVE_SEARCH = True
    logger.info("Brave Search modules loaded successfully")
except ImportError:
    HAS_BRAVE_SEARCH = False
    logger.warning("Brave Search modules not available")

# Set the desired quality threshold
QUALITY_THRESHOLD = 0.1

async def update_persistent_cache_config():
    """Update the persistent cache configuration to use a lower quality threshold."""
    if not HAS_BRAVE_SEARCH:
        logger.error("Brave Search modules not available, cannot update cache configuration")
        return False
    
    # Initialize the persistent cache with the lower quality threshold
    cache_config = {
        "quality_threshold": QUALITY_THRESHOLD  # Lower threshold to allow more entries to be cached
    }
    
    try:
        # Create the cache instance with our custom config
        cache = BraveSearchPersistentCache(config=cache_config)
        
        # Check the current configuration
        logger.info(f"Current cache configuration: {cache.config}")
        
        # Verify the quality threshold was set correctly
        actual_threshold = cache.config.get("quality_threshold", None)
        if actual_threshold == QUALITY_THRESHOLD:
            logger.info(f"Quality threshold set successfully to {QUALITY_THRESHOLD}")
        else:
            logger.warning(f"Quality threshold not set correctly. Expected: {QUALITY_THRESHOLD}, Actual: {actual_threshold}")
        
        # Get the database path
        db_path = cache.config.get("db_path")
        if not db_path:
            # Use the default path if not specified
            db_path = os.path.expanduser("~/.cache/livekit-amanda/brave_persistent_cache.db")
        
        # Check if the database exists
        if os.path.exists(db_path):
            logger.info(f"SQLite database found at {db_path}")
            
            # Connect to the database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if the config table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='config'")
            if cursor.fetchone():
                # Update the config in the database
                config_json = json.dumps({
                    "quality_threshold": QUALITY_THRESHOLD
                })
                
                # Try to update the existing config
                cursor.execute("UPDATE config SET value = ? WHERE key = 'quality_threshold'", (QUALITY_THRESHOLD,))
                
                # If no rows were updated, insert a new config entry
                if cursor.rowcount == 0:
                    cursor.execute("INSERT INTO config (key, value) VALUES (?, ?)", 
                                  ("quality_threshold", QUALITY_THRESHOLD))
                
                conn.commit()
                logger.info(f"Updated quality_threshold in database to {QUALITY_THRESHOLD}")
            else:
                logger.warning("Config table not found in database")
            
            # Close the connection
            conn.close()
        else:
            logger.warning(f"SQLite database not found at {db_path}")
        
        # Set the environment variable for future runs
        os.environ["BRAVE_PERSISTENT_CACHE_QUALITY_THRESHOLD"] = str(QUALITY_THRESHOLD)
        logger.info(f"Set BRAVE_PERSISTENT_CACHE_QUALITY_THRESHOLD={QUALITY_THRESHOLD} in environment")
        
        # Update the .env file if it exists
        env_file = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_file):
            # Read the current .env file
            with open(env_file, "r") as f:
                lines = f.readlines()
            
            # Check if the variable already exists
            found = False
            for i, line in enumerate(lines):
                if line.startswith("BRAVE_PERSISTENT_CACHE_QUALITY_THRESHOLD="):
                    lines[i] = f"BRAVE_PERSISTENT_CACHE_QUALITY_THRESHOLD={QUALITY_THRESHOLD}\n"
                    found = True
                    break
            
            # Add the variable if it doesn't exist
            if not found:
                lines.append(f"\n# Lower quality threshold for persistent cache\nBRAVE_PERSISTENT_CACHE_QUALITY_THRESHOLD={QUALITY_THRESHOLD}\n")
            
            # Write the updated .env file
            with open(env_file, "w") as f:
                f.writelines(lines)
            
            logger.info(f"Updated .env file with BRAVE_PERSISTENT_CACHE_QUALITY_THRESHOLD={QUALITY_THRESHOLD}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating persistent cache configuration: {e}")
        return False
    finally:
        # Clean up resources
        if 'cache' in locals() and hasattr(cache, 'close'):
            await cache.close()
            logger.info("Closed cache connection")

async def test_cache_with_new_config():
    """Test the persistent cache with the new configuration."""
    if not HAS_BRAVE_SEARCH:
        logger.error("Brave Search modules not available, cannot test cache")
        return False
    
    # Get the API key from environment variables
    api_key = os.getenv("BRAVE_WEB_SEARCH_API_KEY")
    if not api_key:
        logger.error("BRAVE_WEB_SEARCH_API_KEY not set in environment variables")
        return False
    
    # Initialize the persistent cache with the lower quality threshold
    cache_config = {
        "quality_threshold": QUALITY_THRESHOLD  # Lower threshold to allow more entries to be cached
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
        logger.error(f"Error testing persistent cache: {e}")
        return False
    finally:
        # Clean up resources
        if 'cache' in locals() and hasattr(cache, 'close'):
            await cache.close()
            logger.info("Closed cache connection")

async def main():
    """Main function to update the persistent cache configuration."""
    logger.info("Starting persistent cache configuration update...")
    
    # Update the persistent cache configuration
    update_success = await update_persistent_cache_config()
    logger.info(f"Configuration update: {'SUCCESS' if update_success else 'FAILED'}")
    
    # Import time here to avoid potential circular imports
    import time
    
    # Test the cache with the new configuration
    test_success = await test_cache_with_new_config()
    logger.info(f"Cache test: {'SUCCESS' if test_success else 'FAILED'}")
    
    logger.info("Configuration update complete!")

if __name__ == "__main__":
    # Import time here to avoid potential circular imports
    import time
    asyncio.run(main())
