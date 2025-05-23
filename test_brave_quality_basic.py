#!/usr/bin/env python3
"""
Basic test script for Brave Search Quality API and Persistent Cache.
This script tests the core functionality without requiring the full RAG integration.
"""

import os
import asyncio
import logging
import time
import json
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_brave_quality_basic")

# Import Brave Search components
from brave_search_quality_api import get_quality_api, high_quality_web_search
from brave_search_persistent_cache import get_persistent_cache

async def test_brave_quality_search():
    """Test Brave Quality Search API."""
    logger.info("Testing Brave Quality Search API")
    
    try:
        # Get the quality API
        quality_api = get_quality_api()
        
        # Create a test context
        context = {"session_id": "test_session"}
        
        # Perform a search
        query = "climate change solutions"
        logger.info(f"Searching for: {query}")
        
        results = await high_quality_web_search(context, query, num_results=3)
        
        if results and len(results) > 100:  # Basic validation
            logger.info("✅ Brave Quality Search API test passed")
            
            # Print a preview of the results
            preview = results[:300] + "..." if len(results) > 300 else results
            logger.info(f"Search results preview: {preview}")
            
            return results
        else:
            logger.error("❌ Brave Quality Search API test failed: insufficient results")
            return None
    except Exception as e:
        logger.error(f"❌ Brave Quality Search API test failed with error: {e}")
        return None

async def test_persistent_cache():
    """Test Persistent Cache."""
    logger.info("Testing Persistent Cache")
    
    try:
        # Get the persistent cache
        persistent_cache = get_persistent_cache()
        
        # Generate a unique cache key
        cache_key = f"test_persistent_cache:{int(time.time())}"
        test_data = {
            "query": "test query",
            "timestamp": time.time(),
            "test_id": "persistent_cache_test"
        }
        
        # Store data in persistent cache
        logger.info(f"Storing data in persistent cache with key: {cache_key}")
        persistent_cache.store(
            cache_key,
            json.dumps(test_data),
            metadata={"test": True}
        )
        
        # Retrieve data from persistent cache
        logger.info(f"Retrieving data from persistent cache with key: {cache_key}")
        cached_data = persistent_cache.get(cache_key)
        
        if cached_data:
            retrieved_data = json.loads(cached_data)
            if retrieved_data.get("test_id") == "persistent_cache_test":
                logger.info("✅ Persistent Cache test passed")
                logger.info(f"Retrieved data: {retrieved_data}")
                return retrieved_data
            else:
                logger.error("❌ Persistent Cache test failed: data mismatch")
                return None
        else:
            logger.error("❌ Persistent Cache test failed: data not retrieved")
            return None
    except Exception as e:
        logger.error(f"❌ Persistent Cache test failed with error: {e}")
        return None

async def test_quality_search_with_cache():
    """Test Brave Quality Search API with Persistent Cache."""
    logger.info("Testing Brave Quality Search API with Persistent Cache")
    
    try:
        # Get the quality API and persistent cache
        quality_api = get_quality_api()
        persistent_cache = get_persistent_cache()
        
        # Create a test context
        context = {"session_id": "test_session"}
        
        # Perform a search
        query = "artificial intelligence ethics"
        cache_key = f"quality_search:{query}:3"
        
        # Clear the cache first to ensure a fresh test
        try:
            persistent_cache.delete(cache_key)
            logger.info(f"Cleared cache key: {cache_key}")
        except:
            pass
        
        logger.info(f"First search for: {query} (should miss cache)")
        start_time = time.time()
        results1 = await high_quality_web_search(context, query, num_results=3)
        end_time = time.time()
        first_search_time = end_time - start_time
        
        logger.info(f"First search completed in {first_search_time:.2f} seconds")
        
        # Check if results were stored in cache
        cached_data = persistent_cache.get(cache_key)
        if cached_data:
            logger.info("✅ Results successfully stored in cache")
        else:
            logger.warning("⚠️ Results not stored in cache")
        
        # Perform the same search again (should hit cache)
        logger.info(f"Second search for: {query} (should hit cache)")
        start_time = time.time()
        results2 = await high_quality_web_search(context, query, num_results=3)
        end_time = time.time()
        second_search_time = end_time - start_time
        
        logger.info(f"Second search completed in {second_search_time:.2f} seconds")
        
        # Compare search times
        if second_search_time < first_search_time:
            logger.info(f"✅ Cache hit confirmed: second search ({second_search_time:.2f}s) was faster than first search ({first_search_time:.2f}s)")
            return True
        else:
            logger.warning(f"⚠️ Cache may not be working: second search ({second_search_time:.2f}s) was not faster than first search ({first_search_time:.2f}s)")
            return False
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return None

async def main():
    """Run all tests."""
    try:
        logger.info("Starting tests for Brave Search Quality API and Persistent Cache")
        
        # Test Brave Quality Search API
        await test_brave_quality_search()
        
        # Test Persistent Cache
        await test_persistent_cache()
        
        # Test Brave Quality Search API with Persistent Cache
        await test_quality_search_with_cache()
        
        logger.info("All tests completed")
    except Exception as e:
        logger.error(f"Error during testing: {e}")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
