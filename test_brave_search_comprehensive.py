#!/usr/bin/env python3
"""
Comprehensive test script for Brave Search Quality API integration.

This script tests all components of the Brave Search ecosystem:
1. Brave Quality Search API
2. Brave API Search (with key)
3. Brave NoKey Search
4. Memory Cache
5. Persistent Cache
6. RAG Data Embedding
7. Persistent Cache Embedding
8. RAG Integration
9. Grounding

Run this script to verify that all components are working correctly.
"""

import os
import sys
import asyncio
import logging
import time
import json
from pprint import pprint
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_brave_search_comprehensive")

# Import Brave Search components
# Define flags for available components
HAS_QUALITY_API = False
HAS_PERSISTENT_CACHE = False
HAS_BRAVE_SEARCH_API = False
HAS_BRAVE_SEARCH_NOKEY = False
HAS_GROUNDING = False

# Import quality API components
try:
    from brave_search_quality_api import get_quality_api, high_quality_web_search, high_quality_ai_search
    HAS_QUALITY_API = True
except ImportError as e:
    logger.warning(f"Brave Search Quality API not available: {e}")

# Import persistent cache
try:
    from brave_search_persistent_cache import get_persistent_cache
    HAS_PERSISTENT_CACHE = True
except ImportError as e:
    logger.warning(f"Brave Search Persistent Cache not available: {e}")

# Import Brave Search API client
try:
    from brave_search_api import get_brave_search_client
    HAS_BRAVE_SEARCH_API = True
except ImportError as e:
    logger.warning(f"Brave Search API client not available: {e}")

# Import Brave Search NoKey
try:
    # Try different possible import paths
    try:
        from brave_search_free_tier import search as brave_search_nokey
    except ImportError:
        try:
            from brave_search_nokey import search as brave_search_nokey
        except ImportError:
            from brave_search_free_tier import brave_search as brave_search_nokey
    HAS_BRAVE_SEARCH_NOKEY = True
except ImportError as e:
    logger.warning(f"Brave Search NoKey not available: {e}")

# Import Grounding
try:
    from brave_search_grounding import ground_search_results
    HAS_GROUNDING = True
except ImportError as e:
    logger.warning(f"Brave Search Grounding not available: {e}")

# Try to import RAG integration components
try:
    from brave_search_quality_rag_integration import (
        get_integration,
        process_data_directory,
        search_with_rag,
        close_integration
    )
except ImportError as e:
    logger.warning(f"RAG integration components not available: {e}")
    HAS_RAG = False
else:
    HAS_RAG = True

# Test data
TEST_QUERIES = [
    "climate change solutions",
    "artificial intelligence ethics",
    "renewable energy technologies"
]

class ComprehensiveTest:
    """Comprehensive test for Brave Search Quality API integration."""
    
    def __init__(self):
        """Initialize test components."""
        logger.info("Initializing test components")
        
        # Initialize Brave Search components if available
        self.quality_api = get_quality_api() if HAS_QUALITY_API else None
        self.persistent_cache = get_persistent_cache() if HAS_PERSISTENT_CACHE else None
        self.brave_search_client = get_brave_search_client() if HAS_BRAVE_SEARCH_API else None
        
        # Test status tracking
        self.test_results = {
            "brave_quality_search": False,
            "brave_api_search": False,
            "brave_nokey_search": False,
            "memory_cache": False,
            "persistent_cache": False,
            "rag_data_embedding": False,
            "persistent_cache_embedding": False,
            "rag_integration": False,
            "grounding": False
        }
        
        logger.info("Test components initialized")
    
    async def test_brave_quality_search(self):
        """Test Brave Quality Search API."""
        logger.info("Testing Brave Quality Search API")
        
        if not HAS_QUALITY_API:
            logger.warning("⚠️ Brave Quality Search API test skipped: component not available")
            return None
        
        try:
            context = {"session_id": "test_session"}
            query = TEST_QUERIES[0]
            
            results = await high_quality_web_search(context, query, num_results=3)
            
            if results and len(results) > 100:  # Basic validation
                logger.info("✅ Brave Quality Search API test passed")
                self.test_results["brave_quality_search"] = True
                return results
            else:
                logger.error("❌ Brave Quality Search API test failed: insufficient results")
                return None
        except Exception as e:
            logger.error(f"❌ Brave Quality Search API test failed with error: {e}")
            return None
    
    async def test_brave_api_search(self):
        """Test Brave API Search (with key)."""
        logger.info("Testing Brave API Search (with key)")
        
        if not HAS_BRAVE_SEARCH_API:
            logger.warning("⚠️ Brave API Search test skipped: component not available")
            return None
        
        try:
            query = TEST_QUERIES[1]
            
            results = await self.brave_search_client.search(query, count=3)
            
            if results and "web" in results and len(results["web"].get("results", [])) > 0:
                logger.info("✅ Brave API Search test passed")
                self.test_results["brave_api_search"] = True
                return results
            else:
                logger.error("❌ Brave API Search test failed: insufficient results")
                return None
        except Exception as e:
            logger.error(f"❌ Brave API Search test failed with error: {e}")
            return None
    
    async def test_brave_nokey_search(self):
        """Test Brave NoKey Search."""
        logger.info("Testing Brave NoKey Search")
        
        if not HAS_BRAVE_SEARCH_NOKEY:
            logger.warning("⚠️ Brave NoKey Search test skipped: component not available")
            return None
        
        try:
            query = TEST_QUERIES[2]
            
            results = await brave_search_nokey(query, num_results=3)
            
            if results and len(results) > 0:
                logger.info("✅ Brave NoKey Search test passed")
                self.test_results["brave_nokey_search"] = True
                return results
            else:
                logger.error("❌ Brave NoKey Search test failed: insufficient results")
                return None
        except Exception as e:
            logger.error(f"❌ Brave NoKey Search test failed with error: {e}")
            return None
    
    async def test_memory_cache(self):
        """Test Memory Cache."""
        logger.info("Testing Memory Cache")
        
        if not HAS_RAG:
            logger.warning("⚠️ Memory Cache test skipped: RAG integration not available")
            return None
        
        try:
            # Get the integration instance
            integration = get_integration()
            
            # Check if memory cache is initialized
            if integration.memory_cache is None:
                logger.error("❌ Memory Cache test failed: memory cache not initialized")
                return None
            
            # Perform a search to populate memory cache
            context = {"session_id": "test_session"}
            query = TEST_QUERIES[0]
            
            # First search to populate cache
            await search_with_rag(context, query, num_results=3)
            
            # Second search should use memory cache
            start_time = time.time()
            results = await search_with_rag(context, query, num_results=3)
            end_time = time.time()
            
            # If memory cache is working, second search should be much faster
            if results and (end_time - start_time) < 1.0:  # Less than 1 second indicates cache hit
                logger.info(f"✅ Memory Cache test passed (response time: {end_time - start_time:.2f}s)")
                self.test_results["memory_cache"] = True
                return results
            else:
                logger.error(f"❌ Memory Cache test failed: slow response time ({end_time - start_time:.2f}s)")
                return None
        except Exception as e:
            logger.error(f"❌ Memory Cache test failed with error: {e}")
            return None
    
    async def test_persistent_cache(self):
        """Test Persistent Cache."""
        logger.info("Testing Persistent Cache")
        
        if not HAS_PERSISTENT_CACHE:
            logger.warning("⚠️ Persistent Cache test skipped: component not available")
            return None
        
        try:
            # Generate a unique cache key
            cache_key = f"test_persistent_cache:{int(time.time())}"
            test_data = {
                "query": TEST_QUERIES[0],
                "timestamp": time.time(),
                "test_id": "persistent_cache_test"
            }
            
            # Store data in persistent cache
            self.persistent_cache.store(
                cache_key,
                json.dumps(test_data),
                metadata={"test": True}
            )
            
            # Retrieve data from persistent cache
            cached_data = self.persistent_cache.get(cache_key)
            
            if cached_data:
                retrieved_data = json.loads(cached_data)
                if retrieved_data.get("test_id") == "persistent_cache_test":
                    logger.info("✅ Persistent Cache test passed")
                    self.test_results["persistent_cache"] = True
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
    
    async def test_rag_data_embedding(self):
        """Test RAG Data Embedding."""
        logger.info("Testing RAG Data Embedding")
        
        if not HAS_RAG:
            logger.warning("⚠️ RAG Data Embedding test skipped: RAG integration not available")
            return None
        
        try:
            # Create a test data file
            data_dir = os.environ.get("DATA_DIR", "./data")
            os.makedirs(data_dir, exist_ok=True)
            
            test_file_path = os.path.join(data_dir, "test_embedding.txt")
            with open(test_file_path, "w") as f:
                f.write(f"This is a test file for RAG data embedding.\n")
                f.write(f"It contains information about {TEST_QUERIES[0]}.\n")
                f.write(f"The file should be processed and embedded in the RAG vector store.\n")
            
            # Process the data directory
            await process_data_directory()
            
            # Get the integration instance
            integration = get_integration()
            
            # Check if the vector store contains documents
            try:
                rag_docs = integration.vector_store_rag.get()
                if rag_docs and len(rag_docs.get("ids", [])) > 0:
                    logger.info(f"✅ RAG Data Embedding test passed: {len(rag_docs.get('ids', []))} documents in vector store")
                    self.test_results["rag_data_embedding"] = True
                    
                    # Clean up test file
                    os.remove(test_file_path)
                    return rag_docs
                else:
                    logger.error("❌ RAG Data Embedding test failed: no documents in vector store")
                    return None
            except Exception as e:
                logger.error(f"❌ RAG Data Embedding test failed with error accessing vector store: {e}")
                return None
        except Exception as e:
            logger.error(f"❌ RAG Data Embedding test failed with error: {e}")
            return None
    
    async def test_persistent_cache_embedding(self):
        """Test Persistent Cache Embedding."""
        logger.info("Testing Persistent Cache Embedding")
        
        if not HAS_RAG:
            logger.warning("⚠️ Persistent Cache Embedding test skipped: RAG integration not available")
            return None
        
        try:
            # Get the integration instance
            integration = get_integration()
            
            # Perform a search to populate persistent cache and vector store
            context = {"session_id": "test_session"}
            query = TEST_QUERIES[2]  # Use a different query
            
            # Perform search to populate cache and trigger embedding
            await search_with_rag(context, query, num_results=3)
            
            # Check if the quality vector store contains documents
            try:
                quality_docs = integration.vector_store_quality.get()
                if quality_docs and len(quality_docs.get("ids", [])) > 0:
                    logger.info(f"✅ Persistent Cache Embedding test passed: {len(quality_docs.get('ids', []))} documents in vector store")
                    self.test_results["persistent_cache_embedding"] = True
                    return quality_docs
                else:
                    logger.error("❌ Persistent Cache Embedding test failed: no documents in vector store")
                    return None
            except Exception as e:
                logger.error(f"❌ Persistent Cache Embedding test failed with error accessing vector store: {e}")
                return None
        except Exception as e:
            logger.error(f"❌ Persistent Cache Embedding test failed with error: {e}")
            return None
    
    async def test_rag_integration(self):
        """Test RAG Integration."""
        logger.info("Testing RAG Integration")
        
        if not HAS_RAG:
            logger.warning("⚠️ RAG Integration test skipped: RAG integration not available")
            return None
        
        try:
            # Perform a search with RAG
            context = {"session_id": "test_session"}
            query = TEST_QUERIES[0]
            
            results = await search_with_rag(context, query, num_results=3)
            
            if results and len(results) > 100:  # Basic validation
                logger.info("✅ RAG Integration test passed")
                self.test_results["rag_integration"] = True
                return results
            else:
                logger.error("❌ RAG Integration test failed: insufficient results")
                return None
        except Exception as e:
            logger.error(f"❌ RAG Integration test failed with error: {e}")
            return None
    
    async def test_grounding(self):
        """Test Grounding."""
        logger.info("Testing Grounding")
        
        if not HAS_GROUNDING or not HAS_QUALITY_API:
            logger.warning("⚠️ Grounding test skipped: component not available")
            return None
        
        try:
            # Get search results to ground
            context = {"session_id": "test_session"}
            query = TEST_QUERIES[1]
            
            # Get search results
            search_results = await high_quality_web_search(context, query, num_results=3)
            
            if not search_results:
                logger.error("❌ Grounding test failed: no search results to ground")
                return None
            
            # Ground the search results
            grounded_results = await ground_search_results(search_results)
            
            if grounded_results and len(grounded_results) > 0:
                logger.info("✅ Grounding test passed")
                self.test_results["grounding"] = True
                return grounded_results
            else:
                logger.error("❌ Grounding test failed: no grounded results")
                return None
        except Exception as e:
            logger.error(f"❌ Grounding test failed with error: {e}")
            return None
    
    def print_summary(self):
        """Print a summary of all test results."""
        logger.info("\n" + "=" * 50)
        logger.info("BRAVE SEARCH QUALITY API INTEGRATION TEST SUMMARY")
        logger.info("=" * 50)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name.ljust(30)}: {status}")
        
        logger.info("=" * 50)
        
        # Calculate overall status
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        if not HAS_RAG:
            # Adjust for RAG-related tests
            rag_tests = ["memory_cache", "rag_data_embedding", "persistent_cache_embedding", "rag_integration"]
            total -= len(rag_tests)
        
        logger.info(f"Overall Status: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        logger.info("=" * 50)

async def main():
    """Run all tests."""
    test = ComprehensiveTest()
    
    try:
        # Print available components
        logger.info("\n" + "=" * 50)
        logger.info("COMPONENT AVAILABILITY")
        logger.info("=" * 50)
        logger.info(f"Brave Quality API: {'✅ Available' if HAS_QUALITY_API else '❌ Not Available'}")
        logger.info(f"Persistent Cache: {'✅ Available' if HAS_PERSISTENT_CACHE else '❌ Not Available'}")
        logger.info(f"Brave Search API: {'✅ Available' if HAS_BRAVE_SEARCH_API else '❌ Not Available'}")
        logger.info(f"Brave NoKey Search: {'✅ Available' if HAS_BRAVE_SEARCH_NOKEY else '❌ Not Available'}")
        logger.info(f"Grounding: {'✅ Available' if HAS_GROUNDING else '❌ Not Available'}")
        logger.info(f"RAG Integration: {'✅ Available' if HAS_RAG else '❌ Not Available'}")
        logger.info("=" * 50 + "\n")
        
        # Run all tests
        await test.test_brave_quality_search()
        await test.test_brave_api_search()
        await test.test_brave_nokey_search()
        await test.test_persistent_cache()
        
        if HAS_RAG:
            await test.test_memory_cache()
            await test.test_rag_data_embedding()
            await test.test_persistent_cache_embedding()
            await test.test_rag_integration()
        
        await test.test_grounding()
        
        # Print summary
        test.print_summary()
    except Exception as e:
        logger.error(f"Error during testing: {e}")
    finally:
        # Close resources
        if HAS_RAG:
            await close_integration()
        logger.info("Test completed")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
