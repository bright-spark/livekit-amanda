"""
Test module for the Brave Search Quality API.

This module tests the functionality of the Brave Search Quality API and its integration
with the persistent cache for high-quality data retrieval.
"""

import os
import json
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import modules to test
from brave_search_quality_api import (
    BraveSearchQualityAPI,
    get_quality_api,
    close_quality_api,
    high_quality_web_search,
    high_quality_ai_search,
    improve_search_quality
)

class MockContext:
    """Mock context for testing."""
    def __init__(self):
        self.data = {}

class TestBraveSearchQualityAPI(unittest.TestCase):
    """Test cases for the Brave Search Quality API."""
    
    def setUp(self):
        """Set up the test environment."""
        pass
    
    async def asyncSetUp(self):
        """Async setup for the test environment."""
        # Create a mock context
        self.context = MockContext()
        
        # Create a test configuration
        self.test_config = {
            "enable_quality_api": True,
            "quality_threshold": 0.7,
            "cache_ttl": 3600,  # 1 hour for testing
            "refresh_threshold": 1800,  # 30 minutes for testing
            "max_retries": 2,
            "enrichment_enabled": True,
            "fallback_to_regular": True,
            "parallel_processing": True,
        }
        
        # Sample web search results
        self.sample_web_results = {
            "query": {"query": "test query"},
            "web": {
                "results": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "description": "This is test result 1",
                        "domain": "example.com",
                        "age": "2 days ago"
                    },
                    {
                        "title": "Test Result 2",
                        "url": "https://example.com/2",
                        "description": "This is test result 2",
                        "domain": "example.com",
                        "age": "1 week ago"
                    }
                ],
                "featured_snippet": {
                    "title": "Featured Snippet",
                    "description": "This is a featured snippet",
                    "url": "https://example.com/featured"
                }
            },
            "news": {
                "results": [
                    {
                        "title": "Test News 1",
                        "url": "https://example.com/news/1",
                        "description": "This is test news 1",
                        "source": "Example News",
                        "age": "1 day ago"
                    }
                ]
            },
            "_metadata": {
                "quality_score": 0.85,
                "enriched_at": datetime.now().isoformat()
            }
        }
        
        # Sample AI search results
        self.sample_ai_results = {
            "query": {"query": "test query"},
            "generated_answer": {
                "answer": "This is a test AI-generated answer.",
                "_confidence_score": 0.9,
                "points": [
                    "Supporting point 1",
                    "Supporting point 2"
                ],
                "sources": [
                    {
                        "title": "Source 1",
                        "url": "https://example.com/source/1"
                    },
                    {
                        "title": "Source 2",
                        "url": "https://example.com/source/2"
                    }
                ]
            },
            "_metadata": {
                "quality_score": 0.9,
                "enriched_at": datetime.now().isoformat()
            }
        }
        
        # Create a patch for the persistent cache
        self.persistent_cache_patch = patch('brave_search_quality_api.get_persistent_cache')
        self.mock_get_persistent_cache = self.persistent_cache_patch.start()
        
        # Create a mock persistent cache
        self.mock_persistent_cache = AsyncMock()
        self.mock_persistent_cache.get_high_quality_result = AsyncMock(return_value=None)
        self.mock_persistent_cache.store_high_quality_result = AsyncMock(return_value=True)
        self.mock_persistent_cache.get_stats = AsyncMock(return_value={"entries": 10, "size_bytes": 1024})
        
        # Set up the mock to return our mock persistent cache
        self.mock_get_persistent_cache.return_value = self.mock_persistent_cache
        
        # Create patches for the Brave Search API functions
        self.web_search_patch = patch('brave_search_quality_api.brave_web_search')
        self.ai_search_patch = patch('brave_search_quality_api.brave_ai_search')
        self.get_web_client_patch = patch('brave_search_quality_api.get_brave_web_search_client')
        self.get_ai_client_patch = patch('brave_search_quality_api.get_brave_ai_search_client')
        
        # Start the patches
        self.mock_web_search = self.web_search_patch.start()
        self.mock_ai_search = self.ai_search_patch.start()
        self.mock_get_web_client = self.get_web_client_patch.start()
        self.mock_get_ai_client = self.get_ai_client_patch.start()
        
        # Set up the mock web search client
        self.mock_web_client = AsyncMock()
        self.mock_web_client.search = AsyncMock(return_value=self.sample_web_results)
        self.mock_get_web_client.return_value = self.mock_web_client
        
        # Set up the mock AI search client
        self.mock_ai_client = AsyncMock()
        self.mock_ai_client.ai_search = AsyncMock(return_value=self.sample_ai_results)
        self.mock_get_ai_client.return_value = self.mock_ai_client
        
        # Set up the mock search functions
        self.mock_web_search.return_value = self.sample_web_results
        self.mock_ai_search.return_value = self.sample_ai_results
        
        # Initialize the quality API
        self.api = BraveSearchQualityAPI(config=self.test_config)
    
    async def asyncTearDown(self):
        """Async teardown for the test environment."""
        # Stop all patches
        self.persistent_cache_patch.stop()
        self.web_search_patch.stop()
        self.ai_search_patch.stop()
        self.get_web_client_patch.stop()
        self.get_ai_client_patch.stop()
        
        # Close the quality API
        await close_quality_api()
    
    async def test_get_high_quality_web_search_from_cache(self):
        """Test getting high-quality web search results from cache."""
        # Set up the mock to return cached results
        self.mock_persistent_cache.get_high_quality_result.return_value = self.sample_web_results
        
        # Call the method
        result = await self.api.get_high_quality_web_search(
            self.context, "test query", num_results=2
        )
        
        # Check that the result is the cached result
        self.assertEqual(result, self.sample_web_results)
        
        # Check that the cache was queried
        self.mock_persistent_cache.get_high_quality_result.assert_called_once()
        
        # Check that the web search was not called
        self.mock_web_client.search.assert_not_called()
    
    async def test_get_high_quality_web_search_from_api(self):
        """Test getting high-quality web search results from the API."""
        # Set up the mock to return no cached results
        self.mock_persistent_cache.get_high_quality_result.return_value = None
        
        # Call the method
        result = await self.api.get_high_quality_web_search(
            self.context, "test query", num_results=2
        )
        
        # Check that the result is the API result
        self.assertEqual(result, self.sample_web_results)
        
        # Check that the cache was queried
        self.mock_persistent_cache.get_high_quality_result.assert_called()
        
        # Check that the web search was called
        self.mock_web_client.search.assert_called_once()
        
        # Check that the result was stored in the cache
        self.mock_persistent_cache.store_high_quality_result.assert_called_once()
    
    async def test_get_high_quality_web_search_force_refresh(self):
        """Test forcing a refresh for high-quality web search results."""
        # Call the method with force_refresh=True
        result = await self.api.get_high_quality_web_search(
            self.context, "test query", num_results=2, force_refresh=True
        )
        
        # Check that the result is the API result
        self.assertEqual(result, self.sample_web_results)
        
        # Check that the cache was not queried
        self.mock_persistent_cache.get_high_quality_result.assert_not_called()
        
        # Check that the web search was called
        self.mock_web_client.search.assert_called_once()
        
        # Check that the result was stored in the cache
        self.mock_persistent_cache.store_high_quality_result.assert_called_once()
    
    async def test_get_high_quality_ai_search_from_cache(self):
        """Test getting high-quality AI search results from cache."""
        # Set up the mock to return cached results
        self.mock_persistent_cache.get_high_quality_result.return_value = self.sample_ai_results
        
        # Call the method
        result = await self.api.get_high_quality_ai_search(
            self.context, "test query"
        )
        
        # Check that the result is the cached result
        self.assertEqual(result, self.sample_ai_results)
        
        # Check that the cache was queried
        self.mock_persistent_cache.get_high_quality_result.assert_called_once()
        
        # Check that the AI search was not called
        self.mock_ai_client.ai_search.assert_not_called()
    
    async def test_get_high_quality_ai_search_from_api(self):
        """Test getting high-quality AI search results from the API."""
        # Set up the mock to return no cached results
        self.mock_persistent_cache.get_high_quality_result.return_value = None
        
        # Call the method
        result = await self.api.get_high_quality_ai_search(
            self.context, "test query"
        )
        
        # Check that the result is the API result
        self.assertEqual(result, self.sample_ai_results)
        
        # Check that the cache was queried
        self.mock_persistent_cache.get_high_quality_result.assert_called()
        
        # Check that the AI search was called
        self.mock_ai_client.ai_search.assert_called_once()
        
        # Check that the result was stored in the cache
        self.mock_persistent_cache.store_high_quality_result.assert_called_once()
    
    async def test_improve_search_quality(self):
        """Test improving search quality by trying different query variations."""
        # Set up the mock to return results with increasing quality scores
        low_quality_result = self.sample_web_results.copy()
        low_quality_result["_metadata"]["quality_score"] = 0.5
        
        medium_quality_result = self.sample_web_results.copy()
        medium_quality_result["_metadata"]["quality_score"] = 0.7
        
        high_quality_result = self.sample_web_results.copy()
        high_quality_result["_metadata"]["quality_score"] = 0.9
        
        # Set up the mock to return different results for different queries
        async def mock_get_high_quality_web_search(context, query, **kwargs):
            if query == "test query":
                return low_quality_result
            elif query == "test query detailed information":
                return medium_quality_result
            elif query == "test query comprehensive guide":
                return high_quality_result
            else:
                return low_quality_result
        
        # Replace the method with our mock
        self.api.get_high_quality_web_search = mock_get_high_quality_web_search
        
        # Call the method
        result = await self.api.improve_search_quality(
            self.context, "test query", "web", max_retries=2
        )
        
        # Check that the result is the highest quality result
        self.assertEqual(result, high_quality_result)
    
    async def test_format_high_quality_web_results(self):
        """Test formatting high-quality web search results."""
        # Call the method
        formatted = await self.api.format_high_quality_web_results(
            self.sample_web_results, num_results=2
        )
        
        # Check that the formatted result contains expected elements
        self.assertIn("HIGH-QUALITY WEB SEARCH GROUNDING INFORMATION", formatted)
        self.assertIn("Query: 'test query'", formatted)
        self.assertIn("Test Result 1", formatted)
        self.assertIn("Test Result 2", formatted)
        self.assertIn("https://example.com/1", formatted)
        self.assertIn("Featured Snippet", formatted)
        self.assertIn("Test News 1", formatted)
    
    async def test_format_high_quality_ai_results(self):
        """Test formatting high-quality AI search results."""
        # Call the method
        formatted = await self.api.format_high_quality_ai_results(
            self.sample_ai_results
        )
        
        # Check that the formatted result contains expected elements
        self.assertIn("HIGH-QUALITY AI SEARCH GROUNDING INFORMATION", formatted)
        self.assertIn("Query: 'test query'", formatted)
        self.assertIn("This is a test AI-generated answer.", formatted)
        self.assertIn("Confidence: 0.90", formatted)
        self.assertIn("Supporting point 1", formatted)
        self.assertIn("Source 1", formatted)
        self.assertIn("https://example.com/source/1", formatted)
    
    async def test_high_quality_web_search_tool(self):
        """Test the high_quality_web_search tool function."""
        # Set up the mock to return formatted results
        formatted_results = "Formatted web search results"
        self.api.format_high_quality_web_results = AsyncMock(return_value=formatted_results)
        
        # Create a patch for get_quality_api
        with patch('brave_search_quality_api.get_quality_api', return_value=self.api):
            # Call the tool function
            result = await high_quality_web_search(self.context, "test query", 2)
            
            # Check that the result is the formatted result
            self.assertIn("GROUNDING INSTRUCTIONS FOR LLM", result)
            self.assertIn(formatted_results, result)
    
    async def test_high_quality_ai_search_tool(self):
        """Test the high_quality_ai_search tool function."""
        # Set up the mock to return formatted results
        formatted_results = "Formatted AI search results"
        self.api.format_high_quality_ai_results = AsyncMock(return_value=formatted_results)
        
        # Create a patch for get_quality_api
        with patch('brave_search_quality_api.get_quality_api', return_value=self.api):
            # Call the tool function
            result = await high_quality_ai_search(self.context, "test query")
            
            # Check that the result is the formatted result
            self.assertIn("GROUNDING INSTRUCTIONS FOR LLM", result)
            self.assertIn(formatted_results, result)
    
    async def test_improve_search_quality_tool(self):
        """Test the improve_search_quality tool function."""
        # Set up the mock to return results
        improved_results = self.sample_web_results.copy()
        improved_results["_metadata"]["quality_score"] = 0.9
        
        # Set up the mock to return improved results
        self.api.improve_search_quality = AsyncMock(return_value=improved_results)
        
        # Set up the mock to return formatted results
        formatted_results = "Formatted improved search results"
        self.api.format_high_quality_web_results = AsyncMock(return_value=formatted_results)
        
        # Create a patch for get_quality_api
        with patch('brave_search_quality_api.get_quality_api', return_value=self.api):
            # Call the tool function
            result = await improve_search_quality(self.context, "test query", "web")
            
            # Check that the result is the formatted result
            self.assertEqual(result, formatted_results)
            
            # Check that the improve_search_quality method was called
            self.api.improve_search_quality.assert_called_once_with(
                self.context, "test query", "web"
            )
    
    async def test_get_stats(self):
        """Test getting statistics about the quality API."""
        # Call the method
        stats = await self.api.get_stats()
        
        # Check that the stats contain expected elements
        self.assertIn("config", stats)
        self.assertIn("persistent_cache", stats)
        self.assertIn("timestamp", stats)
        
        # Check that the config is the test config
        self.assertEqual(stats["config"], self.test_config)
        
        # Check that the persistent cache stats were retrieved
        self.mock_persistent_cache.get_stats.assert_called_once()

def run_tests():
    """Run the tests."""
    unittest.main()

if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBraveSearchQualityAPI)
    
    # Create a test runner
    runner = unittest.TextTestRunner()
    
    # Run the tests
    async def run_all_tests():
        for test in suite:
            if test._testMethodName.startswith("test_"):
                # Get the test method
                test_method = getattr(test, test._testMethodName)
                
                # Check if it's an async test
                if asyncio.iscoroutinefunction(test_method):
                    # Set up async environment
                    await test.asyncSetUp()
                    try:
                        await test_method()
                        print(f"✅ {test._testMethodName} passed")
                    except Exception as e:
                        print(f"❌ {test._testMethodName} failed: {e}")
                        raise
                    finally:
                        await test.asyncTearDown()
                else:
                    # Run the sync test
                    test.setUp()
                    try:
                        test_method()
                        print(f"✅ {test._testMethodName} passed")
                    except Exception as e:
                        print(f"❌ {test._testMethodName} failed: {e}")
                        raise
                    finally:
                        test.tearDown()
    
    # Run the async tests using the new asyncio API
    asyncio.run(run_all_tests())
    
    print("All tests completed successfully!")
