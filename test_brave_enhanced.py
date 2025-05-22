"""
Test script for the enhanced Brave Search API implementation.
This script demonstrates the performance improvements and reduced API usage
with the enhanced caching and rate limiting mechanisms.
"""

import asyncio
import os
import logging
import time
from dotenv import load_dotenv
from brave_search_tools_enhanced import web_search, batch_web_search, get_cache_stats, clear_session_cache
from brave_search_optimized_enhanced import get_enhanced_brave_search_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Create a simple run context for testing
class TestRunContext:
    def __init__(self):
        self.session = None
        self.userdata = {}

async def test_web_search_performance():
    """Test the enhanced web_search function performance"""
    context = TestRunContext()
    
    print("\n=== Testing Enhanced Web Search Performance ===")
    
    # Test queries
    queries = [
        "climate change solutions",
        "machine learning applications",
        "renewable energy technology",
        "artificial intelligence ethics",
        "quantum computing basics"
    ]
    
    # First run - should be cache misses
    print("\n--- First Run (Cache Misses) ---")
    first_run_times = []
    
    for query in queries:
        start_time = time.time()
        result = await web_search(context, query, num_results=3)
        elapsed = time.time() - start_time
        first_run_times.append(elapsed)
        
        print(f"Query: '{query}'")
        print(f"Time: {elapsed:.4f} seconds")
        print(f"Result length: {len(result)} characters")
        print("---")
    
    # Second run - should be cache hits
    print("\n--- Second Run (Cache Hits) ---")
    second_run_times = []
    
    for query in queries:
        start_time = time.time()
        result = await web_search(context, query, num_results=3)
        elapsed = time.time() - start_time
        second_run_times.append(elapsed)
        
        print(f"Query: '{query}'")
        print(f"Time: {elapsed:.4f} seconds")
        print(f"Result length: {len(result)} characters")
        print("---")
    
    # Calculate and print speedup
    avg_first_run = sum(first_run_times) / len(first_run_times)
    avg_second_run = sum(second_run_times) / len(second_run_times)
    speedup = avg_first_run / avg_second_run if avg_second_run > 0 else float('inf')
    
    print(f"\nAverage first run time: {avg_first_run:.4f} seconds")
    print(f"Average second run time: {avg_second_run:.4f} seconds")
    print(f"Average speedup: {speedup:.2f}x")
    
    # Get cache statistics
    stats = await get_cache_stats()
    print(f"\nCache statistics: {stats}")

async def test_batch_search():
    """Test the batch_web_search function"""
    context = TestRunContext()
    
    print("\n=== Testing Batch Web Search ===")
    
    # Test queries
    queries = [
        "python programming best practices",
        "javascript frameworks comparison",
        "data science tools",
        "web development trends",
        "mobile app development"
    ]
    
    # Clear session cache
    await clear_session_cache()
    
    # First run - should be cache misses
    print("\n--- Batch Search First Run ---")
    start_time = time.time()
    results = await batch_web_search(context, queries, num_results=3)
    elapsed = time.time() - start_time
    
    print(f"Total time for 5 queries: {elapsed:.4f} seconds")
    print(f"Average time per query: {elapsed/len(queries):.4f} seconds")
    
    # Print a sample of results
    for query, result in list(results.items())[:2]:
        print(f"\nQuery: '{query}'")
        print(f"Result preview: {result[:200]}...")
    
    # Second run - should be cache hits
    print("\n--- Batch Search Second Run ---")
    start_time = time.time()
    results = await batch_web_search(context, queries, num_results=3)
    elapsed = time.time() - start_time
    
    print(f"Total time for 5 queries: {elapsed:.4f} seconds")
    print(f"Average time per query: {elapsed/len(queries):.4f} seconds")
    
    # Get cache statistics
    stats = await get_cache_stats()
    print(f"\nCache statistics after batch search: {stats}")

async def test_query_preprocessing():
    """Test query preprocessing for improved cache hits"""
    context = TestRunContext()
    
    print("\n=== Testing Query Preprocessing ===")
    
    # Test query pairs (original and variation)
    query_pairs = [
        ("climate change solutions", "CLIMATE CHANGE SOLUTIONS"),
        ("machine learning", "machine  learning"),
        ("artificial intelligence", "the artificial intelligence"),
        ("renewable energy", "renewable energy technology"),
        ("python programming", "programming in python")
    ]
    
    # Clear session cache
    await clear_session_cache()
    client = await get_enhanced_brave_search_client()
    await client.clear_cache()
    
    print("\n--- Testing Query Variations ---")
    
    for original, variation in query_pairs:
        # First query
        start_time = time.time()
        result1 = await web_search(context, original, num_results=2)
        time1 = time.time() - start_time
        
        # Second query (variation)
        start_time = time.time()
        result2 = await web_search(context, variation, num_results=2)
        time2 = time.time() - start_time
        
        print(f"Original query: '{original}' - {time1:.4f}s")
        print(f"Variation: '{variation}' - {time2:.4f}s")
        print(f"Cache hit: {time2 < 0.1}")
        print("---")
    
    # Get cache statistics
    stats = await get_cache_stats()
    print(f"\nCache statistics after query preprocessing test: {stats}")

async def test_disk_cache_persistence():
    """Test disk cache persistence across client restarts"""
    context = TestRunContext()
    
    print("\n=== Testing Disk Cache Persistence ===")
    
    # Test query
    query = "persistent disk cache test query"
    
    # First run with a new client
    print("\n--- First Run (New Client) ---")
    client1 = await get_enhanced_brave_search_client()
    await client1.clear_cache()  # Clear cache to start fresh
    
    start_time = time.time()
    result1 = await web_search(context, query, num_results=3)
    time1 = time.time() - start_time
    
    print(f"Query: '{query}'")
    print(f"Time: {time1:.4f} seconds")
    print(f"Result length: {len(result1)} characters")
    
    # Get cache statistics
    stats1 = await get_cache_stats()
    print(f"Cache statistics: {stats1}")
    
    # Close the client to simulate application restart
    await client1.close()
    
    # Simulate application restart by clearing the global client
    import brave_search_optimized_enhanced
    brave_search_optimized_enhanced._enhanced_brave_search_client = None
    
    # Second run with a new client instance
    print("\n--- Second Run (After Client Restart) ---")
    
    start_time = time.time()
    result2 = await web_search(context, query, num_results=3)
    time2 = time.time() - start_time
    
    print(f"Query: '{query}'")
    print(f"Time: {time2:.4f} seconds")
    print(f"Result length: {len(result2)} characters")
    
    # Get cache statistics
    stats2 = await get_cache_stats()
    print(f"Cache statistics: {stats2}")
    
    # Check if disk cache was used
    print(f"\nDisk cache hit: {time2 < time1 * 0.5}")
    print(f"Speedup from disk cache: {time1 / time2 if time2 > 0 else float('inf'):.2f}x")

async def main():
    """Run all tests"""
    # Check if API key is set
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        print("WARNING: BRAVE_API_KEY is not set in your environment variables.")
        print("Please set it in your .env file and try again.")
        return
    
    print(f"Using Brave API key: {api_key[:5]}...{api_key[-4:]}")
    
    # Run tests
    await test_web_search_performance()
    await test_batch_search()
    await test_query_preprocessing()
    await test_disk_cache_persistence()
    
    # Close client
    client = await get_enhanced_brave_search_client()
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
