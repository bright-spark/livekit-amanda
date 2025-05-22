"""
Test script for the optimized Brave Search API implementations with caching and rate limiting.
This script demonstrates how the caching mechanism reduces API calls for repeated searches.
"""

import asyncio
import os
import logging
import time
from dotenv import load_dotenv
from brave_search_integration import (
    web_search, 
    indeed_job_search, 
    basic_search_locanto, 
    get_cache_stats, 
    clear_cache
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Create a simple run context for testing
class TestRunContext:
    def __init__(self):
        self.session = None
        self.userdata = {}

async def test_web_search_caching():
    """Test the web_search function with caching"""
    context = TestRunContext()
    query = "climate change solutions"
    num_results = 3
    
    print(f"\n--- Testing Web Search with Caching: '{query}' ---")
    
    # First search (should be a cache miss)
    start_time = time.time()
    result1 = await web_search(context, query, num_results)
    first_search_time = time.time() - start_time
    print(f"First search took {first_search_time:.2f} seconds")
    
    # Get cache stats after first search
    stats1 = get_cache_stats()
    print(f"Cache stats after first search: {stats1}")
    
    # Second search with same query (should be a cache hit)
    start_time = time.time()
    result2 = await web_search(context, query, num_results)
    second_search_time = time.time() - start_time
    print(f"Second search took {second_search_time:.2f} seconds")
    
    # Get cache stats after second search
    stats2 = get_cache_stats()
    print(f"Cache stats after second search: {stats2}")
    
    # Print the speedup
    if first_search_time > 0:
        speedup = first_search_time / second_search_time if second_search_time > 0 else float('inf')
        print(f"Speedup from caching: {speedup:.2f}x")
    
    # Print a sample of the results
    print("\nSample of results:")
    lines = result2.split('\n')
    for line in lines[:10]:
        print(line)
    print("...")

async def test_indeed_job_search_caching():
    """Test the indeed_job_search function with caching"""
    query = "software developer"
    location = "Cape Town"
    max_results = 3
    
    print(f"\n--- Testing Indeed Job Search with Caching: '{query}' in '{location}' ---")
    
    # First search (should be a cache miss)
    start_time = time.time()
    result1 = await indeed_job_search(query, location, max_results)
    first_search_time = time.time() - start_time
    print(f"First search took {first_search_time:.2f} seconds")
    
    # Get cache stats after first search
    stats1 = get_cache_stats()
    print(f"Cache stats after first search: {stats1}")
    
    # Second search with same query (should be a cache hit)
    start_time = time.time()
    result2 = await indeed_job_search(query, location, max_results)
    second_search_time = time.time() - start_time
    print(f"Second search took {second_search_time:.2f} seconds")
    
    # Get cache stats after second search
    stats2 = get_cache_stats()
    print(f"Cache stats after second search: {stats2}")
    
    # Print the speedup
    if first_search_time > 0:
        speedup = first_search_time / second_search_time if second_search_time > 0 else float('inf')
        print(f"Speedup from caching: {speedup:.2f}x")
    
    # Print a sample of the results
    print("\nSample of results:")
    lines = result2.split('\n')
    for line in lines[:10]:
        print(line)
    print("...")

async def test_locanto_search_caching():
    """Test the basic_search_locanto function with caching"""
    context = TestRunContext()
    query = "dating"
    location = "Cape Town"
    category = "personals"
    
    print(f"\n--- Testing Locanto Search with Caching: '{query}' in '{location}', category: '{category}' ---")
    
    # First search (should be a cache miss)
    start_time = time.time()
    result1 = await basic_search_locanto(context, query, location, category)
    first_search_time = time.time() - start_time
    print(f"First search took {first_search_time:.2f} seconds")
    
    # Get cache stats after first search
    stats1 = get_cache_stats()
    print(f"Cache stats after first search: {stats1}")
    
    # Second search with same query (should be a cache hit)
    start_time = time.time()
    result2 = await basic_search_locanto(context, query, location, category)
    second_search_time = time.time() - start_time
    print(f"Second search took {second_search_time:.2f} seconds")
    
    # Get cache stats after second search
    stats2 = get_cache_stats()
    print(f"Cache stats after second search: {stats2}")
    
    # Print the speedup
    if first_search_time > 0:
        speedup = first_search_time / second_search_time if second_search_time > 0 else float('inf')
        print(f"Speedup from caching: {speedup:.2f}x")
    
    # Print a sample of the results
    print("\nSample of results:")
    lines = result2.split('\n')
    for line in lines[:10]:
        print(line)
    print("...")

async def test_cache_clearing():
    """Test clearing the cache"""
    print("\n--- Testing Cache Clearing ---")
    
    # Get cache stats before clearing
    stats_before = get_cache_stats()
    print(f"Cache stats before clearing: {stats_before}")
    
    # Clear the cache
    clear_cache()
    
    # Get cache stats after clearing
    stats_after = get_cache_stats()
    print(f"Cache stats after clearing: {stats_after}")

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
    await test_web_search_caching()
    await test_indeed_job_search_caching()
    await test_locanto_search_caching()
    await test_cache_clearing()

if __name__ == "__main__":
    asyncio.run(main())
