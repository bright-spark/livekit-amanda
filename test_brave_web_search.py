"""
Test script for the Brave Search API integration for web search.
This script demonstrates how to use the Brave Search API for general web searches.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv
from brave_search_tools import get_brave_search_client, web_search, fallback_web_search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Create a simple run context for testing
class TestRunContext:
    def __init__(self):
        self.session = None
        self.userdata = {}

async def test_web_search():
    """Test the web_search function"""
    context = TestRunContext()
    query = "renewable energy developments"
    num_results = 5
    
    print(f"\n--- Testing Web Search: '{query}' ---")
    result = await web_search(context, query, num_results)
    print(result)

async def test_fallback_web_search():
    """Test the fallback_web_search function"""
    context = TestRunContext()
    query = "latest AI research papers"
    num_results = 3
    
    print(f"\n--- Testing Fallback Web Search: '{query}' ---")
    result = await fallback_web_search(context, query, num_results)
    print(result)

async def test_direct_api_usage():
    """Test direct usage of the Brave Search client"""
    print("\n--- Testing Direct API Usage ---")
    client = get_brave_search_client()
    
    # Test a simple search
    query = "climate change solutions"
    results = await client.search(query=query, count=3)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    elif "web" in results and "results" in results["web"]:
        print(f"Found {len(results['web']['results'])} results:")
        for idx, result in enumerate(results["web"]["results"], 1):
            print(f"{idx}. {result.get('title', 'No title')}")
            print(f"   URL: {result.get('url', '')}")
            print(f"   Description: {result.get('description', 'No description')[:100]}...")
            print()
    else:
        print("No results found")

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
    await test_web_search()
    await test_fallback_web_search()
    await test_direct_api_usage()

if __name__ == "__main__":
    asyncio.run(main())
