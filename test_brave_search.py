"""
Test script for the Brave Search API integration.
This script demonstrates how to use the Brave Search API for Locanto searches.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv
from brave_search_locanto import get_brave_search_client, brave_search_locanto, brave_search_locanto_by_category
from livekit.agents import RunContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Create a simple run context for testing
class TestRunContext:
    def __init__(self):
        self.session = None
        self.userdata = {}

async def test_basic_search():
    """Test basic search functionality"""
    context = TestRunContext()
    query = "dating"
    location = "Cape Town"
    category = "personals"
    
    print(f"\n--- Testing Basic Search: '{query}' in {location}, category: {category} ---")
    result = await brave_search_locanto(context, query, location, category)
    print(result)

async def test_category_search():
    """Test category-based search functionality"""
    context = TestRunContext()
    category_path = "personals/men-seeking-men"
    location = "western-cape"
    max_pages = 2
    
    print(f"\n--- Testing Category Search: {category_path} in {location} ---")
    result = await brave_search_locanto_by_category(context, category_path, location, max_pages)
    print(result)

async def test_direct_api_usage():
    """Test direct usage of the Brave Search client"""
    print("\n--- Testing Direct API Usage ---")
    client = get_brave_search_client()
    
    # Test a simple search
    query = "locanto dating site:locanto.co.za"
    results = await client.search(query=query, count=5)
    
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
    await test_basic_search()
    await test_category_search()
    await test_direct_api_usage()

if __name__ == "__main__":
    asyncio.run(main())
