"""
Test script for the Brave Search API integration with Indeed job searches.
This script demonstrates how to use the Brave Search API for Indeed job searches.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv
from brave_search_indeed import get_brave_indeed_client, indeed_job_search, search_indeed_jobs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

async def test_indeed_job_search():
    """Test the indeed_job_search function"""
    query = "software developer"
    location = "Cape Town"
    max_results = 5
    
    print(f"\n--- Testing Indeed Job Search: '{query}' in {location} ---")
    result = await indeed_job_search(query, location, max_results)
    print(result)

async def test_search_indeed_jobs():
    """Test the search_indeed_jobs function"""
    query = "data analyst"
    location = "Johannesburg"
    max_pages = 1
    
    print(f"\n--- Testing search_indeed_jobs: '{query}' in {location} ---")
    jobs = await search_indeed_jobs(query, location, max_pages)
    
    print(f"Found {len(jobs)} jobs:")
    for idx, job in enumerate(jobs[:5], 1):  # Show first 5 jobs
        print(f"{idx}. {job.get('title', 'No title')} at {job.get('company', 'Unknown company')}")
        if job.get('location'):
            print(f"   Location: {job['location']}")
        if job.get('salary'):
            print(f"   Salary: {job['salary']}")
        if job.get('summary'):
            summary = job['summary']
            if len(summary) > 100:
                summary = summary[:97] + "..."
            print(f"   Summary: {summary}")
        if job.get('url', {}).get('direct'):
            print(f"   URL: {job['url']['direct']}")
        print()

async def test_direct_api_usage():
    """Test direct usage of the Brave Search Indeed client"""
    print("\n--- Testing Direct API Usage ---")
    client = get_brave_indeed_client()
    
    # Test a simple search
    query = "marketing manager"
    location = "Pretoria"
    results = await client.search(query=query, location=location, count=5)
    
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
    await test_indeed_job_search()
    await test_search_indeed_jobs()
    await test_direct_api_usage()

if __name__ == "__main__":
    asyncio.run(main())
