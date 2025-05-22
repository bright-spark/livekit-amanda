"""
Optimized Brave Search API integration for Indeed job searches.
This module provides functionality to search Indeed jobs using the Brave Search API with caching and rate limiting.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union
from livekit.agents import function_tool, RunContext

# Import the optimized Brave Search client
from brave_search_optimized import get_optimized_brave_search_client

# Handle utils import with try/except
try:
    from agent_utils import sanitize_for_azure, handle_tool_results
except ImportError:
    try:
        from .agent_utils import sanitize_for_azure, handle_tool_results
    except ImportError:
        # Define fallback functions if not available
        def sanitize_for_azure(text):
            return text
            
        async def handle_tool_results(session, text):
            pass

class OptimizedBraveSearchIndeedClient:
    """Optimized client for searching Indeed jobs using Brave Search API with caching and rate limiting."""
    
    def __init__(self):
        """Initialize the optimized Brave Search client for Indeed job searches."""
        self.brave_client = get_optimized_brave_search_client()
    
    async def search(self, 
                    query: str, 
                    location: str = "", 
                    count: int = 10) -> Dict[str, Any]:
        """Search for jobs on Indeed using Brave Search API with caching and rate limiting.
        
        Args:
            query: Job search query (job title, keywords, company)
            location: The location to search for jobs
            count: Number of results to return
            
        Returns:
            Dict containing the search results
        """
        # Build a more specific query for Brave Search
        search_query = f"{query} {location} jobs site:indeed.com"
        
        # Use the optimized client with caching
        return await self.brave_client.search(query=search_query, count=count)
    
    def extract_job_details(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract job details from Brave Search API results.
        
        Args:
            results: Search results from the Brave Search API
            
        Returns:
            List of job dictionaries with extracted information
        """
        jobs = []
        
        if "error" in results:
            logging.error(f"Error in Brave Search API results: {results['error']}")
            return jobs
        
        if "web" not in results or "results" not in results["web"]:
            logging.info("No job results found in Brave Search API response")
            return jobs
        
        for result in results["web"]["results"]:
            title = result.get("title", "")
            url = result.get("url", "")
            description = result.get("description", "")
            
            # Extract company name from title or description
            company = ""
            location_val = ""
            salary = ""
            
            # Try to extract company from title (e.g., "Job Title - Company")
            if " - " in title:
                parts = title.split(" - ")
                if len(parts) >= 2:
                    title = parts[0].strip()
                    company = parts[1].strip()
                    # Sometimes Indeed titles have format "Job Title - Company - Location"
                    if len(parts) >= 3:
                        location_val = parts[2].strip()
            
            # Try to extract salary from description
            salary_match = re.search(r'(\$[\d,]+(?:\s*-\s*\$[\d,]+)?(?:\s*(?:a|per)\s*(?:year|month|hour|week))?)', description)
            if salary_match:
                salary = salary_match.group(1)
            
            # Try to extract location if not already found
            if not location_val:
                location_match = re.search(r'in\s+([A-Za-z\s]+(?:,\s*[A-Za-z\s]+)?)', description)
                if location_match:
                    location_val = location_match.group(1).strip()
            
            job = {
                "title": title,
                "company": company,
                "location": location_val,
                "salary": salary,
                "summary": description,
                "url": {
                    "direct": url
                }
            }
            
            jobs.append(job)
        
        return jobs
    
    def format_job_results(self, jobs: List[Dict[str, Any]], query: str, location: str, max_results: int = 5, include_urls: bool = True) -> str:
        """Format job results into a readable string.
        
        Args:
            jobs: List of job dictionaries
            query: The original search query
            location: The original location query
            max_results: Maximum number of results to include
            include_urls: Whether to include URLs in the output
            
        Returns:
            Formatted string of job results
        """
        if not jobs:
            return f"No jobs found for '{query}' in '{location}'."
        
        result = f"Here are some jobs for '{query}' in '{location}':\n\n"
        
        # Limit the number of results
        max_results = min(max_results, len(jobs))
        for i, job in enumerate(jobs[:max_results], 1):
            result += f"{i}. {job['title']}"
            if job['company']:
                result += f" at {job['company']}"
            result += "\n"
            
            if job['location']:
                result += f"   Location: {job['location']}\n"
                
            if job['salary']:
                result += f"   Salary: {job['salary']}\n"
                
            if job['summary']:
                # Truncate long summaries
                summary = job['summary']
                if len(summary) > 150:
                    summary = summary[:147] + "..."
                result += f"   Summary: {summary}\n"
                
            if include_urls and job['url']['direct']:
                result += f"   URL: {job['url']['direct']}\n"
                
            result += "\n"
            
        return result

# Create a singleton instance
_optimized_brave_indeed_client = None

def get_optimized_brave_indeed_client() -> OptimizedBraveSearchIndeedClient:
    """Get or create a singleton instance of the OptimizedBraveSearchIndeedClient.
    
    Returns:
        OptimizedBraveSearchIndeedClient instance
    """
    global _optimized_brave_indeed_client
    if _optimized_brave_indeed_client is None:
        _optimized_brave_indeed_client = OptimizedBraveSearchIndeedClient()
    return _optimized_brave_indeed_client

# Function tool implementation that matches the original indeed_job_search signature
@function_tool
async def indeed_job_search(
    query: str = "customer service",
    location: str = "Johannesburg, Gauteng",
    max_results: int = 5,
    include_urls: bool = True
) -> str:
    """Search for jobs on Indeed using Brave Search API with caching and rate limiting.
    
    Args:
        query: The job search query (job title, keywords, company)
        location: The location to search for jobs
        max_results: Maximum number of results to return (default: 5)
        include_urls: Whether to include URLs in the results (default: True)
        
    Returns:
        A formatted string with job search results
    """
    logging.info(f"[TOOL] indeed_job_search (Optimized Brave API) called with query: {query}, location: {location}")
    
    try:
        # Get the optimized Brave Search client
        client = get_optimized_brave_indeed_client()
        
        # Search for jobs with caching
        results = await client.search(query=query, location=location, count=max_results * 2)  # Request more to ensure we get enough after filtering
        
        if "error" in results:
            error_msg = f"Error searching Indeed jobs: {results['error']}"
            logging.error(error_msg)
            return sanitize_for_azure(error_msg)
        
        # Extract job details
        jobs = client.extract_job_details(results)
        
        # Format results
        formatted_results = client.format_job_results(
            jobs=jobs,
            query=query,
            location=location,
            max_results=max_results,
            include_urls=include_urls
        )
        
        return sanitize_for_azure(formatted_results)
        
    except Exception as e:
        error_msg = f"Error searching Indeed jobs: {str(e)}"
        logging.error(error_msg)
        return sanitize_for_azure(error_msg)

# Implementation of search_indeed_jobs that matches the original signature
@function_tool
async def search_indeed_jobs(
    query: Optional[str] = None,
    location: Optional[str] = None,
    max_pages: int = 1,
    use_proxy: bool = True  # Parameter kept for compatibility, but not used
) -> List[Dict[str, Any]]:
    """
    Search for job listings on Indeed using Brave Search API with caching and rate limiting.
    
    Args:
        query: Search query string (job title, keywords, etc.)
        location: Location to search in
        max_pages: Maximum number of pages to search (translates to count in API)
        use_proxy: Whether to use proxy for scraping (kept for compatibility, not used)
        
    Returns:
        List of job listing dictionaries
    """
    try:
        # Get the optimized Brave Search client
        client = get_optimized_brave_indeed_client()
        
        # Search for jobs with caching
        results = await client.search(
            query=query or "software developer",
            location=location or "New York",
            count=max_pages * 10  # Approximate number of results per page
        )
        
        # Extract job details
        jobs = client.extract_job_details(results)
        
        return jobs
        
    except Exception as e:
        logging.error(f"Error in search_indeed_jobs: {str(e)}")
        return []
