import asyncio
import logging
import urllib.parse
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup

# Handle imports with try/except for flexibility
try:
    from puppeteer_crawler import crawl_page
except ImportError:
    try:
        from .puppeteer_crawler import crawl_page
    except ImportError:
        logging.warning("puppeteer_crawler module not available")
        # Define a fallback function
        async def crawl_page(*args, **kwargs):
            return "Puppeteer crawler functionality is not available."

# Handle utils import with try/except
try:
    from agent_utils import sanitize_for_azure
except ImportError:
    try:
        from .agent_utils import sanitize_for_azure
    except ImportError:
        # Define a fallback function if not available
        def sanitize_for_azure(text):
            return text

# Define a fallback handle_tool_results function
try:
    from agent_utils import handle_tool_results
except ImportError:
    # Define a fallback function
    async def handle_tool_results(session, text):
        pass

async def indeed_job_search(
    query: str = "customer service",
    location: str = "Johannesburg, Gauteng",
    max_results: int = 5,
    include_urls: bool = True
) -> str:
    """Search for jobs on Indeed using Playwright-powered scraping.
    
    Args:
        query: The job search query (job title, keywords, company)
        location: The location to search for jobs
        max_results: Maximum number of results to return (default: 5)
        include_urls: Whether to include URLs in the results (default: True)
        
    Returns:
        A formatted string with job search results
    """
    logging.info(f"[TOOL] indeed_job_search called with query: {query}, location: {location}")
    
    try:
        # Build the Indeed search URL
        base_url = "https://www.indeed.com"
        search_url = f"{base_url}/jobs?q={urllib.parse.quote_plus(query)}&l={urllib.parse.quote_plus(location)}"
        
        # Use puppeteer to fetch the page (handles JavaScript rendering and bypasses some anti-scraping)
        listings = await crawl_page(search_url)
        
        if not listings or isinstance(listings, str) and "error" in listings.lower():
            error_msg = f"Error fetching Indeed listings: {listings}"
            logging.error(error_msg)
            error_msg = sanitize_for_azure(error_msg)
            return error_msg
        
        # Parse job listings from the HTML
        soup = BeautifulSoup(listings, "html.parser")
        jobs = []
        
        # Find job cards
        job_cards = soup.select("div.job_seen_beacon")
        
        if not job_cards:
            # Try alternative selectors
            job_cards = soup.select("div.jobsearch-ResultsList div[data-jk]")
            
        if not job_cards:
            # Try another alternative
            job_cards = soup.select("div.tapItem")
            
        if job_cards:
            for card in job_cards:
                # Extract job details
                title_elem = card.select_one("h2.jobTitle span") or card.select_one("h2.jobTitle a") or card.select_one("h2.jobTitle")
                company_elem = card.select_one("span.companyName") or card.select_one("div.company_location span.companyName")
                location_elem = card.select_one("div.companyLocation") or card.select_one("div.company_location div.companyLocation")
                salary_elem = card.select_one("div.salary-snippet-container") or card.select_one("div.metadata.salary-snippet-container")
                summary_elem = card.select_one("div.job-snippet") or card.select_one("div.jobCardShelfContainer div.result-footer-content.job-snippet")
                
                # Extract text content
                title = title_elem.get_text(strip=True) if title_elem else ""
                company = company_elem.get_text(strip=True) if company_elem else ""
                location_val = location_elem.get_text(strip=True) if location_elem else ""
                salary = salary_elem.get_text(strip=True) if salary_elem else ""
                summary = summary_elem.get_text(strip=True) if summary_elem else ""
                
                # Extract URL
                job_url = None
                if include_urls:
                    # Try to find the job URL
                    url_elem = card.select_one("h2.jobTitle a") or card.select_one("a.jcs-JobTitle")
                    if url_elem and url_elem.has_attr('href'):
                        job_url = url_elem['href']
                        if not job_url.startswith('http'):
                            job_url = f"{base_url}{job_url}"
                        
                        # Store both the direct URL and the proxied URL
                        url = {
                            "direct": job_url,
                            "proxied": f"https://please.untaint.us/?url={urllib.parse.quote_plus(job_url)}"
                        }
                
                if title and company:
                    jobs.append({
                        "title": title,
                        "company": company,
                        "location": location_val,
                        "salary": salary,
                        "summary": summary,
                        "url": url if include_urls and job_url else None
                    })
        
        if not jobs:
            logging.warning(f"No jobs found for query: {query}, location: {location}")
            no_jobs_msg = f"I couldn't find any jobs matching '{query}' in '{location}'. Try a different search query or location."
            no_jobs_msg = sanitize_for_azure(no_jobs_msg)
            return no_jobs_msg
        
        # Format results for output
        result = f"Here are some jobs for '{query}' in '{location}':\n\n"
        
        # Limit the number of results
        max_results = min(max_results, len(jobs))
        for i, job in enumerate(jobs[:max_results], 1):
            result += f"{i}. {job['title']} at {job['company']}\n"
            
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
                
            if include_urls and job['url']:
                result += f"   URL: {job['url']['direct']}\n"
                
            result += "\n"
            
        return result
    except Exception as e:
        error_msg = f"Error searching Indeed jobs: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Stack trace: {asyncio.format_exception(type(e), e, e.__traceback__)}")
        return error_msg

async def search_indeed_jobs(
    query: Optional[str] = None,
    location: Optional[str] = None,
    max_pages: int = 1,
    use_proxy: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for job listings on Indeed.
    
    Args:
        query: Search query string (job title, keywords, etc.)
        location: Location to search in
        max_pages: Maximum number of pages to scrape
        use_proxy: Whether to use proxy for scraping
        
    Returns:
        List of job listing dictionaries
    """
    try:
        # Call the indeed_job_search function
        result = await indeed_job_search(
            query=query or "software developer",
            location=location or "New York",
            max_results=max_pages * 10,  # Approximate number of results per page
            include_urls=True
        )
        
        # Parse the result string into a list of dictionaries
        listings = []
        current_job = {}
        
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                if current_job and 'title' in current_job:
                    listings.append(current_job)
                    current_job = {}
                continue
                
            if line[0].isdigit() and '. ' in line:
                # This is a new job title line
                if current_job and 'title' in current_job:
                    listings.append(current_job)
                
                # Start a new job entry
                parts = line.split('. ', 1)
                if len(parts) > 1:
                    title_company = parts[1].split(' at ', 1)
                    if len(title_company) > 1:
                        current_job = {
                            'title': title_company[0].strip(),
                            'company': title_company[1].strip()
                        }
                    else:
                        current_job = {'title': parts[1].strip()}
            elif line.startswith('   Location: '):
                current_job['location'] = line.replace('   Location: ', '').strip()
            elif line.startswith('   Salary: '):
                current_job['salary'] = line.replace('   Salary: ', '').strip()
            elif line.startswith('   Summary: '):
                current_job['description'] = line.replace('   Summary: ', '').strip()
            elif line.startswith('   URL: '):
                current_job['url'] = line.replace('   URL: ', '').strip()
        
        # Add the last job if it exists
        if current_job and 'title' in current_job:
            listings.append(current_job)
            
        return listings
        
    except Exception as e:
        logging.error(f"Error in search_indeed_jobs: {str(e)}")
        return []
