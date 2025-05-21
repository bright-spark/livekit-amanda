import asyncio
import logging
import urllib.parse
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup
from livekit.agents import function_tool, RunContext

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
    from utils import sanitize_for_azure, handle_tool_results
except ImportError:
    try:
        from .utils import sanitize_for_azure, handle_tool_results
    except ImportError:
        logging.warning("utils module not available, using fallback definitions")
        # Fallback definitions
        def sanitize_for_azure(text):
            return text
            
        async def handle_tool_results(session, text):
            pass

@function_tool
async def indeed_job_search(
    context: RunContext,
    query: str = "customer service",
    location: str = "Johannesburg, Gauteng",
    max_results: int = 5,
    include_urls: bool = True
) -> str:
    """Search for jobs on Indeed using Playwright-powered scraping.
    
    Args:
        context: The run context for the tool
        query: The job search query (job title, keywords, company)
        location: The location to search for jobs
        max_results: Maximum number of results to return (default: 5)
        include_urls: Whether to include URLs in the results (default: True)
        
    Returns:
        str: Formatted job search results or error message
    """
    logging.info(f"[TOOL] indeed_job_search called with query: {query}, location: {location}")
    
    try:
        # Construct the search URL
        base_url = "https://za.indeed.com/jobs"
        params = {
            "q": query,
            "l": location,
        }
        original_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        # Use the untaint.us proxy to avoid Cloudflare restrictions
        proxy_base = "https://please.untaint.us/?url="
        search_url = f"{proxy_base}{urllib.parse.quote_plus(original_url)}"
        
        logging.info(f"[TOOL] indeed_job_search original URL: {original_url}")
        logging.info(f"[TOOL] indeed_job_search proxied URL: {search_url}")
        
        # Use crawl_page to fetch the search results page
        try:
            listings = await crawl_page(search_url, wait_selector=".jobsearch-ResultsList", timeout=45000, extract_text=False)
        except Exception as crawl_error:
            logging.error(f"[TOOL] indeed_job_search crawl_page error: {crawl_error}")
            error_msg = f"I couldn't access Indeed to search for jobs: {str(crawl_error)}"
            error_msg = sanitize_for_azure(error_msg)
            
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't search for jobs on Indeed right now."
            return error_msg
        
        # Parse job listings from the HTML
        soup = BeautifulSoup(listings, "html.parser")
        jobs = []
        
        # Try different selectors for job cards (Indeed changes their HTML structure frequently)
        job_cards = soup.find_all('div', class_='job_seen_beacon') or \
                   soup.find_all('div', class_='jobCard') or \
                   soup.find_all('div', {'data-testid': 'jobCard'}) or \
                   soup.find_all('div', class_='tapItem')
        
        for div in job_cards:
            try:
                # Try different selectors for job elements
                title_elem = div.find('h2', class_='jobTitle') or div.find('h2') or div.find('a', {'data-testid': 'jobTitle'})
                company_elem = div.find('span', class_='companyName') or div.find('div', class_='company')
                location_elem = div.find('div', class_='companyLocation') or div.find('div', class_='location')
                summary_elem = div.find('div', class_='job-snippet') or div.find('div', class_='summary')
                link_elem = div.find('a', href=True) or title_elem if hasattr(title_elem, 'href') else None
                salary_elem = div.find('div', class_='salary-snippet') or div.find('span', class_='salaryText')
                
                # Extract text from elements
                title = title_elem.get_text(strip=True) if title_elem else None
                company = company_elem.get_text(strip=True) if company_elem else None
                location_val = location_elem.get_text(strip=True) if location_elem else None
                summary = summary_elem.get_text(strip=True) if summary_elem else None
                salary = salary_elem.get_text(strip=True) if salary_elem else None
                
                # Get URL
                url = None
                if link_elem and 'href' in link_elem.attrs:
                    href = link_elem['href']
                    if href.startswith('/'):
                        job_url = f"https://za.indeed.com{href}"
                    elif href.startswith('http'):
                        job_url = href
                    else:
                        job_url = None
                        
                    # If we have a valid job URL, create both direct and proxied versions
                    if job_url:
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
                        "summary": summary,
                        "salary": salary,
                        "url": url
                    })
            except Exception as parse_error:
                logging.warning(f"[TOOL] indeed_job_search parse error for job card: {parse_error}")
                continue
        
        if not jobs:
            no_jobs_msg = f"No jobs found for '{query}' in '{location}'. Try different search terms or location."
            no_jobs_msg = sanitize_for_azure(no_jobs_msg)
            
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, no_jobs_msg)
                return "I couldn't find any jobs matching your search criteria."
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
                # Use the proxied URL by default, but also include the direct URL
                result += f"   URL (Proxied): {job['url']['proxied']}\n"
                result += f"   URL (Direct): {job['url']['direct']}\n"
                
            result += "\n"
        
        # Add a note about the total number of results
        if len(jobs) > max_results:
            result += f"\n{len(jobs) - max_results} more jobs found. Refine your search for more specific results."
        
        result = sanitize_for_azure(result)
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, result)
            return f"I found {len(jobs)} jobs matching your search. I'll read some of them to you."
        
        return result
        
    except Exception as e:
        logging.error(f"[TOOL] indeed_job_search exception: {e}", exc_info=True)
        error_msg = f"Sorry, I couldn't search for jobs right now: {str(e)}"
        error_msg = sanitize_for_azure(error_msg)
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I encountered an error while searching for jobs."
        
        return error_msg