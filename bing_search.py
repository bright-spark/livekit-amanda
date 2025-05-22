import asyncio
import logging
import requests
import time
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from livekit.agents import function_tool, RunContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache to store search results
_search_cache = {}
_last_request_time = 0
_rate_limit = 0.5  # 500ms between requests

def scrape_bing(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Scrape Bing search results using requests (fast method).
    
    Args:
        query: The search query
        num_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing search results with title, link, and snippet
    """
    global _last_request_time
    
    # Check cache first
    cache_key = f"{query}:{num_results}"
    if cache_key in _search_cache:
        logger.info(f"Cache hit for query: {query}")
        return _search_cache[cache_key]
    
    # Rate limiting
    current_time = time.time()
    time_since_last_request = current_time - _last_request_time
    if time_since_last_request < _rate_limit:
        time.sleep(_rate_limit - time_since_last_request)
    
    # Bing search URL
    url = "https://www.bing.com/search"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.bing.com/'
    }
    
    # Parameters for the search query
    params = {
        'q': query,  # the search query
        'count': str(min(num_results * 2, 50)),  # request more than needed in case some results are filtered
        'form': 'QBLH',
        'sp': '-1',
        'pq': query,
        'sc': '8-5',
        'qs': 'n',
        'sk': '',
        'cvid': time.time()
    }
    
    results = []
    try:
        # Make the GET request to Bing search
        response = requests.get(url, headers=headers, params=params, timeout=5)
        _last_request_time = time.time()
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all search results
            result_elements = soup.find_all('li', {'class': 'b_algo'}) or soup.find_all('div', {'class': 'b_algo'})
            
            # Extract information from each result
            for result in result_elements[:num_results]:
                try:
                    title_elem = result.find('h2') or result.find('h3')
                    link_elem = result.find('a', href=True)
                    snippet_elem = result.find('p')
                    
                    if title_elem and link_elem:
                        title = title_elem.text.strip()
                        link = link_elem['href']
                        snippet = snippet_elem.text.strip() if snippet_elem else ""
                        
                        results.append({
                            "title": title,
                            "link": link,
                            "snippet": snippet
                        })
                except (AttributeError, KeyError) as e:
                    logger.warning(f"Error parsing result: {e}")
                    continue
        else:
            logger.error(f"Failed to retrieve search results: {response.status_code}")
    except Exception as e:
        logger.error(f"Error in scrape_bing: {e}")
    
    # Cache the results
    _search_cache[cache_key] = results
    
    return results

async def async_scrape_bing(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Asynchronous wrapper for scrape_bing.
    """
    # Run the synchronous function in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: scrape_bing(query, num_results))

def format_bing_results(results: List[Dict[str, str]], query: str) -> str:
    """
    Format Bing search results into a readable string.
    
    Args:
        results: List of search result dictionaries
        query: The original search query
        
    Returns:
        Formatted string with search results
    """
    if not results:
        return f"No results found for '{query}' on Bing."
    
    formatted = f"Bing search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        link = result.get('link', 'No link')
        snippet = result.get('snippet', '')
        
        formatted += f"{i}. {title}\n   {link}\n"
        if snippet:
            formatted += f"   {snippet}\n"
        formatted += "\n"
    
    return formatted

@function_tool
async def bing_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """
    Search Bing and return formatted results (optimized for speed).
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        Formatted search results as a string
    """
    try:
        logger.info(f"[TOOL] bing_search called with query: {query}, num_results: {num_results}")
        
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)
            
        # Ensure num_results is an integer
        if not isinstance(num_results, int):
            try:
                num_results = int(num_results)
            except (ValueError, TypeError):
                num_results = 5
        
        # Limit number of results to a reasonable range
        num_results = max(1, min(num_results, 20))
        
        results = await async_scrape_bing(query, num_results)
        formatted_results = format_bing_results(results, query)
        
        # Handle session output for voice responses if available
        session = getattr(context, 'session', None)
        if session and hasattr(session, 'add_message'):
            await session.add_message(role="assistant", content=formatted_results)
            return "I've found some results on Bing and will read them to you now."
        
        return formatted_results
    except Exception as e:
        logger.error(f"[TOOL] bing_search error: {str(e)}")
        return f"An error occurred while searching Bing: {str(e)}"

# Example usage
if __name__ == "__main__":
    results = scrape_bing("python programming tutorial", 5)
    print(format_bing_results(results, "python programming tutorial"))