# Standard library imports
import asyncio
import logging
import re
import urllib.parse
from typing import List, Dict, Any, Optional, Union

# Third-party imports
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Error
from livekit.agents import function_tool, RunContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 45000  # 45 seconds
PROXY_URL = "https://please.untaint.us/?url="

async def scrape_bing(query: str, num_results: int = 10, use_proxy: bool = True, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, str]]:
    """
    Scrape Bing search results using Playwright.
    
    Args:
        query: The search query
        num_results: Maximum number of results to return
        use_proxy: Whether to use the untaint.us proxy to avoid restrictions
        timeout: Timeout in milliseconds for page navigation
        
    Returns:
        List of dictionaries containing search results with title, link, and snippet
    """
    logger.info(f"Scraping Bing for query: {query}, num_results: {num_results}")
    
    # --- AI logic for Bing parameters ---
    safesearch = "Moderate"  # Default safesearch setting
    mkt = "en-US"
    setlang = "EN"
    freshness = None
    response_filter = "Webpages"  # Default to webpages
    
    # Lowercase and clean query for checks
    q_lower = query.lower().strip()
    
    # Smart parameter selection based on query content
    # Time-sensitive queries
    if any(word in q_lower for word in ["news", "today", "latest", "breaking", "update", "recent", "current"]):
        freshness = "Day"  # More recent results for news queries
        logger.info(f"Query appears time-sensitive, setting freshness to {freshness}")
    
    # Media type queries
    if any(word in q_lower for word in ["image", "photo", "picture", "gallery"]):
        response_filter = "Images"
        logger.info(f"Query appears image-related, setting filter to {response_filter}")
    elif any(word in q_lower for word in ["video", "clip", "movie", "watch"]):
        response_filter = "Videos"
        logger.info(f"Query appears video-related, setting filter to {response_filter}")
    
    # Content filtering
    if re.search(r"\b(sex|porn|adult|xxx|nude|nsfw)\b", q_lower):
        safesearch = "Off"
        logger.info("Query contains adult content terms, disabling safesearch")
    
    # Build Bing URL with parameters
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://www.bing.com/search?q={encoded_query}&count={num_results}&safesearch={safesearch}&mkt={mkt}&setlang={setlang}"
    
    # Add optional parameters if specified
    if freshness:
        url += f"&freshness={freshness}"
    
    if response_filter and response_filter != "Webpages":
        url += f"&filters={response_filter}"
    
    # Apply proxy if requested
    if use_proxy:
        original_url = url
        url = f"{PROXY_URL}{urllib.parse.quote_plus(url)}"
        logger.info(f"Using proxy. Original URL: {original_url}")
    
    logger.info(f"Final URL: {url}")
    results = []
    
    try:
        async with async_playwright() as p:
            # Launch browser with appropriate settings
            browser_args = {"headless": True}
            
            # Launch browser
            browser = await p.chromium.launch(**browser_args)
            
            try:
                # Create context with realistic browser settings
                context = await browser.new_context(
                    user_agent=USER_AGENT,
                    viewport={"width": 1280, "height": 800},
                    locale="en-US",
                    extra_http_headers={
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                        "Referer": "https://www.bing.com/"
                    }
                )
                
                # Create page and navigate
                page = await context.new_page()
                response = await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                
                if not response:
                    logger.error("Failed to get response from page navigation")
                    return []
                    
                if not response.ok:
                    logger.error(f"HTTP error: {response.status} {response.status_text}")
                    return []
                
                # Handle cookie consent and other popups
                await _handle_popups(page)
                
                # Wait for search results to load
                try:
                    # Wait for either search results or no results message
                    await page.wait_for_selector('li.b_algo, .b_no', timeout=10000)
                except Error as e:
                    logger.warning(f"Timeout waiting for results: {e}")
                
                # Get page content
                html = await page.content()
                
                # Parse results
                results = _parse_bing_results(html, num_results)
                
                if not results:
                    logger.warning("No results found or parsing failed")
                    # Take screenshot for debugging if needed
                    # await page.screenshot(path="bing_debug.png")
                
            finally:
                await browser.close()
    
    except Exception as e:
        logger.error(f"Error during Bing scraping: {str(e)}")
    
    return results

async def _handle_popups(page: Page) -> None:
    """
    Handle various popups, consent dialogs, and overlays that might appear on Bing.
    
    Args:
        page: The Playwright page object
    """
    try:
        # Common consent/cookie button selectors
        consent_selectors = [
            'button[aria-label*="accept"]',
            'button[aria-label*="Accept"]',
            'button[title*="Accept"]',
            'button[aria-label*="Agree"]',
            'button[title*="Agree"]',
            'button:has-text("Accept")',
            'button:has-text("Agree")',
            '#bnp_btn_accept',
            '#consent-banner button',
            '.bnp_btn_accept'
        ]
        
        for selector in consent_selectors:
            if await page.query_selector(selector):
                logger.info(f"Found consent popup with selector: {selector}")
                await page.click(selector)
                # Wait a moment for the popup to disappear
                await asyncio.sleep(1)
                break
    except Exception as e:
        logger.warning(f"Error handling popups: {str(e)}")

def _parse_bing_results(html: str, num_results: int) -> List[Dict[str, str]]:
    """
    Parse Bing search results from HTML.
    
    Args:
        html: The HTML content of the Bing search results page
        num_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing search results
    """
    results = []
    soup = BeautifulSoup(html, "html.parser")
    
    # Try multiple selectors for result items (Bing changes its HTML structure occasionally)
    items = soup.find_all('li', {'class': 'b_algo'}) or soup.find_all('div', {'class': 'b_algo'})
    
    if not items:
        logger.warning("No items found with primary selectors, trying alternative selectors")
        items = soup.find_all('div', {'class': 'b_title'}) or soup.find_all('div', {'class': 'b_attribution'})
    
    # Process each result item
    for item in items[:num_results]:
        try:
            # Try different selectors for title and link
            title_elem = item.find('h2') or item.find('h3') or item.find('a')
            link_elem = item.find('a', href=True) or title_elem if hasattr(title_elem, 'href') else None
            
            # Try different selectors for snippet
            snippet_elem = item.find('p') or item.find('div', {'class': 'b_caption'}) or item.find_next('p')
            
            if title_elem and link_elem and link_elem.has_attr('href'):
                title = title_elem.get_text(strip=True)
                link = link_elem['href']
                
                # Skip internal Bing links
                if link.startswith('/') or 'bing.com' in link:
                    continue
                    
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Clean up snippet text
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                
                results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet
                })
        except Exception as e:
            logger.warning(f"Error parsing result item: {str(e)}")
            continue
    
    return results

@function_tool
async def bing_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """
    Search Bing and return formatted results.
    
    Args:
        context: The run context
        query: The search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        Formatted search results as a string
    """
    try:
        logger.info(f"[TOOL] bing_search called with query: {query}, num_results: {num_results}")
        results = await scrape_bing(query, num_results=num_results)
        
        if not results:
            return f"No results found for '{query}'. Try a different search query."
        
        # Format results
        formatted_results = f"Search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   URL: {result['link']}\n"
            if result['snippet']:
                formatted_results += f"   {result['snippet']}\n"
            formatted_results += "\n"
        
        return formatted_results
    except Exception as e:
        logger.error(f"[TOOL] bing_search error: {str(e)}")
        return f"An error occurred while searching: {str(e)}"

# Example usage
if __name__ == "__main__":
    async def test():
        results = await scrape_bing("python programming tutorial", num_results=5)
        print(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['link']}")
            print(f"   {result['snippet']}")
            print()
    
    asyncio.run(test())