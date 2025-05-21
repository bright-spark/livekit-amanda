# Standard library imports
import asyncio
import logging
import re
import urllib.parse
from typing import Dict, List, Any, Optional, Union, Set
from urllib.parse import urljoin, urlparse, parse_qs

# Third-party imports
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 45000  # 45 seconds
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
PROXY_URL = "https://please.untaint.us/?url="
MAX_CONCURRENT_REQUESTS = 3

async def crawl_page(
    url: str, 
    wait_selector: Optional[str] = None, 
    timeout: int = DEFAULT_TIMEOUT, 
    user_agent: Optional[str] = None, 
    extra_headers: Optional[Dict[str, str]] = None, 
    extract_text: bool = True,
    use_proxy: bool = True,
    retry_count: int = 2
) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Fetches and optionally extracts readable content from any web page using Playwright.
    If the URL is a Locanto listing, returns structured data with listing details.
    
    Args:
        url: The URL to crawl
        wait_selector: CSS selector to wait for before extracting content
        timeout: Max timeout for page load in milliseconds
        user_agent: Custom user agent string
        extra_headers: Additional HTTP headers
        extract_text: If True, returns visible text, else returns raw HTML
        use_proxy: Whether to use the untaint.us proxy to avoid restrictions
        retry_count: Number of retry attempts if the initial request fails
        
    Returns:
        Extracted text/HTML as string or structured dict for listings
    """
    # Apply proxy if requested and not already proxied
    original_url = url
    if use_proxy and not url.startswith(PROXY_URL):
        url = f"{PROXY_URL}{urllib.parse.quote_plus(url)}"
        logger.info(f"Using proxy URL: {url}")
    
    # Set default user agent if not provided
    if not user_agent:
        user_agent = DEFAULT_USER_AGENT
    
    # Initialize retry counter
    attempts = 0
    last_error = None
    
    while attempts <= retry_count:
        try:
            logger.info(f"Crawling page (attempt {attempts+1}/{retry_count+1}): {url}")
            async with async_playwright() as p:
                # Launch browser with appropriate settings
                browser_args = {"headless": True}
                
                # Launch browser
                browser = await p.chromium.launch(**browser_args)
                
                try:
                    # Prepare context arguments
                    context_args = {
                        'user_agent': user_agent,
                        'viewport': {"width": 1280, "height": 800},
                        'locale': "en-US"
                    }
                    
                    if extra_headers:
                        context_args['extra_http_headers'] = extra_headers
                    else:
                        context_args['extra_http_headers'] = {
                            "Accept-Language": "en-US,en;q=0.9",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                        }
                    
                    # Create context and page
                    context = await browser.new_context(**context_args)
                    page = await context.new_page()
                    
                    # Navigate to URL with appropriate wait strategy
                    response = await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                    
                    if not response:
                        logger.error(f"Failed to get response from page navigation: {url}")
                        raise Exception("No response from page navigation")
                        
                    if not response.ok:
                        logger.error(f"HTTP error: {response.status} {response.status_text}")
                        raise Exception(f"HTTP error: {response.status} {response.status_text}")
                    
                    # Handle cookie consent and other popups
                    await _handle_popups(page)
                    
                    # Wait for specific selector if provided
                    if wait_selector:
                        try:
                            await page.wait_for_selector(wait_selector, timeout=timeout)
                        except Error as e:
                            logger.warning(f"Timeout waiting for selector '{wait_selector}': {e}")
                    else:
                        # Default wait to ensure page is loaded
                        await page.wait_for_timeout(2000)
                    
                    # Get page content
                    html = await page.content()
                    
                    # Process the HTML based on the URL and extraction preferences
                    result = await _process_html(html, original_url, extract_text, page)
                    return result
                    
                finally:
                    await browser.close()
                    
        except Exception as e:
            last_error = e
            logger.error(f"Error during crawling (attempt {attempts+1}): {str(e)}")
            attempts += 1
            if attempts <= retry_count:
                # Exponential backoff between retries
                wait_time = 2 ** attempts
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {attempts} attempts: {str(e)}")
                return {"error": f"Failed to crawl page: {str(e)}", "url": original_url}

async def _handle_popups(page: Page) -> None:
    """
    Handle various popups, consent dialogs, and overlays that might appear on websites.
    
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
            '.bnp_btn_accept',
            '.cookie-consent__btn',
            '.consent-btn',
            '.gdpr-consent-button'
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

async def _process_html(
    html: str, 
    url: str, 
    extract_text: bool, 
    page: Optional[Page] = None
) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process the HTML content based on the URL and extraction preferences.
    
    Args:
        html: The HTML content to process
        url: The original URL (without proxy)
        extract_text: Whether to extract text or return structured data
        page: The Playwright page object (optional)
        
    Returns:
        Processed content as string or structured data
    """
    # Check if this is a Locanto page
    if 'locanto.co.za' in url:
        return await _process_locanto_page(html, url, page)
    
    # For other websites, process based on extract_text preference
    if extract_text:
        return _extract_readable_text(html)
    else:
        return html

async def _process_locanto_page(html: str, url: str, page: Optional[Page] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process Locanto page HTML, handling both listing collections and detail pages.
    
    Args:
        html: The HTML content to process
        url: The original URL (without proxy)
        page: The Playwright page object (optional)
        
    Returns:
        Structured data for Locanto page
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Detect if this is a listing collection (search results) or detail page
    articles = soup.find_all('article', class_='posting_listing')
    
    # Case 1: Search results page with article listings
    if articles:
        return await _process_locanto_listing_collection(soup, url, articles)
    
    # Case 2: Try to find listing links in other formats
    listing_links = _find_locanto_listing_links(soup, url)
    if listing_links:
        return await _process_locanto_listing_links(listing_links)
    
    # Case 3: This is a detail page
    return _process_locanto_detail_page(soup, url)

async def _process_locanto_listing_collection(soup: BeautifulSoup, url: str, articles: List[Any]) -> List[Dict[str, Any]]:
    """
    Process a collection of Locanto listings from search results.
    
    Args:
        soup: BeautifulSoup object of the page
        url: The original URL
        articles: List of article elements containing listings
        
    Returns:
        List of structured listing data
    """
    listings = []
    
    # Extract basic info from each listing
    for art in articles:
        try:
            # Get listing URL
            a_elem = art.find('a', class_='posting_listing__title') or art.find('a', class_='js-ad_link')
            if not a_elem or not a_elem.get('href'): continue
                
            listing_url = urljoin(url, a_elem['href'])
            
            # Get title
            title_div = a_elem.find('div', class_='h3') or a_elem.find('div', class_='js-result_title')
            title = title_div.get_text(strip=True) if title_div else a_elem.get_text(strip=True)
            
            # Get location
            loc_span = art.find('span', class_='posting_listing__city') or art.find('span', class_='js-result_location')
            location = loc_span.get_text(strip=True) if loc_span else None
            
            # Get age
            age_span = art.find('span', class_='posting_listing__age')
            age = age_span.get_text(strip=True) if age_span else None
            
            # Get category
            cat_span = art.find('span', class_='posting_listing__category')
            category = cat_span.get_text(strip=True) if cat_span else None
            
            # Get description
            desc_div = art.find('div', class_='posting_listing__description') or art.find('div', class_='js-description_snippet')
            description = desc_div.get_text(strip=True) if desc_div else None
            
            # Get image
            img_elem = art.find('img', class_='posting_listing__image')
            image_url = img_elem.get('src') if img_elem and img_elem.get('src') else None
            images = [image_url] if image_url else []
            
            # Create listing object
            listing = {
                'listing_url': listing_url,
                'title': title,
                'location': location,
                'age': age,
                'category': category,
                'description': description,
                'images': images
            }
            
            listings.append(listing)
        except Exception as e:
            logger.warning(f"Error processing listing article: {str(e)}")
    
    # If we have listings, fetch more details for each
    if listings:
        return await _fetch_listing_details(listings)
    
    return listings

def _find_locanto_listing_links(soup: BeautifulSoup, url: str) -> List[str]:
    """
    Find Locanto listing links in a page that doesn't use the standard article format.
    
    Args:
        soup: BeautifulSoup object of the page
        url: The original URL
        
    Returns:
        List of listing URLs
    """
    listing_links = []
    
    # Try to find links matching Locanto listing patterns
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Match common Locanto listing URL patterns
        if re.search(r'(/ID_\d+/|/ID_\d+\.html|/\d+\.html|/\d+/)', href):
            abs_url = urljoin(url, href)
            if abs_url not in listing_links:
                listing_links.append(abs_url)
    
    return list(dict.fromkeys(listing_links))  # Remove duplicates

async def _process_locanto_listing_links(listing_links: List[str]) -> List[Dict[str, Any]]:
    """
    Process a list of Locanto listing links by fetching details for each.
    
    Args:
        listing_links: List of listing URLs to process
        
    Returns:
        List of structured listing data
    """
    results = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def fetch_listing(listing_url: str) -> Dict[str, Any]:
        async with sem:
            try:
                # Use crawl_page with extract_text=False to get structured data
                return await crawl_page(listing_url, extract_text=False, retry_count=1)
            except Exception as e:
                logger.error(f"Error fetching listing {listing_url}: {str(e)}")
                return {'listing_url': listing_url, 'error': f'Failed to extract: {str(e)}'}
    
    # Create tasks for all listings
    tasks = [fetch_listing(url) for url in listing_links]
    
    # Process results as they complete
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            if result and isinstance(result, dict):
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing listing task: {str(e)}")
    
    return results

async def _fetch_listing_details(listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fetch additional details for a list of listings.
    
    Args:
        listings: List of basic listing data
        
    Returns:
        Enhanced list of listings with additional details
    """
    results = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def fetch_listing(listing: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            try:
                if 'listing_url' not in listing or not listing['listing_url']:
                    return listing
                    
                # Use crawl_page with extract_text=False to get structured data
                details = await crawl_page(listing['listing_url'], extract_text=False, retry_count=1)
                
                if isinstance(details, dict):
                    # Merge details with listing data, preserving existing values
                    for key, value in details.items():
                        if key not in listing or not listing[key]:
                            listing[key] = value
                            
                return listing
            except Exception as e:
                logger.error(f"Error fetching details for {listing.get('listing_url')}: {str(e)}")
                listing['error'] = f'Failed to fetch details: {str(e)}'
                return listing
    
    # Create tasks for all listings
    tasks = [fetch_listing(listing) for listing in listings]
    
    # Process results as they complete
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing detail task: {str(e)}")
    
    return results

def _process_locanto_detail_page(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """
    Process a Locanto detail page.
    
    Args:
        soup: BeautifulSoup object of the page
        url: The original URL
        
    Returns:
        Structured data for the listing
    """
    # Extract title
    title_tag = soup.find('h1') or soup.find('h2', class_='app_title') or soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else None
    
    # Extract location
    location = None
    loc_selectors = [
        'span[itemprop="addressLocality"]',
        '.vap_posting_details__address',
        '.js-result_location',
        'span.posting_listing__city',
        'div[class*="location"]',
        'span[class*="location"]'
    ]
    
    for selector in loc_selectors:
        loc_tag = soup.select_one(selector)
        if loc_tag:
            location = loc_tag.get_text(strip=True)
            break
    
    # Extract category from breadcrumb
    category = None
    breadcrumb = soup.find('ul', class_=re.compile(r'breadcrumb', re.I)) or soup.find('nav', class_=re.compile(r'breadcrumb', re.I))
    if breadcrumb:
        cats = [li.get_text(strip=True) for li in breadcrumb.find_all('li')]
        if cats:
            category = ' > '.join(cats)
    
    # Extract description
    description = None
    desc_selectors = [
        '.vap__description',
        '.vap_user_content__description',
        '.js-description_snippet',
        '.posting_listing__description',
        'div[class*="description"]',
        'div[class*="content"]'
    ]
    
    for selector in desc_selectors:
        desc_tag = soup.select_one(selector)
        if desc_tag:
            description = desc_tag.get_text(strip=True)
            break
    
    # Extract contact info
    contact_info = None
    body_text = soup.get_text(separator=' ', strip=True)
    
    # Look for phone numbers
    phone_match = re.search(r'(\+?\d[\d\s\-]{7,}\d)', body_text)
    if phone_match:
        contact_info = phone_match.group(1)
    else:
        # Look for email addresses
        email_match = re.search(r'([\w\.-]+@[\w\.-]+)', body_text)
        if email_match:
            contact_info = email_match.group(1)
    
    # Extract images
    images = []
    img_selectors = [
        '.user_images__img',
        'img[src*="locanto"]',
        'img.posting_listing__image',
        '.gallery img',
        '.image-gallery img'
    ]
    
    for selector in img_selectors:
        for img in soup.select(selector):
            src = img.get('src') or img.get('data-src')
            if src and not src.endswith('.svg'):
                if not src.startswith('http'):
                    src = urljoin(url, src)
                images.append(src)
    
    # Remove duplicates while preserving order
    images = list(dict.fromkeys(images))
    
    # Extract age
    age = None
    age_selectors = [
        '.header-age',
        '.vap_user_content__feature_value',
        'span.posting_listing__age',
        'span[class*="age"]'
    ]
    
    for selector in age_selectors:
        age_tag = soup.select_one(selector)
        if age_tag:
            age = age_tag.get_text(strip=True)
            break
    
    # If age not found in specific elements, try to extract from title or description
    if not age and (title or description):
        search_text = f"{title or ''} {description or ''}"
        age_match = re.search(r'\b(\d{2})\s*(?:yo|year|yr)s?\b', search_text)
        if age_match:
            age = age_match.group(1)
    
    # Return structured data
    return {
        'title': title,
        'location': location,
        'category': category,
        'description': description,
        'contact_info': contact_info,
        'listing_url': url,
        'images': images,
        'age': age
    }

def _extract_readable_text(html: str) -> str:
    """
    Extract readable text from HTML content.
    
    Args:
        html: The HTML content
        
    Returns:
        Extracted readable text
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script, style, and other non-content elements
    for tag in soup(['script', 'style', 'noscript', 'iframe', 'head', 'meta', 'link']):
        tag.decompose()
    
    # Extract text with spacing between elements
    text = ' '.join(soup.stripped_strings)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example usage:
# asyncio.run(crawl_page('https://example.com'))
