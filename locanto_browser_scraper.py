"""
LocantoBrowserScraper: Async, Playwright-powered scraper for locanto.co.za

Usage:
    from locanto_browser_scraper import LocantoBrowserScraper
    
    # Basic search
    listings = await LocantoBrowserScraper().search_listings("dating", "Cape Town", max_pages=2)
    
    # Get details for a specific listing
    details = await LocantoBrowserScraper().get_listing_details(listings[0]['url'])
    
    # Search with proxy
    scraper = LocantoBrowserScraper(use_proxy=True)
    listings = await scraper.search_listings("dating", "Cape Town", max_pages=2)

Requirements:
    pip install playwright beautifulsoup4 playwright-stealth
    playwright install
"""
import asyncio
import json
import logging
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qsl, parse_qs, urlencode, urljoin, urlparse, urlunparse, quote, unquote

# Playwright imports
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError

# Optional stealth mode
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False
    logging.warning("playwright-stealth not available. Browser fingerprinting protection disabled.")

# Optional BeautifulSoup for fallback parsing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available. Fallback HTML parsing disabled.")

# Optional import for agent integration
try:
    from .agent_utils import speak_chunks
    from .locanto_constants import LOCANTO_CATEGORY_SLUGS, LOCANTO_LOCATION_SLUGS, LOCANTO_SECTION_IDS
    AGENT_INTEGRATION = True
except ImportError:
    speak_chunks = None
    LOCANTO_CATEGORY_SLUGS = []
    LOCANTO_LOCATION_SLUGS = []
    LOCANTO_SECTION_IDS = {}
    AGENT_INTEGRATION = False

# Try to import puppeteer crawler for enhanced scraping
try:
    from .puppeteer_crawler import crawl_page as puppeteer_crawl
    PUPPETEER_AVAILABLE = True
except ImportError:
    PUPPETEER_AVAILABLE = False
    logging.warning("puppeteer_crawler not available. Enhanced scraping disabled.")

# Constants
PROXY_PREFIX = "https://please.untaint.us/?url="
BASE_URL = "https://www.locanto.co.za"

# Scraping constants
MAX_ARTICLES_PER_PAGE = 10
PAGE_TIMEOUT = 30000  # 30 seconds
WAIT_AFTER_NAV = 2000  # 2 seconds
SELECTOR_TIMEOUT = 8000  # 8 seconds
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = 4

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LocantoBrowserScraper")

# --- Helper functions ---
def is_valid_locanto_location(location: str) -> bool:
    """
    Check if a location string is a valid Locanto location slug.
    
    Args:
        location: Location string to check
        
    Returns:
        True if valid, False otherwise
    """
    if not location:
        return False
        
    # Normalize location string
    normalized = location.lower().strip().replace(' ', '-')
    
    # Check if it's in the predefined slugs
    if normalized in LOCANTO_LOCATION_SLUGS:
        return True
        
    # Check for common South African locations if not in predefined list
    common_locations = [
        'cape-town', 'johannesburg', 'pretoria', 'durban', 'port-elizabeth',
        'bloemfontein', 'east-london', 'kimberley', 'polokwane', 'nelspruit',
        'western-cape', 'gauteng', 'kwazulu-natal', 'eastern-cape', 'free-state',
        'north-west', 'mpumalanga', 'limpopo', 'northern-cape'
    ]
    
    return normalized in common_locations

# --- Selectors for robust extraction ---
LISTING_SELECTORS = {
    'container': [
        'article.posting_listing',
        '.js-posting_listing',
        '.result-item',
    ],
    'url': [
        'a.posting_listing__title.js-result_title.js-ad_link',
        'a.posting_listing__title',
        'a.js-ad_link',
        'a[href*="/ID_"]',
    ],
    'title': [
        'a.posting_listing__title.js-result_title.js-ad_link div.h3.js-result_title',
        'div.h3.js-result_title',
        '.posting_listing__title',
        '.result-title',
    ],
    'location': [
        'span.js-result_location.posting_listing__city',
        'span.js-result_location',
        'span.posting_listing__city',
        '.result-location',
    ],
    'description': [
        'div.posting_listing__description.js-description_snippet',
        '.js-description_snippet',
        'div.posting_listing__description',
        '.result-description',
    ],
    'age': [
        'span.posting_listing__age',
        '.result-age',
        '.age-info',
    ],
    'category': [
        'span.posting_listing__category',
        '.result-category',
        '.category-info',
    ],
    'image': [
        'img.posting_listing__image',
        '.result-image img',
        'img[src*="locanto"]',
    ],
}

DETAIL_SELECTORS = {
    'title': [
        'h1.app_title',
        'h1',
        '.vap_header__title',
        '.listing-title',
    ],
    'description': [
        '.vap__description',
        '.vap_user_content__description',
        '.js-description_snippet',
        '.posting_listing__description',
        'div[class*="description"]',
        'div[class*="content"]',
        '.listing-description',
    ],
    'price': [
        'div.price',
        'span.price',
        'div[class*="price"]',
        'span[class*="price"]',
        '.listing-price',
    ],
    'location': [
        'span[itemprop="addressLocality"]',
        '.vap_posting_details__address',
        '.js-result_location',
        'div[class*="location"]',
        'span[class*="location"]',
        '.listing-location',
    ],
    'date_posted': [
        'meta[name="dcterms.date"]',
        '.vap_user_content__date',
        'time[datetime]',
        '.listing-date',
    ],
    'images': [
        '.user_images__img',
        'img[src*="locanto"]',
        'img.posting_listing__image',
        '.gallery img',
        '.listing-images img',
    ],
    'contact_info': [
        '.vap__description',
        '.contact_buttons__button--call',
        'div[class*="contact"]',
        'div[class*="phone"]',
        'a[href^="tel:"]',
        '.listing-contact',
    ],
    'age': [
        '.header-age',
        '.vap_user_content__feature_value',
        'span.posting_listing__age',
        'span[class*="age"]',
        '.listing-age',
    ],
}

def clean_url(url: str) -> str:
    """
    Clean and normalize Locanto URLs, removing proxy prefixes and decoding repeatedly.
    
    Args:
        url: URL to clean
        
    Returns:
        Cleaned URL
    """
    if not url:
        return ""
        
    # Decode repeatedly until stable
    prev = None
    while prev != url:
        prev = url
        url = unquote(url)
    
    # Handle common proxy patterns
    proxy_patterns = [
        # Standard proxy prefix
        PROXY_PREFIX,
        # URL-encoded proxy prefix
        quote(PROXY_PREFIX),
        # Double-encoded proxy prefix
        quote(quote(PROXY_PREFIX)),
        # Alternative proxy formats
        "https://please.untaint.us?url=",
        "http://please.untaint.us/?url=",
    ]
    
    # Remove all proxy prefixes (even nested ones)
    for pattern in proxy_patterns:
        while pattern in url:
            url = url.replace(pattern, "")
    
    # Remove any ?url= or &url= parameters
    url = re.sub(r'([&?])url=([^&]*)', r'\1', url)
    
    # Remove any leading ? or & left over
    url = re.sub(r'^[?&]+', '', url)
    
    # Fix protocol and double slashes
    url = re.sub(r'https?:/{1,}', 'https://', url)
    
    # Ensure www. is present for locanto.co.za
    if 'locanto.co.za' in url and 'www.' not in url:
        url = url.replace('locanto.co.za', 'www.locanto.co.za')
    
    # Remove any trailing whitespace
    return url.strip()

def add_proxy_to_url(url: str, use_proxy: bool = True) -> str:
    """
    Add proxy prefix to URL if use_proxy is True.
    
    Args:
        url: URL to modify
        use_proxy: Whether to add proxy prefix
        
    Returns:
        URL with proxy prefix if use_proxy is True, otherwise original URL
    """
    if not url or not use_proxy:
        return url
        
    # Clean the URL first
    clean = clean_url(url)
    
    # Add proxy prefix
    return f"{PROXY_PREFIX}{clean}"

def build_locanto_url(
    query: Optional[str] = None, 
    location: Optional[str] = None, 
    category: Optional[str] = None, 
    tag: Optional[str] = None, 
    section: Optional[str] = None,
    use_proxy: bool = False
) -> str:
    """
    Build a Locanto URL based on search parameters.
    
    Args:
        query: Search query
        location: Location to search in
        category: Category to search in
        tag: Tag to search for
        section: Section to search in
        use_proxy: Whether to add proxy prefix
        
    Returns:
        Constructed Locanto URL
    """
    base = "https://www.locanto.co.za"
    result_url = ""
    
    # Tag search has priority
    if tag:
        result_url = f"{base}/g/tag/{tag}/"
    
    # Location-based search
    elif location and is_valid_locanto_location(location):
        # Normalize location
        loc_slug = location.lower().replace(' ', '-')
        
        # Build URL with category or section if provided
        if category:
            result_url = f"{base}/{loc_slug}/{category}/"
        elif section:
            result_url = f"{base}/{loc_slug}/{section}/"
        else:
            result_url = f"{base}/{loc_slug}/"
            
        # Add query parameter if provided
        if query:
            result_url += f"?query={quote(query)}"
    
    # Generic search
    elif query:
        result_url = f"{base}/g/q/?query={quote(query)}"
    
    # Default to base URL
    else:
        result_url = base
    
    # Add proxy if requested
    return add_proxy_to_url(result_url, use_proxy)

class LocantoBrowserScraper:
    """
    A robust, Playwright-powered scraper for Locanto.co.za with proxy support.
    
    Features:
    - Proxy support via https://please.untaint.us
    - Stealth mode to avoid detection
    - Fallback mechanisms for different page structures
    - Concurrent fetching of listing details
    - Cookie persistence
    """
    def __init__(self, 
                 cookies_path: Optional[str] = None, 
                 detail_fetch_concurrency: int = 4,
                 use_proxy: bool = False,
                 headless: bool = True,
                 timeout: int = PAGE_TIMEOUT,
                 max_retries: int = MAX_RETRIES):
        """
        Initialize the Locanto browser scraper.
        
        Args:
            cookies_path: Path to cookies file for authentication persistence
            detail_fetch_concurrency: Number of concurrent requests for fetching listing details
            use_proxy: Whether to route requests through the proxy
            headless: Whether to run the browser in headless mode
            timeout: Page load timeout in milliseconds
            max_retries: Maximum number of retries for failed requests
        """
        # Browser components
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None
        
        # Configuration
        self.cookies_path = cookies_path
        self.detail_fetch_concurrency = min(detail_fetch_concurrency, MAX_CONCURRENT_REQUESTS)
        self.use_proxy = use_proxy
        self.headless = headless
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Stats and cache
        self.request_count = 0
        self.error_count = 0
        self.cache = {}
        
        # User agent rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
        ]

    async def start(self):
        """
        Start the browser session.
        """
        try:
            # Start Playwright
            self.playwright = await async_playwright().start()
            
            # Launch browser with appropriate settings
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials'
                ]
            )
            
            # Select a random user agent
            user_agent = self.user_agents[int(time.time()) % len(self.user_agents)]
            
            # Create browser context with appropriate settings
            self.context = await self.browser.new_context(
                user_agent=user_agent,
                viewport={'width': 1920, 'height': 1080},
                locale="en-US",
                timezone_id="Africa/Johannesburg",
                geolocation={"latitude": -33.9249, "longitude": 18.4241},  # Cape Town
                permissions=["geolocation"]
            )
            
            # Load cookies if provided
            if self.cookies_path and os.path.exists(self.cookies_path):
                try:
                    logger.info(f"Loading cookies from {self.cookies_path}")
                    with open(self.cookies_path, "r") as f:
                        cookies = json.load(f)
                    await self.context.add_cookies(cookies)
                except Exception as e:
                    logger.warning(f"Failed to load cookies: {str(e)}")
            
            # Apply stealth mode if available
            if STEALTH_AVAILABLE:
                await stealth_async(self.context)
            
            # Create page with appropriate settings
            self.page = await self.context.new_page()
            
            # Set default timeout
            self.page.set_default_timeout(self.timeout)
            
            # Handle dialog automatically
            self.page.on("dialog", lambda dialog: asyncio.create_task(dialog.dismiss()))
            
            logger.info("Browser session started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start browser session: {str(e)}")
            await self.close()
            return False

    async def close(self):
        """
        Close the browser session and clean up resources.
        """
        try:
            if self.page:
                await self.page.close()
                self.page = None
                
            if self.context:
                # Save cookies if path is provided
                if self.cookies_path:
                    try:
                        cookies = await self.context.cookies()
                        os.makedirs(os.path.dirname(os.path.abspath(self.cookies_path)), exist_ok=True)
                        with open(self.cookies_path, "w") as f:
                            json.dump(cookies, f)
                        logger.info(f"Saved cookies to {self.cookies_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save cookies: {str(e)}")
                
                await self.context.close()
                self.context = None
                
            if self.browser:
                await self.browser.close()
                self.browser = None
                
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
                
            logger.info("Browser session closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser session: {str(e)}")

    async def try_selectors_playwright(self, element: Page, selectors: List[str], attr: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Try multiple selectors to extract text or attribute value from an element.
        
        Args:
            element: Playwright page or element to query
            selectors: List of CSS selectors to try
            attr: Optional attribute name to extract
            
        Returns:
            Tuple of (extracted value, successful selector)
        """
        for sel in selectors:
            try:
                # Find the element
                found = await element.query_selector(sel)
                if not found:
                    continue
                    
                # Extract attribute or text
                if attr:
                    val = await found.get_attribute(attr)
                    if val:
                        logger.debug(f"Selector '{sel}' succeeded for attr '{attr}'")
                        return val.strip(), sel
                else:
                    val = await found.inner_text()
                    if val:
                        logger.debug(f"Selector '{sel}' succeeded for text")
                        return val.strip(), sel
            except Exception as e:
                logger.debug(f"Selector '{sel}' failed: {str(e)}")
                continue
                
        return None, None
        
    async def handle_popups(self):
        """
        Handle various popups, consent dialogs, and overlays that might appear on Locanto.
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
                'button:has-text("I agree")',
                'button:has-text("OK")',
                '#bnp_btn_accept',
                '#consent-banner button',
                '.bnp_btn_accept',
                '.cookie-consent__btn',
                '.consent-btn',
                '.gdpr-consent-button'
            ]
            
            for selector in consent_selectors:
                try:
                    if await self.page.query_selector(selector):
                        logger.info(f"Found consent popup with selector: {selector}")
                        await self.page.click(selector)
                        # Wait a moment for the popup to disappear
                        await asyncio.sleep(1)
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Error handling popups: {str(e)}")
            
    async def navigate_to_url(self, url: str, wait_for_selector: Optional[str] = None, retry_count: int = 0) -> bool:
        """
        Navigate to a URL with retry logic and popup handling.
        
        Args:
            url: URL to navigate to
            wait_for_selector: Optional selector to wait for after navigation
            retry_count: Current retry attempt
            
        Returns:
            True if navigation was successful, False otherwise
        """
        if not self.page:
            if not await self.start():
                return False
                
        # Add proxy if needed
        if self.use_proxy and not url.startswith(PROXY_PREFIX):
            url = add_proxy_to_url(url, True)
            
        try:
            # Increment request counter
            self.request_count += 1
            
            # Navigate to the URL
            logger.info(f"Navigating to: {url}")
            response = await self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
            
            if not response:
                logger.warning(f"No response received for {url}")
                if retry_count < self.max_retries:
                    return await self.navigate_to_url(url, wait_for_selector, retry_count + 1)
                return False
                
            # Check if the page loaded successfully
            status = response.status
            if status >= 400:
                logger.warning(f"Received status code {status} for {url}")
                if retry_count < self.max_retries:
                    return await self.navigate_to_url(url, wait_for_selector, retry_count + 1)
                return False
                
            # Wait for network to be idle
            await self.page.wait_for_load_state("networkidle", timeout=self.timeout)
            
            # Handle popups
            await self.handle_popups()
            
            # Wait for specific selector if provided
            if wait_for_selector:
                try:
                    await self.page.wait_for_selector(wait_for_selector, timeout=SELECTOR_TIMEOUT)
                except PlaywrightTimeoutError:
                    logger.warning(f"Selector '{wait_for_selector}' not found on page")
                    # Continue anyway, as the page might have loaded but the selector is missing
            
            # Additional wait to ensure JavaScript has executed
            await asyncio.sleep(WAIT_AFTER_NAV / 1000)  # Convert ms to seconds
            
            return True
        except Exception as e:
            self.error_count += 1
            logger.error(f"Navigation error for {url}: {str(e)}")
            
            if retry_count < self.max_retries:
                logger.info(f"Retrying navigation to {url} (attempt {retry_count + 1}/{self.max_retries})")
                return await self.navigate_to_url(url, wait_for_selector, retry_count + 1)
                
            return False

    async def search_listings(self, 
                         query: Optional[str] = None, 
                         location: Optional[str] = None, 
                         max_pages: int = 1, 
                         tag: Optional[str] = None, 
                         category: Optional[str] = None, 
                         section: Optional[str] = None, 
                         url: Optional[str] = None, 
                         age_min: Optional[int] = None, 
                         age_max: Optional[int] = None, 
                         query_description: Optional[bool] = None, 
                         dist: Optional[int] = None, 
                         sort: Optional[str] = None,
                         fetch_details: bool = True,
                         use_puppeteer: bool = True) -> List[Dict[str, Any]]:
        """
        Search Locanto listings with robust error handling and proxy support.
        
        Args:
            query: Search query string
            location: Location to search in
            max_pages: Maximum number of pages to scrape
            tag: Tag to search for
            category: Category to search in
            section: Section to search in
            url: Direct URL to search (overrides other parameters)
            age_min: Minimum age filter
            age_max: Maximum age filter
            query_description: Whether to search in descriptions
            dist: Distance filter in km
            sort: Sort order (date, price, etc.)
            fetch_details: Whether to fetch detailed information for each listing
            use_puppeteer: Whether to try using puppeteer crawler first
            
        Returns:
            List of listing dictionaries
        """
        # Initialize results and counters
        results = []
        page_num = 1
        listings_found = 0
        
        # Set default values for parameters
        if query_description is None:
            query_description = True
        if sort is None:
            sort = "date"
        if age_min is None:
            age_min = 18
        if age_max is None:
            age_max = 40
        if dist is None:
            dist = 30
            
        # Try using puppeteer crawler first if available and requested
        if use_puppeteer and PUPPETEER_AVAILABLE:
            try:
                logger.info("Attempting to use puppeteer crawler for initial search")
                
                # Build the search URL
                if url:
                    search_url = clean_url(url)
                else:
                    search_url = build_locanto_url(query, location, category, tag, section, self.use_proxy)
                
                # Add query parameters
                parsed = urlparse(search_url)
                query_params = dict(parse_qsl(parsed.query))
                query_params['query_description'] = '1' if query_description else '0'
                query_params['sort'] = sort
                query_params['age[min]'] = str(age_min)
                query_params['age[max]'] = str(age_max)
                query_params['dist'] = str(dist)
                new_query = urlencode(query_params)
                parsed = parsed._replace(query=new_query)
                search_url = urlunparse(parsed)
                
                # Use puppeteer crawler to fetch results
                puppeteer_results = await puppeteer_crawl(search_url, extract_text=False)
                
                # Check if we got valid results
                if isinstance(puppeteer_results, list) and len(puppeteer_results) > 0:
                    logger.info(f"Puppeteer crawler found {len(puppeteer_results)} listings")
                    return puppeteer_results
                    
                logger.info("Puppeteer crawler didn't return valid results, falling back to Playwright")
            except Exception as e:
                logger.warning(f"Puppeteer crawler failed: {str(e)}. Falling back to Playwright.")
        
        # Start browser session if needed
        if not self.page:
            if not await self.start():
                logger.error("Failed to start browser session")
                return []
                
        try:
            # Process each page up to max_pages
            while page_num <= max_pages:
                # Build the search URL for this page
                if url:
                    search_url = clean_url(url)
                    parsed = urlparse(search_url)
                    query_params = dict(parse_qsl(parsed.query))
                    
                    # Set search parameters
                    query_params.setdefault('query', query or '')
                    query_params['query_description'] = '1' if query_description else '0'
                    query_params['sort'] = sort
                    query_params['age[min]'] = str(age_min)
                    query_params['age[max]'] = str(age_max)
                    query_params['dist'] = str(dist)
                    
                    # Add page number if not the first page
                    if page_num > 1:
                        query_params['page'] = str(page_num)
                        
                    # Rebuild the URL
                    new_query = urlencode(query_params)
                    parsed = parsed._replace(query=new_query)
                    search_url = urlunparse(parsed)
                else:
                    # Build URL from components
                    base_url = build_locanto_url(query, location, category, tag, section)
                    
                    # Add query parameters
                    parsed = urlparse(base_url)
                    query_params = dict(parse_qsl(parsed.query))
                    query_params['query_description'] = '1' if query_description else '0'
                    query_params['sort'] = sort
                    query_params['age[min]'] = str(age_min)
                    query_params['age[max]'] = str(age_max)
                    query_params['dist'] = str(dist)
                    
                    # Add page number if not the first page
                    if page_num > 1:
                        query_params['page'] = str(page_num)
                        
                    # Rebuild the URL
                    new_query = urlencode(query_params)
                    parsed = parsed._replace(query=new_query)
                    search_url = urlunparse(parsed)
                
                # Log the search URL
                logger.info(f"Searching page {page_num}/{max_pages}: {search_url}")
                
                # Navigate to the search URL
                if not await self.navigate_to_url(search_url, wait_for_selector="article.posting_listing"):
                    logger.warning(f"Failed to navigate to page {page_num}")
                    # Try next page
                    page_num += 1
                    continue
                
                # Extract listings from the page
                page_listings = await self._extract_listings_from_page(search_url)
                
                if not page_listings:
                    logger.info(f"No listings found on page {page_num}")
                    # Check if there's a next page button
                    next_page = await self.page.query_selector('a.pagination__next')
                    if not next_page:
                        logger.info("No more pages available")
                        break
                else:
                    # Add listings to results
                    results.extend(page_listings)
                    listings_found += len(page_listings)
                    logger.info(f"Found {len(page_listings)} listings on page {page_num}. Total: {listings_found}")
                    
                    # Check if we've reached the maximum number of listings per page
                    # If not, there are no more pages
                    if len(page_listings) < MAX_ARTICLES_PER_PAGE:
                        logger.info(f"Found fewer than {MAX_ARTICLES_PER_PAGE} listings, assuming no more pages")
                        break
                
                # Move to next page
                page_num += 1
                
                # Add a small delay between pages to avoid rate limiting
                await asyncio.sleep(1)
                
                # Process contact info for new listings if needed
                async def fetch_contact_info(listing, sem):
                    async with sem:
                        contact_info = None
                        try:
                            detail_url = listing['url']
                            if detail_url and detail_url.startswith(PROXY_PREFIX):
                                detail = await self.get_listing_details(detail_url)
                                contact_info = detail.get('contact_info')
                        except Exception as e:
                            logging.error(f"[DEBUG] Error fetching contact info for {listing.get('url')}: {e}")
                        listing['contact_info'] = contact_info
                sem = asyncio.Semaphore(self.detail_fetch_concurrency)
                await asyncio.gather(*(fetch_contact_info(listing, sem) for listing in results[-len(articles):]))
                page_num += 1
            
            if results and debug_info:
                results[0].update(debug_info)
            elif debug_info:
                results.append(debug_info)
                
            logger.info(f"Search completed. Found {len(results)} listings across {page_num-1} pages")
            
            # Fetch details for each listing if requested
            if results and fetch_details:
                results = await self.fetch_listing_details(results)
                
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            logger.error(traceback.format_exc())
            return results
        finally:
            # Don't close the browser here to allow reuse for detail fetching
            pass
            
    async def _extract_listings_from_page(self, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract listings from the current page.
        
        Args:
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of listing dictionaries
        """
        page_listings = []
        
        try:
            # Try multiple container selectors
            articles = []
            for container_selector in LISTING_SELECTORS['container']:
                found = await self.page.query_selector_all(container_selector)
                if found:
                    articles.extend(found)
                    break
            
            if not articles:
                logger.warning("No listing containers found on page")
                return []
                
            # Process each article
            for article in articles:
                try:
                    listing = {}
                    
                    # Extract URL
                    url_val, url_sel = await self.try_selectors_playwright(article, LISTING_SELECTORS['url'], "href")
                    if url_val:
                        listing['url'] = urljoin(base_url, url_val)
                        listing['listing_url'] = listing['url']  # For compatibility
                    else:
                        # Skip listings without URL
                        continue
                        
                    # Extract title
                    title_val, _ = await self.try_selectors_playwright(article, LISTING_SELECTORS['title'])
                    if title_val:
                        listing['title'] = title_val
                    else:
                        # Skip listings without title
                        continue
                        
                    # Extract location
                    location_val, _ = await self.try_selectors_playwright(article, LISTING_SELECTORS['location'])
                    if location_val:
                        listing['location'] = location_val
                        
                    # Extract description
                    desc_val, _ = await self.try_selectors_playwright(article, LISTING_SELECTORS['description'])
                    if desc_val:
                        listing['description'] = desc_val
                        
                    # Extract age
                    age_val, _ = await self.try_selectors_playwright(article, LISTING_SELECTORS['age'])
                    if age_val:
                        listing['age'] = age_val
                        
                    # Extract category
                    category_val, _ = await self.try_selectors_playwright(article, LISTING_SELECTORS['category'])
                    if category_val:
                        listing['category'] = category_val
                        
                    # Extract image
                    image_val, _ = await self.try_selectors_playwright(article, LISTING_SELECTORS['image'], "src")
                    if image_val:
                        listing['images'] = [image_val]
                        
                    # Add timestamp
                    listing['timestamp'] = int(time.time())
                    
                    # Add to results if we have at least URL and title
                    if 'url' in listing and 'title' in listing:
                        page_listings.append(listing)
                        
                except Exception as e:
                    logger.warning(f"Error extracting listing details: {str(e)}")
                    continue
                    
            return page_listings
            
        except Exception as e:
            logger.error(f"Error extracting listings from page: {str(e)}")
            return []
            
    async def fetch_listing_details(self, listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch detailed information for each listing in parallel.
        
        Args:
            listings: List of listing dictionaries
        
        Returns:
            List of listing dictionaries with detailed information
        """
        if not listings:
            return []
            
        results = []
        sem = asyncio.Semaphore(self.detail_fetch_concurrency)
        
        async def fetch_detail(listing: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                try:
                    url = listing.get('url') or listing.get('listing_url')
                    if not url:
                        return listing
                        
                    # Use puppeteer crawler if available
                    if PUPPETEER_AVAILABLE:
                        try:
                            details = await puppeteer_crawl(url, extract_text=False)
                            if isinstance(details, dict):
                                # Merge details with listing data
                                for key, value in details.items():
                                    if key not in listing or not listing[key]:
                                        listing[key] = value
                                return listing
                        except Exception as e:
                            logger.warning(f"Puppeteer crawler failed for {url}: {str(e)}")
                    
                    # Fallback to regular detail fetching
                    details = await self.get_listing_details(url)
                    if details:
                        # Merge details with listing data
                        for key, value in details.items():
                            if key not in listing or not listing[key]:
                                listing[key] = value
                                
                    return listing
                except Exception as e:
                    logger.error(f"Error fetching details for {listing.get('url')}: {str(e)}")
                    listing['error'] = f"Failed to fetch details: {str(e)}"
                    return listing
        
        # Create tasks for all listings
        tasks = [fetch_detail(listing) for listing in listings]
        
        # Process results as they complete
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing detail task: {str(e)}")
                
        return results

    async def handle_popups(self):
        """
        Handle various popups, consent dialogs, and overlays that might appear on Locanto.
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
                'button:has-text("I agree")',
                'button:has-text("OK")',
                '#bnp_btn_accept',
                '#consent-banner button',
                '.bnp_btn_accept',
                '.cookie-consent__btn',
                '.consent-btn',
                '.gdpr-consent-button'
            ]
            
            for selector in consent_selectors:
                try:
                    if await self.page.query_selector(selector):
                        logger.info(f"Found consent popup with selector: {selector}")
                        await self.page.click(selector)
                        # Wait a moment for the popup to disappear
                        await asyncio.sleep(1)
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Error handling popups: {str(e)}")
            
    async def get_listing_details(self, url: str) -> Dict[str, Any]:
        """
        Get detailed information for a single listing.
        
        Args:
            url: URL of the listing to fetch details for
            
        Returns:
            Dictionary with listing details
        """
        # Try using puppeteer crawler first if available
        if PUPPETEER_AVAILABLE:
            try:
                logger.info(f"Attempting to use puppeteer crawler for listing details: {url}")
                details = await puppeteer_crawl(url, extract_text=False)
                if isinstance(details, dict) and 'title' in details:
                    logger.info(f"Successfully fetched listing details with puppeteer crawler")
                    return details
                logger.info("Puppeteer crawler didn't return valid results, falling back to Playwright")
            except Exception as e:
                logger.warning(f"Puppeteer crawler failed: {str(e)}. Falling back to Playwright.")
                
        # Start browser if needed
        if not self.page:
            if not await self.start():
                logger.error("Failed to start browser session")
                return {}
                
        try:
            # Clean and possibly add proxy to URL
            clean = clean_url(url)
            target_url = add_proxy_to_url(clean, self.use_proxy)
            
            # Navigate to the listing page
            if not await self.navigate_to_url(target_url):
                logger.warning(f"Failed to navigate to listing: {url}")
                return {}
                
            # Extract listing details
            details = {}
            
            # Title
            title_val, _ = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['title'])
            if title_val:
                details['title'] = title_val
                
            # Description
            desc_val, _ = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['description'])
            if desc_val:
                details['description'] = desc_val
                
            # Location
            location_val, _ = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['location'])
            if location_val:
                details['location'] = location_val
                
            # Age
            age_val, _ = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['age'])
            if age_val:
                details['age'] = age_val
                
            # Price
            price_val, _ = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['price'])
            if price_val:
                details['price'] = price_val
                
            # Date posted
            date_val, date_sel = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['date_posted'])
            if date_val:
                details['date_posted'] = date_val
                
            # Contact info
            contact_val, _ = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['contact_info'])
            if contact_val:
                details['contact_info'] = contact_val
                
            # Images
            images = []
            for img_sel in DETAIL_SELECTORS['images']:
                try:
                    img_elems = await self.page.query_selector_all(img_sel)
                    for img in img_elems:
                        src = await img.get_attribute('src')
                        if src and not src.endswith('.svg'):
                            images.append(src)
                except Exception:
                    continue
                    
            if images:
                details['images'] = list(dict.fromkeys(images))  # Remove duplicates
                
            # Add URL and timestamp
            details['url'] = clean
            details['listing_url'] = clean  # For compatibility
            details['timestamp'] = int(time.time())
            
            return details
            
        except Exception as e:
            logger.error(f"Error fetching listing details: {str(e)}")
            return {}
            
# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        scraper = LocantoBrowserScraper(use_proxy=True)
        listings = await scraper.search_listings("dating", "Cape Town", max_pages=1)
        print(f"Found {len(listings)} listings")
        await scraper.close()
            
        asyncio.run(main())

    async def get_listing_details(self, url: str) -> Dict:
        """Get detailed info from a Locanto listing page using the same browser context. Extracts only relevant content. Retries on failure."""
        url = clean_url(url)
        if not url.startswith(PROXY_PREFIX):
            if url.startswith("/"):
                url = BASE_URL + url
            if url.startswith(BASE_URL):
                url = PROXY_PREFIX + url
            elif url.startswith('http'):
                url = PROXY_PREFIX + url
        logging.info(f"[DEBUG] Navigating to detail page: {url}")
        details = {}
        if not self.page:
            await self.start()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT)
                await asyncio.sleep(2)
                html = await self.page.content()
                # Title
                title, title_sel = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['title'])
                details['title'] = title or ""
                # Description
                description, desc_sel = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['description'])
                details['description'] = description or ""
                # Price
                price, price_sel = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['price'])
                details['price'] = price or ""
                # Location
                location, loc_sel = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['location'])
                details['location'] = location or ""
                # Date posted
                date_posted, date_sel = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['date_posted'])
                details['date_posted'] = date_posted or ""
                # Images
                img_urls = []
                for sel in DETAIL_SELECTORS['images']:
                    imgs = await self.page.query_selector_all(sel)
                    for img in imgs:
                        src = await img.get_attribute('src')
                        if src:
                            img_urls.append(src)
                details['images'] = img_urls
                # Contact info
                contact, contact_sel = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['contact_info'])
                details['contact_info'] = contact or ""
                # Age
                age, age_sel = await self.try_selectors_playwright(self.page, DETAIL_SELECTORS['age'])
                if age:
                    age_match = re.search(r'(\d{2})', age)
                    details['age'] = age_match.group(1) if age_match else age
                else:
                    details['age'] = ""
                # Ad ID
                id_match = re.search(r'ID_(\d+)', url)
                if not id_match:
                    id_match = re.search(r'ID_(\d+)', html)
                details['ad_id'] = id_match.group(1) if id_match else ""
                logging.info(f"[DEBUG] Details extracted: {details}")
                return details
            except Exception as e:
                logging.error(f"[DEBUG] Error fetching details (attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2)
                else:
                    details['error'] = f"Failed to fetch details after {MAX_RETRIES} attempts: {e}"
                    return details

    async def recursive_map_site(self, start_url: str = None, max_depth: int = 2, visited=None, depth=0, save_html=False):
        """
        Recursively map Locanto site structure for fine-tuning selectors.
        - start_url: The URL to start mapping from (default: Locanto search page).
        - max_depth: How deep to recurse (default: 2).
        - visited: Set of visited URLs to avoid loops.
        - save_html: If True, saves each page's HTML for manual inspection.
        """
        if visited is None:
            visited = set()
        if not start_url:
            start_url = f"{PROXY_PREFIX}{BASE_URL}/g/q/?query=dating"
        if start_url in visited or depth > max_depth:
            return
        visited.add(start_url)
        logging.info(f"[MAP] Visiting (depth={depth}): {start_url}")
        if not self.page:
            await self.start()
        try:
            await self.page.goto(start_url, wait_until="domcontentloaded", timeout=60000)
            await self.page.wait_for_timeout(2000)
            html = await self.page.content()
            if save_html:
                fname = f"map_debug_depth{depth}_{re.sub(r'[^a-zA-Z0-9]', '_', start_url)[:50]}.html"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(html)
            # Find all listing containers
            articles = await self.page.query_selector_all('article.posting_listing, li.listing, div.listing')
            logging.info(f"[MAP] Found {len(articles)} listing elements at depth {depth}")
            # Find all category links (sidebars, navs, etc.)
            cat_links = await self.page.query_selector_all('a[href*="/g/"], a[href*="/personals/"], a[href*="/dating/"]')
            next_links = await self.page.query_selector_all('a[rel="next"], a.js-pagination-next')
            # Print found category and next links
            for a in cat_links + next_links:
                href = await a.get_attribute('href')
                if href and not href.startswith('http'):
                    href = BASE_URL + href
                if href and not href.startswith(PROXY_PREFIX):
                    href = PROXY_PREFIX + href
                if href and href not in visited:
                    logging.info(f"[MAP] Queuing: {href}")
                    await self.recursive_map_site(href, max_depth, visited, depth+1, save_html)
        except Exception as e:
            logging.error(f"[MAP] Error at {start_url}: {e}")

@function_tool
async def search_locanto_browser(context: RunContext, query: str = "dating", location: str = "Cape Town", max_pages: int = 1, tag: str = None, category: str = None, section: str = None, url: str = None) -> str:
    """Search Locanto.co.za for listings by tag, category, section, generic query, or direct URL. Uses Playwright browser and proxy. Supports tag, category, section, and direct URLs."""
    try:
        scraper = LocantoBrowserScraper()
        listings = await scraper.search_listings(query=query, location=location, max_pages=max_pages, tag=tag, category=category, section=section, url=url)
        if not listings:
            return f"No Locanto listings found for '{query}' in '{location}'."
        # If the first result is an error dict, return the error message
        if isinstance(listings[0], dict) and 'error' in listings[0]:
            debug_url = listings[0].get('_debug_url')
            debug_proxied_url = listings[0].get('_debug_proxied_url')
            debug_msg = f"\n[DEBUG] Search URL: {debug_url}" if debug_url else ""
            debug_msg += f"\n[DEBUG] Proxied URL: {debug_proxied_url}" if debug_proxied_url else ""
            return f"Error: {listings[0]['error']}{debug_msg}"
        summary = f"Found {len(listings)} Locanto listings for '{query}' in '{location}':\n\n"
        for i, listing in enumerate(listings[:5], 1):
            if not isinstance(listing, dict) or 'title' not in listing:
                continue
            summary += f"{i}. {listing['title']}\n"
            if listing.get('age'):
                summary += f"   Age: {listing['age']}\n"
            if listing.get('location'):
                summary += f"   Location: {listing['location']}\n"
            if listing.get('description'):
                desc = listing['description'][:120] + ('...' if len(listing['description']) > 120 else '')
                summary += f"   Description: {desc}\n"
            # Do NOT include the URL in the summary
            summary += "\n"
        return summary
    except Exception as e:
        return f"Error searching Locanto with Playwright: {e}" 