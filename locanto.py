# Locanto configuration
from typing import List, Dict, Any, Optional, TypedDict, Union, Tuple
from urllib.parse import urljoin
import logging
import difflib
import re
import asyncio
from bs4 import BeautifulSoup
from livekit.agents import function_tool, RunContext

# Import utils directly instead of using relative import
try:
    from utils import sanitize_for_azure, clean_spoken, handle_tool_results
except ImportError:
    # Fallback definitions if utils module is not available
    def sanitize_for_azure(text):
        return text
    
    def clean_spoken(text):
        return text
    
    async def handle_tool_results(session, text):
        pass

# Import puppeteer_crawler for enhanced web scraping
try:
    from puppeteer_crawler import crawl_page
except ImportError:
    try:
        from .puppeteer_crawler import crawl_page
    except ImportError:
        logging.warning("puppeteer_crawler module not available, using fallback scraping methods")
        # Define a fallback function
        async def crawl_page(*args, **kwargs):
            return "Puppeteer crawler functionality is not available."

# Create a singleton instance of the LocantoClient
_locanto_client = None

def get_locanto_client(cookies_path="locanto_cookies.json"):
    """Get or create a singleton instance of the LocantoClient.
    
    Args:
        cookies_path: Path to the cookies JSON file
        
    Returns:
        LocantoClient instance
    """
    global _locanto_client
    if _locanto_client is None:
        _locanto_client = LocantoClient(cookies_path=cookies_path)
    return _locanto_client
LOCANTO_LOCATION_SLUGS = ["western-cape", "gauteng", "kwazulu-natal", "eastern-cape",
                         "free-state", "limpopo", "mpumalanga", "north-west", "northern-cape"]
LOCANTO_CATEGORY_SLUGS = ["personals", "men-seeking-men", "women-seeking-men", "men-seeking-women", "women-seeking-women"]
LOCANTO_SECTION_IDS = ["dating", "casual-encounters", "missed-connections", "friends"]
LOCANTO_TAG_SLUGS = ["gay", "lesbian", "bisexual", "transgender", "queer", "lgbtq+", "straight"]

class LocantoCategory(TypedDict):
    name: str
    url: str
    count: int

class LocantoListing(TypedDict):
    title: str
    description: str
    location: str
    price: str
    date_posted: str
    url: str
    images: List[str]
    contact_info: Optional[str]
    poster_info: Optional[str]
    full_description: Optional[str]
    category_path: List[str]
    age: Optional[str]
    reply_count: Optional[int]
    ad_id: Optional[str]

def is_valid_locanto_location(location):
    if not location:
        return False
    return location.lower().replace(' ', '-') in LOCANTO_LOCATION_SLUGS

def is_valid_locanto_category(category):
    if not category:
        return False
    return category.replace(' ', '-').replace('_', '-').title() in LOCANTO_CATEGORY_SLUGS

def is_valid_locanto_section(section):
    if not section:
        return False
    return section in LOCANTO_SECTION_IDS

def is_valid_locanto_tag(tag):
    if not tag:
        return False
    return tag.replace(' ', '-').lower() in {t.lower() for t in LOCANTO_TAG_SLUGS}

def suggest_closest_slug(input_str, valid_slugs):
    import difflib
    matches = difflib.get_close_matches(input_str.lower().replace(' ', '-'), [s.lower() for s in valid_slugs], n=3, cutoff=0.6)
    return matches

class LocantoClient:
    """Client for interacting with Locanto website."""
    
    def __init__(self, cookies_path=None):
        """Initialize the Locanto client.
        
        Args:
            cookies_path: Optional path to a cookies JSON file
        """
        self.cookies_path = cookies_path
        self.cookies = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self._load_cookies()
    
    def _load_cookies(self):
        """Load cookies from file if available."""
        if not self.cookies_path:
            return
            
        try:
            import json
            import os
            if os.path.exists(self.cookies_path):
                with open(self.cookies_path, 'r') as f:
                    self.cookies = json.load(f)
                logging.info(f"Loaded cookies from {self.cookies_path}")
        except Exception as e:
            logging.error(f"Error loading cookies: {e}")
    
    def _save_cookies(self):
        """Save cookies to file if path is specified."""
        if not self.cookies_path:
            return
            
        try:
            import json
            with open(self.cookies_path, 'w') as f:
                json.dump(self.cookies, f)
            logging.info(f"Saved cookies to {self.cookies_path}")
        except Exception as e:
            logging.error(f"Error saving cookies: {e}")
    
    def _update_headers(self, url):
        """Update headers with referer based on URL."""
        headers = self.headers.copy()
        headers['Referer'] = url
        return headers
    
    async def _update_cookies(self, response):
        """Update cookies from response."""
        if 'set-cookie' in response.headers:
            # Parse cookies from response
            cookies_header = response.headers.get('set-cookie')
            if cookies_header:
                import http.cookies
                cookie = http.cookies.SimpleCookie()
                cookie.load(cookies_header)
                for key, morsel in cookie.items():
                    self.cookies[key] = morsel.value
            self._save_cookies()
    
    async def _get_client(self):
        """Get an HTTP client with the current cookies and proxy configuration."""
        import httpx
        
        # Use the untaint.us proxy for all requests
        return httpx.AsyncClient(
            cookies=self.cookies, 
            follow_redirects=True,
            # Configure proxy for all requests
            proxies=None,  # We'll handle proxy at the URL level instead
            timeout=45.0  # Increased timeout for proxy requests
        )
        
    async def get_categories(self, base_url: str) -> List[LocantoCategory]:
        """Get available categories from a Locanto page.

        Args:
            base_url: The URL to get categories from

        Returns:
            List of LocantoCategory objects
        """
        categories: List[LocantoCategory] = []
        async with await self._get_client() as client:
            try:
                response = await client.get(base_url, headers=self._update_headers(base_url))
                await self._update_cookies(response)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find category links in the sidebar navigation
                category_elements = soup.select('nav.sidebar a[href*="/c/"]')
                for elem in category_elements:
                    name = elem.get_text(strip=True)
                    url = urljoin(base_url, elem['href'])
                    count_elem = elem.find('span', class_='count')
                    count = int(count_elem.get_text(strip=True)) if count_elem else 0
                    
                    categories.append({
                        'name': name,
                        'url': url,
                        'count': count
                    })

            except Exception as e:
                logging.error(f"Error getting categories: {str(e)}")

        return categories

    async def get_listing_details(self, url: str) -> Dict[str, Any]:
        """Get detailed information from a single listing page.

        Args:
            url: The URL of the listing to scrape

        Returns:
            Dictionary containing detailed listing information
        """
        details = {
            'contact_info': None,
            'poster_info': None,
            'full_description': None,
            'age': None,
            'reply_count': None,
            'ad_id': None,
            'images': []
        }

        # Try to use puppeteer_crawler first for better data extraction
        try:
            logging.info(f"Using puppeteer crawler to fetch listing details from {url}")
            # Use untaint.us proxy to avoid Cloudflare restrictions
            proxy_url = f"https://please.untaint.us/?url={urllib.parse.quote_plus(url)}"
            puppeteer_result = await crawl_page(
                proxy_url,
                wait_selector=".ad-content__description, .posting__description",
                timeout=45000,
                extract_text=False
            )
            
            # Check if we got a valid result from puppeteer
            if isinstance(puppeteer_result, dict) and 'title' in puppeteer_result:
                logging.info(f"Successfully extracted structured data using puppeteer crawler")
                # Map puppeteer result to our details format
                details['full_description'] = puppeteer_result.get('description')
                details['contact_info'] = puppeteer_result.get('contact_info')
                details['images'] = puppeteer_result.get('images', [])
                
                # Try to extract age from the title or description
                if puppeteer_result.get('title'):
                    age_match = re.search(r'\b(\d{2})\s*(?:yo|year|yr)s?\b', 
                                         puppeteer_result['title'] + ' ' + (puppeteer_result.get('description') or ''))
                    if age_match:
                        details['age'] = age_match.group(1)
                
                # Return early if we have good data
                if details['full_description'] and details['images']:
                    return details
        except Exception as e:
            logging.warning(f"Error using puppeteer crawler for listing details: {str(e)}")
            # Continue with traditional method if puppeteer fails

        # Fallback to traditional method
        try:
            logging.info(f"Falling back to traditional method for listing details from {url}")
            # Use proxy URL for traditional method too
            proxied_url = self._proxy_url(url)
            logging.info(f"Using proxy URL for listing details: {proxied_url}")
            
            async with await self._get_client() as client:
                try:
                    response = await client.get(proxied_url, headers=self._update_headers(url))
                    await self._update_cookies(response)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract full description from the ad details
                    desc_elem = soup.select_one('div.ad-content__description, div.posting__description')
                    
                    # Get age if available
                    age_elem = soup.select_one('span.age')
                    if age_elem:
                        details['age'] = age_elem.get_text(strip=True)
                    
                    # Get reply count
                    reply_elem = soup.select_one('span.reply-count')
                    if reply_elem:
                        try:
                            details['reply_count'] = int(reply_elem.get_text(strip=True))
                        except ValueError:
                            details['reply_count'] = 0
                    
                    # Get ad ID
                    ad_id_elem = soup.select_one('span.ad-id')
                    if ad_id_elem:
                        details['ad_id'] = ad_id_elem.get_text(strip=True)
                    
                    # Extract full description
                    if desc_elem:
                        details['full_description'] = desc_elem.get_text(strip=True)

                    # Extract contact information from the contact section
                    contact_elem = soup.select_one('div.contact-box, div.contact-info')
                    if contact_elem:
                        details['contact_info'] = contact_elem.get_text(strip=True)

                    # Extract poster information from the user section
                    poster_elem = soup.select_one('div.user-info, div.poster-info')
                    if poster_elem:
                        details['poster_info'] = poster_elem.get_text(strip=True)
                        
                    # Extract images
                    img_elems = soup.select('div.gallery img, div.ad-gallery img')
                    for img in img_elems:
                        if 'src' in img.attrs:
                            img_url = img['src']
                            if not img_url.startswith('http'):
                                img_url = urljoin(url, img_url)
                            details['images'].append(img_url)
                except Exception as e:
                    logging.error(f"Error in traditional method for listing details: {str(e)}")
        except Exception as e:
            logging.error(f"Error getting listing details: {str(e)}")

        return details

    def _proxy_url(self, url: str) -> str:
        """Wrap a URL with the untaint.us proxy.
        
        Args:
            url: The original URL to proxy
            
        Returns:
            The proxied URL
        """
        return f"https://please.untaint.us/?url={urllib.parse.quote_plus(url)}"
    
    async def locanto_search(self, category_path: List[str] = ['personals', 'men-seeking-men'], location: str = 'western-cape', max_pages: int = 3) -> List[LocantoListing]:
        """Search Locanto.co.za for listings in a specific category and location.
        
        Args:
            category_path: List of category segments to search in (default: ['personals', 'men-seeking-men'])
            location: The location to search in (default: 'western-cape')
            max_pages: Maximum number of pages to scrape (default: 3)
            
        Returns:
            List of LocantoListing objects containing the scraped data
        """
        # Construct the URL based on category path
        category_url = '/'.join(category_path)
        base_url = f'https://locanto.co.za/{location}/{category_url}/'
        listings: List[LocantoListing] = []
        
        for page in range(1, max_pages + 1):
            url = f'{base_url}?page={page}' if page > 1 else base_url
            logging.info(f"Searching Locanto page {page} at URL: {url}")
            
            # Try puppeteer crawler first for better results
            try:
                # Use untaint.us proxy to avoid Cloudflare restrictions
                proxy_url = f"https://please.untaint.us/?url={urllib.parse.quote_plus(url)}"
                logging.info(f"Using puppeteer crawler with proxy: {proxy_url}")
                
                puppeteer_result = await crawl_page(
                    proxy_url,
                    wait_selector=".resultlist__listing, .posting_listing",
                    timeout=45000,
                    extract_text=False
                )
                
                # Check if we got a collection of listings
                if isinstance(puppeteer_result, list) and len(puppeteer_result) > 0:
                    logging.info(f"Successfully extracted {len(puppeteer_result)} listings using puppeteer crawler")
                    
                    # Process each listing from puppeteer results
                    for item in puppeteer_result:
                        if not isinstance(item, dict) or 'listing_url' not in item:
                            continue
                            
                        try:
                            # Get detailed information for this listing
                            details = await self.get_listing_details(item['listing_url'])
                            
                            # Create listing object combining puppeteer data and details
                            listing: LocantoListing = {
                                'title': item.get('title', ''),
                                'description': item.get('description', ''),
                                'location': item.get('location', ''),
                                'price': '',  # Puppeteer doesn't extract price
                                'date_posted': '',  # Puppeteer doesn't extract date
                                'url': item['listing_url'],
                                'images': item.get('images', []) or details.get('images', []),
                                'contact_info': details.get('contact_info'),
                                'poster_info': details.get('poster_info'),
                                'full_description': details.get('full_description'),
                                'category_path': category_path,
                                'age': details.get('age'),
                                'reply_count': details.get('reply_count'),
                                'ad_id': details.get('ad_id')
                            }
                            
                            listings.append(listing)
                        except Exception as e:
                            logging.error(f"Error processing puppeteer listing: {str(e)}")
                            continue
                    
                    # If we got results, continue to next page
                    if len(puppeteer_result) > 0:
                        await asyncio.sleep(1)  # Small delay between pages
                        continue
            except Exception as e:
                logging.warning(f"Error using puppeteer crawler for search page {page}: {str(e)}")
                # Fall back to traditional method
            
            # Traditional method as fallback
            try:
                logging.info(f"Falling back to traditional method for search page {page}")
                # Use proxy URL for traditional method too
                proxied_url = self._proxy_url(url)
                logging.info(f"Using proxy URL: {proxied_url}")
                
                async with await self._get_client() as client:
                    response = await client.get(proxied_url, headers=self._update_headers(url))
                    await self._update_cookies(response)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find all listing containers (try different selectors)
                    listing_containers = soup.select('div.resultlist__listing, article.posting_listing')
                    logging.info(f"Found {len(listing_containers)} listing containers with traditional method")
                    
                    for container in listing_containers:
                        try:
                            # Extract listing details
                            title_elem = container.select_one('h3.resultlist__title a, a.posting_listing__title')
                            title = title_elem.get_text(strip=True) if title_elem else ''
                            listing_url = urljoin(base_url, title_elem['href']) if title_elem and 'href' in title_elem.attrs else ''
                            
                            if not listing_url:
                                continue
                            
                            description = ''
                            desc_elem = container.select_one('div.resultlist__description, div.posting_listing__description')
                            if desc_elem:
                                description = desc_elem.get_text(strip=True)
                            
                            location_val = ''
                            loc_elem = container.select_one('div.resultlist__location, span.posting_listing__city')
                            if loc_elem:
                                location_val = loc_elem.get_text(strip=True)
                            
                            price = ''
                            price_elem = container.select_one('span.resultlist__price, span.posting_listing__price')
                            if price_elem:
                                price = price_elem.get_text(strip=True)
                            
                            date_posted = ''
                            date_elem = container.select_one('time.resultlist__date, span.posting_listing__date')
                            if date_elem:
                                date_posted = date_elem.get_text(strip=True)
                            
                            # Extract images
                            images = []
                            img_elems = container.select('img.resultlist__image, img.posting_listing__image')
                            for img in img_elems:
                                if 'src' in img.attrs:
                                    img_url = img['src']
                                    if not img_url.startswith('http'):
                                        img_url = urljoin(base_url, img_url)
                                    images.append(img_url)
                            
                            # Get detailed information for this listing
                            details = await self.get_listing_details(listing_url)

                            listing: LocantoListing = {
                                'title': title,
                                'description': description,
                                'location': location_val,
                                'price': price,
                                'date_posted': date_posted,
                                'url': listing_url,
                                'images': images or details.get('images', []),
                                'contact_info': details.get('contact_info'),
                                'poster_info': details.get('poster_info'),
                                'full_description': details.get('full_description'),
                                'category_path': category_path,
                                'age': details.get('age'),
                                'reply_count': details.get('reply_count'),
                                'ad_id': details.get('ad_id')
                            }
                            
                            listings.append(listing)
                            
                        except Exception as e:
                            logging.error(f"Error processing listing: {str(e)}")
                            continue
            except Exception as e:
                logging.error(f"Error fetching page {page}: {str(e)}")
                break
            
            # Small delay between pages to be respectful
            await asyncio.sleep(1)
        
        return listings


    async def locanto_search_by_category(self, category_path: List[str] = ['personals', 'men-seeking-men'], location: str = 'western-cape', max_pages: int = 3) -> List[LocantoListing]:
        """Search Locanto.co.za for listings in a specific category and location.
        
        Args:
            category: The category to search in (default: 'personals')
            location: The location to search in (default: 'western-cape')
            max_pages: Maximum number of pages to scrape (default: 3)
            
        Returns:
            List of LocantoListing objects containing the scraped data
        """
        # Construct the URL based on category path
        category_url = '/'.join(category_path)
        base_url = f'https://locanto.co.za/{location}/{category_url}/'
        listings: List[LocantoListing] = []
        
        for page in range(1, max_pages + 1):
            url = f'{base_url}?page={page}' if page > 1 else base_url
            try:
                response = await self.client.get(url, headers=self._update_headers(url))
                await self._update_cookies(response)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find all listing containers
                listing_containers = soup.select('div.resultlist__listing')
                
                for container in listing_containers:
                    try:
                        # Extract listing details
                        title_elem = container.select_one('h3.resultlist__title a')
                        title = title_elem.get_text(strip=True) if title_elem else ''
                        url = urljoin(base_url, title_elem['href']) if title_elem else ''
                        
                        description = ''
                        desc_elem = container.select_one('div.resultlist__description')
                        if desc_elem:
                            description = desc_elem.get_text(strip=True)
                        
                        location = ''
                        loc_elem = container.select_one('div.resultlist__location')
                        if loc_elem:
                            location = loc_elem.get_text(strip=True)
                        
                        price = ''
                        price_elem = container.select_one('span.resultlist__price')
                        if price_elem:
                            price = price_elem.get_text(strip=True)
                        
                        date_posted = ''
                        date_elem = container.select_one('time.resultlist__date')
                        if date_elem:
                            date_posted = date_elem.get_text(strip=True)
                        
                        images = []
                        img_elems = container.select('img.resultlist__image')
                        for img in img_elems:
                            if 'src' in img.attrs:
                                img_url = urljoin(base_url, img['src'])
                                images.append(img_url)
                        
                        # Get detailed information for this listing
                        details = await self.get_listing_details(url)

                        listing: LocantoListing = {
                            'title': title,
                            'description': description,
                            'location': location,
                            'price': price,
                            'date_posted': date_posted,
                            'url': url,
                            'images': images,
                            'contact_info': details['contact_info'],
                            'poster_info': details['poster_info'],
                            'full_description': details['full_description'],
                            'category_path': category_path,
                            'age': details.get('age'),
                            'reply_count': details.get('reply_count'),
                            'ad_id': details.get('ad_id')
                        }
                        
                        listings.append(listing)
                        
                    except Exception as e:
                        print(f"Error processing listing: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error fetching page {page}: {str(e)}")
                break
            
            # Small delay between pages to be respectful
            await asyncio.sleep(1)
        
        return listings

class LocantoScraper:
    @function_tool
    async def get_categories(self, base_url: str) -> List[LocantoCategory]:
        """Get available categories from a Locanto page.

        Args:
            base_url: The URL to get categories from

        Returns:
            List of LocantoCategory objects
        """
        categories: List[LocantoCategory] = []
        async with await self._get_client() as client:
            try:
                response = await client.get(base_url, headers=self._update_headers(base_url))
                await self._update_cookies(response)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find category links in the sidebar navigation
                category_elements = soup.select('nav.sidebar a[href*="/c/"]')
                for elem in category_elements:
                    name = elem.get_text(strip=True)
                    url = urljoin(base_url, elem['href'])
                    count_elem = elem.find('span', class_='count')
                    count = int(count_elem.get_text(strip=True)) if count_elem else 0
                    
                    categories.append({
                        'name': name,
                        'url': url,
                        'count': count
                    })

            except Exception as e:
                print(f"Error getting categories: {str(e)}")

        return categories

async def get_listing_details(self, url: str) -> Dict[str, Any]:
    """Get detailed information from a single listing page.

    Args:
        url: The URL of the listing to scrape

    Returns:
        Dictionary containing detailed listing information
    """
    details = {
        'contact_info': None,
        'poster_info': None,
        'full_description': None
    }

    async with await self._get_client() as client:
        try:
            response = await client.get(url, headers=self._update_headers(url))
            await self._update_cookies(response)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract full description from the ad details
            desc_elem = soup.select_one('div.ad-content__description')
            
            # Get age if available
            age_elem = soup.select_one('span.age')
            if age_elem:
                details['age'] = age_elem.get_text(strip=True)
            
            # Get reply count
            reply_elem = soup.select_one('span.reply-count')
            if reply_elem:
                try:
                    details['reply_count'] = int(reply_elem.get_text(strip=True))
                except ValueError:
                    details['reply_count'] = 0
            
            # Get ad ID
            ad_id_elem = soup.select_one('span.ad-id')
            if ad_id_elem:
                details['ad_id'] = ad_id_elem.get_text(strip=True)
            
            # Extract full description
            if desc_elem:
                details['full_description'] = desc_elem.get_text(strip=True)

            # Extract contact information from the contact section
            contact_elem = soup.select_one('div.contact-box')
            if contact_elem:
                details['contact_info'] = contact_elem.get_text(strip=True)

            # Extract poster information from the user section
            poster_elem = soup.select_one('div.user-info')
            if poster_elem:
                details['poster_info'] = poster_elem.get_text(strip=True)
        except Exception as e:
            print(f"Error getting listing details: {str(e)}")

    return details

    @function_tool
    async def search_locanto(self, context: RunContext, query: str, location: str = "", category: str = "") -> str:
        """Search Locanto listings.
        
        Args:
            context: The run context for the tool
            query: The search query
            location: Optional location filter
            category: Optional category filter
            
        Returns:
            str: Formatted search results
        """
        try:
            # Build the search URL
            base_url = "https://www.locanto.com/search"
            params = {
                'q': query,
                'button': '',
                'submit=1': '',
            }
            
            if location:
                params['loc'] = location
            if category:
                params['category'] = category
            
            # Use the web_search tool to get results
            search_url = f"{base_url}?{urllib.parse.urlencode(params)}"
            return await self.scrape_website(context, search_url, ".regular-ad", text_only=True)
            
        except Exception as e:
            logging.error(f"Locanto search error: {e}", exc_info=True)
            return f"I couldn't search Locanto: {str(e)}"

    async def locanto_search_by_category(self, category_path: List[str] = ['personals', 'men-seeking-men'], location: str = 'western-cape', max_pages: int = 3) -> List[LocantoListing]:
        """Search Locanto.co.za for listings in a specific category and location.
        
        Args:
            category: The category to search in (default: 'personals')
            location: The location to search in (default: 'western-cape')
            max_pages: Maximum number of pages to scrape (default: 3)
            
        Returns:
            List of LocantoListing objects containing the scraped data
        """
        # Construct the URL based on category path
        category_url = '/'.join(category_path)
        base_url = f'https://locanto.co.za/{location}/{category_url}/'
        listings: List[LocantoListing] = []
        
        for page in range(1, max_pages + 1):
            url = f'{base_url}?page={page}' if page > 1 else base_url
            try:
                response = await self.client.get(url, headers=self._update_headers(url))
                await self._update_cookies(response)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find all listing containers
                listing_containers = soup.select('div.resultlist__listing')
                
                for container in listing_containers:
                    try:
                        # Extract listing details
                        title_elem = container.select_one('h3.resultlist__title a')
                        title = title_elem.get_text(strip=True) if title_elem else ''
                        url = urljoin(base_url, title_elem['href']) if title_elem else ''
                        
                        description = ''
                        desc_elem = container.select_one('div.resultlist__description')
                        if desc_elem:
                            description = desc_elem.get_text(strip=True)
                        
                        location = ''
                        loc_elem = container.select_one('div.resultlist__location')
                        if loc_elem:
                            location = loc_elem.get_text(strip=True)
                        
                        price = ''
                        price_elem = container.select_one('span.resultlist__price')
                        if price_elem:
                            price = price_elem.get_text(strip=True)
                        
                        date_posted = ''
                        date_elem = container.select_one('time.resultlist__date')
                        if date_elem:
                            date_posted = date_elem.get_text(strip=True)
                        
                        images = []
                        img_elems = container.select('img.resultlist__image')
                        for img in img_elems:
                            if 'src' in img.attrs:
                                img_url = urljoin(base_url, img['src'])
                                images.append(img_url)
                        
                        # Get detailed information for this listing
                        details = await self.get_listing_details(url)

                        listing: LocantoListing = {
                            'title': title,
                            'description': description,
                            'location': location,
                            'price': price,
                            'date_posted': date_posted,
                            'url': url,
                            'images': images,
                            'contact_info': details['contact_info'],
                            'poster_info': details['poster_info'],
                            'full_description': details['full_description'],
                            'category_path': category_path,
                            'age': details.get('age'),
                            'reply_count': details.get('reply_count'),
                            'ad_id': details.get('ad_id')
                        }
                        
                        listings.append(listing)
                        
                    except Exception as e:
                        print(f"Error processing listing: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error fetching page {page}: {str(e)}")
                break
            
            # Small delay between pages to be respectful
            await asyncio.sleep(1)
        
        return listings


    async def locanto_search_by_category(self, category_path: List[str] = ['personals', 'men-seeking-men'], location: str = 'western-cape', max_pages: int = 3) -> List[LocantoListing]:
        """Search Locanto.co.za for listings in a specific category and location.
        
        Args:
            category: The category to search in (default: 'personals')
            location: The location to search in (default: 'western-cape')
            max_pages: Maximum number of pages to scrape (default: 3)
            
        Returns:
            List of LocantoListing objects containing the scraped data
        """
        # Construct the URL based on category path
        category_url = '/'.join(category_path)
        base_url = f'https://locanto.co.za/{location}/{category_url}/'
        listings: List[LocantoListing] = []
        
        for page in range(1, max_pages + 1):
            url = f'{base_url}?page={page}' if page > 1 else base_url
            try:
                response = await self.client.get(url, headers=self._update_headers(url))
                await self._update_cookies(response)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find all listing containers
                listing_containers = soup.select('div.resultlist__listing')
                
                for container in listing_containers:
                    try:
                        # Extract listing details
                        title_elem = container.select_one('h3.resultlist__title a')
                        title = title_elem.get_text(strip=True) if title_elem else ''
                        url = urljoin(base_url, title_elem['href']) if title_elem else ''
                        
                        description = ''
                        desc_elem = container.select_one('div.resultlist__description')
                        if desc_elem:
                            description = desc_elem.get_text(strip=True)
                        
                        location = ''
                        loc_elem = container.select_one('div.resultlist__location')
                        if loc_elem:
                            location = loc_elem.get_text(strip=True)
                        
                        price = ''
                        price_elem = container.select_one('span.resultlist__price')
                        if price_elem:
                            price = price_elem.get_text(strip=True)
                        
                        date_posted = ''
                        date_elem = container.select_one('time.resultlist__date')
                        if date_elem:
                            date_posted = date_elem.get_text(strip=True)
                        
                        images = []
                        img_elems = container.select('img.resultlist__image')
                        for img in img_elems:
                            if 'src' in img.attrs:
                                img_url = urljoin(base_url, img['src'])
                                images.append(img_url)
                        
                        # Get detailed information for this listing
                        details = await self.get_listing_details(url)

                        listing: LocantoListing = {
                            'title': title,
                            'description': description,
                            'location': location,
                            'price': price,
                            'date_posted': date_posted,
                            'url': url,
                            'images': images,
                            'contact_info': details['contact_info'],
                            'poster_info': details['poster_info'],
                            'full_description': details['full_description'],
                            'category_path': category_path,
                            'age': details.get('age'),
                            'reply_count': details.get('reply_count'),
                            'ad_id': details.get('ad_id')
                        }
                        
                        listings.append(listing)
                        
                    except Exception as e:
                        print(f"Error processing listing: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error fetching page {page}: {str(e)}")
                break
            
            # Small delay between pages to be respectful
            await asyncio.sleep(1)
        
        return listings

# --- Web Tools ---
@function_tool
async def basic_search_locanto(self, context: RunContext, query: str, location: str = "", category: str = "") -> str:
    """Search Locanto listings.
    
    Args:
        context: The run context for the tool
        query: The search query
        location: Optional location filter
        category: Optional category filter
        
    Returns:
        str: Formatted search results
    """
    try:
        # Build the search URL
        base_url = "https://www.locanto.com/search"
        params = {
            'q': query,
            'button': '',
            'submit=1': '',
        }
        
        if location:
            params['loc'] = location
        if category:
            params['category'] = category
        
        # Use the web_search tool to get results
        search_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        return await self.scrape_website(context, search_url, ".regular-ad", text_only=True)
        
    except Exception as e:
        logging.error(f"Locanto search error: {e}", exc_info=True)
        return f"I couldn't search Locanto: {str(e)}"

@function_tool
async def search_locanto(context: RunContext, category_path: str = 'personals/men-seeking-men', location: str = 'western-cape', max_pages: int = 3, return_url: bool = False) -> str:
    try:
        import html
        ddg_query = f"{category_path.replace('/', ' ')} {location} site:locanto.co.za"
        import asyncio
        loop = asyncio.get_event_loop()
        results = []
        assistant = AIVoiceAssistant()
        categories = category_path.split('/')
        listings = await assistant.locanto_search(categories, location, max_pages)
        if not listings:
            summary = "No listings found matching your criteria."
        else:
            first_url = None
            url_map = {}
            summary = f"Found {len(listings)} listings on Locanto:\n\n"
            for idx, listing in enumerate(listings, 1):
                if not isinstance(listing, dict):
                    continue
                title = listing.get('title', 'No title')
                title = clean_spoken(title)
                summary += f"{idx}. {title}\n"
                ad_id = listing.get('ad_id')
                if ad_id:
                    summary += f"Ad ID: {clean_spoken(str(ad_id))}\n"
                age = listing.get('age')
                if age:
                    summary += f"Age: {clean_spoken(str(age))}\n"
                category_path = listing.get('category_path', [])
                if category_path:
                    summary += f"Category: {clean_spoken(' > '.join(category_path))}\n"
                price = listing.get('price')
                if price:
                    summary += f"Price: {clean_spoken(str(price))}\n"
                loc = listing.get('location')
                if loc:
                    summary += f"Location: {clean_spoken(str(loc))}\n"
                date_posted = listing.get('date_posted')
                if date_posted:
                    summary += f"Posted: {clean_spoken(str(date_posted))}\n"
                url = listing.get('url')
                if url:
                    url_map[idx] = url
                    if not first_url:
                        first_url = url
                contact_info = listing.get('contact_info')
                if contact_info:
                    summary += f"Contact: {clean_spoken(str(contact_info))}\n"
                poster_info = listing.get('poster_info')
                if poster_info:
                    summary += f"Poster: {clean_spoken(str(poster_info))}\n"
                reply_count = listing.get('reply_count')
                if reply_count is not None:
                    summary += f"Replies: {clean_spoken(str(reply_count))}\n"
                description = listing.get('description')
                if description:
                    desc = description[:200] + '...' if len(description) > 200 else description
                    summary += f"Description: {clean_spoken(desc)}\n"
                summary += "\n"
            # Store mapping in session.userdata for later use
            session = getattr(context, 'session', None)
            if session is not None:
                session.userdata['last_locanto_urls'] = url_map
            if first_url and return_url:
                return first_url
            if first_url:
                summary += f"Would you like to open the first listing in your browser?"
            if session:
                await handle_tool_results(session, summary)
                return "I've found some results and will read them to you now."
            else:
                return summary
        summary = sanitize_for_azure(summary)
        summary = clean_spoken(summary)
        logging.info(f"[TOOL] search_locanto summary: {summary}")
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, summary)
            return "I've found some results and will read them to you now."
        else:
            return summary
    except Exception as e:
        logging.error(f"[TOOL] search_locanto exception: {e}")
        return sanitize_for_azure(f"Sorry, there was a problem searching Locanto: {e}")

@function_tool
async def search_locanto_browser(context: RunContext, query: str = "dating", location: str = "Cape Town", max_pages: int = 1, tag: str = None, category: str = None, section: str = None, url: str = None, return_url: bool = False) -> str:
    try:
        from locanto_browser_scraper import search_locanto_browser, LocantoBrowserScraper
        loc_valid = is_valid_locanto_location(location)
        cat_valid = is_valid_locanto_category(category) if category else True
        sec_valid = is_valid_locanto_section(section) if section else True
        if tag and not is_valid_locanto_tag(tag):
            suggestions = suggest_closest_slug(tag, LOCANTO_TAG_SLUGS)
            msg = f"Tag '{tag}' is not valid. Did you mean: {', '.join(suggestions)}?" if suggestions else f"Tag '{tag}' is not valid. Please choose from: {', '.join(sorted(list(LOCANTO_TAG_SLUGS))[:10])}..."
            msg = sanitize_for_azure(msg)
            msg = clean_spoken(msg)
            logging.info(f"[TOOL] search_locanto_browser tag error: {msg}")
            return msg
        suggestions = []
        if location and not loc_valid:
            suggestions = suggest_closest_slug(location, LOCANTO_LOCATION_SLUGS)
            msg = f"Location '{location}' is not valid. Did you mean: {', '.join(suggestions)}?" if suggestions else f"Location '{location}' is not valid. Please choose from: {', '.join(sorted(list(LOCANTO_LOCATION_SLUGS))[:10])}..."
            msg = sanitize_for_azure(msg)
            msg = clean_spoken(msg)
            logging.info(f"[TOOL] search_locanto_browser location error: {msg}")
            return msg
        if category and not cat_valid:
            suggestions = suggest_closest_slug(category, LOCANTO_CATEGORY_SLUGS)
            msg = f"Category '{category}' is not valid. Did you mean: {', '.join(suggestions)}?" if suggestions else f"Category '{category}' is not valid. Please choose from: {', '.join(sorted(list(LOCANTO_CATEGORY_SLUGS))[:10])}..."
            msg = sanitize_for_azure(msg)
            msg = clean_spoken(msg)
            logging.info(f"[TOOL] search_locanto_browser category error: {msg}")
            return msg
        if section and not sec_valid:
            suggestions = suggest_closest_slug(section, LOCANTO_SECTION_IDS)
            msg = f"Section '{section}' is not valid. Did you mean: {', '.join(suggestions)}?" if suggestions else f"Section '{section}' is not valid. Please choose from: {', '.join(sorted(list(LOCANTO_SECTION_IDS))[:10])}..."
            msg = sanitize_for_azure(msg)
            msg = clean_spoken(msg)
            logging.info(f"[TOOL] search_locanto_browser section error: {msg}")
            return msg
        scraper = LocantoBrowserScraper()
        listings = await scraper.search_listings(query=query, location=location, max_pages=max_pages, tag=tag, category=category, section=section, url=url)
        first_url = None
        url_map = {}
        if not listings:
            summary = f"No Locanto listings found for '{query}' in '{location}'."
        elif isinstance(listings[0], dict) and 'error' in listings[0]:
            summary = f"Error: {listings[0]['error']}"
        else:
            summary = f"Found {len(listings)} Locanto listings for '{query}' in '{location}':\n\n"
            for idx, listing in enumerate(listings[:5], 1):
                if not isinstance(listing, dict):
                    continue
                title = listing.get('title', 'No title')
                title = clean_spoken(title)
                summary += f"{idx}. {title}\n"
                age = listing.get('age')
                if age:
                    summary += f"   Age: {clean_spoken(str(age))}\n"
                loc = listing.get('location')
                if loc:
                    summary += f"   Location: {clean_spoken(str(loc))}\n"
                description = listing.get('description')
                if description:
                    desc = description[:120] + ('...' if len(description) > 120 else '')
                    summary += f"   Description: {clean_spoken(desc)}\n"
                url = listing.get('url')
                if url:
                    url_map[idx] = url
                    if not first_url:
                        first_url = url
                summary += "\n"
            # Store mapping in session.userdata for later use
            session = getattr(context, 'session', None)
            if session is not None:
                session.userdata['last_locanto_urls'] = url_map
            if first_url:
                if return_url:
                    return first_url
                summary += f"Would you like to open the first listing in your browser?"
        summary = sanitize_for_azure(summary)
        summary = clean_spoken(summary)
        logging.info(f"[TOOL] search_locanto_browser summary: {summary}")
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, summary)
            return "I've found some results and will read them to you now."
        else:
            return summary
    except Exception as e:
        logging.error(f"[TOOL] search_locanto_browser exception: {e}")
        return sanitize_for_azure(f"Error searching Locanto with Playwright: {e}")

@function_tool
async def locanto_matchmaking(
    context: RunContext,
    query: Optional[str] = None,  # parameter1
    gender: Optional[str] = None,
    seeking: Optional[str] = None,
    age: Optional[str] = None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    query_description: Optional[bool] = None,
    location: Optional[str] = None,
    tag: Optional[str] = None,
    category: Optional[str] = None,
    section: Optional[str] = None,
    dist: Optional[int] = None,
    sort: Optional[str] = None,
    max_pages: int = 1,
    return_url: bool = False
) -> str:
    try:
        from locanto_browser_scraper import LocantoBrowserScraper
        import urllib.parse
        missing = []
        if not query:
            missing.append("search term (e.g. companion, relationship, love, etc.)")
        if not location:
            missing.append("location (e.g. randburg, johannesburg, etc.)")
        if age_max is None:
            missing.append("maximum age (age_max)")
        if missing:
            summary = ("To find matches, please provide: " + ', '.join(missing) + ". "
                    "For example: 'Find companionship in Randburg, max age 40'.")
            summary = sanitize_for_azure(summary)
            summary = clean_spoken(summary)
            logging.info(f"[TOOL] locanto_matchmaking missing params: {summary}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, summary)
                return "I'll read the requirements to you."
            else:
                return summary
        unsafe_terms = {"sex", "nsa", "hookup", "hookups", "anal", "blowjob", "quickie", "incalls", "outcalls", "massage", "MILF"}
        if any(term in query.lower() for term in unsafe_terms):
            summary = ("For your safety, please use respectful, safe-for-work search terms. "
                    "Try words like 'companionship', 'relationship', 'friendship', or 'meeting people'.")
            summary = sanitize_for_azure(summary)
            summary = clean_spoken(summary)
            logging.info(f"[TOOL] locanto_matchmaking unsafe terms: {summary}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, summary)
                return "I'll read the safety message to you."
            else:
                return summary
        if age_min is None:
            age_min = 18
        if dist is None:
            dist = 30
        if sort is None:
            sort = "date"
        if category is None:
            category = "Personals"
        if section is None:
            section = "P"
        if query_description is None:
            query_description = True
        location_slug = location.lower().replace(' ', '-')
        loc_valid = is_valid_locanto_location(location_slug)
        if location and not loc_valid:
            suggestions = suggest_closest_slug(location, LOCANTO_LOCATION_SLUGS)
            summary = f"Location '{location}' is not valid. Did you mean: {', '.join(suggestions)}?" if suggestions else f"Location '{location}' is not valid. Please choose from: {', '.join(sorted(list(LOCANTO_LOCATION_SLUGS))[:10])}..."
            summary = sanitize_for_azure(summary)
            summary = clean_spoken(summary)
            logging.info(f"[TOOL] locanto_matchmaking location error: {summary}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, summary)
                return "I'll read the location suggestion to you."
            else:
                return summary
        cat_valid = is_valid_locanto_category(category) if category else True
        if category and not cat_valid:
            suggestions = suggest_closest_slug(category, LOCANTO_CATEGORY_SLUGS)
            summary = f"Category '{category}' is not valid. Did you mean: {', '.join(suggestions)}?" if suggestions else f"Category '{category}' is not valid. Please choose from: {', '.join(sorted(list(LOCANTO_CATEGORY_SLUGS))[:10])}..."
            summary = sanitize_for_azure(summary)
            summary = clean_spoken(summary)
            logging.info(f"[TOOL] locanto_matchmaking category error: {summary}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, summary)
                return "I'll read the category suggestion to you."
            else:
                return summary
        sec_valid = is_valid_locanto_section(section) if section else True
        if section and not sec_valid:
            suggestions = suggest_closest_slug(section, LOCANTO_SECTION_IDS)
            summary = f"Section '{section}' is not valid. Did you mean: {', '.join(suggestions)}?" if suggestions else f"Section '{section}' is not valid. Please choose from: {', '.join(sorted(list(LOCANTO_SECTION_IDS))[:10])}..."
            summary = sanitize_for_azure(summary)
            summary = clean_spoken(summary)
            logging.info(f"[TOOL] locanto_matchmaking section error: {summary}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, summary)
                return "I'll read the section suggestion to you."
            else:
                return summary
        scraper = LocantoBrowserScraper()
        listings = await scraper.search_listings(
            query=query,
            location=location_slug,
            age_min=age_min,
            age_max=age_max,
            query_description=query_description,
            dist=dist,
            sort=sort,
            category=category,
            section=section,
            max_pages=max_pages
        )
        summary = f"{query} in {location_slug}\n\n"
        first_url = None
        url_map = {}
        if not listings or (isinstance(listings[0], dict) and 'error' in listings[0]):
            debug_url = listings[0].get('_debug_url') if listings and isinstance(listings[0], dict) else None
            debug_proxied_url = listings[0].get('_debug_proxied_url') if listings and isinstance(listings[0], dict) else None
            debug_msg = ""
            if debug_url:
                debug_msg += f"\n[DEBUG] Search URL: {debug_url}"
            if debug_proxied_url:
                debug_msg += f"\n[DEBUG] Proxied URL: {debug_proxied_url}"
            summary += f"No results found. {debug_msg.strip()}"
        else:
            for idx, listing in enumerate(listings, 1):
                if not isinstance(listing, dict):
                    continue
                title = listing.get('title', 'No title')
                title = clean_spoken(title)
                location_val = listing.get('location', '')
                age = listing.get('age', '')
                desc = listing.get('description', '')
                url = listing.get('url', '')
                if url:
                    url_map[idx] = url
                    if not first_url:
                        first_url = url
                summary += f"{idx}. {title} ({clean_spoken(location_val)}, {clean_spoken(str(age))})\n{clean_spoken(desc)}\n"
            # Store mapping in session.userdata for later use
            session = getattr(context, 'session', None)
            if session is not None:
                try:
                    session.userdata['last_locanto_urls'] = url_map
                except Exception as e:
                    logging.warning(f"Could not set session.userdata['last_locanto_urls']: {e}")
            if first_url:
                if return_url:
                    return first_url
                summary += f"Would you like to open the first listing in your browser?"
        summary = sanitize_for_azure(summary)
        summary = clean_spoken(summary)
        logging.info(f"[TOOL] locanto_matchmaking summary: {summary}")
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, summary)
            return "I've found some results and will read them to you now."
        else:
            return summary
    except Exception as e:
        import traceback
        logging.error(f"locanto_matchmaking failed: {e}")
        logging.error(traceback.format_exc())
        summary = ("Sorry, I can't help with that request right now. "
                "Please try rephrasing, or ask for help with meeting people, making friends, or finding companionship.")
        summary = sanitize_for_azure(summary)
        summary = clean_spoken(summary)
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, summary)
            return "I'll read the error message to you."
        else:
            return summary

@function_tool
async def show_top_locanto_categories_and_tags(context: RunContext, location: str = None) -> str:
    from locanto_constants import LOCANTO_CATEGORY_SLUGS, LOCANTO_TAG_SLUGS
    try:
        from locanto_constants import LOCANTO_TAGS_BY_LOCATION
    except ImportError:
        LOCANTO_TAGS_BY_LOCATION = {}
    def humanize(slug):
        return slug.replace('-', ' ').replace('_', ' ')
    def clean_spoken(text):
        import re
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'#+', '', text)
        text = re.sub(r'[\*\_`~\[\]\(\)\>\!]', '', text)
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        # Add a period at the end of lists/multi-line outputs if not already present
        if text and not text.endswith('.'):
            # If the text is a list (multiple lines), add a period at the end
            if '\n' in text:
                text += '.'
        return text
    # Try to get location-specific tags
    tags = None
    location_slug = location.lower().replace(' ', '-') if location else None
    if location_slug and LOCANTO_TAGS_BY_LOCATION and location_slug in LOCANTO_TAGS_BY_LOCATION:
        tags = LOCANTO_TAGS_BY_LOCATION[location_slug]
    if tags:
        top_tags = [clean_spoken(sanitize_for_azure(humanize(tag))) for tag in tags[:10]]
        tag_note = f"Top Locanto tags in {location}:"
    else:
        top_tags = [clean_spoken(sanitize_for_azure(humanize(slug))) for slug in list(LOCANTO_TAG_SLUGS)[:10]]
        tag_note = "Top Locanto tags (global):"
    top_categories = [clean_spoken(sanitize_for_azure(humanize(slug))) for slug in list(LOCANTO_CATEGORY_SLUGS)[:10]]
    summary = "Top Locanto Categories:\n" + "\n".join(top_categories)
    summary += f"\n{tag_note}\n" + "\n".join(top_tags)
    summary = clean_spoken(sanitize_for_azure(summary))
    logging.info(f"[TOOL] show_top_locanto_categories_and_tags: {summary}")
    session = getattr(context, 'session', None)
    if session:
        await handle_tool_results(session, summary)
        return f"Here are the top Locanto categories and tags{' in ' + location if location else ''}. I'll read them to you."
    return summary    