# Standard library imports
import asyncio
import json
import logging
import os
import re
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union
from urllib.parse import quote_plus, urljoin

# Third-party imports
import brotli
import httpx
import pytz
import redis as redis_pkg
import uvicorn
import wikipediaapi
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from lxml_html_clean import Cleaner
from playwright.async_api import async_playwright
from pydantic import BaseModel
from googlesearch import search as google_search

# LiveKit imports
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.llm import ChatChunk, ChatContext, ChatMessage
from livekit.plugins import azure, openai, silero

# Local application imports
from mcp_client import MCPServerSse
from mcp_client.agent_tools import MCPToolsIntegration

load_dotenv()

class FunctionAgent(Agent):
    """A LiveKit agent that uses MCP tools from one or more MCP servers."""

    def __init__(self):
        super().__init__(
            instructions="""
                You are Amanda, an advanced AI assistant with access to a comprehensive set of tools and capabilities.
                Your primary goal is to be helpful, informative, and efficient in your responses.
                
                ===== AVAILABLE TOOLS =====
                
                [SEARCH & BROWSING]
                - web_search(query): Search the web for information
                - bing_web_search(query): Alternative search using Bing
                - wiki_lookup(topic): Get detailed information from Wikipedia
                - web_crawl(url, selector, max_pages): Extract content from web pages
                - open_website(url): Open a specific website
                - open_known_website(site_name, query): Open a well-known website (e.g., 'google', 'wikipedia')
                
                [LOCANTO INTEGRATION]
                - search_locanto(category_path, location, max_pages): Search Locanto listings
                - search_locanto_browser(query, location, max_pages, tag, category, section, url): Advanced Locanto search
                - locanto_matchmaking(query, gender, seeking, age, location, tag, category, section, max_pages): Find matches on Locanto
                - show_top_locanto_categories_and_tags(location): Browse Locanto categories
                
                [UTILITIES]
                - get_weather(location): Get current weather conditions
                - get_news_headlines(topic, country): Fetch latest news
                - calculate(expression): Evaluate mathematical expressions
                - get_current_datetime(): Get current date and time
                - get_fun_content(content_type): Get jokes, facts, or trivia
                - indeed_job_search(query, location): Search for jobs on Indeed
                
                [ADVANCED]
                - extract_links(url, filter_pattern): Extract links from a webpage
                - clean_html(html_content): Sanitize HTML content
                - take_screenshot(url, selector): Capture webpage screenshots
                
                ===== GUIDELINES =====
                1. TOOL SELECTION: Choose the most appropriate tool for each task
                2. EFFICIENCY: Use the most direct tool that can answer the query
                3. VERIFICATION: Cross-reference information when possible
                4. PRIVACY: Never share sensitive personal information
                5. ATTRIBUTION: Cite sources when using external information
                6. CLARITY: Ask for clarification if a request is unclear
                7. CONTEXT: Maintain context from previous interactions
                8. LIMITATIONS: Be clear about the limitations of your knowledge
                
                ===== BEST PRACTICES =====
                - For general knowledge: Use wiki_lookup() first
                - For current information: Use web_search() or get_news_headlines()
                - For calculations: Use calculate()
                - For Locanto-related queries: Use the appropriate Locanto tools
                - For website interaction: Use open_website() or web_crawl()
                
                Always present information in a clear, organized, and helpful manner.
                """,
            stt=azure.STT(
                speech_key=os.environ["AZURE_STT_API_KEY"],
                speech_region=os.environ["AZURE_STT_REGION"]
            ),
            llm=openai.LLM.with_azure(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                api_version=os.environ["AZURE_OPENAI_VERSION"],
                temperature=0.3,  # Lower temperature for more focused and consistent responses
            ),
            tts=azure.TTS(
                speech_key=os.environ["AZURE_TTS_API_KEY"],
                speech_region=os.environ["AZURE_TTS_REGION"]
            ),
            vad=silero.VAD.load(
                activation_threshold=0.4,  # Slightly lower threshold for better voice detection
                min_speech_duration=0.15,  # Shorter minimum speech duration (in seconds)
                min_silence_duration=0.5,  # Shorter silence duration for quicker response (in seconds)
                sample_rate=16000,  # Explicitly set sample rate
                force_cpu=True,  # Use CPU for better stability
                prefix_padding_duration=0.3  # Add small padding before speech
            ),
            allow_interruptions=True
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Override the llm_node to say a message when a tool call is detected."""
        activity = self._activity
        tool_call_detected = False

        # Get the original response from the parent class
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            # Check if this chunk contains a tool call
            if isinstance(chunk, ChatChunk) and chunk.delta and chunk.delta.tool_calls and not tool_call_detected:
                # Say the checking message only once when we detect the first tool call
                tool_call_detected = True
                # Use the chat context to send a text response that will be spoken
                chat_ctx.add_message(role="assistant", content="Sure, I'll check that for you.")

            yield chunk
            
# --- END: FunctionAgent ---

# --- BEGIN: TOOLS ---
# Import tools and utilities from tools.py
from .tools import (
    # Tool functions
    search_locanto,
    search_locanto_browser,
    locanto_matchmaking,
    web_crawl,
    web_search,
    get_current_datetime,
    wiki_lookup,
    get_news_headlines,
    get_weather,
    calculate,
    get_fun_content,
    show_top_locanto_categories_and_tags,
    open_website,
    indeed_job_search,
    open_known_website,
    bing_web_search,
    
    # Utility functions
    handle_tool_results,
    sanitize_for_azure,
    clean_spoken,
    get_current_date_and_timezone,
    is_sequence_but_not_str,
    
    # Data models
    LocantoCategory,
    LocantoListing,
    
    # Validation functions
    is_valid_locanto_location,
    is_valid_locanto_category,
    is_valid_locanto_section,
    is_valid_locanto_tag,
    suggest_closest_slug,
    
    # Constants and configurations
    WELL_KNOWN_WEBSITES,
    
    # Main class
    AIVoiceAssistant
)

# --- TOOL DEFINITIONS (single-source, top-level) ---
# All tools are now imported from tools.py

# Fallback URL for error cases
FALLBACK_URL = "https://fallback"

# Locanto configuration
LOCANTO_LOCATION_SLUGS = ["western-cape", "gauteng", "kwazulu-natal", "eastern-cape",
                         "free-state", "limpopo", "mpumalanga", "north-west", "northern-cape"]
LOCANTO_CATEGORY_SLUGS = ["personals", "men-seeking-men", "women-seeking-men", "men-seeking-women", "women-seeking-women"]
LOCANTO_SECTION_IDS = ["dating", "casual-encounters", "missed-connections", "friends"]
LOCANTO_TAG_SLUGS = ["gay", "lesbian", "bisexual", "transgender", "queer", "lgbtq+", "straight"]

# --- END: TOOLS ---

# --- BEGIN: entrypoint ---

class AIVoiceAssistant:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIVoiceAssistant, cls).__new__(cls)
            cls._instance.vad = None
            cls._instance.session = None
            cls._instance.agent = None
            # Initialize Wikipedia API
            cls._instance.wiki_wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                user_agent='AIVoiceAssistant/1.0'
            )
            # Initialize geocoder with a user agent for weather and location services
            cls._instance.geolocator = Nominatim(user_agent="AIVoiceAssistant/1.0")
            # Cache dictionaries
            cls._instance.weather_cache = {}  # Cache weather data
            cls._instance.wiki_cache = {}     # Cache wikipedia lookups
            cls._instance.news_cache = {}     # Cache news lookups
            cls._instance.crawl_cache = {}    # Cache web crawl results
            print("All search and lookup clients initialized")
        return cls._instance

    def initialize_vad(self, proc: JobProcess):
        """Initialize Voice Activity Detection with all relevant parameters from env vars"""
        if self.vad is None:
            import os
            threshold = float(os.environ.get("VAD_THRESHOLD", 0.5))
            min_speech = float(os.environ.get("VAD_MIN_SPEECH", 0.1))
            min_silence = float(os.environ.get("VAD_MIN_SILENCE", 0.5))
            debug = os.environ.get("VAD_DEBUG", "false").lower() in ("1", "true", "yes", "on")
            try:
                proc.userdata["vad"] = silero.VAD.load(
                    threshold=threshold,
                    min_speech_duration=min_speech,
                    min_silence_duration=min_silence,
                    debug=debug
                )
                print(f"[VAD] Loaded with threshold={threshold}, min_speech={min_speech}, min_silence={min_silence}, debug={debug}")
            except TypeError:
                # Fallback if silero.VAD.load does not accept these params
                proc.userdata["vad"] = silero.VAD.load()
                print(f"[VAD] Loaded with default params (full config not supported)")
            self.vad = proc.userdata["vad"]

    def setup_session(self, vad):
        """Setup agent session with all required components"""
        if self.session is None:
            self.session = AgentSession(
                vad=vad,
                stt=azure.STT(
                    speech_key=os.environ["AZURE_STT_API_KEY"],
                    speech_region=os.environ["AZURE_STT_REGION"]
                ),
                llm=openai.LLM.with_azure(
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                    api_version=os.environ["AZURE_OPENAI_VERSION"],
                    temperature=1,
                    parallel_tool_calls=True,
                    tool_choice="auto",
                    timeout=httpx.Timeout(connect=15.0, read=10.0, write=5.0, pool=5.0),
                    user="martin",
                    organization=os.environ.get("redbuilder"),
                    project=os.environ.get("kiki"),
                ),
                tts=azure.TTS(
                    speech_key=os.environ["AZURE_TTS_API_KEY"],
                    speech_region=os.environ["AZURE_TTS_REGION"]
                )
            )
        return self.session

    def __init__(self):
        """Initialize the AIVoiceAssistant with necessary components"""
        if not hasattr(self, '_instance'):
            self._instance = None
            self.vad = None
            self.session = None
            self.agent = None
            # Initialize Wikipedia API
            self.wiki_wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                user_agent='AIVoiceAssistant/1.0'
            )
            # Initialize geocoder
            self.geolocator = Nominatim(user_agent="AIVoiceAssistant/1.0")
            # Cache dictionaries
            self.weather_cache = {}
            self.wiki_cache = {}
            self.news_cache = {}
            self.crawl_cache = {}
            # Store cookies between requests
            self.cookies = {}
            # Default headers for HTTP requests
            self.default_headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'DNT': '1'
            }
            print("All search and lookup clients initialized")

    def _update_headers(self, url: str) -> Dict[str, str]:
        """Update headers for each request to maintain session-like behavior"""
        headers = self.client.headers.copy()
        headers.update({
            'Referer': url,
            'Cookie': '; '.join([f'{k}={v}' for k, v in self.cookies.items()])
        })
        return headers

    async def _update_cookies(self, response: httpx.Response) -> None:
        """Update stored cookies from response"""
        if 'set-cookie' in response.headers:
            for cookie in response.headers.getlist('set-cookie'):
                if '=' in cookie:
                    name, value = cookie.split('=', 1)
                    value = value.split(';')[0]
                    self.cookies[name] = value

    async def _get_client(self) -> httpx.AsyncClient:
        """Get an HTTP client for making requests"""
        return httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers=self.default_headers
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

    async def locanto_search(self, category_path: List[str] = ['personals', 'men-seeking-men'], location: str = 'western-cape', max_pages: int = 3) -> List[LocantoListing]:
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
    async def web_search(self, context: RunContext, query: str, num_results: int = 5) -> str:
        """Search the web for information using Google Search.
        
        Args:
            context: The run context for the tool
            query: The search query
            num_results: Number of results to return (1-10)
            
        Returns:
            str: Formatted search results with titles and URLs
        """
        try:
            num_results = max(1, min(10, int(num_results)))
            search_results = []
            
            for result in google_search(query, num_results=num_results):
                title = result.split('/')[-1].replace('-', ' ').title()
                search_results.append({
                    'title': title,
                    'url': result
                })
            
            if not search_results:
                return "I couldn't find any results for that query."
            
            response = f"Here are the top {len(search_results)} results for '{query}':\n\n"
            for i, result in enumerate(search_results, 1):
                response += f"{i}. {result['title']}\n   {result['url']}\n\n"
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Web search error: {e}", exc_info=True)
            return "I encountered an error while searching the web. Please try again later."
    
    @function_tool
    async def scrape_website(self, context: RunContext, url: str, selector: str = "body", text_only: bool = True) -> str:
        """Scrape content from a website using Playwright.
        
        Args:
            context: The run context for the tool
            url: The URL to scrape
            selector: CSS selector to target specific elements
            text_only: Whether to return only text content
            
        Returns:
            str: Extracted content from the webpage
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                # Set a reasonable timeout
                page.set_default_timeout(30000)  # 30 seconds
                
                # Navigate to the page
                response = await page.goto(url, wait_until="domcontentloaded")
                if not response or not response.ok:
                    return f"Failed to load {url}. Status: {response.status if response else 'No response'}"
                
                # Wait for the selector to be present
                try:
                    await page.wait_for_selector(selector, timeout=10000)
                except Exception as e:
                    logging.warning(f"Selector {selector} not found: {e}")
                
                # Extract content based on parameters
                if text_only:
                    content = await page.locator(selector).inner_text()
                else:
                    content = await page.locator(selector).inner_html()
                
                # Clean up
                await browser.close()
                
                # Clean and truncate the content
                content = ' '.join(content.split())
                if len(content) > 4000:  # Limit response length
                    content = content[:4000] + '... [content truncated]'
                
                return content
                
        except Exception as e:
            logging.error(f"Web scraping error: {e}", exc_info=True)
            return f"I encountered an error while scraping the website: {str(e)}"
    
    # --- Information Tools ---
    
    @function_tool
    async def wikipedia_search(self, context: RunContext, query: str) -> str:
        """Search for information on Wikipedia.
        
        Args:
            context: The run context for the tool
            query: The topic to look up
            
        Returns:
            str: Summary of the Wikipedia article
        """
        try:
            wiki_wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                user_agent='AmandaAI/1.0 (your-email@example.com)'
            )
            
            page = wiki_wiki.page(query)
            if not page.exists():
                return f"I couldn't find a Wikipedia article about '{query}'."
            
            # Get the first two paragraphs of the summary
            summary = '\n\n'.join(page.summary.split('\n\n')[:2])
            
            return f"According to Wikipedia: {summary}\n\nRead more: {page.fullurl}"
            
        except Exception as e:
            logging.error(f"Wikipedia search error: {e}", exc_info=True)
            return "I encountered an error while searching Wikipedia."
    
    @function_tool
    async def get_weather(self, context: RunContext, location: str) -> str:
        """Get current weather information for a location.
        
        Args:
            context: The run context for the tool
            location: The city and country (e.g., 'New York, US')
            
        Returns:
            str: Weather information or error message
        """
        try:
            # First, get coordinates for the location
            geolocator = Nominatim(user_agent="amanda_weather")
            location_data = geolocator.geocode(location)
            
            if not location_data:
                return f"Could not find location: {location}"
                
            # In a real implementation, you would call a weather API here
            # This is a placeholder response
            return (
                f"Weather information for {location_data.address}:\n"
                "Current: 22°C (72°F), Partly Cloudy\n"
                "Humidity: 65%\n"
                "Wind: 12 km/h\n"
                "\nNote: This is sample data. Integrate with a weather API for real data."
            )
            
        except Exception as e:
            logging.error(f"Weather lookup error: {e}", exc_info=True)
            return f"I couldn't get the weather for {location}. Please try again later."
    
    # --- Utility Tools ---
    
    @function_tool
    async def calculate_math(self, context: RunContext, expression: str) -> str:
        """Evaluate a mathematical expression.
        
        Args:
            context: The run context for the tool
            expression: The mathematical expression to evaluate
            
        Returns:
            str: The result or an error message
        """
        try:
            # Basic safety check
            if not re.match(r'^[0-9+\-*/().\s^%]+$', expression):
                return "Invalid expression. Only numbers and basic arithmetic operators are allowed."
                
            # Replace ^ with ** for exponentiation
            expression = expression.replace('^', '**')
            
            # Evaluate in a restricted environment
            result = eval(expression, {"__builtins__": None}, {})
            return f"The result of {expression} is {result}."
            
        except Exception as e:
            logging.error(f"Calculation error: {e}", exc_info=True)
            return "I couldn't evaluate that expression. Please check the format and try again."
    
    @function_tool
    async def get_current_time(self, context: RunContext, timezone: str = "UTC") -> str:
        """Get the current time in the specified timezone.
        
        Args:
            context: The run context for the tool
            timezone: The timezone (e.g., 'UTC', 'America/New_York')
            
        Returns:
            str: Current time in the specified timezone
        """
        try:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
            return f"The current time in {timezone} is {now.strftime('%Y-%m-%d %H:%M:%S %Z')}."
        except Exception as e:
            logging.error(f"Time lookup error: {e}", exc_info=True)
            return f"I couldn't get the time for timezone {timezone}. Please check the timezone name and try again."
    
    # --- Browser Automation ---
    
    @function_tool
    async def take_screenshot(self, context: RunContext, url: str, selector: str = "body") -> str:
        """Take a screenshot of a webpage or specific element.
        
        Args:
            context: The run context for the tool
            url: The URL to capture
            selector: CSS selector for specific element (default: whole page)
            
        Returns:
            str: Path to the screenshot or error message
        """
        try:
            # Create screenshots directory if it doesn't exist
            os.makedirs('screenshots', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshots/screenshot_{timestamp}.png"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(viewport={'width': 1280, 'height': 800})
                page = await context.new_page()
                
                # Navigate to the page
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if not response or not response.ok:
                    return f"Failed to load {url}. Status: {response.status if response else 'No response'}"
                
                # Wait for the page to be fully loaded
                await page.wait_for_load_state('networkidle')
                
                # Wait for the selector to be present
                try:
                    await page.wait_for_selector(selector, state='attached', timeout=10000)
                    # Scroll to the element to ensure it's in view
                    element = page.locator(selector)
                    await element.scroll_into_view_if_needed()
                    # Add a small delay to ensure any lazy-loaded content is visible
                    await asyncio.sleep(1)
                except Exception as e:
                    logging.warning(f"Selector {selector} not found: {e}")
                    # If specific selector not found, take full page screenshot
                    selector = 'body'
                    element = page.locator(selector)
                
                # Take screenshot with retry logic
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        await element.screenshot(
                            path=screenshot_path,
                            type='png',
                            timeout=10000
                        )
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        await asyncio.sleep(1)
                
                # Clean up
                await browser.close()
                
                # Verify the screenshot was created
                if os.path.exists(screenshot_path):
                    file_size = os.path.getsize(screenshot_path) / 1024  # Size in KB
                    if file_size < 1:  # If file is too small, it might be empty
                        raise Exception("Screenshot file is too small")
                    return f"Screenshot saved as {os.path.abspath(screenshot_path)}"
                else:
                    raise Exception("Failed to save screenshot")
                
        except Exception as e:
            logging.error(f"Screenshot error: {e}", exc_info=True)
            return f"I couldn't take a screenshot: {str(e)}. Please try again or check the URL and selector."
    
    # --- Data Processing Tools ---
    
    @function_tool
    async def clean_html(self, context: RunContext, html_content: str) -> str:
        """Clean and sanitize HTML content, removing scripts and unwanted tags.
        
        Args:
            context: The run context for the tool
            html_content: The HTML content to clean
            
        Returns:
            str: Cleaned HTML or text content
        """
        try:
            # Create a cleaner that removes scripts, styles, etc.
            from lxml import html
            from lxml_html_clean import Cleaner
            
            cleaner = Cleaner(
                scripts=True,
                javascript=True,
                style=True,
                links=True,
                meta=True,
                page_structure=False,
                safe_attrs_only=True,
                safe_attrs=frozenset(['src', 'alt', 'href', 'title', 'width', 'height'])
            )
            
            # Parse the HTML
            doc = html.document_fromstring(html_content)
            
            # Clean the document
            cleaned_doc = cleaner.clean_html(doc)
            
            # Convert back to string
            result = html.tostring(cleaned_doc, encoding='unicode', pretty_print=True)
            
            # Remove multiple spaces and newlines
            result = ' '.join(result.split())
            
            # Truncate if too long
            if len(result) > 4000:
                result = result[:4000] + '... [content truncated]'
                
            return result
            
        except Exception as e:
            logging.error(f"HTML cleaning error: {e}", exc_info=True)
            return f"I couldn't clean the HTML content: {str(e)}"
    
    @function_tool
    async def extract_links(self, context: RunContext, url: str, filter_pattern: str = None) -> str:
        """Extract all links from a webpage, optionally filtered by a pattern.
        
        Args:
            context: The run context for the tool
            url: The URL to extract links from
            filter_pattern: Optional regex pattern to filter links
            
        Returns:
            str: Formatted list of links
        """
        try:
            # Fetch the page content
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                html_content = response.text
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html5lib')
            
            # Extract all links
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True) or '[No text]'
                
                # Make relative URLs absolute
                if not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                    href = urljoin(url, href)
                
                # Filter if pattern is provided
                if filter_pattern:
                    if re.search(filter_pattern, href, re.IGNORECASE):
                        links.append((text, href))
                else:
                    links.append((text, href))
            
            # Format the results
            if not links:
                return "No links found" + (f" matching pattern '{filter_pattern}'" if filter_pattern else "")
            
            result = f"Found {len(links)} links"
            if filter_pattern:
                result += f" matching pattern '{filter_pattern}'"
            result += ":\n\n"
            
            for i, (text, href) in enumerate(links[:20], 1):  # Limit to 20 links
                result += f"{i}. {text}\n   {href}\n"
            
            if len(links) > 20:
                result += f"\n... and {len(links) - 20} more links not shown."
            
            return result
            
        except Exception as e:
            logging.error(f"Link extraction error: {e}", exc_info=True)
            return f"I couldn't extract links from the page: {str(e)}"
    
    # --- Integration Tools ---
    
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
    
    @function_tool
    async def get_news_headlines(self, context: RunContext, topic: str = "", country: str = "us") -> str:
        """Get the latest news headlines.
        
        Args:
            context: The run context for the tool
            topic: The topic to search for (optional)
            country: Two-letter country code (default: us)
            
        Returns:
            str: Formatted news headlines
        """
        try:
            # Use Google News search
            query = f"{topic} news" if topic else "latest news"
            search_url = f"https://news.google.com/search?q={urllib.parse.quote_plus(query)}&hl=en-{country.upper()}&gl={country.upper()}&ceid={country.upper()}:en"
            
            # Scrape Google News
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()
                
                # Navigate to Google News
                await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                
                # Wait for articles to load
                await page.wait_for_selector("article", timeout=10000)
                
                # Extract headlines and links
                articles = await page.evaluate('''() => {
                    const articles = [];
                    document.querySelectorAll('article').forEach(article => {
                        const link = article.querySelector('a[href^="./article/"]');
                        if (link) {
                            articles.push({
                                title: link.textContent.trim(),
                                url: new URL(link.href, window.location.href).href
                            });
                        }
                    });
                    return articles.slice(0, 5); // Return top 5 articles
                }''')
                
                await browser.close()
                
                # Format the results
                if not articles:
                    return "No news articles found."
                
                result = f"Here are the latest {topic + ' ' if topic else ''}news headlines:\n\n"
                for i, article in enumerate(articles, 1):
                    result += f"{i}. {article['title']}\n   {article['url']}\n\n"
                
                return result.strip()
                
        except Exception as e:
            logging.error(f"News lookup error: {e}", exc_info=True)
            return f"I couldn't fetch the news: {str(e)}"
    
    @function_tool
    async def get_current_datetime(self, context: RunContext) -> str:
        """Get the current date and time.
        
        Args:
            context: The run context for the tool
            
        Returns:
            str: The current date and time
        """
        try:
            now = datetime.now(pytz.utc)
            local_tz = pytz.timezone(os.environ.get('TZ', 'UTC'))
            local_now = now.astimezone(local_tz)
            
            return f"The current date and time is {local_now.strftime('%A, %B %d, %Y at %I:%M %p %Z')}."
            
        except Exception as e:
            logging.error(f"Datetime error: {e}")
            return "I couldn't get the current date and time."
    
    @function_tool
    async def calculate(self, context: RunContext, expression: str) -> str:
        """Evaluate a mathematical expression.
        
        Args:
            context: The run context for the tool
            expression: The mathematical expression to evaluate
            
        Returns:
            str: The result or an error message
        """
        try:
            # Basic safety check
            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return "Invalid expression. Only numbers and basic arithmetic operators are allowed."
                
            result = eval(expression, {"__builtins__": None}, {})
            return f"The result of {expression} is {result}."
            
        except Exception as e:
            logging.error(f"Calculation error: {e}")
            return "I couldn't evaluate that expression. Please check the format and try again."
    
    @function_tool
    async def get_weather(self, context: RunContext, location: str) -> str:
        """Get the current weather for a location.
        
        Args:
            context: The run context for the tool
            location: The location to get weather for
            
        Returns:
            str: Weather information or an error message
        """
        try:
            # This is a placeholder - in a real implementation, you would call a weather API
            return f"I can't provide real-time weather data for {location} right now. Please check a weather service for the most accurate information."
            
        except Exception as e:
            logging.error(f"Weather lookup error: {e}")
            return "I encountered an error while getting the weather."
    
    @function_tool
    async def get_news_headlines(self, context: RunContext, topic: str = "") -> str:
        """Get the latest news headlines.
        
        Args:
            context: The run context for the tool
            topic: Optional topic to search for
            
        Returns:
            str: News headlines or an error message
        """
        try:
            # This is a placeholder - in a real implementation, you would call a news API
            if topic:
                return f"I can't provide real-time news about {topic} right now. Please check a news service for the latest updates."
            return "I can't provide real-time news right now. Please check a news service for the latest updates."
            
        except Exception as e:
            logging.error(f"News lookup error: {e}")
            return "I encountered an error while getting the news."


# --- END: TOOLS ---    

# --- BEGIN: entrypoint ---

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent application."""
    mcp_server = MCPServerSse(
        params={"url": os.environ.get("ZAPIER_MCP_URL")},
        cache_tools_list=True,
        name="SSE MCP Server"
    )

    agent = await MCPToolsIntegration.create_agent_with_tools(
        agent_class=FunctionAgent,
        mcp_servers=[mcp_server]
    )

    await ctx.connect()

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
