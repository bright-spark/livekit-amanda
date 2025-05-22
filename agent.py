# Standard library imports
import asyncio
import json
import logging
import os
import re
import sys
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
import uvicorn
import wikipediaapi

# Load environment variables first to ensure they're available for all imports
from dotenv import load_dotenv
load_dotenv()

# Local application imports
from agent_utils import speak_chunks
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from lxml_html_clean import Cleaner
from playwright.async_api import async_playwright
from pydantic import BaseModel
from googlesearch import search as google_search

# Local imports
# Import Locanto module with Brave Search API integration
try:
    # Try to import the optimized Brave Search Locanto implementation first
    try:
        from brave_search_locanto_optimized import search_locanto, basic_search_locanto
        from locanto import is_valid_locanto_location, is_valid_locanto_category
        logging.info("Using optimized Brave Search API for Locanto searches")
        HAS_LOCANTO = True
    except ImportError:
        # Fall back to original implementation if optimized version is not available
        from locanto import get_locanto_client, LocantoClient, is_valid_locanto_location, is_valid_locanto_category
        # Initialize locanto client
        locanto_client = get_locanto_client()
        logging.info("Using original Locanto search implementation")
        HAS_LOCANTO = True
        
        # Create a wrapper function for the locanto search
        async def search_locanto(context: RunContext, category_path='personals/men-seeking-men', location='western-cape', max_pages=3, return_url=False):
            """Search Locanto for listings in a specific category and location.
            
            Args:
                context: The run context for the tool
                category_path: Category path string (e.g., 'personals/men-seeking-men')
                location: Location to search in (default: 'western-cape')
                max_pages: Maximum number of pages to scrape (default: 3)
                return_url: Whether to return the first URL instead of formatted results
                
            Returns:
                List of listings or error message
            """
            try:
                # Convert string category path to list if needed
                category_path_list = category_path.split('/') if isinstance(category_path, str) else category_path
                
                if not is_valid_locanto_location(location):
                    return f"Invalid location: {location}. Please use one of the valid locations."
                    
                if len(category_path_list) > 0 and not is_valid_locanto_category(category_path_list[0]):
                    return f"Invalid category: {category_path_list[0]}. Please use one of the valid categories."
                    
                # Get the client and perform the search
                client = get_locanto_client()
                listings = await client.locanto_search(category_path=category_path_list, location=location, max_pages=max_pages)
                
                return listings
            except Exception as e:
                logging.error(f"Error searching Locanto: {e}")
                return f"Error searching Locanto: {str(e)}"
            
except ImportError as e:
    logging.warning(f"Locanto module not available: {e}")
    locanto_client = None
    HAS_LOCANTO = False
    
    # Create a dummy function when Locanto is not available
    async def search_locanto(*args, **kwargs):
        return "Locanto search functionality is not available."

# Import Indeed module with Brave Search API integration
try:
    # Try to import the optimized Brave Search Indeed implementation first
    try:
        from brave_search_indeed_optimized import indeed_job_search
        logging.info("Using optimized Brave Search API for Indeed job searches")
        HAS_INDEED = True
    except ImportError:
        # Fall back to original implementation if optimized version is not available
        from indeed import indeed_job_search
        logging.info("Using original Indeed job search implementation")
        HAS_INDEED = True
except ImportError as e:
    logging.warning(f"Indeed module not available: {e}")
    HAS_INDEED = False
    
    # Create a dummy function when Indeed is not available
    @function_tool
    async def indeed_job_search(context: RunContext, query: str = "customer service", location: str = "Johannesburg, Gauteng") -> str:
        """Search for jobs on Indeed using Brave Search API or scraping.
        
        Args:
            context: The run context for the tool
            query: The job search query (job title, keywords, company)
            location: The location to search for jobs
            
        Returns:
            Formatted job search results or error message
        """
        return "Indeed job search functionality is not available."
            
except ImportError as e:
    logging.warning(f"Locanto module not available: {e}")
    locanto_client = None
    HAS_LOCANTO = False
    
    # Create a dummy function when Locanto is not available
    async def search_locanto(*args, **kwargs):
        return "Locanto search functionality is not available."

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

# Fallback URL for error cases
FALLBACK_URL = "https://fallback"

# --- BEGIN: Well-known Websites Mapping and Tool ---
WELL_KNOWN_WEBSITES = {
    "google": "https://www.google.com/",
    "bing": "https://www.bing.com/",
    "yahoo": "https://www.yahoo.com/",
    "cnn": "https://www.cnn.com/",
    "bbc": "https://www.bbc.com/",
    "nytimes": "https://www.nytimes.com/",
    "fox": "https://www.foxnews.com/",
    "wikipedia": "https://en.wikipedia.org/",
    "youtube": "https://www.youtube.com/",
    "reddit": "https://www.reddit.com/",
    "twitter": "https://twitter.com/",
    "facebook": "https://facebook.com/",
    "linkedin": "https://linkedin.com/",
    "instagram": "https://instagram.com/",
    "tiktok": "https://tiktok.com/",
    "indeed": "https://www.indeed.com/",
    "locanto": "https://www.locanto.co.za/"
}

# Set of sites that support bang-style queries (e.g., @site query)
BROWSER_TOOL = {"gemini"}

# Chunking parameters for voice responses
CHUNK_ENABLE = os.environ.get("CHUNK_ENABLE", "true").lower() in ("true", "1", "yes")
MAX_AUTO_CHUNKS = int(os.environ.get("MAX_AUTO_CHUNKS", 10))
CHUNK_PAUSE = float(os.environ.get("CHUNK_PAUSE", 1.0))

# LLM timeout and retry configuration
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", 60.0))  # Default 60 seconds
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", 3))  # Default 3 retries
LLM_RETRY_DELAY = float(os.environ.get("LLM_RETRY_DELAY", 2.0))  # Default 2 seconds
LLM_RETRY_BACKOFF = float(os.environ.get("LLM_RETRY_BACKOFF", 1.5))  # Default exponential backoff factor

# Search result configuration
SEARCH_RESULT_TRUNCATE = os.environ.get("SEARCH_RESULT_TRUNCATE", "true").lower() in ("true", "1", "yes")
SEARCH_RESULT_MAX_CHARS = int(os.environ.get("SEARCH_RESULT_MAX_CHARS", 1000))  # Default 1000 characters per result

# Debug output function for search results
def debug_search_result(search_engine: str, query: str, results: any) -> None:
    """Print debug output for search results to the console.
    
    Args:
        search_engine: Name of the search engine
        query: The search query
        results: The search results (string or other format)
    """
    if not SEARCH_RESULT_TRUNCATE:
        return
    
    # Convert results to string if not already
    if not isinstance(results, str):
        try:
            results_str = str(results)
        except Exception as e:
            results_str = f"[Error converting results to string: {e}]"
    else:
        results_str = results
        
    separator = "=" * 80
    print(separator)
    print(f"[DEBUG] {search_engine} RESULTS FOR: '{query}'")
    print(separator)
    
    # Truncate results if they're too long
    if len(results_str) > SEARCH_RESULT_MAX_CHARS:
        print(f"{results_str[:SEARCH_RESULT_MAX_CHARS]}...")
        print(f"[TRUNCATED - {len(results_str)} total characters]")
    else:
        print(results_str)
    
    print(separator)
    # Force flush stdout to ensure output is displayed immediately
    sys.stdout.flush()

# Log configuration parameters
logging.info(f"Voice response chunking configured with CHUNK_ENABLE={CHUNK_ENABLE}, MAX_AUTO_CHUNKS={MAX_AUTO_CHUNKS}, CHUNK_PAUSE={CHUNK_PAUSE}s")
logging.info(f"LLM timeout configuration: TIMEOUT={LLM_TIMEOUT}s, MAX_RETRIES={LLM_MAX_RETRIES}, RETRY_DELAY={LLM_RETRY_DELAY}s, BACKOFF={LLM_RETRY_BACKOFF}")
logging.info(f"Search result configuration: TRUNCATE={SEARCH_RESULT_TRUNCATE}, MAX_CHARS={SEARCH_RESULT_MAX_CHARS}")


def sanitize_for_azure(text: str) -> str:
    """Reword or mask terms that may trigger Azure OpenAI's content filter."""
    unsafe_terms = {
        "sex": "intimacy",
        "sexual": "romantic",
        "hookup": "meeting",
        "hookups": "meetings",
        "anal": "[redacted]",
        "blowjob": "[redacted]",
        "quickie": "[redacted]",
        "incalls": "meetings",
        "outcalls": "meetings",
        "massage": "relaxation",
        "MILF": "person",
        "fuck": "love",
        "cunt": "[redacted]",
        "penis": "[redacted]",
        "oral": "[redacted]",
        "wank": "[redacted]",
        "finger": "[redacted]",
        "date": "meet",
        "love": "companionship",
        "kiss": "affection",
        "look": "search",
        "find": "discover",
        "girl": "woman",
    }
    for term, replacement in unsafe_terms.items():
        # Replace whole words only, case-insensitive
        text = re.sub(rf'\\b{re.escape(term)}\\b', replacement, text, flags=re.IGNORECASE)
    return text

# Add this utility function for robust tool output handling

def is_sequence_but_not_str(obj):
    import collections.abc
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, (str, bytes, bytearray))

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

def debug_search_result(search_tool_name, query, result):
    """Print truncated search results to the console for debugging."""
    if not result:
        print(f"\n[DEBUG] {search_tool_name} - No results for query: '{query}'\n")
        return
        
    # Truncate the result if needed
    truncated = result[:SEARCH_RESULT_MAX_CHARS]
    if len(result) > SEARCH_RESULT_MAX_CHARS:
        truncated += f"... [truncated, {len(result) - SEARCH_RESULT_MAX_CHARS} more characters]"
        
    # Print to console with clear formatting
    print(f"\n{'=' * 80}")
    print(f"[DEBUG] {search_tool_name} RESULTS FOR: '{query}'")
    print(f"{'=' * 80}")
    print(truncated)
    print(f"{'=' * 80}\n")

async def handle_tool_results(session, results) -> None:
    """Speak tool results: if a single result, speak it; if multiple, combine and speak once."""
    if not CHUNK_ENABLE:
        # If chunking is disabled, just send the full message
        if is_sequence_but_not_str(results):
            combined = '\n\n'.join(str(r) for r in results if r)
            combined = clean_spoken(combined)
            return await session.add_message(role="assistant", content=combined)
        else:
            results = clean_spoken(results)
            return await session.add_message(role="assistant", content=results)
    else:
        # Use chunking as configured
        if is_sequence_but_not_str(results):
            combined = '\n\n'.join(str(r) for r in results if r)
            combined = clean_spoken(combined)
            return speak_chunks(session, combined, max_auto_chunks=MAX_AUTO_CHUNKS, pause=CHUNK_PAUSE)
        else:
            results = clean_spoken(results)
            return speak_chunks(session, results, max_auto_chunks=MAX_AUTO_CHUNKS, pause=CHUNK_PAUSE) # type: ignore

class FunctionAgent(Agent):
    """A LiveKit agent that uses MCP tools from one or more MCP servers."""

    def __init__(self):
        # Get current date for the instructions
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        super().__init__(
            instructions=f"""
                You are Amanda, an advanced AI assistant with access to a comprehensive set of tools and capabilities.
                Your primary goal is to be helpful, informative, and efficient in your responses.
                
                IMPORTANT: It is {current_date} now. Your own data only includes information up to August 2024.
                For any questions about people, events, developments, or any information after August 2024, you MUST use web search tools
                to provide accurate and up-to-date information. Never rely solely on your training data for time-sensitive questions.
                
                TIME-SENSITIVE QUERY PROTOCOL:
                1. IDENTIFY: Recognize when a query relates to events, people, products, or information that may have changed after August 2024
                2. SEARCH FIRST: For such queries, ALWAYS use web_search() BEFORE responding
                3. VERIFY: Cross-check information from multiple web sources when possible
                4. CITE SOURCES: Always mention the sources of your information
                5. BE TRANSPARENT: If search tools fail, clearly state that you cannot provide up-to-date information
                
                Examples of time-sensitive queries that REQUIRE web search:
                - "Who is the current President of [country]?"
                - "What is the latest iPhone model?"
                - "How did [recent sports event] turn out?"
                - "What are the current COVID-19 guidelines?"
                - "What's the status of [ongoing situation]?"
                - "What happened in [recent news event]?"
                - "What's the current price of [product/stock/cryptocurrency]?"
                
                ===== AVAILABLE TOOLS =====
                
                [SEARCH & BROWSING]
                - web_search(query): Search the web for information
                - google_search(query, num_results): Search using Google
                - wikipedia_search(query): Search Wikipedia for information
                - wiki_lookup(topic): Get detailed information from Wikipedia
                - fallback_web_search(query, num_results): Alternative search when primary methods fail
                - web_crawl(url, selector, max_pages): Extract content from web pages
                - scrape_website(url, selector, text_only): Scrape content from websites
                - open_website(url): Open a specific website
                - open_known_website(site_name, query): Open a well-known website (e.g., 'google', 'wikipedia')
                
                [LOCANTO INTEGRATION]
                - search_locanto(category_path, location, max_pages): Search Locanto listings
                - search_locanto_browser(query, location, max_pages, tag, category, section, url): Advanced Locanto search
                - locanto_matchmaking(query, gender, seeking, age, location, tag, category, section, max_pages): Find matches on Locanto
                - show_top_locanto_categories_and_tags(location): Browse Locanto categories
                - basic_search_locanto(query, location, category): Basic Locanto search
                
                [UTILITIES]
                - get_weather(location): Get current weather conditions
                - get_news(topic, country): Fetch latest news
                - calculate(expression): Evaluate mathematical expressions
                - calculate_math(expression): Alternative math evaluation
                - evaluate_expression(expression): Evaluate complex expressions
                - get_current_time(timezone): Get current time in specific timezone
                - get_current_date(): Get current date
                - get_current_date_and_timezone(): Get date with timezone
                - get_fun_content(content_type): Get jokes, facts, or trivia
                - indeed_job_search(query, location): Search for jobs on Indeed
                
                [ADVANCED]
                - extract_links(url, filter_pattern): Extract links from a webpage
                - clean_html(html_content): Sanitize HTML content
                - take_screenshot(url, selector): Capture webpage screenshots
                
                ===== TOOL USAGE GUIDELINES =====
                1. PROACTIVE TOOL USE: Automatically use tools whenever they can help answer a query without asking for permission
                2. AUTO SELECTION: Select and use the most appropriate tool immediately based on the query
                3. MULTIPLE TOOLS: Chain multiple tools together when necessary to provide comprehensive answers
                4. VERIFICATION: Cross-reference information from multiple tools when possible
                5. FALLBACKS: If one tool fails, automatically try alternative tools
                6. EFFICIENCY: Use the most direct path to answer each query
                7. CONTEXT AWARENESS: Use previous conversation context to inform tool selection
                8. TRANSPARENCY: Briefly mention which tools you used to find information
                
                ===== BEST PRACTICES =====
                - For general knowledge: Automatically use wiki_lookup() or wikipedia_search()
                - For current information: ALWAYS use web_search(), google_search() or get_news() without asking
                - For post-August 2024 information: MANDATORY to use web search tools and cite sources
                - For calculations: Immediately use calculate() or evaluate_expression()
                - For Locanto queries: Directly use the appropriate Locanto tools
                - For website interaction: Use open_website() or web_crawl() proactively
                - For weather: Use get_weather() without asking for confirmation
                - For time/date: Use the appropriate time/date functions automatically
                - For news events: ALWAYS use get_news() to ensure current information
                
                Always be proactive, efficient, and helpful. Use tools automatically without asking for permission.
                Present information in a clear, organized manner after using the appropriate tools.
                
                FINAL REMINDER: You are operating in {current_date}, which is AFTER your knowledge cutoff of August 2024.
                Your primary responsibility is to provide accurate, up-to-date information. For ANY time-sensitive query,
                you MUST use web search tools before responding. This is not optional - it is a core requirement of your operation.
                Failure to use web search for time-sensitive information would result in potentially outdated or incorrect responses.
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
                parallel_tool_calls=True,
            ),
            tts=azure.TTS(
                speech_key=os.environ["AZURE_TTS_API_KEY"],
                speech_region=os.environ["AZURE_TTS_REGION"]
            ),
            vad=silero.VAD.load(
                activation_threshold=0.7,  # Slightly lower threshold for better voice detection
                min_speech_duration=0.2,  # Shorter minimum speech duration (in seconds)
                min_silence_duration=0.2,  # Shorter silence duration for quicker response (in seconds)
                sample_rate=16000,  # Explicitly set sample rate
                force_cpu=True,  # Use CPU for better stability
                prefix_padding_duration=0.3  # Add small padding before speech
            ),
            allow_interruptions=True
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Override the llm_node to say a message when a tool call is detected."""
        tool_call_detected = False

        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            if isinstance(chunk, ChatChunk) and chunk.delta and chunk.delta.tool_calls and not tool_call_detected:
                tool_call_detected = True
                chat_ctx.add_message(role="assistant", content="Sure, I'll check that for you.")

            yield chunk

# Try to import the fallback search system first
try:
    from fallback_search_system import unified_web_search, fallback_search
    HAS_FALLBACK_SEARCH = True
    logging.info("Using unified fallback search system with multiple search engines")
except ImportError:
    HAS_FALLBACK_SEARCH = False
    logging.warning("Fallback search system not available, will try individual search methods")

# Try to import optimized Brave Search API implementations
try:
    import brave_search_free_tier
    from brave_search_free_tier import web_search as brave_web_search
    from brave_search_free_tier import get_cache_stats as brave_get_cache_stats
    
    BRAVE_SEARCH_ENABLE_CACHE = os.environ.get("BRAVE_SEARCH_ENABLE_CACHE", "true").lower() == "true"
    BRAVE_SEARCH_ENABLE_PERSISTENCE = os.environ.get("BRAVE_SEARCH_ENABLE_PERSISTENCE", "true").lower() == "true"
    BRAVE_SEARCH_RATE_LIMIT = int(os.environ.get("BRAVE_SEARCH_RATE_LIMIT", "1"))
    
    HAS_BRAVE_SEARCH = True
    logging.info(f"Using configurable Brave Search API implementation with settings: cache={BRAVE_SEARCH_ENABLE_CACHE}, persistence={BRAVE_SEARCH_ENABLE_PERSISTENCE}, rate_limit={BRAVE_SEARCH_RATE_LIMIT}")
except ImportError:
    HAS_BRAVE_SEARCH = False
    logging.warning("Brave Search implementation not available, will use fallback")

# Try to import DuckDuckGo Search
try:
    from duckduckgo_search import ddg_web_search
    HAS_DUCKDUCKGO = True
    logging.info("DuckDuckGo Search available as fallback")
except ImportError:
    HAS_DUCKDUCKGO = False
    logging.warning("DuckDuckGo Search not available")

# Import tools and functions from tools.py
from tools import (
    get_current_time,
    get_current_date,
    get_current_date_and_timezone,
    get_weather,
    get_news,
    calculate,
    calculate_math,
    evaluate_expression,
    get_fun_content,
    take_screenshot,
    clean_html,
    extract_links,
    open_website,
    google_search,
    wikipedia_search,
    wiki_lookup,
    sanitize_for_azure,
    clean_spoken,
    handle_tool_results,
)

# Conditionally import web_search and fallback_web_search from tools if Brave Search is not available
if not HAS_BRAVE_SEARCH:
    from tools import web_search, fallback_web_search

# Import Locanto-related functions directly from locanto.py
try:
    from locanto import (
        is_valid_locanto_location,
        is_valid_locanto_category,
        is_valid_locanto_section,
        is_valid_locanto_tag,
        suggest_closest_slug,
        show_top_locanto_categories_and_tags,
        basic_search_locanto
    )
    HAS_LOCANTO_UTILS = True
    logging.info("Imported Locanto utility functions from locanto.py")
except ImportError:
    HAS_LOCANTO_UTILS = False
    logging.warning("Could not import Locanto utility functions from locanto.py")


class AIVoiceAssistant:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIVoiceAssistant, cls).__new__(cls)
            cls._instance.vad = None
            cls._instance.session = None
            cls._instance.agent = None
            cls._instance.wiki_wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                user_agent='AIVoiceAssistant/1.0'
            )
            cls._instance.geolocator = Nominatim(user_agent="AIVoiceAssistant/1.0")
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
                    temperature=0.7,
                    tool_choice="auto",
                    timeout=httpx.Timeout(connect=15.0, read=10.0, write=5.0, pool=5.0),
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
            self.wiki_wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                user_agent='AIVoiceAssistant/1.0'
            )
            self.geolocator = Nominatim(user_agent="AIVoiceAssistant/1.0")
            self.weather_cache = {}
            self.wiki_cache = {}
            self.news_cache = {}
            self.crawl_cache = {}
            self.cookies = {}
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

# --- BEGIN: entrypoint ---

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent application."""
    agent = FunctionAgent()
    
    web_search_tools = []
    from livekit.agents import function_tool
    
    # Individual search tools for each provider that can be called explicitly by the user
    
    # 1. Brave Search Tool
    @function_tool
    async def brave_search(context: RunContext, query: str, num_results: int = 5) -> str:
        """Search the web using Brave Search API specifically.
        
        Args:
            context: The run context for the tool
            query: The search query
            num_results: Number of results to return (1-10)
            
        Returns:
            str: Formatted search results with titles and URLs
        """
        try:
            if not HAS_BRAVE_SEARCH:
                return "Brave Search API is not available. Please try another search tool."
                
            if not isinstance(query, str):
                query = str(query)
                
            # Ensure num_results is an integer
            if not isinstance(num_results, int):
                try:
                    num_results = int(num_results)
                except (ValueError, TypeError):
                    num_results = 5
            
            # Limit number of results to a reasonable range
            num_results = max(1, min(num_results, 10))
            
            logging.info(f"Using Brave Search API explicitly for query: '{query}'")
            results = await brave_web_search(query, num_results)
            
            # Print debug output to console
            debug_search_result("BRAVE SEARCH", query, results)
            
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, results)
                return "I've found some results using Brave Search and will read them to you now."
            return results
        except Exception as e:
            logging.error(f"Error in brave_search: {e}")
            error_msg = f"I couldn't find any results for '{query}' using Brave Search. Try a different query or search tool."
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't find any results using Brave Search."
            return error_msg
    
    # 2. Bing Search Tool
    @function_tool
    async def bing_search(context: RunContext, query: str, num_results: int = 5) -> str:
        """Search the web using Bing specifically.
        
        Args:
            context: The run context for the tool
            query: The search query
            num_results: Number of results to return (1-10)
            
        Returns:
            str: Formatted search results with titles and URLs
        """
        try:
            # Try to import from bing_search.py first (fast version)
            try:
                from bing_search import bing_search as bing_search_fast
                logging.info(f"Using fast Bing search for query: '{query}'")
                results = await bing_search_fast(context, query, num_results)
                
                # Print debug output to console
                debug_search_result("BING SEARCH (FAST)", query, results)
                
                return results
            except ImportError:
                # Try to import from bing_extended.py (quality version)
                try:
                    from bing_extended import bing_search as bing_search_quality
                    logging.info(f"Using quality Bing search for query: '{query}'")
                    results = await bing_search_quality(context, query, num_results)
                    
                    # Print debug output to console
                    debug_search_result("BING SEARCH (QUALITY)", query, results)
                    
                    return results
                except ImportError:
                    return "Bing Search is not available. Please try another search tool."
        except Exception as e:
            logging.error(f"Error in bing_search: {e}")
            error_msg = f"I couldn't find any results for '{query}' using Bing Search. Try a different query or search tool."
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't find any results using Bing Search."
            return error_msg
    
    # 3. DuckDuckGo Search Tool
    @function_tool
    async def duckduckgo_search(context: RunContext, query: str, num_results: int = 5) -> str:
        """Search the web using DuckDuckGo specifically.
        
        Args:
            context: The run context for the tool
            query: The search query
            num_results: Number of results to return (1-10)
            
        Returns:
            str: Formatted search results with titles and URLs
        """
        try:
            # Try to import from duckduckgo_search.py
            try:
                from duckduckgo_search import ddg_web_search
                logging.info(f"Using DuckDuckGo search for query: '{query}'")
                results = await ddg_web_search(query, num_results)
                
                # Print debug output to console
                debug_search_result("DUCKDUCKGO SEARCH", query, results)
                
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, results)
                    return "I've found some results using DuckDuckGo and will read them to you now."
                return results
            except ImportError:
                return "DuckDuckGo Search is not available. Please try another search tool."
        except Exception as e:
            logging.error(f"Error in duckduckgo_search: {e}")
            error_msg = f"I couldn't find any results for '{query}' using DuckDuckGo Search. Try a different query or search tool."
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't find any results using DuckDuckGo Search."
            return error_msg
    
    # 4. Google Search Tool
    @function_tool
    async def google_search(context: RunContext, query: str, num_results: int = 5) -> str:
        """Search the web using Google specifically.
        
        Args:
            context: The run context for the tool
            query: The search query
            num_results: Number of results to return (1-10)
            
        Returns:
            str: Formatted search results with titles and URLs
        """
        try:
            # Try to import from googlesearch
            try:
                from googlesearch import search as google_search_func
                logging.info(f"Using Google search for query: '{query}'")
                
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
                num_results = max(1, min(num_results, 10))
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                results_list = await loop.run_in_executor(
                    None,
                    lambda: list(google_search_func(query, num_results=num_results))
                )
                
                if not results_list:
                    return f"No results found for '{query}' on Google."
                
                formatted = f"Google search results for '{query}':\n\n"
                
                for i, result in enumerate(results_list, 1):
                    formatted += f"{i}. {result}\n\n"
                
                # Print debug output to console
                debug_search_result("GOOGLE SEARCH", query, formatted)
                
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, formatted)
                    return "I've found some results using Google and will read them to you now."
                return formatted
            except ImportError:
                return "Google Search is not available. Please try another search tool."
        except Exception as e:
            logging.error(f"Error in google_search: {e}")
            error_msg = f"I couldn't find any results for '{query}' using Google Search. Try a different query or search tool."
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't find any results using Google Search."
            return error_msg
    
    # Define the web_search function based on available search methods
    if HAS_BRAVE_SEARCH:
        # Use Brave Search API as the primary search method
        @function_tool
        async def web_search(context: RunContext, query: str, num_results: int = 5) -> str:
            """Search the web and return formatted results.
            
            Args:
                context: The run context for the tool
                query: The search query
                num_results: Number of results to return (1-10)
                
            Returns:
                str: Formatted search results with titles and URLs
            """
            logging.info(f"Using Brave Search API for query: '{query}'")
            logging.info(f"[TOOL] web_search called for query: {query}, num_results: {num_results}")
            
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
            num_results = max(1, min(num_results, 10))
            
            start_time = time.time()
            try:
                # Use Brave Search API
                results = await brave_web_search(query, num_results)
                
                # Print debug output to console
                debug_search_result("WEB SEARCH (BRAVE PRIMARY)", query, results)
                
                # Handle session output for voice responses if available
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, results)
                    return "I've found some results and will read them to you now."
                
                return results
            except Exception as e:
                logging.error(f"Error in web_search: {e}")
                return await fallback_web_search(context, query, num_results)
            finally:
                end_time = time.time()
                logging.info(f"[PERFORMANCE] web_search completed in {end_time - start_time:.4f}s")
        
        @function_tool
        async def fallback_web_search(context: RunContext, query: str, num_results: int = 10) -> str:
            """Alternative search when primary methods fail, using multiple search engines.
            
            Args:
                context: The run context for the tool
                query: The search query
                num_results: Number of results to return (1-20)
                
            Returns:
                str: Formatted search results with titles and URLs
            """
            # If fallback search system is available, use it
            if HAS_FALLBACK_SEARCH:
                try:
                    logging.info(f"Using fallback search system for query: '{query}'")
                    results = await fallback_search(context, query, max(num_results, 10))
                    
                    # Print debug output to console
                    debug_search_result("FALLBACK SEARCH SYSTEM", query, results)
                    
                    return results
                except Exception as e:
                    logging.error(f"Fallback search error: {e}")
            
            # Otherwise, just use the primary web_search with more results
            logging.info(f"Fallback search system not available, using primary web_search for query: '{query}'")
            results = await web_search(context, query, max(num_results, 10))
            
            # Print debug output to console
            debug_search_result("FALLBACK TO PRIMARY", query, results)
            
            return results
    
    elif HAS_BRAVE_SEARCH:
        # Use Brave Search API if available
        @function_tool
        async def web_search(context: RunContext, query: str, num_results: int = 5) -> str:
            """Search the web for information using optimized Brave Search API.
            
            Args:
                context: The run context for the tool
                query: The search query
                num_results: Number of results to return (1-10)
                
            Returns:
                str: Formatted search results with titles and URLs
            """
            try:
                if not isinstance(query, str):
                    query = str(query)
                results = await brave_web_search(query, num_results)
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, results)
                    return "I've found some results and will read them to you now."
                return results
            except Exception as e:
                logging.error(f"Error in web_search using Brave Search API: {e}")
                # Try DuckDuckGo as fallback if available
                if HAS_DUCKDUCKGO:
                    try:
                        logging.info(f"Trying DuckDuckGo fallback for query: '{query}'")
                        results = await ddg_web_search(query, num_results)
                        session = getattr(context, 'session', None)
                        if session:
                            await handle_tool_results(session, results)
                            return "I've found some results using DuckDuckGo and will read them to you now."
                        return results
                    except Exception as ddg_error:
                        logging.error(f"Error in DuckDuckGo fallback: {ddg_error}")
                
                error_msg = f"I couldn't find any results for '{query}'. Try a different query."
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, error_msg)
                    return "I couldn't find any results for your search."
                return error_msg
        
        @function_tool
        async def fallback_web_search(context: RunContext, query: str, num_results: int = 10) -> str:
            """Alternative search when primary methods fail, using DuckDuckGo or other sources.
            
            Args:
                context: The run context for the tool
                query: The search query
                num_results: Number of results to return (1-20)
                
            Returns:
                str: Formatted search results with titles and URLs
            """
            if HAS_DUCKDUCKGO:
                try:
                    results = await ddg_web_search(query, num_results)
                    session = getattr(context, 'session', None)
                    if session:
                        await handle_tool_results(session, results)
                        return "I've found some results using DuckDuckGo and will read them to you now."
                    return results
                except Exception as e:
                    logging.error(f"Error in fallback_web_search using DuckDuckGo: {e}")
            
            # If DuckDuckGo fails or is not available, try Brave as a last resort
            return await web_search(context, query, num_results)
    
    else:
        # Use the original tools.py implementations if no specialized search is available
        @function_tool
        async def web_search(context: RunContext, query: str, num_results: int = 5) -> str:
            """Search the web for information.
            
            Args:
                context: The run context for the tool
                query: The search query
                num_results: Number of results to return (1-10)
                
            Returns:
                str: Formatted search results with titles and URLs
            """
            # Try DuckDuckGo first if available
            if HAS_DUCKDUCKGO:
                try:
                    results = await ddg_web_search(query, num_results)
                    session = getattr(context, 'session', None)
                    if session:
                        await handle_tool_results(session, results)
                        return "I've found some results using DuckDuckGo and will read them to you now."
                    return results
                except Exception as e:
                    logging.error(f"Error in web_search using DuckDuckGo: {e}")
            
            # Fall back to tools.py implementation
            from tools import web_search as tools_web_search
            return await tools_web_search(context, query)
        
        @function_tool
        async def fallback_web_search(context: RunContext, query: str, num_results: int = 10) -> str:
            """Alternative search when primary methods fail.
            
            Args:
                context: The run context for the tool
                query: The search query
                num_results: Number of results to return (1-20)
                
            Returns:
                str: Formatted search results with titles and URLs
            """
            # Try tools.py implementation
            from tools import fallback_web_search as tools_fallback_web_search
            return await tools_fallback_web_search(context, query, num_results)
    
    # Add the web search tools to the list
    web_search_tools = [web_search, fallback_web_search]
    
    # Initialize agent with MCP tools
    agent = FunctionAgent()
    
    # Check if local tools are enabled
    local_tools_enabled = os.environ.get("LOCAL_TOOLS", "true").lower() in ("true", "1", "yes")
    
    # Initialize tools list
    tools_to_register = []
    
    # Register local tools if enabled
    if local_tools_enabled:
        logging.info("Local tools are enabled")
        tools_to_register.extend([
            get_current_time,
            get_current_date,
            get_current_date_and_timezone,
            
            # Weather and news tools
            get_weather,
            get_news,
            
            # Math calculation tool
            calculate,
            
            # Fun content tools
            get_fun_content,
            
            # Web tools
            take_screenshot,
            clean_html,
            extract_links,
            open_website,
            
            # Search tools
            wikipedia_search,
            wiki_lookup,
            
            # Job search tools (if enabled)
            *([indeed_job_search] if os.environ.get("INDEED_ENABLE", "true").lower() in ("true", "1", "yes") else []),
            *([search_locanto] if os.environ.get("LOCANTO_ENABLE", "true").lower() in ("true", "1", "yes") else [])
        ])
    else:
        logging.info("Local tools are disabled via environment variable")
    
    # Add the main web search tools
    tools_to_register.append(web_search)
    tools_to_register.append(fallback_web_search)
    
    # Add the individual provider-specific search tools based on environment variables
    brave_search_enabled = os.environ.get("BRAVE_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
    bing_search_enabled = os.environ.get("BING_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
    duckduckgo_search_enabled = os.environ.get("DUCKDUCKGO_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
    google_search_enabled = os.environ.get("GOOGLE_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
    wikipedia_enabled = os.environ.get("WIKIPEDIA_ENABLE", "true").lower() in ("true", "1", "yes")
    
    # Check job search tool status
    indeed_enabled = os.environ.get("INDEED_ENABLE", "true").lower() in ("true", "1", "yes")
    locanto_enabled = os.environ.get("LOCANTO_ENABLE", "true").lower() in ("true", "1", "yes")
    
    # Log job search tool status
    if indeed_enabled:
        logging.info("Indeed job search tool is enabled")
    else:
        logging.info("Indeed job search tool is disabled via environment variable")
        
    if locanto_enabled:
        logging.info("Locanto search tool is enabled")
    else:
        logging.info("Locanto search tool is disabled via environment variable")
    
    if brave_search_enabled:
        tools_to_register.append(brave_search)
        logging.info("Brave Search tool enabled")
    
    if bing_search_enabled:
        tools_to_register.append(bing_search)
        logging.info("Bing Search tool enabled")
    
    if duckduckgo_search_enabled:
        tools_to_register.append(duckduckgo_search)
        logging.info("DuckDuckGo Search tool enabled")
    
    if google_search_enabled:
        tools_to_register.append(google_search)
        logging.info("Google Search tool enabled")
    
    if HAS_LOCANTO_UTILS:
        tools_to_register.append(basic_search_locanto)
        tools_to_register.append(show_top_locanto_categories_and_tags)
    
    if hasattr(agent, '_tools') and isinstance(agent._tools, list):
        agent._tools.extend(tools_to_register)
        logging.info(f"Added {len(tools_to_register)} tools to agent")
    logging.info(f"Registered {len(tools_to_register)} local tools with agent")
    
    logging.info("Created agent with local tools")
    
    # Initialize tool sources list
    tool_sources = ["Local"]
    
    # Check if MCP_CLIENT is enabled and ZAPIER_MCP_URL is set
    mcp_client_enabled = os.environ.get("MCP_CLIENT", "false").lower() in ("true", "1", "yes")
    zapier_mcp_url = os.environ.get("ZAPIER_MCP_URL")
    
    if mcp_client_enabled:
        if zapier_mcp_url and zapier_mcp_url.strip():
            logging.info(f"MCP_CLIENT is enabled and Zapier MCP URL found, attempting to connect")
            try:
                mcp_server = MCPServerSse(
                    params={"url": zapier_mcp_url},
                    cache_tools_list=True,
                    name="Zapier MCP Server"
                )
                
                await mcp_server.connect()
                
                available_tools = await mcp_server.list_tools()
                logging.info(f"Found {len(available_tools)} tools on Zapier MCP server")
                
                if available_tools:
                    mcp_tools = await MCPToolsIntegration.register_with_agent(
                        agent=agent,
                        mcp_servers=[mcp_server],
                        convert_schemas_to_strict=True,
                        auto_connect=False
                    )
                    logging.info(f"Successfully registered {len(mcp_tools)} MCP tools from Zapier")
                    tool_sources.append("Zapier MCP")
                    tool_names = [getattr(t, '__name__', str(t)) for t in mcp_tools]
                    logging.info(f"Registered MCP tool names: {', '.join(tool_names)}")
                else:
                    logging.warning("No tools found on Zapier MCP server")
            except Exception as e:
                import traceback
                logging.error(f"Failed to register MCP tools: {e}")
                logging.debug(traceback.format_exc())
                logging.warning("MCP tools will not be available")
        else:
            logging.warning("MCP_CLIENT is enabled but no ZAPIER_MCP_URL found in environment variables. MCP tools will not be available.")
    else:
        logging.info("MCP_CLIENT is disabled, skipping MCP tool registration")
        
    logging.info(f"Agent initialized with tools from: {', '.join(tool_sources)}")
    
    tool_sources_str = ", ".join(tool_sources)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    brave_info = ""
    if HAS_BRAVE_SEARCH:
        brave_info = "\n    You are using the optimized Brave Search API for web searches, which is highly efficient with caching."
    
    instructions = f"""
        You are Amanda, an advanced AI assistant with access to a comprehensive set of tools and capabilities.
        Your primary goal is to be helpful, informative, and efficient in your responses.
        
        It is now {current_date}.{brave_info}
        
        IMPORTANT: When asked to search for information or look something up, ALWAYS use the web_search tool by default.
        The web_search tool automatically prioritizes Brave Search for the best results.
        
        You also have access to provider-specific search tools that can be used when explicitly requested:
        - brave_search: Uses Brave Search API specifically (preferred for most searches)
        - bing_search: Uses Bing search specifically
        - duckduckgo_search: Uses DuckDuckGo search specifically
        - google_search: Uses Google search specifically (use only when explicitly requested)
        
        You have access to tools from the following sources: {tool_sources_str}
        
        Available tools include:
        - Web search tools (web_search, fallback_web_search, brave_search, bing_search, duckduckgo_search, google_search, wikipedia_search, wiki_lookup)
        - Time and date tools (get_current_time, get_current_date)
        - Weather and news tools (get_weather, get_news)
        - Math calculation tools (calculate)
        - Web tools (take_screenshot, clean_html, extract_links, open_website)
        - Fun content (get_fun_content)
        - Job search tools (indeed_job_search, search_locanto)
        - External tools (if Zapier MCP is available)
    """
    
    await agent.update_instructions(instructions)
    
    logging.info("Updated agent instructions to prioritize web search tools")
    
    await ctx.connect()
    
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
