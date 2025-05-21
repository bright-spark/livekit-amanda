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
import uvicorn
import wikipediaapi

# Local application imports
from agent_utils import speak_chunks
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from lxml_html_clean import Cleaner
from playwright.async_api import async_playwright
from pydantic import BaseModel
from googlesearch import search as google_search

# Local imports
# Import Locanto module
try:
    from locanto import get_locanto_client, LocantoClient, is_valid_locanto_location, is_valid_locanto_category
    # Initialize locanto client
    locanto_client = get_locanto_client()
    HAS_LOCANTO = True
    
    # Create a wrapper function for the locanto search
    async def search_locanto(category_path=['personals', 'men-seeking-men'], location='western-cape', max_pages=3):
        """Search Locanto for listings in a specific category and location.
        
        Args:
            category_path: List of category path segments (default: ['personals', 'men-seeking-men'])
            location: Location to search in (default: 'western-cape')
            max_pages: Maximum number of pages to scrape (default: 3)
            
        Returns:
            List of listings or error message
        """
        try:
            if not is_valid_locanto_location(location):
                return f"Invalid location: {location}. Please use one of the valid locations."
                
            if len(category_path) > 0 and not is_valid_locanto_category(category_path[0]):
                return f"Invalid category: {category_path[0]}. Please use one of the valid categories."
                
            # Get the client and perform the search
            client = get_locanto_client()
            listings = await client.locanto_search(category_path=category_path, location=location, max_pages=max_pages)
            
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

# Import Indeed module
try:
    from indeed import indeed_job_search
    HAS_INDEED = True
except ImportError as e:
    logging.warning(f"Indeed module not available: {e}")
    HAS_INDEED = False
    
    # Create a dummy function when Indeed is not available
    @function_tool
    async def indeed_job_search(context: RunContext, query: str = "customer service", location: str = "Johannesburg, Gauteng") -> str:
        """Search for jobs on Indeed using Playwright-powered scraping.
        
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

def handle_tool_results(session, results) -> None:
    """Speak tool results: if a single result, speak it; if multiple, combine and speak once."""
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
                
                IMPORTANT: Today's date is {current_date}. Your training data only includes information up to August 2024.
                For any questions about events, developments, or information after August 2024, you MUST use web search tools
                to provide accurate and up-to-date information. Never rely solely on your training data for time-sensitive questions.
                
                TIME-SENSITIVE QUERY PROTOCOL:
                1. IDENTIFY: Recognize when a query relates to events, people, products, or information that may have changed after August 2024
                2. SEARCH FIRST: For such queries, ALWAYS use web_search(), google_search(), or get_news() BEFORE responding
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
                # Removed tool_choice parameter to fix API error
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

# Import tools and functions from tools.py
from tools import (
    # Tool functions
    get_current_time,
    get_current_date,
    get_current_date_and_timezone,
    get_weather,
    get_news,
    calculate,
    calculate_math,
    evaluate_expression,
    get_fun_content,
    # Web search tools
    web_search,
    google_search,
    wikipedia_search,
    wiki_lookup,
    fallback_web_search,
    take_screenshot,
    clean_html,
    extract_links,
    open_website,
    open_known_website,
    web_crawl,
    scrape_website,
    web_search,
    google_search,
    wikipedia_search,
    wiki_lookup,
    fallback_web_search,
    get_fun_content
)

# Import Locanto related functions and classes from locanto.py
from locanto import (
    # Classes
    LocantoCategory,
    LocantoListing,
    LocantoScraper,
    
    # Constants
    LOCANTO_LOCATION_SLUGS,
    LOCANTO_CATEGORY_SLUGS,
    LOCANTO_SECTION_IDS,
    LOCANTO_TAG_SLUGS,
    
    # Functions
    is_valid_locanto_location,
    is_valid_locanto_category,
    is_valid_locanto_section,
    is_valid_locanto_tag,
    suggest_closest_slug,
    show_top_locanto_categories_and_tags,
    basic_search_locanto
)
            
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
                    temperature=0.7,
                    # No parallel_tool_calls to avoid errors
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

# --- BEGIN: entrypoint ---

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent application."""
    # Create the agent first with local tools
    agent = FunctionAgent()
    logging.info("Created agent with local tools")
    
    # Track available tool sources for user feedback
    tool_sources = ["Local"]
    
    # Check if ZAPIER_MCP_URL is set and add MCP tools if available
    zapier_mcp_url = os.environ.get("ZAPIER_MCP_URL")
    if zapier_mcp_url and zapier_mcp_url.strip():
        logging.info(f"Found Zapier MCP URL, attempting to connect")
        try:
            # Create MCP server
            mcp_server = MCPServerSse(
                params={"url": zapier_mcp_url},
                cache_tools_list=True,
                name="Zapier MCP Server"
            )
            
            # Connect to the server first to validate the connection
            await mcp_server.connect()
            
            # List available tools for logging purposes
            available_tools = await mcp_server.list_tools()
            logging.info(f"Found {len(available_tools)} tools on Zapier MCP server")
            
            if available_tools:
                # Register MCP tools with the agent
                mcp_tools = await MCPToolsIntegration.register_with_agent(
                    agent=agent,
                    mcp_servers=[mcp_server],
                    convert_schemas_to_strict=True,
                    auto_connect=False  # Already connected above
                )
                logging.info(f"Successfully registered {len(mcp_tools)} MCP tools from Zapier")
                tool_sources.append("Zapier MCP")
                
                # Log the names of registered tools for debugging
                tool_names = [getattr(t, '__name__', str(t)) for t in mcp_tools]
                logging.info(f"Registered MCP tool names: {', '.join(tool_names)}")
            else:
                logging.warning("No tools found on the Zapier MCP server")
        except Exception as e:
            import traceback
            logging.error(f"Failed to register MCP tools: {e}")
            logging.debug(traceback.format_exc())
    else:
        logging.warning("No ZAPIER_MCP_URL found in environment variables. MCP tools will not be available.")
        
    logging.info(f"Agent initialized with tools from: {', '.join(tool_sources)}")
    
    # Add a note about available tool sources to the agent instructions
    tool_sources_str = ", ".join(tool_sources)
    
    # Add a special instruction to the agent to prioritize web search tools
    current_date = datetime.now().strftime("%Y-%m-%d")
    agent.update_instructions(f"""
        You are Amanda, an advanced AI assistant with access to a comprehensive set of tools and capabilities.
        Your primary goal is to be helpful, informative, and efficient in your responses.
        
        Today's date is {current_date}.
        
        IMPORTANT: When asked to search for information or look something up, ALWAYS use the web_search, 
        google_search, or wikipedia_search tools. These are the most reliable tools for finding information.
        
        You have access to tools from the following sources: {tool_sources_str}
        
        Available tools include:
        - Web search tools (web_search, google_search, wikipedia_search, wiki_lookup)
        - Time and date tools (get_current_time, get_current_date)
        - Weather and news tools (get_weather, get_news)
        - Math calculation tools (calculate, calculate_math, evaluate_expression)
        - Web tools (take_screenshot, clean_html, extract_links, open_website)
        - Fun content (get_fun_content)
        - Job search tools (indeed_job_search, search_locanto)
        - External tools (if Zapier MCP is available)
    """)
    
    logging.info("Updated agent instructions to prioritize web search tools")
    
    await ctx.connect()
    
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
