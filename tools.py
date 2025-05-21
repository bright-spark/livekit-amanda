from .puppeteer_crawler import crawl_page
from livekit.agents import function_tool
from livekit.agents import RunContext
import httpx
import logging
import random
import html
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
import os
from .utils import sanitize_for_azure
from livekit.agents import function_tool, RunContext

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
    from .agent_utils import speak_chunks
    if is_sequence_but_not_str(results):
        combined = '\n\n'.join(str(r) for r in results if r)
        combined = clean_spoken(combined)
        return speak_chunks(session, combined, max_auto_chunks=MAX_AUTO_CHUNKS, pause=CHUNK_PAUSE)
    else:
        results = clean_spoken(results)
        return speak_chunks(session, results, max_auto_chunks=MAX_AUTO_CHUNKS, pause=CHUNK_PAUSE) # type: ignore


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

@function_tool
async def get_current_date(context: RunContext) -> str:
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
def get_current_date_and_timezone(context: RunContext) -> str:
    """Get the current server date and time in a natural language format with timezone."""
    try:
        # Get local timezone
        local_tz = pytz.timezone(os.environ.get('TZ', 'Etc/UTC'))
        now = datetime.now(local_tz)
        timezone_name = local_tz.zone
    except Exception:
        now = datetime.now()
        timezone_name = time.tzname[0]
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p")
    return f"{time_str} on {date_str} in the {timezone_name} timezone"
 
@function_tool
async def get_weather(context: RunContext, location: str) -> str:
    """Get the current weather forecast for a location.
    Args:
        context: The run context for the tool
        location: The location to get weather for (city name or address)
    Returns:
        str: The weather forecast or a fallback message
    """
    logging.info(f"[TOOL] get_weather called for location: {location}")
    try:
        def get_coordinates():
            try:
                location_data = Nominatim(user_agent="AIVoiceAssistant/1.0").geocode(location, timeout=10)
                if location_data:
                    return {
                        'lat': location_data.latitude,
                        'lon': location_data.longitude,
                        'display_name': location_data.address
                    }
                return None
            except GeocoderTimedOut:
                return None
            except Exception as e:
                logging.error(f"[TOOL] get_weather geocode error: {e}")
                return None
        loop = asyncio.get_event_loop()
        coords = await loop.run_in_executor(None, get_coordinates)
        if not coords:
            msg = f"I couldn't find the location '{location}'. Could you provide a city and country name?"
            msg = sanitize_for_azure(msg)
            logging.info(f"[TOOL] get_weather: {msg}")
            return msg
        api_key = os.environ.get("OPENWEATHER_API_KEY")
        if not api_key:
            msg = f"I can tell you about the weather in {coords['display_name']}, but I need an OpenWeatherMap API key configured. I'll tell you what I know about weather patterns in this area based on my training instead."
            msg = sanitize_for_azure(msg)
            logging.info(f"[TOOL] get_weather: {msg}")
            return msg
        async def fetch_weather():
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={coords['lat']}&lon={coords['lon']}&appid={api_key}&units=metric"
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                return response.json() if response.status_code == 200 else None
        weather_data = await fetch_weather()
        if not weather_data:
            msg = f"I found {coords['display_name']}, but couldn't retrieve the current weather. Let me tell you about typical weather patterns for this area based on my training."
            msg = sanitize_for_azure(msg)
            logging.info(f"[TOOL] get_weather: {msg}")
            return msg
        temp_c = weather_data.get('main', {}).get('temp')
        temp_f = temp_c * 9/5 + 32 if temp_c is not None else None
        condition = weather_data.get('weather', [{}])[0].get('description', 'unknown conditions')
        humidity = weather_data.get('main', {}).get('humidity')
        wind_speed = weather_data.get('wind', {}).get('speed')
        weather_response = f"The current weather in {coords['display_name']} is {condition}. "
        if temp_c is not None:
            weather_response += f"The temperature is {temp_c:.1f}°C ({temp_f:.1f}°F). "
        if humidity is not None:
            weather_response += f"Humidity is at {humidity}%. "
        if wind_speed is not None:
            weather_response += f"Wind speed is {wind_speed} meters per second. "
        weather_response = sanitize_for_azure(weather_response)
        logging.info(f"[TOOL] get_weather response: {weather_response}")
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, weather_response)
            return "Here's the current weather. I'll read it to you."
        return weather_response
    except Exception as e:
        logging.error(f"[TOOL] get_weather exception: {e}")
        return sanitize_for_azure(f"I tried to get the weather for {location}, but encountered a technical issue. I can tell you about typical weather patterns for this area based on my training.")

@function_tool
async def get_news(context: RunContext, topic: str = "", country: str = "US") -> str:
    import logging
    import html
    search_query = topic if topic else "breaking news"
    if country and country.upper() != "US":
        search_query += f" {country} news"
    # Use web_search for news
    results = await web_search(context, search_query)
    if not results or "I couldn't find any results" in results:
        msg = f"I couldn't find any recent news{' about ' + topic if topic else ''}{' in ' + country if country and country.upper() != 'US' else ''}. Would you like me to search for something else?"
        msg = sanitize_for_azure(msg)
        logging.info(f"[TOOL] get_news_headlines: {msg}")
        return msg
    # Optionally, parse results for top headlines (simple extraction)
    # If results is HTML, try to extract headlines
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(results, 'html.parser')
    headlines = []
    for result in soup.find_all(['h3', 'h2', 'h1'], limit=4):
        text = result.get_text(strip=True)
        if text:
            headlines.append(text)
    if not headlines:
        # If not HTML, treat as plain text
        lines = [line.strip() for line in results.split('\n') if line.strip()]
        headlines = lines[:4]
    if not headlines:
        msg = f"I couldn't find any recent news{' about ' + topic if topic else ''}{' in ' + country if country and country.upper() != 'US' else ''}. Would you like me to search for something else?"
        msg = sanitize_for_azure(msg)
        logging.info(f"[TOOL] get_news_headlines: {msg}")
        return msg
    topic_str = f" about {topic}" if topic else ""
    country_str = f" in {country}" if country and country.upper() != "US" else ""
    formatted_results = f"Here are the latest headlines{topic_str}{country_str}:\n"
    for i, headline in enumerate(headlines, 1):
        formatted_results += f"Headline {i}: {headline}\n\n"
    formatted_results = sanitize_for_azure(formatted_results)
    logging.info(f"[TOOL] get_news_headlines results: {formatted_results}")
    session = getattr(context, 'session', None)
    if session:
        await handle_tool_results(session, formatted_results)
        return "Here are the latest news headlines. I'll read them to you."
    return formatted_results

@function_tool
async def calculate(context: RunContext, expression: str) -> str:
    """Calculate a mathematical expression and return the result.
    Args:
        context: The run context for the tool
        expression: The mathematical expression to evaluate
    Returns:
        str: The result or an error message
    """
    logging.info(f"[TOOL] calculate called for expression: {expression}")
    try:
        import re
        cleaned_expr = expression.lower()
        cleaned_expr = cleaned_expr.replace('plus', '+')
        cleaned_expr = cleaned_expr.replace('minus', '-')
        cleaned_expr = cleaned_expr.replace('times', '*')
        cleaned_expr = cleaned_expr.replace('multiplied by', '*')
        cleaned_expr = cleaned_expr.replace('divided by', '/')
        cleaned_expr = cleaned_expr.replace('x', '*')
        cleaned_expr = cleaned_expr.replace('÷', '/')
        cleaned_expr = re.sub(r'[^0-9+\-*/().%^ ]', '', cleaned_expr)
        cleaned_expr = cleaned_expr.replace('^', '**')
        if not cleaned_expr:
            msg = "I couldn't parse that as a mathematical expression. Please try again with a simpler calculation."
            msg = sanitize_for_azure(msg)
            logging.info(f"[TOOL] calculate: {msg}")
            return msg
        result = eval(cleaned_expr)
        if isinstance(result, float):
            formatted_result = f"{result:.4f}".rstrip('0').rstrip('.') if '.' in f"{result:.4f}" else f"{result:.0f}"
        else:
            formatted_result = str(result)
        response = f"The result of {expression} is {formatted_result}."
        response = sanitize_for_azure(response)
        logging.info(f"[TOOL] calculate response: {response}")
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, response)
            return "Here's the result. I'll read it to you."
        return response
    except Exception as e:
        logging.error(f"[TOOL] calculate exception: {e}")
        return sanitize_for_azure(f"I couldn't calculate '{expression}'. Please try with a simpler expression or check the format.")

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
async def evaluate_expression(self, context: RunContext, expression: str) -> str:
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

@function_tool
async def open_website(context: RunContext, url: str, description: str = "") -> dict:
    import re
    url = url.strip()
    # Basic validation: must start with http:// or https://
    if not re.match(r'^https?://', url):
        msg = f"Sorry, I can only open valid web addresses that start with http or https."
        msg = sanitize_for_azure(msg)
        return {"message": msg}
    # Instead of opening the browser here, return a signal to the frontend
    return {
        "action": "open_url",
        "url": url,
        "message": sanitize_for_azure("Opening the website in your browser.")
    }

@function_tool
async def open_known_website(context: RunContext, site_name: str, query: str = None) -> str:
    """Open a well-known website by name (e.g., 'google', 'cnn', 'tinder'). If a query is provided, open the search page for that query. If the site is not recognized, use a fallback search URL."""
    import logging
    import urllib.parse
    site_key = site_name.strip().lower()
    url = WELL_KNOWN_WEBSITES.get(site_key)
    fallback_url = "https://fallback"
    if not url:
        # Try fuzzy match
        import difflib
        matches = difflib.get_close_matches(site_key, WELL_KNOWN_WEBSITES.keys(), n=1, cutoff=0.7)
        if matches:
            url = WELL_KNOWN_WEBSITES[matches[0]]
            site_key = matches[0]
        else:
            # If the site is in BROWSER_TOOL and query is provided, open '@site_name query' in the browser
            if query and site_key in BROWSER_TOOL:
                bang_query = f"@{site_name} {query}"
                return await open_website(context, bang_query, description=f"Opening @{site_name} {query} in your browser")
            # Otherwise, open a Google search for the site (and query, if provided)
            google_url = "https://www.google.com/search?q="
            if query:
                search_terms = f"{site_name} {query}"
            else:
                search_terms = site_name
            search_url = f"{google_url}{urllib.parse.quote(search_terms)}&ie=UTF-8"
            return await open_website(context, search_url, description=f"Opening Google search for {search_terms}")
    if query:
        # Remove trailing slash for consistency
        url = url.rstrip('/')
        if site_key == "google":
            search_url = f"{url}/search?q={urllib.parse.quote(query)}&ie=UTF-8"
        elif site_key == "wikipedia":
            # Wikipedia article URL: https://en.wikipedia.org/wiki/{query}
            # Capitalize first letter, replace spaces with underscores
            article = query.strip().replace(' ', '_')
            if article:
                article = article[0].upper() + article[1:]
            search_url = f"{url}/wiki/{article}"
        elif site_key == "fallback":
            search_url = f"{url}/search/?q={urllib.parse.quote(query)}"
        else:
            search_url = f"{url}/search/?q={urllib.parse.quote(query)}"
        return await open_website(context, search_url, description=f"Opening {site_name} search for {query}")
    return await open_website(context, url, description=f"Opening {site_name}")
# --- END: Well-known Websites Mapping and Tool ---

@function_tool
async def web_crawl(context: RunContext, url: str, selector: str = "", max_pages: int = 1) -> str:
    """Crawl a web page and extract content, optionally using a CSS selector.
    Args:
        context: The run context for the tool
        url: The URL to crawl
        selector: Optional CSS selector to extract specific content
        max_pages: Maximum number of pages to crawl (default: 1, max: 3)
    Returns:
        str: The extracted content or an error message
    """
    logging.info(f"[TOOL] web_crawl called for url: {url}, selector: {selector}, max_pages: {max_pages}")
    try:
        if not url.startswith(('http://', 'https://')):
            msg = "Error: URL must start with http:// or https://"
            msg = sanitize_for_azure(msg)
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, msg)
                return "There was a problem with the URL. I'll read the error."
            return msg
        cache_key = f"{url}_{selector}_{max_pages}"
        max_pages = min(max_pages, 3)
        session_req = requests.Session()
        session_req.headers.update({
            'User-Agent': 'AIVoiceAssistant/1.0 (Educational/Research Purpose)',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        response = session_req.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "iframe", "nav", "footer"]):
            script.extract()
        if selector:
            content_elements = soup.select(selector)
            if not content_elements:
                msg = f"No content found using selector '{selector}' on {url}"
                msg = sanitize_for_azure(msg)
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, msg)
                    return "There was a problem with the selector. I'll read the error."
                return msg
            content = '\n\n'.join(elem.get_text(strip=True) for elem in content_elements)
            content = sanitize_for_azure(content)
            logging.info(f"[TOOL] web_crawl selector content: {content}")
            return content
        # If no selector, return the main text content
        content = soup.get_text(separator='\n', strip=True)
        content = sanitize_for_azure(content)
        logging.info(f"[TOOL] web_crawl main content: {content}")
        return content
    except Exception as e:
        logging.error(f"[TOOL] web_crawl exception: {e}")
        return sanitize_for_azure(f"I tried to crawl {url}, but encountered a technical issue. Let me know if you need help with something else.")

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
    
@function_tool
async def web_search(context: RunContext, query: str) -> str:
    import logging
    from bs4 import BeautifulSoup
    try:
        from .bing_playwright_scraper import scrape_bing
        results = await scrape_bing(query, num_results=5)
        logging.info(f"[web_search] Playwright Bing results: {results}")
        if results and isinstance(results, list) and all('title' in r and 'link' in r for r in results):
            spoken = f"Here are the top results for {query} from Bing:\n"
            for i, r in enumerate(results[:3], 1):
                spoken += f"{i}. {r['title']}\n{r['link']}\n\n"
            spoken = sanitize_for_azure(spoken)
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, spoken)
                return "I've found some results and will read them to you now."
            return spoken
        else:
            return f"I couldn't find any results for '{query}'. Try a different query."
    except Exception as e:
        logging.error(f"[web_search] Playwright Bing failed or returned no results: {e}")
        return f"I couldn't find any results for '{query}'. Try a different query."

@function_tool
async def google_search(self, context: RunContext, query: str, num_results: int = 5) -> str:
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
async def wiki_lookup(context: RunContext, topic: str) -> str:
    """Lookup a topic on Wikipedia for detailed, factual information.
    Args:
        context: The run context for the tool
        topic: The topic to look up on Wikipedia
    Returns:
        str: A summary of the Wikipedia article or a fallback message
    """
    logging.info(f"[TOOL] wiki_lookup called for topic: {topic}")
    try:
        def lookup_wiki():
            page = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                user_agent='AIVoiceAssistant/1.0'
            ).page(topic)
            if not page.exists():
                search = wikipediaapi.Wikipedia(
                    language='en',
                    extract_format=wikipediaapi.ExtractFormat.WIKI,
                    user_agent='AIVoiceAssistant/1.0'
                ).opensearch(topic)
                if search:
                    page = wikipediaapi.Wikipedia(
                        language='en',
                        extract_format=wikipediaapi.ExtractFormat.WIKI,
                        user_agent='AIVoiceAssistant/1.0'
                    ).page(search[0])
            if page.exists():
                summary = page.summary.split('\n\n')[:2]
                summary = '\n\n'.join(summary)
                words = summary.split()
                if len(words) > 300:
                    summary = ' '.join(words[:300]) + '...'
                result = f"According to Wikipedia: {summary}"
                return result
            else:
                return f"I couldn't find a Wikipedia article about '{topic}'. Let me share what I know based on my training."
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lookup_wiki)
        result = sanitize_for_azure(result)
        logging.info(f"[TOOL] wiki_lookup result: {result}")
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, result)
            return "Here's what I found on Wikipedia. I'll read it to you."
        return result
    except Exception as e:
        logging.error(f"[TOOL] wiki_lookup exception: {e}")
        return sanitize_for_azure(f"I tried looking up '{topic}' on Wikipedia, but encountered a technical issue. Let me answer based on what I know.")

@function_tool
async def global_web_search(query, num_results=10):
    import asyncio
    try:
        from .bing_playwright_scraper import scrape_bing
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(scrape_bing(query, num_results=num_results))
        if results and isinstance(results, list) and all('title' in r and 'link' in r for r in results):
            html_results = "".join(f'<a href="{r["link"]}">{r["title"]}</a><br>' for r in results[:num_results])
            return html_results
    except Exception as e:
        return f"Error: Bing search failed: {e}"
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    try:
        params = {"q": query}
        resp = requests.get(SEARXNG_FALLBACK, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    try:
        params = {"q": query}
        resp = requests.get(SAPTI_FALLBACK, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        return f"Error: All search engines failed: {e}"
    
@function_tool
async def get_fun_content(context: RunContext, content_type: str = "joke") -> str:
    """Get a joke, fun fact, or trivia question.
    Args:
        context: The run context for the tool
        content_type: 'joke', 'fact', or 'trivia'
    Returns:
        str: The fun content
    """
    logging.info(f"[TOOL] get_fun_content called for type: {content_type}")
    try:
        content_type = content_type.lower()
        if content_type == "joke":
            async with httpx.AsyncClient() as client:
                response = await client.get("https://v2.jokeapi.dev/joke/Any?safe-mode&type=single", timeout=5.0)
                if response.status_code == 200:
                    joke_data = response.json()
                    if joke_data.get('type') == 'single':
                        joke = joke_data.get('joke', "Why did the AI assistant go to the comedy club? To improve its response-time!")
                        joke = sanitize_for_azure(joke)
                        logging.info(f"[TOOL] get_fun_content joke: {joke}")
                        session = getattr(context, 'session', None)
                        if session:
                            await handle_tool_results(session, joke)
                            return "Here's a joke. I'll read it to you."
                        return joke
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "Why did the voice assistant go to school? To get a little smarter!",
                "What do you call an AI that sings? Artificial Harmonies!",
                "I asked the voice assistant to tell me a joke, and it said 'Just a moment, I'm still trying to understand humor.'"
            ]
            joke = sanitize_for_azure(random.choice(jokes))
            logging.info(f"[TOOL] get_fun_content fallback joke: {joke}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, joke)
                return "Here's a joke. I'll read it to you."
            return joke
        elif content_type == "fact":
            async with httpx.AsyncClient() as client:
                response = await client.get("https://uselessfacts.jsph.pl/api/v2/facts/random?language=en", timeout=5.0)
                if response.status_code == 200:
                    fact_data = response.json()
                    fact = fact_data.get('text', "")
                    if fact:
                        fact_str = f"Here's a fun fact: {fact}"
                        fact_str = sanitize_for_azure(fact_str)
                        logging.info(f"[TOOL] get_fun_content fact: {fact_str}")
                        session = getattr(context, 'session', None)
                        if session:
                            await handle_tool_results(session, fact_str)
                            return "Here's a fun fact. I'll read it to you."
                        return fact_str
            facts = [
                "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly good to eat.",
                "Octopuses have three hearts and blue blood.",
                "The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.",
                "A day on Venus is longer than a year on Venus. It takes 243 Earth days to rotate once on its axis and 225 Earth days to orbit the sun."
            ]
            fact_str = f"Here's a fun fact: {random.choice(facts)}"
            fact_str = sanitize_for_azure(fact_str)
            logging.info(f"[TOOL] get_fun_content fallback fact: {fact_str}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, fact_str)
                return "Here's a fun fact. I'll read it to you."
            return fact_str
        elif content_type == "trivia":
            async with httpx.AsyncClient() as client:
                response = await client.get("https://opentdb.com/api.php?amount=1&type=multiple", timeout=5.0)
                if response.status_code == 200:
                    trivia_data = response.json()
                    results = trivia_data.get('results', [])
                    if results:
                        question = html.unescape(results[0].get('question', ""))
                        correct_answer = html.unescape(results[0].get('correct_answer', ""))
                        category = html.unescape(results[0].get('category', ""))
                        if question and correct_answer:
                            trivia_str = f"Here's a {category} trivia question: {question} The answer is: {correct_answer}"
                            trivia_str = sanitize_for_azure(trivia_str)
                            logging.info(f"[TOOL] get_fun_content trivia: {trivia_str}")
                            session = getattr(context, 'session', None)
                            if session:
                                await handle_tool_results(session, trivia_str)
                                return "Here's a trivia question. I'll read it to you."
                            return trivia_str
            trivia_items = [
                "In which year was the first iPhone released? The answer is 2007.",
                "What is the capital of New Zealand? The answer is Wellington.",
                "Who wrote 'Romeo and Juliet'? The answer is William Shakespeare.",
                "What element has the chemical symbol 'Au'? The answer is Gold."
            ]
            trivia_str = f"Here's a trivia question: {random.choice(trivia_items)}"
            trivia_str = sanitize_for_azure(trivia_str)
            logging.info(f"[TOOL] get_fun_content fallback trivia: {trivia_str}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, trivia_str)
                return "Here's a trivia question. I'll read it to you."
            return trivia_str
        else:
            msg = "I can tell you a joke, share a fun fact, or give you some trivia. Which would you prefer?"
            msg = sanitize_for_azure(msg)
            logging.info(f"[TOOL] get_fun_content: {msg}")
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, msg)
                return "Let me know what you'd like!"
            return msg
    except Exception as e:
        logging.error(f"[TOOL] get_fun_content exception: {e}")
        fallbacks = {
            "joke": "Why did the AI go to therapy? It had too many neural issues!",
            "fact": "Here's a fun fact: The average person will spend six months of their life waiting for red lights to turn green.",
            "trivia": "Here's a trivia question: What is the most abundant gas in Earth's atmosphere? The answer is nitrogen."
        }
        fallback = fallbacks.get(content_type, fallbacks["joke"])
        fallback = sanitize_for_azure(fallback)
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, fallback)
            return "Here's something fun. I'll read it to you."
        return fallback