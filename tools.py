# Standard library imports
import asyncio
import html
import logging
import os
import random
import re
import time
from datetime import datetime
from urllib.parse import urljoin

# Third-party imports
import httpx
import pytz
import requests
from bs4 import BeautifulSoup
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from livekit.agents import function_tool, RunContext

# Handle relative imports with try/except for flexibility
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
    from utils import sanitize_for_azure, clean_spoken, handle_tool_results
except ImportError:
    try:
        from .utils import sanitize_for_azure, clean_spoken, handle_tool_results
    except ImportError:
        logging.warning("utils module not available, using fallback definitions")
        # Fallback definitions
        def sanitize_for_azure(text):
            return text
            
        def clean_spoken(text):
            return text
            
        async def handle_tool_results(session, text):
            pass

@function_tool
async def get_current_time(context: RunContext, timezone: str = "UTC") -> str:
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
async def get_current_date_and_timezone(context: RunContext) -> str:
    """Get the current server date and time in a natural language format with timezone."""
    try:
        # Get local timezone
        local_tz = pytz.timezone(os.environ.get('TZ', 'Etc/UTC'))
        now = datetime.now(local_tz)
        timezone_name = local_tz.zone
    except Exception as e:
        logging.error(f"Timezone error: {e}")
        # Fallback to UTC
        now = datetime.now(pytz.UTC)
        timezone_name = "UTC"
        
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p")
    response = f"{time_str} on {date_str} in the {timezone_name} timezone"
    
    # Handle speech output if session is available
    session = getattr(context, 'session', None)
    if session:
        await handle_tool_results(session, response)
        return "Here's the current date and time. I'll read it to you."
    return response
 
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
        # Use safer calculation method with ast
        import ast
        import operator
        
        # Define safe operations
        safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,  # Unary negation
        }
        
        def safe_eval(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                return safe_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = safe_eval(node.operand)
                return safe_operators[type(node.op)](operand)
            else:
                raise TypeError(f"Unsupported operation: {node.__class__.__name__}")
        
        # Parse the expression into an AST
        parsed_expr = ast.parse(cleaned_expr, mode='eval').body
        # Evaluate the expression safely
        result = safe_eval(parsed_expr)
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
async def calculate_math(context: RunContext, expression: str) -> str:
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
        
        # Use safer calculation method
        import ast
        import operator
        
        # Define safe operations
        safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,  # Unary negation
        }
        
        def safe_eval(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                return safe_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = safe_eval(node.operand)
                return safe_operators[type(node.op)](operand)
            else:
                raise TypeError(f"Unsupported operation: {node.__class__.__name__}")
        
        try:
            # Parse the expression into an AST
            parsed_expr = ast.parse(expression, mode='eval').body
            # Evaluate the expression safely
            result = safe_eval(parsed_expr)
            return f"The result of {expression} is {result}."
        except Exception as e:
            logging.error(f"AST evaluation error: {e}")
            return sanitize_for_azure(f"I couldn't calculate '{expression}'. Please try with a simpler expression.")
        
    except Exception as e:
        logging.error(f"Calculation error: {e}", exc_info=True)
        return "I couldn't evaluate that expression. Please check the format and try again."

@function_tool
async def evaluate_expression(context: RunContext, expression: str) -> str:
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
            error_msg = "Invalid expression. Only numbers and basic arithmetic operators are allowed."
            error_msg = sanitize_for_azure(error_msg)
            
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I can't process that expression. Please try again with basic math operators."
            return error_msg
            
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        # Use safer calculation method with ast
        import ast
        import operator
        
        # Define safe operations
        safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,  # Unary negation
        }
        
        def safe_eval(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                return safe_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = safe_eval(node.operand)
                return safe_operators[type(node.op)](operand)
            else:
                raise TypeError(f"Unsupported operation: {node.__class__.__name__}")
        
        # Parse the expression into an AST
        parsed_expr = ast.parse(expression, mode='eval').body
        # Evaluate the expression safely
        result = safe_eval(parsed_expr)
        
        # Format the result
        if isinstance(result, float):
            formatted_result = f"{result:.4f}".rstrip('0').rstrip('.') if '.' in f"{result:.4f}" else f"{result:.0f}"
        else:
            formatted_result = str(result)
            
        response = f"The result of {expression.replace('**', '^')} is {formatted_result}."
        response = sanitize_for_azure(response)
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, response)
            return "Here's the result. I'll read it to you."
        return response
        
    except Exception as e:
        logging.error(f"Calculation error: {e}", exc_info=True)
        error_msg = f"I couldn't evaluate '{expression}'. Please check the format and try again."
        error_msg = sanitize_for_azure(error_msg)
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I couldn't calculate that. Please try a different expression."
        return error_msg

@function_tool
async def take_screenshot(context: RunContext, url: str, selector: str = "body") -> str:
    """Take a screenshot of a webpage and return a description.
    
    Args:
        context: The run context for the tool
        url: The URL of the webpage to screenshot
        selector: CSS selector to screenshot (defaults to entire page)
        
    Returns:
        str: A description of the screenshot or an error message
    """
    try:
        from playwright.async_api import async_playwright
        import os
        import time
        
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        logging.info(f"[TOOL] take_screenshot: Taking screenshot of {url} with selector {selector}")
        
        # Create screenshots directory if it doesn't exist
        os.makedirs('screenshots', exist_ok=True)
        screenshot_path = f"screenshots/screenshot_{int(time.time())}.png"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={'width': 1280, 'height': 800})
            page = await context.new_page()
            
            # Navigate to the page with timeout
            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if not response or not response.ok:
                    await browser.close()
                    error_msg = f"Failed to load {url}. Status: {response.status if response else 'No response'}"
                    error_msg = sanitize_for_azure(error_msg)
                    
                    session = getattr(context, 'session', None)
                    if session:
                        await handle_tool_results(session, error_msg)
                        return "I couldn't load that webpage."
                    return error_msg
            except Exception as nav_error:
                await browser.close()
                logging.error(f"Navigation error: {nav_error}")
                error_msg = f"Error loading {url}: {str(nav_error)}"
                error_msg = sanitize_for_azure(error_msg)
                
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, error_msg)
                    return "I had trouble loading that webpage."
                return error_msg
            
            # Wait for the page to be fully loaded
            try:
                await page.wait_for_load_state('networkidle', timeout=10000)
            except Exception as load_error:
                logging.warning(f"Page load state timeout: {load_error}")
                # Continue anyway, as the page might still be usable
            
            # Try to find the specified element
            try:
                await page.wait_for_selector(selector, state='attached', timeout=5000)
                element = page.locator(selector)
                await element.scroll_into_view_if_needed()
                # Add a small delay to ensure any lazy-loaded content is visible
                await asyncio.sleep(1)
            except Exception as selector_error:
                logging.warning(f"Selector {selector} not found: {selector_error}")
                # If specific selector not found, take full page screenshot
                selector = 'body'
                element = page.locator(selector)
            
            # Take screenshot with retry logic
            max_retries = 2
            success = False
            for attempt in range(max_retries):
                try:
                    await element.screenshot(path=screenshot_path, timeout=10000)
                    success = True
                    break
                except Exception as screenshot_error:
                    if attempt == max_retries - 1:
                        await browser.close()
                        logging.error(f"Screenshot error: {screenshot_error}")
                        error_msg = f"Failed to take screenshot: {str(screenshot_error)}"
                        error_msg = sanitize_for_azure(error_msg)
                        
                        session = getattr(context, 'session', None)
                        if session:
                            await handle_tool_results(session, error_msg)
                            return "I couldn't take a screenshot of that webpage."
                        return error_msg
                    await asyncio.sleep(1)
            
            # Clean up
            await browser.close()
            
            if success and os.path.exists(screenshot_path):
                file_size = os.path.getsize(screenshot_path) / 1024  # Size in KB
                if file_size < 1:  # If file is too small, it might be empty
                    error_msg = "Screenshot file is too small or empty"
                    error_msg = sanitize_for_azure(error_msg)
                    
                    session = getattr(context, 'session', None)
                    if session:
                        await handle_tool_results(session, error_msg)
                        return "I couldn't capture a proper screenshot of that webpage."
                    return error_msg
                
                response = f"I've taken a screenshot of {url} focusing on the '{selector}' element. The screenshot has been saved as {os.path.abspath(screenshot_path)}"
                response = sanitize_for_azure(response)
                
                # Handle session output for voice responses
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, response)
                    return "I've taken a screenshot of that webpage."
                return response
            else:
                error_msg = "Failed to save screenshot"
                error_msg = sanitize_for_azure(error_msg)
                
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, error_msg)
                    return "I couldn't save the screenshot."
                return error_msg
            
    except Exception as e:
        logging.error(f"Screenshot error: {e}", exc_info=True)
        error_msg = f"I couldn't take a screenshot: {str(e)}. Please try again or check the URL and selector."
        error_msg = sanitize_for_azure(error_msg)
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I encountered an error while taking the screenshot."
        return error_msg

@function_tool
async def clean_html(context: RunContext, html_content: str) -> str:
    """Clean and sanitize HTML content, removing scripts and unwanted tags.
    
    Args:
        context: The run context for the tool
        html_content: The HTML content to clean
        
    Returns:
        str: The cleaned HTML content
    """
    try:
        from bs4 import BeautifulSoup
        
        if not html_content or not isinstance(html_content, str):
            error_msg = "Invalid HTML content provided. Please provide valid HTML content."
            error_msg = sanitize_for_azure(error_msg)
            
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't process that HTML content."
            return error_msg
        
        logging.info(f"[TOOL] clean_html: Cleaning HTML content of length {len(html_content)}")
        
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and iframe tags for security
        for tag in soup(["script", "style", "iframe", "noscript"]):
            tag.extract()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if too long
        if len(text) > 8000:
            text = text[:8000] + "... [content truncated]"
        
        response = f"Cleaned HTML content:\n\n{text}"
        response = sanitize_for_azure(response)
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, response)
            return "I've cleaned the HTML content. Here's a summary of what I found."
        return response
        
    except Exception as e:
        logging.error(f"HTML cleaning error: {e}", exc_info=True)
        error_msg = f"I couldn't clean the HTML content: {str(e)}"
        error_msg = sanitize_for_azure(error_msg)
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I encountered an error while cleaning the HTML content."
        return error_msg

@function_tool
async def extract_links(context: RunContext, url: str, filter_pattern: str = None) -> str:
    """Extract all links from a webpage, optionally filtered by a pattern.
    
    Args:
        context: The run context for the tool
        url: The URL to extract links from
        filter_pattern: Optional regex pattern to filter links
        
    Returns:
        str: A list of extracted links
    """
    try:
        import re
        import httpx
        from bs4 import BeautifulSoup
        
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        logging.info(f"[TOOL] extract_links: Extracting links from {url} with filter {filter_pattern if filter_pattern else 'None'}")
        
        # Fetch the webpage
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()
        except httpx.HTTPError as http_error:
            error_msg = f"Failed to fetch {url}: {str(http_error)}"
            error_msg = sanitize_for_azure(error_msg)
            
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, error_msg)
                return "I couldn't access that webpage."
            return error_msg
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                href = url.rstrip('/') + href
            elif not href.startswith(('http://', 'https://')):
                # Handle other relative URLs
                from urllib.parse import urljoin
                href = urljoin(url, href)
            
            # Filter links if pattern provided
            if filter_pattern:
                try:
                    if not re.search(filter_pattern, href, re.IGNORECASE):
                        continue
                except re.error as regex_error:
                    error_msg = f"Invalid regex pattern '{filter_pattern}': {str(regex_error)}"
                    error_msg = sanitize_for_azure(error_msg)
                    
                    session = getattr(context, 'session', None)
                    if session:
                        await handle_tool_results(session, error_msg)
                        return "The filter pattern you provided isn't valid. Please try again with a valid regex pattern."
                    return error_msg
                
            links.append(href)
        
        # Remove duplicates
        links = list(dict.fromkeys(links))
        
        if not links:
            no_links_msg = f"No links found on {url}" + (f" matching pattern '{filter_pattern}'" if filter_pattern else "")
            no_links_msg = sanitize_for_azure(no_links_msg)
            
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, no_links_msg)
                return "I couldn't find any links on that webpage."
            return no_links_msg
        
        result = f"Found {len(links)} links" + (f" matching pattern '{filter_pattern}'" if filter_pattern else "") + f" on {url}:\n\n"
        for i, link in enumerate(links[:20], 1):
            result += f"{i}. {link}\n"
            
        if len(links) > 20:
            result += f"\n... and {len(links) - 20} more links."
        
        result = sanitize_for_azure(result)
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, result)
            return f"I found {len(links)} links on that webpage. I'll read some of them to you."
        return result
        
    except Exception as e:
        logging.error(f"Link extraction error: {e}", exc_info=True)
        error_msg = f"I couldn't extract links from {url}: {str(e)}"
        error_msg = sanitize_for_azure(error_msg)
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I encountered an error while extracting links from that webpage."
        return error_msg

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
async def web_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """Search the web for information on a given query.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of search results to return (default: 5, max: 10)
        
    Returns:
        str: The search results or an error message
    """
    logging.info(f"[TOOL] web_search called for query: {query}, num_results: {num_results}")
    try:
        # Limit the number of results
        num_results = min(num_results, 10)
        
        # Use Google search to get results
        search_results = []
        for result in google_search(query, num=num_results, stop=num_results, pause=1.0):
            search_results.append(result)
        
        if not search_results:
            msg = f"No search results found for query: {query}"
            msg = sanitize_for_azure(msg)
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, msg)
                return "I couldn't find any search results for that query."
            return msg
        
        # Format the results
        formatted_results = f"Search results for '{query}':\n\n"
        for i, result in enumerate(search_results, 1):
            formatted_results += f"{i}. {result}\n"
        
        formatted_results = sanitize_for_azure(formatted_results)
        logging.info(f"[TOOL] web_search results: {formatted_results}")
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, formatted_results)
            return f"I've found {len(search_results)} search results for '{query}'. I'll read them to you."
        return formatted_results
    except Exception as e:
        logging.error(f"[TOOL] web_search exception: {e}")
        error_msg = f"I encountered an error while searching for '{query}': {str(e)}"
        error_msg = sanitize_for_azure(error_msg)
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I encountered an error while searching. I'll read the error message to you."
        return error_msg

@function_tool
async def google_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """Search Google for information on a given query.
    
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of search results to return (default: 5, max: 10)
        
    Returns:
        str: The search results or an error message
    """
    logging.info(f"[TOOL] google_search called for query: {query}, num_results: {num_results}")
    try:
        # Limit the number of results
        num_results = min(num_results, 10)
        
        # Use Google search to get results
        search_results = []
        for result in google_search(query, num=num_results, stop=num_results, pause=1.0):
            search_results.append(result)
        
        if not search_results:
            msg = f"No Google search results found for query: {query}"
            msg = sanitize_for_azure(msg)
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, msg)
                return "I couldn't find any Google search results for that query."
            return msg
        
        # Format the results
        formatted_results = f"Google search results for '{query}':\n\n"
        for i, result in enumerate(search_results, 1):
            formatted_results += f"{i}. {result}\n"
        
        formatted_results = sanitize_for_azure(formatted_results)
        logging.info(f"[TOOL] google_search results: {formatted_results}")
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, formatted_results)
            return f"I've found {len(search_results)} Google search results for '{query}'. I'll read them to you."
        return formatted_results
    except Exception as e:
        logging.error(f"[TOOL] google_search exception: {e}")
        error_msg = f"I encountered an error while searching Google for '{query}': {str(e)}"
        error_msg = sanitize_for_azure(error_msg)
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I encountered an error while searching Google. I'll read the error message to you."
        return error_msg

@function_tool
async def wikipedia_search(context: RunContext, query: str, max_results: int = 3) -> str:
    """Search Wikipedia for information on a given query.
    
    Args:
        context: The run context for the tool
        query: The search query
        max_results: Maximum number of results to return (default: 3, max: 5)
        
    Returns:
        str: The search results or an error message
    """
    logging.info(f"[TOOL] wikipedia_search called for query: {query}, max_results: {max_results}")
    try:
        # Limit the number of results
        max_results = min(max_results, 5)
        
        # Use Wikipedia API to search
        wiki_wiki = wikipediaapi.Wikipedia('en')
        search_results = wiki_wiki.page(query)
        
        if not search_results.exists():
            msg = f"No Wikipedia article found for query: {query}"
            msg = sanitize_for_azure(msg)
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, msg)
                return "I couldn't find any Wikipedia article for that query."
            return msg
        
        # Get the summary
        summary = search_results.summary[0:1000] + "..." if len(search_results.summary) > 1000 else search_results.summary
        
        formatted_results = f"Wikipedia search result for '{query}':\n\n{summary}\n\nURL: {search_results.fullurl}"
        formatted_results = sanitize_for_azure(formatted_results)
        logging.info(f"[TOOL] wikipedia_search results: {formatted_results}")
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, formatted_results)
            return f"I've found a Wikipedia article for '{query}'. I'll read the summary to you."
        return formatted_results
    except Exception as e:
        logging.error(f"[TOOL] wikipedia_search exception: {e}")
        error_msg = f"I encountered an error while searching Wikipedia for '{query}': {str(e)}"
        error_msg = sanitize_for_azure(error_msg)
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I encountered an error while searching Wikipedia. I'll read the error message to you."
        return error_msg

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
async def scrape_website(context: RunContext, url: str, selector: str = "body", text_only: bool = True) -> str:
    """Scrape content from a website using a CSS selector.
    
    Args:
        context: The run context for the tool
        url: The URL to scrape
        selector: CSS selector to target specific content (default: body)
        text_only: Whether to return only text or HTML (default: True)
        
    Returns:
        str: The scraped content
    """
    try:
        from playwright.async_api import async_playwright
        
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        logging.info(f"[TOOL] scrape_website: Scraping {url} with selector {selector}, text_only={text_only}")
            
        async with async_playwright() as p:
            # Launch browser with a timeout
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={'width': 1280, 'height': 800})
            page = await context.new_page()
            
            # Navigate to the page with timeout
            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if not response or not response.ok:
                    await browser.close()
                    error_msg = f"Failed to load {url}. Status: {response.status if response else 'No response'}"
                    error_msg = sanitize_for_azure(error_msg)
                    
                    session = getattr(context, 'session', None)
                    if session:
                        await handle_tool_results(session, error_msg)
                        return "I couldn't access that webpage."
                    return error_msg
            except Exception as nav_error:
                await browser.close()
                logging.error(f"Navigation error: {nav_error}")
                error_msg = f"Error loading {url}: {str(nav_error)}"
                error_msg = sanitize_for_azure(error_msg)
                
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, error_msg)
                    return "I had trouble loading that webpage."
                return error_msg
            
            # Wait for the page to be fully loaded
            try:
                await page.wait_for_load_state('networkidle', timeout=10000)
            except Exception as load_error:
                logging.warning(f"Page load state timeout: {load_error}")
                # Continue anyway, as the page might still be usable
            
            # Wait for selector
            try:
                await page.wait_for_selector(selector, timeout=10000)
            except Exception as e:
                logging.warning(f"Selector {selector} not found: {e}")
                await browser.close()
                error_msg = f"Could not find element matching selector '{selector}' on the page."
                error_msg = sanitize_for_azure(error_msg)
                
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, error_msg)
                    return "I couldn't find that element on the webpage."
                return error_msg
            
            # Extract content based on parameters
            try:
                if text_only:
                    content = await page.locator(selector).inner_text()
                else:
                    content = await page.locator(selector).inner_html()
            except Exception as extract_error:
                await browser.close()
                logging.error(f"Content extraction error: {extract_error}")
                error_msg = f"Error extracting content from selector '{selector}': {str(extract_error)}"
                error_msg = sanitize_for_azure(error_msg)
                
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, error_msg)
                    return "I had trouble extracting content from that webpage."
                return error_msg
            
            # Clean up
            await browser.close()
            
            # Clean and truncate the content
            import re
            content = re.sub(r'\s+', ' ', content).strip()
            if len(content) > 8000:  # Limit response length
                content = content[:8000] + '... [content truncated]'
            
            if not content.strip():
                error_msg = f"No content found in the selected element on {url}"
                error_msg = sanitize_for_azure(error_msg)
                
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, error_msg)
                    return "I couldn't find any content in that element."
                return error_msg
            
            response = f"Content from {url} (selector: {selector}):\n\n{content}"
            response = sanitize_for_azure(response)
            
            # Handle session output for voice responses
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, response)
                return "I've scraped the content from that webpage. Here's what I found."
            return response
            
    except Exception as e:
        logging.error(f"Web scraping error: {e}", exc_info=True)
        error_msg = f"I encountered an error while scraping {url}: {str(e)}"
        error_msg = sanitize_for_azure(error_msg)
        
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I encountered an error while scraping that webpage."
        return error_msg
    
@function_tool
async def web_search(context: RunContext, query: str) -> str:
    import logging
    from bs4 import BeautifulSoup
    try:
        try:
            from bing_playwright_scraper import scrape_bing
        except ImportError:
            try:
                from .bing_playwright_scraper import scrape_bing
            except ImportError:
                logging.error("bing_playwright_scraper module not available")
                return f"I couldn't find any results for '{query}'. The search functionality is currently unavailable."
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
        error_msg = f"I couldn't find any results for '{query}'. Try a different query or approach."
        error_msg = sanitize_for_azure(error_msg)
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I couldn't find any results for your search."
        return error_msg

@function_tool
async def google_search(context: RunContext, query: str, num_results: int = 5) -> str:
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
        
        # Import the library locally to avoid name collision
        from googlesearch import search as google_search_lib
        for result in google_search_lib(query, num_results=num_results):
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
        
        response = sanitize_for_azure(response.strip())
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, response)
            return "I've found some results and will read them to you now."
        return response
        
    except Exception as e:
        logging.error(f"Web search error: {e}", exc_info=True)
        return "I encountered an error while searching the web. Please try again later."

@function_tool
async def wikipedia_search(context: RunContext, query: str) -> str:
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
        
        response = f"According to Wikipedia: {summary}\n\nRead more: {page.fullurl}"
        response = sanitize_for_azure(response)
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, response)
            return "Here's what I found on Wikipedia. I'll read it to you."
        return response
        
    except Exception as e:
        logging.error(f"Wikipedia search error: {e}", exc_info=True)
        error_msg = f"I encountered an error while searching Wikipedia for '{query}'. Let me answer based on what I know."
        error_msg = sanitize_for_azure(error_msg)
        
        # Handle session output for voice responses
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, error_msg)
            return "I'll share what I know about this topic."
        return error_msg

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
async def fallback_web_search(context: RunContext, query: str, num_results: int = 10) -> str:
    """Search the web for information using multiple search engines when the primary search fails.
        
    Args:
        context: The run context for the tool
        query: The search query
        num_results: Number of results to return (1-10)
        
    Returns:
        str: Formatted search results with titles and URLs
    """
    # Define fallback search URLs
    SEARXNG_FALLBACK = "https://searx.be/search"
    SAPTI_FALLBACK = "https://search.sapti.me/search"
    FALLBACK_URL = "https://duckduckgo.com/"
    
    logging.info(f"[TOOL] fallback_web_search called for query: {query}")
    
    # Try Bing search first
    try:
        try:
            from bing_playwright_scraper import scrape_bing
        except ImportError:
            try:
                from .bing_playwright_scraper import scrape_bing
            except ImportError:
                logging.error("bing_playwright_scraper module not available")
                raise ImportError("bing_playwright_scraper module not available")
        results = await scrape_bing(query, num_results=num_results)
        if results and isinstance(results, list) and all('title' in r and 'link' in r for r in results):
            response = f"Here are the top {len(results)} results for '{query}':\n\n"
            for i, result in enumerate(results[:num_results], 1):
                response += f"{i}. {result['title']}\n   {result['link']}\n\n"
            
            response = sanitize_for_azure(response)
            session = getattr(context, 'session', None)
            if session:
                await handle_tool_results(session, response)
                return "I've found some results and will read them to you now."
            return response
    except Exception as e:
        logging.error(f"[TOOL] fallback_web_search Bing error: {e}")
    
    # Try SearXNG
    try:
        async with httpx.AsyncClient() as client:
            params = {"q": query}
            response = await client.get(SEARXNG_FALLBACK, params=params, timeout=10.0)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                for result in soup.select('.result')[:num_results]:
                    title_elem = result.select_one('.result-title')
                    url_elem = result.select_one('.result-url')
                    if title_elem and url_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'url': url_elem.get_text(strip=True)
                        })
                
                if results:
                    response_text = f"Here are the top {len(results)} results for '{query}':\n\n"
                    for i, result in enumerate(results, 1):
                        response_text += f"{i}. {result['title']}\n   {result['url']}\n\n"
                    
                    response_text = sanitize_for_azure(response_text)
                    session = getattr(context, 'session', None)
                    if session:
                        await handle_tool_results(session, response_text)
                        return "I've found some results and will read them to you now."
                    return response_text
    except Exception as e:
        logging.error(f"[TOOL] fallback_web_search SearXNG error: {e}")
    
    # Try final fallback
    try:
        async with httpx.AsyncClient() as client:
            params = {"q": query}
            response = await client.get(FALLBACK_URL, params=params, timeout=10.0)
            if response.status_code == 200:
                fallback_msg = f"I found some results for '{query}', but I can't display them in detail. You might want to try searching for this query directly."
                fallback_msg = sanitize_for_azure(fallback_msg)
                session = getattr(context, 'session', None)
                if session:
                    await handle_tool_results(session, fallback_msg)
                    return "I'll share what I found."
                return fallback_msg
    except Exception as e:
        logging.error(f"[TOOL] fallback_web_search final fallback error: {e}")
    
    # All searches failed
    error_msg = f"I couldn't find any results for '{query}' using any available search engines. Please try a different query or approach."
    error_msg = sanitize_for_azure(error_msg)
    session = getattr(context, 'session', None)
    if session:
        await handle_tool_results(session, error_msg)
        return "I couldn't find any results for your search."
    return error_msg
    
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