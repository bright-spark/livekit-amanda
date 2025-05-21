import os
import logging
import asyncio
import json
import time
import httpx
import pytz
import random
import wikipediaapi
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, Agent, AgentSession, function_tool, RunContext
from livekit.agents.llm import ChatContext, ChatMessage, ChatChunk, function_tool
from livekit.plugins import azure, openai, silero
from mcp_client import MCPServerSse
from mcp_client.agent_tools import MCPToolsIntegration
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
from typing import List, Dict, Any, Optional, Union, Literal, TypedDict, Tuple
import re
import webbrowser
import threading
import urllib.parse
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import redis as redis_pkg
from playwright.async_api import async_playwright
import brotli
from lxml_html_clean import Cleaner
import html5lib

load_dotenv()

class FunctionAgent(Agent):
    """A LiveKit agent that uses MCP tools from one or more MCP servers."""

    def __init__(self):
        super().__init__(
            instructions="""
                You are Amanda, an advanced AI assistant with access to a wide range of tools and capabilities.
                Your primary goal is to be helpful, informative, and efficient in your responses.
                
                CAPABILITIES:
                - Web Search: Find information from across the internet
                - Web Scraping: Extract and analyze content from web pages
                - Locanto Integration: Search and browse Locanto listings
                - Wikipedia Lookup: Get detailed information on various topics
                - Weather Information: Get current weather conditions
                - Time and Date: Provide accurate time and date information
                - Mathematical Calculations: Perform complex calculations
                - News: Fetch the latest headlines on any topic
                - Browser Automation: Interact with websites as needed
                - Data Processing: Clean and process text/data
                
                GUIDELINES:
                1. Always verify information when possible
                2. Be concise but thorough in responses
                3. Use the most appropriate tool for each request
                4. Handle sensitive information with care
                5. Provide sources when available
                6. Ask for clarification if a request is unclear
                7. Respect user privacy and data protection
                
                When using web search or scraping tools, be mindful of the website's terms of service.
                Always present information in a clear, organized manner.
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
                temperature=0.7,
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
