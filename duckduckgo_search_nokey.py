"""
DuckDuckGo Search implementation without API key using web scraping.
This module provides web search functionality by scraping DuckDuckGo Search:
1. No API key required
2. Caching to minimize requests
3. Rate limiting to avoid being blocked
4. Detailed search results with web content
"""

import asyncio
import logging
import os
import time
import json
import hashlib
import re
import random
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from livekit.agents import RunContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple in-memory cache
_cache = {}
_cache_ttl = 3600  # 1 hour in seconds

# Create a custom DDGS class to avoid the circular import
class DDGS:
    """Custom DuckDuckGo Search class to avoid circular imports."""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def text(self, query: str, max_results: int = 5):
        """Search DuckDuckGo for text results."""
        results = []
        
        # Check cache first
        cache_key = f"ddg_nokey:{query}:{max_results}"
        if cache_key in _cache:
            logger.info(f"Cache hit for query: {query}")
            return _cache[cache_key]
        
        try:
            # Respect rate limits
            await asyncio.sleep(1.0 + random.random())
            
            # Prepare the search URL
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            # Make the request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            async with self.session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Error: DuckDuckGo Search returned status code {response.status}")
                    return []
                
                html = await response.text()
                
                # Parse the HTML
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract search results
                search_results = soup.select('.result')
                
                for i, result in enumerate(search_results):
                    if i >= max_results:
                        break
                        
                    title_elem = result.select_one('.result__title')
                    url_elem = result.select_one('.result__url')
                    desc_elem = result.select_one('.result__snippet')
                    
                    title = title_elem.get_text() if title_elem else "No title"
                    url = url_elem.get_text() if url_elem else "No URL"
                    description = desc_elem.get_text() if desc_elem else "No description"
                    
                    result_dict = {
                        "title": title,
                        "href": url,
                        "body": description
                    }
                    
                    results.append(result_dict)
            
            # Cache results
            _cache[cache_key] = results
            
            return results
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {str(e)}")
            return []

async def duckduckgo_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo Search without requiring an API key.
    
    Args:
        context: The run context from the agent
        query: The search query string
        num_results: Maximum number of results to return
        
    Returns:
        A formatted string containing search results
    """
    logger.info(f"Performing DuckDuckGo search for: {query}")
    
    try:
        # Create DDGS instance without using it as an async context manager
        ddgs = DDGS()
        # Create a session manually
        ddgs.session = aiohttp.ClientSession()
        
        try:
            # Perform the search
            results = await ddgs.text(query, max_results=num_results)
            
            if not results:
                return "No results found."
                
            # Format the results
            formatted_results = []
            for result in results:
                title = result.get("title", "No title")
                url = result.get("href", "No URL")
                description = result.get("body", "No description")
                
                formatted_results.append(f"Title: {title}\nURL: {url}\nDescription: {description}\n")
            
            return "\n".join(formatted_results)
        finally:
            # Close the session manually
            if ddgs.session:
                await ddgs.session.close()
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        return f"Error performing search: {str(e)}"
