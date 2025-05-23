"""
Google Search implementation without API key using the google-search module.
This module provides web search functionality using the google-search package:
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
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# The google-search package has a different import structure than expected
# Try different import approaches
try:
    from googlesearch import search
except ImportError:
    try:
        from google_search import search
    except ImportError:
        try:
            from google import search
        except ImportError:
            # Fallback implementation if the package structure is different
            import requests
            from bs4 import BeautifulSoup
            import urllib.parse
            
            def search(query, num_results=10, lang='en', advanced=False, sleep_interval=1, timeout=5):
                """Fallback implementation of Google search"""
                search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}&num={num_results}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                results = []
                try:
                    response = requests.get(search_url, headers=headers, timeout=timeout)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for g in soup.select('.g'):
                        title_elem = g.select_one('h3')
                        link_elem = g.select_one('a')
                        desc_elem = g.select_one('.VwiC3b')
                        
                        if title_elem and link_elem and 'href' in link_elem.attrs:
                            title = title_elem.get_text()
                            link = link_elem['href']
                            if link.startswith('/url?q='):
                                link = link.split('/url?q=')[1].split('&')[0]
                            description = desc_elem.get_text() if desc_elem else ""
                            
                            class Result:
                                def __init__(self, title, url, description):
                                    self.title = title
                                    self.url = url
                                    self.description = description
                            
                            results.append(Result(title, link, description))
                            
                            if len(results) >= num_results:
                                break
                except Exception as e:
                    logger.error(f"Error in fallback Google search: {str(e)}")
                
                return results
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

async def google_search(context: RunContext, query: str, num_results: int = 5) -> str:
    """Search the web using Google Search without requiring an API key.
    
    Args:
        context: The run context from the agent
        query: The search query string
        num_results: Maximum number of results to return
        
    Returns:
        A formatted string containing search results
    """
    logger.info(f"Performing Google search (nokey) for: {query}")
    
    # Check cache first
    cache_key = f"google_nokey:{query}:{num_results}"
    if cache_key in _cache:
        logger.info(f"Cache hit for query: {query}")
        return _cache[cache_key]
    
    try:
        # Use the google-search module to perform the search
        # This needs to be run in a thread pool since it's a blocking operation
        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(
            None,
            lambda: list(search(
                query,
                num_results=num_results,
                lang='en',
                advanced=True,
                sleep_interval=1,  # Be respectful with rate limits
                timeout=10
            ))
        )
        
        # Format the results
        results = []
        for result in search_results:
            title = result.title if hasattr(result, 'title') and result.title else "No title"
            url = result.url if hasattr(result, 'url') and result.url else "No URL"
            description = result.description if hasattr(result, 'description') and result.description else "No description"
            
            results.append(f"Title: {title}\nURL: {url}\nDescription: {description}\n")
        
        formatted_results = "\n".join(results) if results else "No results found."
        
        # Cache results
        _cache[cache_key] = formatted_results
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error in Google search (nokey): {str(e)}")
        return f"Error performing search: {str(e)}"
