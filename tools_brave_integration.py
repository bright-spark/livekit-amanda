"""
Integration module to connect tools.py web search functionality with Brave Search API.
This module provides drop-in replacements for the web search functions in tools.py.
"""

import logging
from typing import List, Dict, Any, Optional
from livekit.agents import function_tool, RunContext

# Import the Brave Search web search functionality
from brave_search_tools import web_search, fallback_web_search

# Re-export the functions with the same names as the original tools.py
__all__ = ['web_search', 'fallback_web_search']

# Log that we're using the Brave Search API implementation
logging.info("Using Brave Search API implementation for web search functions")

# The functions are already decorated with @function_tool in brave_search_tools.py,
# so we can directly use them as replacements for the tools.py functions.
