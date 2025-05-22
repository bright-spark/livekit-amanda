"""
Integration module to connect Indeed functionality with Brave Search API.
This module provides drop-in replacements for existing Indeed search functions.
"""

import logging
from typing import List, Dict, Any, Optional
from livekit.agents import function_tool, RunContext

# Import the Brave Search Indeed functionality
from brave_search_indeed import indeed_job_search, search_indeed_jobs

# Re-export the functions with the same names as the original indeed.py
__all__ = ['indeed_job_search', 'search_indeed_jobs']

# Log that we're using the Brave Search API implementation
logging.info("Using Brave Search API implementation for Indeed job searches")

# Add any additional helper functions or wrappers here if needed in the future
