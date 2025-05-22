"""
Enhanced Search Module for LiveKit Amanda.

This module integrates our no-API-key search implementations with Azure OpenAI
to provide enhanced search capabilities:

1. Intelligent query reformulation
2. Result summarization and extraction
3. Multi-source search aggregation
4. Contextual search based on conversation history

All while respecting the environment variable controls for search engines.
"""

import asyncio
import logging
import os
from pathlib import Path
import json
import pickle
import numpy as np
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple

# Import Azure OpenAI and other required libraries
import openai
import numpy as np
from dotenv import load_dotenv

# Import libraries for document processing
import PyPDF2
import docx
import pandas as pd

# Import libraries for real-time file monitoring and background processing
import threading
import queue
import concurrent.futures
import watchdog.observers
import watchdog.events
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor

# Import our no-API-key search implementations
try:
    from brave_search_nokey import web_search as brave_search
    from brave_search_nokey import get_api_config as brave_get_api_config
    HAS_BRAVE_SEARCH = True
except ImportError:
    try:
        from brave_search_api import web_search as brave_search
        from brave_search_api import get_api_config as brave_get_api_config
        HAS_BRAVE_SEARCH = True
    except ImportError:
        HAS_BRAVE_SEARCH = False

try:
    from duckduckgo_nokey import web_search as duckduckgo_search
    from duckduckgo_nokey import get_api_config as duckduckgo_get_api_config
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False

try:
    from bing_search_nokey import web_search as bing_search
    from bing_search_nokey import get_api_config as bing_get_api_config
    HAS_BING = True
except ImportError:
    HAS_BING = False

# Import fallback search system if available
try:
    from fallback_search_system import (
        brave_web_search,
        ddg_web_search,
        bing_web_search,
        google_web_search,
        web_search,
    )
    HAS_FALLBACK_SYSTEM = True
except ImportError:
    HAS_FALLBACK_SYSTEM = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if search engines are enabled via environment variables
BRAVE_SEARCH_ENABLED = os.environ.get("BRAVE_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
DUCKDUCKGO_SEARCH_ENABLED = os.environ.get("DUCKDUCKGO_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
BING_SEARCH_ENABLED = os.environ.get("BING_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")
GOOGLE_SEARCH_ENABLED = os.environ.get("GOOGLE_SEARCH_ENABLE", "true").lower() in ("true", "1", "yes")

# Enhanced search configuration
ENABLE_QUERY_REFORMULATION = os.environ.get("ENHANCED_SEARCH_QUERY_REFORMULATION", "true").lower() in ("true", "1", "yes")
ENABLE_RESULT_SUMMARIZATION = os.environ.get("ENHANCED_SEARCH_RESULT_SUMMARIZATION", "true").lower() in ("true", "1", "yes")
ENABLE_MULTI_SOURCE_SEARCH = os.environ.get("ENHANCED_SEARCH_MULTI_SOURCE", "true").lower() in ("true", "1", "yes")
MAX_SOURCES = int(os.environ.get("ENHANCED_SEARCH_MAX_SOURCES", "2"))

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_VERSION")

# Azure OpenAI Embedding configuration
AZURE_OPENAI_EMBEDDING_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")     
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_API_VERSION = os.environ.get("AZURE_OPENAI_VERSION")

# Configure Azure OpenAI client
from openai import AzureOpenAI

# Create Azure OpenAI client for chat completions
chat_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

logger.info(f"Configured Azure OpenAI chat client with model: {AZURE_OPENAI_DEPLOYMENT}, version: {AZURE_OPENAI_API_VERSION}")
logger.info(f"Configured Azure OpenAI embedding client with model: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}, version: {AZURE_OPENAI_EMBEDDING_API_VERSION}")

# Simple in-memory cache
_cache = {}
_cache_ttl = 86400  # 1 day in seconds
_cache_max_size = 100

# Persistent cache for RAG
ENABLE_RAG = os.environ.get("ENHANCED_SEARCH_ENABLE_RAG", "true").lower() in ("true", "1", "yes")
RAG_CACHE_DIR = os.path.expanduser(os.environ.get("ENHANCED_SEARCH_RAG_CACHE_DIR", "~/.enhanced_search_rag_cache"))
os.makedirs(RAG_CACHE_DIR, exist_ok=True)
RAG_VECTOR_DB_PATH = os.path.join(RAG_CACHE_DIR, "vector_db.json")
RAG_MAX_ENTRIES = int(os.environ.get("ENHANCED_SEARCH_RAG_MAX_ENTRIES", "1000"))
RAG_MIN_SIMILARITY = float(os.environ.get("ENHANCED_SEARCH_RAG_MIN_SIMILARITY", "0.75"))

# Vector database for RAG
_vector_db = []

# Data directory configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ENABLE_LOCAL_DATA = os.environ.get("ENABLE_LOCAL_DATA", "true").lower() in ("true", "1", "yes")
ENABLE_REALTIME_MONITORING = os.environ.get("ENABLE_REALTIME_MONITORING", "true").lower() in ("true", "1", "yes")
ENABLE_BACKGROUND_EMBEDDING = os.environ.get("ENABLE_BACKGROUND_EMBEDDING", "true").lower() in ("true", "1", "yes")
MAX_EMBEDDING_WORKERS = int(os.environ.get("MAX_EMBEDDING_WORKERS", "4"))
EMBEDDING_PROGRESS_INTERVAL = float(os.environ.get("EMBEDDING_PROGRESS_INTERVAL", "5.0"))  # Progress reporting interval in percentage
VECTOR_CACHE_PATH = os.path.join(DATA_DIR, "vector_cache.pkl")
EMBEDDING_STATE_PATH = os.path.join(DATA_DIR, "embedding_state.pkl")

def _get_cache_key(query: str, **kwargs) -> str:
    """Generate a cache key for the search query and parameters.
    
    Args:
        query: The search query
        kwargs: Additional search parameters
        
    Returns:
        Cache key string
    """
    # Normalize the query to improve cache hit rate
    normalized_query = " ".join(query.lower().split())
    
    # Create a key from the query and relevant parameters
    key_parts = [normalized_query]
    
    # Add other parameters that affect results
    for param, value in sorted(kwargs.items()):
        if value is not None:
            key_parts.append(f"{param}:{value}")
    
    # Join and hash to create a fixed-length key
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def _clean_cache():
    """Remove expired entries from the cache and enforce size limits."""
    global _cache
    
    # Remove expired entries
    current_time = time.time()
    expired_keys = [k for k, v in _cache.items() if current_time > v.get('expires', 0)]
    for key in expired_keys:
        del _cache[key]
    
    # Enforce size limit
    if len(_cache) > _cache_max_size:
        # Sort by access time and remove oldest entries
        sorted_keys = sorted(_cache.keys(), key=lambda k: _cache[k].get('last_access', 0))
        keys_to_remove = sorted_keys[:len(_cache) - _cache_max_size]
        for key in keys_to_remove:
            del _cache[key]

async def _get_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get a value from the cache if it exists and is not expired.
    
    Args:
        cache_key: The cache key
        
    Returns:
        The cached value or None if not found or expired
    """
    # Check memory cache
    current_time = time.time()
    if cache_key in _cache and current_time < _cache[cache_key].get('expires', 0):
        _cache[cache_key]['last_access'] = current_time
        return _cache[cache_key]['result']
    
    return None

async def _save_to_cache(cache_key: str, result: Dict[str, Any], ttl: int = _cache_ttl) -> None:
    """Save a value to the cache.
    
    Args:
        cache_key: The cache key
        result: The result to cache
        ttl: Time-to-live in seconds
    """
    current_time = time.time()
    expires = current_time + ttl
    
    # Save to memory cache
    _cache[cache_key] = {
        'result': result,
        'expires': expires,
        'last_access': current_time
    }
    
    # Clean cache periodically
    _clean_cache()

# RAG functions
def _load_vector_db():
    """Load the vector database from disk."""
    global _vector_db
    if not ENABLE_RAG:
        return
    
    try:
        if os.path.exists(RAG_VECTOR_DB_PATH):
            with open(RAG_VECTOR_DB_PATH, 'r') as f:
                _vector_db = json.load(f)
            logger.info(f"Loaded {len(_vector_db)} entries from vector database")
        else:
            _vector_db = []
    except Exception as e:
        logger.error(f"Error loading vector database: {e}")
        _vector_db = []

def _save_vector_db():
    """Save the vector database to disk."""
    global _vector_db
    if not ENABLE_RAG:
        return
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(RAG_VECTOR_DB_PATH), exist_ok=True)
        
        with open(RAG_VECTOR_DB_PATH, 'w') as f:
            json.dump(_vector_db, f)
        logger.info(f"Saved {len(_vector_db)} entries to vector database")
    except Exception as e:
        logger.error(f"Error saving vector database: {e}")

async def _get_embedding(text: str) -> List[float]:
    """Get embedding for a text using Azure OpenAI.
    
    Args:
        text: The text to embed
        
    Returns:
        Text embedding as a list of floats
    """
    try:
        # Truncate text if too long
        if len(text) > 8000:
            text = text[:8000]
        
        # First try with the dedicated embedding model
        try:
            # Create a separate OpenAI client for embeddings
            embedding_client = openai.AzureOpenAI(
                api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
                api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
                azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT
            )
            
            response = embedding_client.embeddings.create(
                input=text,
                model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            
            embedding = response.data[0].embedding
            return embedding
        except Exception as embedding_error:
            logger.warning(f"Error using embedding model: {embedding_error}. Falling back to chat model for embeddings.")
            
            # Fall back to using the chat model for embeddings via extraction
            system_message = """You are an embedding extraction assistant. Your task is to generate a list of 1536 floating point numbers that represent the semantic meaning of the input text. These numbers should be between -1 and 1 and should capture the key semantic features of the text.

Output ONLY the array of numbers in valid JSON format, nothing else. The output should be exactly 1536 numbers."""
            
            response = chat_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Generate embedding for: {text}"}
                ],
                temperature=0.0,  # Use deterministic output
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            try:
                # Try to parse the response as JSON
                import json
                content = response.choices[0].message.content.strip()
                result = json.loads(content)
                
                # Extract the embedding from the JSON response
                if isinstance(result, dict) and "embedding" in result:
                    embedding = result["embedding"]
                elif isinstance(result, dict) and "values" in result:
                    embedding = result["values"]
                elif isinstance(result, list):
                    embedding = result
                else:
                    # If we can't find a proper embedding, create a random one
                    logger.warning("Could not extract embedding from model response, using random embedding")
                    import random
                    embedding = [random.uniform(-1, 1) for _ in range(1536)]
                
                # Ensure we have exactly 1536 dimensions
                if len(embedding) < 1536:
                    # Pad with zeros if too short
                    embedding.extend([0.0] * (1536 - len(embedding)))
                elif len(embedding) > 1536:
                    # Truncate if too long
                    embedding = embedding[:1536]
                
                return embedding
            except Exception as json_error:
                logger.error(f"Error parsing embedding from chat model: {json_error}")
                # Return a random vector as fallback
                import random
                return [random.uniform(-1, 1) for _ in range(1536)]
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * 1536  # Ada embedding dimension

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

async def add_to_rag_cache(query: str, search_results: str) -> Optional[str]:
    """Add search results to the RAG cache.
    
    Args:
        query: The search query
        search_results: The search results
        
    Returns:
        ID of the added entry, or None if not added
    """
    if not ENABLE_RAG or not AZURE_OPENAI_API_KEY:
        return
    
    try:
        global _vector_db
        # Load vector DB if not loaded
        if not _vector_db:
            _load_vector_db()
        
        # Get embeddings
        query_embedding = await _get_embedding(query)
        results_embedding = await _get_embedding(search_results)
        
        # Create entry with unique ID
        entry_id = hashlib.md5(f"{query}:{time.time()}".encode()).hexdigest()
        entry = {
            "id": entry_id,
            "query": query,
            "results": search_results,
            "query_embedding": query_embedding,
            "results_embedding": results_embedding,
            "timestamp": time.time(),
            "is_valid": True
        }
        
        # Add to vector DB
        _vector_db.append(entry)
        
        # Limit size
        if len(_vector_db) > RAG_MAX_ENTRIES:
            # Sort by timestamp (oldest first) and remove
            sorted_db = sorted(_vector_db, key=lambda x: x.get("timestamp", 0))
            _vector_db = sorted_db[-RAG_MAX_ENTRIES:]
        
        # Save to disk
        _save_vector_db()
        
        logger.info(f"Added search results for '{query}' to RAG cache with ID: {entry_id}")
        return entry_id
    except Exception as e:
        logger.error(f"Error adding to RAG cache: {e}")

async def invalidate_rag_entry(entry_id: str) -> bool:
    """Invalidate a specific entry in the RAG cache by its ID.
    
    Args:
        entry_id: The ID of the entry to invalidate
        
    Returns:
        True if the entry was found and invalidated, False otherwise
    """
    if not ENABLE_RAG:
        return False
    
    try:
        global _vector_db
        # Load vector DB if not loaded
        if not _vector_db:
            _load_vector_db()
        
        # Find the entry by ID
        for i, entry in enumerate(_vector_db):
            if str(entry.get("id", "")) == entry_id:
                # Remove the entry
                _vector_db.pop(i)
                # Save the updated DB
                _save_vector_db()
                logger.info(f"Invalidated RAG entry with ID: {entry_id}")
                return True
        
        logger.warning(f"RAG entry with ID: {entry_id} not found")
        return False
    except Exception as e:
        logger.error(f"Error invalidating RAG entry: {e}")
        return False

async def invalidate_rag_entries_by_query(query: str, similarity_threshold: float = 0.9) -> int:
    """Invalidate entries in the RAG cache that match the given query with high similarity.
    
    Args:
        query: The query to match against
        similarity_threshold: Minimum similarity threshold (0-1)
        
    Returns:
        Number of entries invalidated
    """
    if not ENABLE_RAG or not AZURE_OPENAI_API_KEY:
        return 0
    
    try:
        global _vector_db
        # Load vector DB if not loaded
        if not _vector_db:
            _load_vector_db()
        
        if not _vector_db:
            return 0
        
        # Get query embedding
        query_embedding = await _get_embedding(query)
        
        # Find entries with high similarity
        entries_to_remove = []
        for i, entry in enumerate(_vector_db):
            # Calculate similarity with query embedding
            query_similarity = _cosine_similarity(query_embedding, entry.get("query_embedding", []))
            if query_similarity >= similarity_threshold:
                entries_to_remove.append(i)
        
        # Remove entries in reverse order to avoid index issues
        for i in sorted(entries_to_remove, reverse=True):
            _vector_db.pop(i)
        
        # Save the updated DB if any entries were removed
        if entries_to_remove:
            _save_vector_db()
            logger.info(f"Invalidated {len(entries_to_remove)} RAG entries matching query: '{query}'")
        
        return len(entries_to_remove)
    except Exception as e:
        logger.error(f"Error invalidating RAG entries by query: {e}")
        return 0

async def retrieve_from_rag_cache(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant entries from the RAG cache.
    
    Args:
        query: The search query
        max_results: Maximum number of results to retrieve
        
    Returns:
        List of relevant entries
    """
    if not ENABLE_RAG or not AZURE_OPENAI_API_KEY:
        return []
    
    try:
        global _vector_db
        # Load vector DB if not loaded
        if not _vector_db:
            _load_vector_db()
        
        if not _vector_db:
            return []
        
        # Get query embedding
        query_embedding = await _get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, entry in enumerate(_vector_db):
            # Calculate similarity with query embedding
            query_similarity = _cosine_similarity(query_embedding, entry.get("query_embedding", []))
            # Calculate similarity with results embedding
            results_similarity = _cosine_similarity(query_embedding, entry.get("results_embedding", []))
            # Use the higher similarity
            similarity = max(query_similarity, results_similarity)
            similarities.append((i, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum similarity
        similarities = [(i, sim) for i, sim in similarities if sim >= RAG_MIN_SIMILARITY]
        
        # Get top results
        top_results = []
        for i, similarity in similarities[:max_results]:
            entry = _vector_db[i].copy()
            entry["similarity"] = similarity
            # Remove embeddings to save space
            entry.pop("query_embedding", None)
            entry.pop("results_embedding", None)
            top_results.append(entry)
        
        logger.info(f"Retrieved {len(top_results)} relevant entries from RAG cache for '{query}'")
        return top_results
    except Exception as e:
        logger.error(f"Error retrieving from RAG cache: {e}")
        return []

async def reformulate_query(query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Use Azure OpenAI to reformulate the search query for better results.
    
    Args:
        query: The original search query
        conversation_history: Optional conversation history for context
        
    Returns:
        Reformulated query
    """
    if not ENABLE_QUERY_REFORMULATION or not AZURE_OPENAI_API_KEY:
        return query
    
    try:
        # Create a cache key
        cache_key = _get_cache_key(query, type="reformulate", history_length=len(conversation_history or []))
        
        # Check cache first
        cached_result = await _get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query reformulation: {query}")
            return cached_result.get('reformulated_query', query)
        
        # Prepare the system message
        system_message = """You are a search query reformulation assistant. Your task is to reformulate the user's search query to make it more effective for web search engines. Follow these guidelines:
1. Make the query more specific and detailed
2. Add relevant keywords that might help find better results
3. Remove unnecessary words or phrases
4. Format the query in a way that search engines understand well
5. DO NOT change the intent or meaning of the original query
6. Return ONLY the reformulated query, nothing else"""
        
        # Prepare the user message
        user_message = f"Original query: {query}"
        
        # Add conversation context if available
        if conversation_history and len(conversation_history) > 0:
            context = "\n\nConversation context:\n"
            for i, message in enumerate(conversation_history[-3:]):  # Use last 3 messages for context
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                context += f"{role}: {content}\n"
            user_message += context
        
        # Call Azure OpenAI
        response = chat_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Extract the reformulated query
        reformulated_query = response.choices[0].message.content.strip()
        
        # Cache the result
        await _save_to_cache(cache_key, {'reformulated_query': reformulated_query})
        
        logger.info(f"Reformulated query: '{query}' -> '{reformulated_query}'")
        return reformulated_query
    except Exception as e:
        logger.error(f"Error reformulating query: {e}")
        return query  # Return original query on error

async def summarize_search_results(query: str, search_results: str) -> str:
    """Use Azure OpenAI to summarize search results.
    
    Args:
        query: The search query
        search_results: The raw search results
        
    Returns:
        Summarized search results
    """
    if not ENABLE_RESULT_SUMMARIZATION or not AZURE_OPENAI_API_KEY:
        return search_results
    
    try:
        # Create a cache key
        cache_key = _get_cache_key(query, type="summarize", results_hash=hashlib.md5(search_results.encode()).hexdigest())
        
        # Check cache first
        cached_result = await _get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit for result summarization: {query}")
            return cached_result.get('summarized_results', search_results)
        
        # Prepare the system message
        system_message = """You are a search result summarization assistant. Your task is to summarize the search results in a concise, informative way. Follow these guidelines:
1. Extract the most relevant information from the search results
2. Organize the information in a clear, structured format
3. Highlight key facts, figures, and quotes
4. Include source URLs for important information
5. Maintain factual accuracy and avoid adding information not in the results
6. Preserve the grounding information at the beginning of the results"""
        
        # Prepare the user message
        user_message = f"Search query: {query}\n\nSearch results:\n{search_results}"
        
        # Call Azure OpenAI
        response = chat_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Extract the summarized results
        summarized_results = response.choices[0].message.content.strip()
        
        # Cache the result
        await _save_to_cache(cache_key, {'summarized_results': summarized_results})
        
        logger.info(f"Summarized search results for query: '{query}'")
        return summarized_results
    except Exception as e:
        logger.error(f"Error summarizing search results: {e}")
        return search_results  # Return original results on error

async def multi_source_search(query: str, num_results: int = 5) -> str:
    """Perform a search using multiple search engines and combine the results.
    
    Args:
        query: The search query
        num_results: Number of results to return from each source
        
    Returns:
        Combined search results
    """
    if not ENABLE_MULTI_SOURCE_SEARCH:
        # Fall back to the fallback search system if available
        if HAS_FALLBACK_SYSTEM:
            return await web_search(query, num_results)
        # Otherwise use the first available search engine
        if HAS_BRAVE_SEARCH and BRAVE_SEARCH_ENABLED:
            return await brave_search(query, num_results)
        if HAS_DUCKDUCKGO and DUCKDUCKGO_SEARCH_ENABLED:
            return await duckduckgo_search(query, num_results)
        if HAS_BING and BING_SEARCH_ENABLED:
            return await bing_search(query, num_results)
        return f"No search engines available for query: '{query}'"
    
    # Create a list of available search engines
    search_engines = []
    if HAS_BRAVE_SEARCH and BRAVE_SEARCH_ENABLED:
        search_engines.append(("brave", brave_search))
    if HAS_DUCKDUCKGO and DUCKDUCKGO_SEARCH_ENABLED:
        search_engines.append(("duckduckgo", duckduckgo_search))
    if HAS_BING and BING_SEARCH_ENABLED:
        search_engines.append(("bing", bing_search))
    
    # Limit to MAX_SOURCES
    search_engines = search_engines[:MAX_SOURCES]
    
    if not search_engines:
        return f"No search engines available for query: '{query}'"
    
    # Create a cache key
    cache_key = _get_cache_key(
        query, 
        type="multi_source", 
        num_results=num_results,
        engines=",".join([engine[0] for engine in search_engines])
    )
    
    # Check cache first
    cached_result = await _get_from_cache(cache_key)
    if cached_result:
        logger.info(f"Cache hit for multi-source search: {query}")
        return cached_result.get('combined_results', f"No results found for '{query}'")
    
    # Perform searches in parallel
    tasks = []
    for engine_name, search_func in search_engines:
        tasks.append(search_func(query, num_results))
    
    # Wait for all searches to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error from {search_engines[i][0]}: {result}")
        else:
            valid_results.append((search_engines[i][0], result))
    
    if not valid_results:
        return f"No results found for '{query}'"
    
    # Combine results
    if AZURE_OPENAI_API_KEY:
        # Use Azure OpenAI to combine results
        try:
            # Prepare the system message
            system_message = """You are a search result aggregation assistant. Your task is to combine search results from multiple sources into a coherent, comprehensive response. Follow these guidelines:
1. Combine information from all sources, removing duplicates
2. Organize the information by relevance and topic
3. Highlight key facts, figures, and quotes
4. Include source URLs and indicate which search engine provided each piece of information
5. Maintain factual accuracy and avoid adding information not in the results
6. Preserve the grounding information at the beginning of the results"""
            
            # Prepare the user message
            user_message = f"Search query: {query}\n\n"
            for engine_name, result in valid_results:
                user_message += f"Results from {engine_name.upper()}:\n{result}\n\n"
            
            # Call Azure OpenAI
            response = chat_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Extract the combined results
            combined_results = response.choices[0].message.content.strip()
            
            # Cache the result
            await _save_to_cache(cache_key, {'combined_results': combined_results})
            
            logger.info(f"Combined search results from multiple sources for query: '{query}'")
            return combined_results
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            # Fall back to simple concatenation
    
    # Simple concatenation if Azure OpenAI is not available or fails
    combined_results = f"[SEARCH GROUNDING INFORMATION]\n- Query: '{query}'\n- Results retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- Search API: Multiple Search Engines\n\n"
    
    for engine_name, result in valid_results:
        # Extract just the results part (skip the grounding information)
        result_parts = result.split("\n\n")
        result_content = "\n\n".join(result_parts[3:]) if len(result_parts) > 3 else result
        combined_results += f"RESULTS FROM {engine_name.upper()}:\n{result_content}\n\n"
    
    # Cache the result
    await _save_to_cache(cache_key, {'combined_results': combined_results})
    
    return combined_results

async def enhanced_search(
    query: str, 
    num_results: int = 5, 
    conversation_history: Optional[List[Dict[str, str]]] = None,
    use_rag: bool = True
) -> str:
    """Perform an enhanced search with query reformulation, multi-source search, and result summarization.
    
    Args:
        query: The search query
        num_results: Number of results to return
        conversation_history: Optional conversation history for context
        use_rag: Whether to use RAG (Retrieval-Augmented Generation)
        
    Returns:
        Enhanced search results
    """
    start_time = time.time()
    logger.info(f"Enhanced search called for query: '{query}'")
    
    # Create a cache key for the entire enhanced search
    cache_key = _get_cache_key(
        query, 
        type="enhanced_search", 
        num_results=num_results,
        history_length=len(conversation_history or []),
        use_rag=use_rag
    )
    
    # Check cache first
    cached_result = await _get_from_cache(cache_key)
    if cached_result:
        logger.info(f"Cache hit for enhanced search: {query}")
        elapsed = time.time() - start_time
        logger.info(f"Enhanced search completed in {elapsed:.4f}s (cache hit)")
        return cached_result.get('enhanced_results', f"No results found for '{query}'")
    
    try:
        # Step 1: Check RAG cache first if enabled
        rag_results = []
        if use_rag and ENABLE_RAG and AZURE_OPENAI_API_KEY:
            rag_results = await retrieve_from_rag_cache(query)
            if rag_results:
                logger.info(f"Found {len(rag_results)} relevant entries in RAG cache for '{query}'")
        
        # Step 2: Reformulate the query if enabled
        if ENABLE_QUERY_REFORMULATION and AZURE_OPENAI_API_KEY:
            reformulated_query = await reformulate_query(query, conversation_history)
        else:
            reformulated_query = query
        
        # Step 3: Perform multi-source search
        search_results = await multi_source_search(reformulated_query, num_results)
        
        # Step 4: Add search results to RAG cache
        if use_rag and ENABLE_RAG and AZURE_OPENAI_API_KEY:
            await add_to_rag_cache(query, search_results)
        
        # Step 5: Combine search results with RAG results if available
        combined_results = search_results
        if rag_results:
            # Use Azure OpenAI to combine new search results with RAG results
            if AZURE_OPENAI_API_KEY:
                try:
                    # Prepare the system message
                    system_message = """You are a search result integration assistant. Your task is to combine new search results with previously cached relevant results into a coherent, comprehensive response. Follow these guidelines:
1. Prioritize the most recent and relevant information from both sources
2. Remove duplicates and contradictions
3. Organize the information by relevance and topic
4. Highlight key facts, figures, and quotes
5. Include source URLs and indicate which search engine provided each piece of information
6. Maintain factual accuracy and avoid adding information not in the results
7. Preserve the grounding information at the beginning of the results"""
                    
                    # Prepare the user message
                    user_message = f"Search query: {query}\n\n"
                    user_message += f"NEW SEARCH RESULTS:\n{search_results}\n\n"
                    user_message += "PREVIOUSLY CACHED RELEVANT RESULTS:\n"
                    for i, result in enumerate(rag_results):
                        user_message += f"[Relevance: {result.get('similarity', 0):.2f}] {result.get('query')}:\n{result.get('results')}\n\n"
                    
                    # Call Azure OpenAI
                    response = chat_client.chat.completions.create(
                        model=AZURE_OPENAI_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=0.3,
                        max_tokens=2000,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    
                    # Extract the combined results
                    combined_results = response.choices[0].message.content.strip()
                    logger.info(f"Combined search results with RAG cache for query: '{query}'")
                except Exception as e:
                    logger.error(f"Error combining search results with RAG cache: {e}")
                    # Fall back to simple concatenation
                    combined_results = f"[SEARCH GROUNDING INFORMATION]\n- Query: '{query}'\n- Results retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- Search API: Enhanced Search with RAG\n\n"
                    combined_results += f"NEW SEARCH RESULTS:\n{search_results}\n\n"
                    combined_results += "PREVIOUSLY CACHED RELEVANT RESULTS:\n"
                    for i, result in enumerate(rag_results):
                        combined_results += f"[Relevance: {result.get('similarity', 0):.2f}] {result.get('query')}:\n{result.get('results')}\n\n"
            else:
                # Simple concatenation if Azure OpenAI is not available
                combined_results = f"[SEARCH GROUNDING INFORMATION]\n- Query: '{query}'\n- Results retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- Search API: Enhanced Search with RAG\n\n"
                combined_results += f"NEW SEARCH RESULTS:\n{search_results}\n\n"
                combined_results += "PREVIOUSLY CACHED RELEVANT RESULTS:\n"
                for i, result in enumerate(rag_results):
                    combined_results += f"[Relevance: {result.get('similarity', 0):.2f}] {result.get('query')}:\n{result.get('results')}\n\n"
        
        # Step 6: Summarize the results if enabled
        if ENABLE_RESULT_SUMMARIZATION and AZURE_OPENAI_API_KEY:
            enhanced_results = await summarize_search_results(reformulated_query, combined_results)
        else:
            enhanced_results = combined_results
        
        # Cache the result
        await _save_to_cache(cache_key, {'enhanced_results': enhanced_results})
        
        elapsed = time.time() - start_time
        logger.info(f"Enhanced search completed in {elapsed:.4f}s")
        
        return enhanced_results
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        # Fall back to the fallback search system if available
        if HAS_FALLBACK_SYSTEM:
            return await web_search(query, num_results)
        # Otherwise use the first available search engine
        if HAS_BRAVE_SEARCH and BRAVE_SEARCH_ENABLED:
            return await brave_search(query, num_results)
        if HAS_DUCKDUCKGO and DUCKDUCKGO_SEARCH_ENABLED:
            return await duckduckgo_search(query, num_results)
        if HAS_BING and BING_SEARCH_ENABLED:
            return await bing_search(query, num_results)
        return f"Error performing search for '{query}': {str(e)}"

class DataFileHandler(FileSystemEventHandler):
    """Handler for file system events in the data directory."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.processing_lock = threading.Lock()
        self.pending_files = set()
        self.processing_timer = None
    
    def on_created(self, event):
        """Handle file creation event."""
        if not event.is_directory:
            self._schedule_processing(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification event."""
        if not event.is_directory:
            self._schedule_processing(event.src_path)
    
    def on_moved(self, event):
        """Handle file move event."""
        if not event.is_directory:
            # Remove old file if it was in our data
            if event.src_path in self.data_manager.data_files:
                with self.processing_lock:
                    file_id = os.path.relpath(event.src_path, DATA_DIR)
                    if file_id in self.data_manager.data_files:
                        del self.data_manager.data_files[file_id]
                    if file_id in self.data_manager.embeddings:
                        del self.data_manager.embeddings[file_id]
            # Process the new file
            self._schedule_processing(event.dest_path)
    
    def on_deleted(self, event):
        """Handle file deletion event."""
        if not event.is_directory:
            with self.processing_lock:
                file_id = os.path.relpath(event.src_path, DATA_DIR)
                if file_id in self.data_manager.data_files:
                    del self.data_manager.data_files[file_id]
                if file_id in self.data_manager.embeddings:
                    del self.data_manager.embeddings[file_id]
                logger.info(f"Removed deleted file from index: {file_id}")
                # Save the updated cache
                self.data_manager._save_vector_cache()
    
    def _schedule_processing(self, file_path):
        """Schedule file processing with debouncing."""
        if not file_path.startswith(DATA_DIR):
            return
            
        with self.processing_lock:
            self.pending_files.add(file_path)
            
            # Cancel existing timer if any
            if self.processing_timer:
                self.processing_timer.cancel()
            
            # Schedule processing after a short delay to debounce multiple events
            self.processing_timer = threading.Timer(2.0, self._process_pending_files)
            self.processing_timer.daemon = True
            self.processing_timer.start()
    
    def _process_pending_files(self):
        """Process all pending files."""
        files_to_process = set()
        
        with self.processing_lock:
            files_to_process = self.pending_files.copy()
            self.pending_files.clear()
            self.processing_timer = None
        
        if files_to_process:
            logger.info(f"Processing {len(files_to_process)} new or modified files")
            processed_files = []
            
            for file_path in files_to_process:
                try:
                    # Process the file metadata but don't generate embeddings yet
                    result = self.data_manager.process_file(file_path, generate_embedding=False)
                    if result:
                        processed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
            
            # Save the updated cache for file metadata
            self.data_manager._save_vector_cache()
            
            # Queue files for background embedding if any were processed
            if processed_files and ENABLE_BACKGROUND_EMBEDDING:
                for file_path in processed_files:
                    try:
                        file_id = os.path.relpath(file_path, DATA_DIR)
                        if file_id in self.data_manager.data_files and file_id not in self.data_manager.embeddings:
                            # Add to embedding queue
                            self.data_manager.embedding_queue.put(file_id)
                            self.data_manager.embedding_state[file_id] = {
                                'path': file_path,
                                'timestamp': time.time()
                            }
                    except Exception as e:
                        logger.error(f"Error queueing file for embedding: {file_path}: {e}")
                
                # Update the progress tracker if it exists, or create a new one
                if self.data_manager.embedding_progress:
                    # Update total files count
                    with self.data_manager.embedding_progress.lock:
                        self.data_manager.embedding_progress.total_files += len(processed_files)
                else:
                    # Create a new progress tracker
                    self.data_manager.embedding_progress = EmbeddingProgressTracker(len(processed_files))
                
                # Save the embedding state
                self.data_manager.save_embedding_state()
                
                logger.info(f"Queued {len(processed_files)} files for background embedding")

async def invalidate_local_data(file_id: str) -> bool:
    """Invalidate a specific document in the local data store.
    
    Args:
        file_id: The ID (relative path) of the document to invalidate
        
    Returns:
        True if the document was found and invalidated, False otherwise
    """
    if not ENABLE_LOCAL_DATA:
        return False
    
    try:
        # Remove from data_manager if it exists
        if file_id in data_manager.data_files:
            del data_manager.data_files[file_id]
            if file_id in data_manager.embeddings:
                del data_manager.embeddings[file_id]
            data_manager._save_vector_cache()
            logger.info(f"Invalidated local document: {file_id}")
            return True
        else:
            logger.warning(f"Local document not found: {file_id}")
            return False
    except Exception as e:
        logger.error(f"Error invalidating local document: {e}")
        return False

class EmbeddingProgressTracker:
    """Tracks progress of embedding generation."""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.last_report_percentage = 0
        self.lock = threading.Lock()
        self.paused = False
        self.completed = False
    
    def update(self, files_processed: int = 1) -> None:
        """Update the progress tracker."""
        with self.lock:
            self.processed_files += files_processed
            current_percentage = (self.processed_files / self.total_files) * 100 if self.total_files > 0 else 100
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Report progress if the percentage has increased by at least EMBEDDING_PROGRESS_INTERVAL
            if (current_percentage - self.last_report_percentage >= EMBEDDING_PROGRESS_INTERVAL or 
                current_percentage >= 100 and not self.completed):
                
                # Calculate throughput (files per second)
                throughput = self.processed_files / elapsed_time if elapsed_time > 0 else 0
                
                # Log progress
                logger.info(f"Embedding progress: {current_percentage:.1f}% complete ({self.processed_files}/{self.total_files} files)")
                logger.info(f"Embedding throughput: {throughput:.2f} files/sec, Time elapsed: {elapsed_time:.2f} seconds")
                
                # Update last report time and percentage
                self.last_report_time = current_time
                self.last_report_percentage = current_percentage
                
                if current_percentage >= 100:
                    self.completed = True
                    logger.info("Embedding generation completed successfully")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get the current progress."""
        with self.lock:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            percentage = (self.processed_files / self.total_files) * 100 if self.total_files > 0 else 100
            throughput = self.processed_files / elapsed_time if elapsed_time > 0 else 0
            
            return {
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "percentage": percentage,
                "elapsed_time": elapsed_time,
                "throughput": throughput,
                "paused": self.paused,
                "completed": self.completed
            }

class DataManager:
    """Manages loading, processing, and retrieving data from the data directory."""
    
    def __init__(self):
        self.data_files = {}
        self.embeddings = {}
        self.embedding_client = None
        self.file_observer = None
        self.embedding_executor = None
        self.embedding_queue = queue.Queue()
        self.embedding_state = {}
        self.embedding_progress = None
        self.embedding_thread = None
        self.embedding_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Initialize components
        self.initialize_embedding_client()
        self.load_embedding_state()
        self.load_data_directory()
        
        # Start background services
        if ENABLE_LOCAL_DATA:
            if ENABLE_BACKGROUND_EMBEDDING:
                self.start_background_embedding()
            if ENABLE_REALTIME_MONITORING:
                self.start_file_monitoring()
    
    def initialize_embedding_client(self):
        """Initialize the Azure OpenAI embedding client."""
        if AZURE_OPENAI_EMBEDDING_API_KEY and AZURE_OPENAI_EMBEDDING_ENDPOINT:
            try:
                self.embedding_client = openai.AzureOpenAI(
                    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
                    api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
                    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT
                )
                logger.info("Azure OpenAI embedding client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI embedding client: {e}")
        else:
            logger.warning("Azure OpenAI embedding credentials not provided")
    
    def load_embedding_state(self):
        """Load the embedding state from disk."""
        try:
            if os.path.exists(EMBEDDING_STATE_PATH):
                with open(EMBEDDING_STATE_PATH, 'rb') as f:
                    self.embedding_state = pickle.load(f)
                logger.info(f"Loaded embedding state with {len(self.embedding_state)} pending files")
        except Exception as e:
            logger.warning(f"Failed to load embedding state: {e}")
            self.embedding_state = {}
    
    def save_embedding_state(self):
        """Save the embedding state to disk."""
        try:
            os.makedirs(os.path.dirname(EMBEDDING_STATE_PATH), exist_ok=True)
            with open(EMBEDDING_STATE_PATH, 'wb') as f:
                pickle.dump(self.embedding_state, f)
            logger.info(f"Saved embedding state with {len(self.embedding_state)} pending files")
        except Exception as e:
            logger.error(f"Failed to save embedding state: {e}")
    
    def start_background_embedding(self):
        """Start the background embedding thread."""
        if self.embedding_thread and self.embedding_thread.is_alive():
            logger.info("Background embedding thread is already running")
            return
        
        try:
            # Create a thread pool for embedding generation
            self.embedding_executor = ThreadPoolExecutor(max_workers=MAX_EMBEDDING_WORKERS)
            
            # Start the background thread
            self.embedding_thread = threading.Thread(
                target=self._background_embedding_worker,
                daemon=True
            )
            self.embedding_thread.start()
            
            logger.info(f"Started background embedding thread with {MAX_EMBEDDING_WORKERS} workers")
        except Exception as e:
            logger.error(f"Failed to start background embedding thread: {e}")
    
    def start_file_monitoring(self):
        """Start real-time monitoring of the data directory."""
        if self.file_observer:
            # Already monitoring
            return
            
        try:
            # Create data directory if it doesn't exist
            os.makedirs(DATA_DIR, exist_ok=True)
            
            # Set up the file system event handler
            event_handler = DataFileHandler(self)
            self.file_observer = Observer()
            self.file_observer.schedule(event_handler, DATA_DIR, recursive=True)
            self.file_observer.start()
            
            logger.info(f"Started real-time monitoring of data directory: {DATA_DIR}")
        except Exception as e:
            logger.error(f"Failed to start file monitoring: {e}")
    
    def stop_file_monitoring(self):
        """Stop real-time monitoring of the data directory."""
        if self.file_observer:
            try:
                self.file_observer.stop()
                self.file_observer.join()
                self.file_observer = None
                logger.info("Stopped real-time monitoring of data directory")
            except Exception as e:
                logger.error(f"Error stopping file monitoring: {e}")
    
    def stop_background_embedding(self):
        """Stop the background embedding thread."""
        if self.embedding_thread and self.embedding_thread.is_alive():
            try:
                # Signal the thread to stop
                self.shutdown_event.set()
                
                # Wait for the thread to finish (with timeout)
                self.embedding_thread.join(timeout=5.0)
                
                # Shutdown the executor
                if self.embedding_executor:
                    self.embedding_executor.shutdown(wait=False)
                
                # Save the current state
                self.save_embedding_state()
                self._save_vector_cache()
                
                logger.info("Stopped background embedding thread")
            except Exception as e:
                logger.error(f"Error stopping background embedding thread: {e}")
    
    def get_embedding_progress(self) -> Dict[str, Any]:
        """Get the current embedding progress."""
        if self.embedding_progress:
            return self.embedding_progress.get_progress()
        else:
            return {
                "total_files": 0,
                "processed_files": 0,
                "percentage": 100.0,
                "elapsed_time": 0.0,
                "throughput": 0.0,
                "paused": False,
                "completed": True
            }
    
    def process_file(self, file_path, generate_embedding=True):
        """Process a single file and update the index.
        
        Args:
            file_path: Path to the file to process
            generate_embedding: Whether to generate embedding immediately or queue for background processing
            
        Returns:
            True if the file was processed successfully, False otherwise
        """
        try:
            path_obj = Path(file_path)
            relative_path = str(path_obj.relative_to(DATA_DIR))
            
            # Skip the vector cache and embedding state files
            if (relative_path == os.path.basename(VECTOR_CACHE_PATH) or 
                relative_path == os.path.basename(EMBEDDING_STATE_PATH)):
                return False
                
            content = self._load_file_content(path_obj)
            if content:
                self.data_files[relative_path] = {
                    'content': content,
                    'path': str(file_path),
                    'type': path_obj.suffix,
                    'last_modified': os.path.getmtime(file_path)
                }
                
                if generate_embedding:
                    # Generate embedding immediately
                    if ENABLE_BACKGROUND_EMBEDDING:
                        # Queue for background processing
                        self.embedding_queue.put(relative_path)
                        self.embedding_state[relative_path] = {
                            'path': str(file_path),
                            'timestamp': time.time()
                        }
                        logger.info(f"Queued file for background embedding: {relative_path}")
                    else:
                        # Process synchronously
                        if len(content) > 8000:
                            chunks = self._chunk_text(content)
                            embeddings = []
                            for chunk in chunks:
                                embedding = self._get_embedding(chunk)
                                if embedding:
                                    embeddings.append(embedding)
                            if embeddings:
                                self.embeddings[relative_path] = np.mean(embeddings, axis=0)
                        else:
                            embedding = self._get_embedding(content)
                            if embedding is not None:
                                self.embeddings[relative_path] = embedding
                
                logger.info(f"Processed file metadata: {relative_path}")
                return True
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        return False
    
    def load_data_directory(self):
        """Load all data files from the data directory."""
        if not ENABLE_LOCAL_DATA:
            logger.info("Local data usage is disabled")
            return
            
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
            
        # Try to load cached vectors first
        if os.path.exists(VECTOR_CACHE_PATH):
            try:
                with open(VECTOR_CACHE_PATH, 'rb') as f:
                    cache = pickle.load(f)
                    self.data_files = cache.get('data_files', {})
                    self.embeddings = cache.get('embeddings', {})
                    logger.info(f"Loaded {len(self.data_files)} documents from vector cache")
                    
                    # Verify files still exist and haven't changed
                    files_to_remove = []
                    for file_id, file_data in self.data_files.items():
                        file_path = file_data.get('path')
                        if not os.path.exists(file_path):
                            files_to_remove.append(file_id)
                        elif 'last_modified' in file_data:
                            # Check if file has been modified since last cached
                            current_mtime = os.path.getmtime(file_path)
                            if current_mtime > file_data['last_modified']:
                                files_to_remove.append(file_id)
                    
                    # Remove stale entries
                    for file_id in files_to_remove:
                        del self.data_files[file_id]
                        if file_id in self.embeddings:
                            del self.embeddings[file_id]
                    
                    if files_to_remove:
                        logger.info(f"Removed {len(files_to_remove)} stale entries from cache")
                    
                    # Only scan for new files if we had a valid cache
                    self._scan_for_new_files()
                    return
            except Exception as e:
                logger.warning(f"Failed to load vector cache: {e}")
        
        # Load all files from data directory
        for file_path in Path(DATA_DIR).rglob('*'):
            if file_path.is_file():
                self.process_file(str(file_path))
        
        logger.info(f"Loaded {len(self.data_files)} documents from data directory")
        
        # Cache the vectors
        self._save_vector_cache()
    
    def _scan_for_new_files(self):
        """Scan for new files that aren't in the cache yet."""
        new_files_count = 0
        for file_path in Path(DATA_DIR).rglob('*'):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(DATA_DIR))
                if relative_path not in self.data_files:
                    if self.process_file(str(file_path)):
                        new_files_count += 1
        
        if new_files_count > 0:
            logger.info(f"Added {new_files_count} new files to the index")
            self._save_vector_cache()
    
    def _load_file_content(self, file_path: Path) -> Optional[str]:
        """Load content from a file based on its type."""
        suffix = file_path.suffix.lower()
        try:
            # Text-based files
            if suffix in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv', '.xml', '.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # JSON files
            elif suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.dumps(json.load(f), indent=2)
            
            # PDF files
            elif suffix == '.pdf':
                return self._extract_pdf_text(file_path)
            
            # Word documents
            elif suffix in ['.docx', '.doc']:
                return self._extract_docx_text(file_path)
            
            # Excel files
            elif suffix in ['.xlsx', '.xls']:
                return self._extract_excel_text(file_path)
            
            else:
                logger.warning(f"Unsupported file type: {suffix} for {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def _extract_excel_text(self, file_path: Path) -> str:
        """Extract text from an Excel file."""
        try:
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"Error extracting text from Excel {file_path}: {e}")
            return ""
    
    def _background_embedding_worker(self):
        """Background worker for processing embedding queue."""
        logger.info("Background embedding worker started")
        
        # Process any pending files from previous runs
        pending_files = list(self.embedding_state.keys())
        if pending_files:
            logger.info(f"Found {len(pending_files)} pending files from previous run")
            for file_id in pending_files:
                if file_id in self.data_files and file_id not in self.embeddings:
                    self.embedding_queue.put(file_id)
        
        # Initialize progress tracker if we have pending files
        if not self.embedding_queue.empty():
            self.embedding_progress = EmbeddingProgressTracker(self.embedding_queue.qsize())
        
        while not self.shutdown_event.is_set():
            try:
                # Get a file from the queue with a timeout
                try:
                    file_id = self.embedding_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Skip if the file is already processed or no longer exists
                if file_id in self.embeddings or file_id not in self.data_files:
                    if file_id in self.embedding_state:
                        del self.embedding_state[file_id]
                    if self.embedding_progress:
                        self.embedding_progress.update()
                    self.embedding_queue.task_done()
                    continue
                
                # Process the file
                file_data = self.data_files[file_id]
                content = file_data['content']
                
                # Add to embedding state to track progress
                self.embedding_state[file_id] = {
                    'path': file_data['path'],
                    'timestamp': time.time()
                }
                
                # Submit embedding task to the thread pool
                future = self.embedding_executor.submit(
                    self._process_file_embedding, file_id, content
                )
                
                # Add callback to handle completion
                future.add_done_callback(lambda f, fid=file_id: self._embedding_callback(f, fid))
                
            except Exception as e:
                logger.error(f"Error in background embedding worker: {e}")
                time.sleep(1.0)  # Avoid tight loop on error
        
        logger.info("Background embedding worker stopped")
    
    def _process_file_embedding(self, file_id: str, content: str) -> Optional[np.ndarray]:
        """Process a single file for embedding."""
        try:
            # For large files, chunk them
            if len(content) > 8000:
                chunks = self._chunk_text(content)
                embeddings = []
                for chunk in chunks:
                    embedding = self._get_embedding(chunk)
                    if embedding:
                        embeddings.append(embedding)
                if embeddings:
                    # Average the embeddings
                    return np.mean(embeddings, axis=0)
            else:
                embedding = self._get_embedding(content)
                if embedding is not None:
                    return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for {file_id}: {e}")
        return None
    
    def _embedding_callback(self, future, file_id: str):
        """Callback for when an embedding task is completed."""
        try:
            # Get the result (or re-raise any exception)
            embedding = future.result()
            
            # Update the embeddings dictionary
            if embedding is not None:
                with self.embedding_lock:
                    self.embeddings[file_id] = embedding
                    # Remove from embedding state since it's now complete
                    if file_id in self.embedding_state:
                        del self.embedding_state[file_id]
                
                # Save the vector cache periodically
                if len(self.embeddings) % 10 == 0:  # Save every 10 embeddings
                    self._save_vector_cache()
                    self.save_embedding_state()
            
            # Update progress tracker
            if self.embedding_progress:
                self.embedding_progress.update()
            
            # Mark task as done
            self.embedding_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in embedding callback for {file_id}: {e}")
            # Mark task as done even on error
            self.embedding_queue.task_done()
    
    def _generate_embeddings(self):
        """Queue documents for embedding generation."""
        if not self.embedding_client or not self.data_files:
            return
        
        # Count files that need embedding
        files_to_embed = [file_id for file_id in self.data_files if file_id not in self.embeddings]
        
        if not files_to_embed:
            logger.info("No files need embedding generation")
            return
        
        logger.info(f"Queueing {len(files_to_embed)} files for embedding generation")
        
        # Initialize progress tracker
        self.embedding_progress = EmbeddingProgressTracker(len(files_to_embed))
        
        # Queue files for embedding
        for file_id in files_to_embed:
            self.embedding_queue.put(file_id)
            self.embedding_state[file_id] = {
                'path': self.data_files[file_id]['path'],
                'timestamp': time.time()
            }
        
        # Save embedding state
        self.save_embedding_state()
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text using Azure OpenAI."""
        if not self.embedding_client:
            return None
            
        try:
            response = self.embedding_client.embeddings.create(
                input=text,
                model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def _chunk_text(self, text: str, chunk_size: int = 4000) -> List[str]:
        """Split text into chunks of specified size."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    def _save_vector_cache(self):
        """Save the vectors and data to cache file."""
        if not self.embeddings:
            return
            
        try:
            cache = {
                'data_files': self.data_files,
                'embeddings': self.embeddings
            }
            os.makedirs(os.path.dirname(VECTOR_CACHE_PATH), exist_ok=True)
            with open(VECTOR_CACHE_PATH, 'wb') as f:
                pickle.dump(cache, f)
            logger.info(f"Saved vector cache with {len(self.embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error saving vector cache: {e}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity."""
        if not self.embedding_client or not self.embeddings:
            return []
            
        try:
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []
                
            results = []
            for file_id, embedding in self.embeddings.items():
                similarity = self._cosine_similarity(query_embedding, embedding)
                results.append({
                    'file_id': file_id,
                    'similarity': similarity,
                    'content': self.data_files[file_id]['content'],
                    'path': self.data_files[file_id]['path']
                })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Initialize the data manager
data_manager = DataManager()

# Function to invalidate entries in the Brave Search persistent cache
async def invalidate_brave_search_cache(query: str = None, url: str = None) -> bool:
    """Invalidate entries in the Brave Search persistent cache.
    
    Args:
        query: Optional query to invalidate (all entries matching this query)
        url: Optional URL to invalidate (all entries containing this URL)
        
    Returns:
        True if any entries were invalidated, False otherwise
    """
    try:
        # Import the Brave Search cache module
        from brave_search_cache import invalidate_cache_entry, invalidate_cache_by_query, invalidate_cache_by_url
        
        if query and url:
            # Invalidate by both query and URL
            result1 = await invalidate_cache_by_query(query)
            result2 = await invalidate_cache_by_url(url)
            return result1 or result2
        elif query:
            # Invalidate by query
            return await invalidate_cache_by_query(query)
        elif url:
            # Invalidate by URL
            return await invalidate_cache_by_url(url)
        else:
            logger.warning("No query or URL provided for cache invalidation")
            return False
    except ImportError:
        logger.error("Brave Search cache module not available")
        return False
    except Exception as e:
        logger.error(f"Error invalidating Brave Search cache: {e}")
        return False

async def enhanced_search_with_local_data(query: str, conversation_history: Optional[List[Dict[str, str]]] = None, 
                                         max_results: int = 10) -> Dict[str, Any]:
    """
    Enhanced search that combines web search with local data retrieval.
    
    Args:
        query: The search query
        conversation_history: Optional conversation history for context
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing search results and metadata
    """
    # Get results from web search
    web_results = await enhanced_search(query, conversation_history, max_results)
    
    # Get results from local data
    local_results = []
    if ENABLE_LOCAL_DATA:
        local_results = data_manager.search(query, top_k=3)
    
    # Combine results
    combined_results = {
        "web_results": web_results.get("results", []),
        "local_results": local_results,
        "query": query,
        "reformulated_query": web_results.get("reformulated_query", query),
        "summary": web_results.get("summary", ""),
        "sources": web_results.get("sources", []) + ["local_data"] if local_results else web_results.get("sources", []),
        "timestamp": time.time()
    }
    
    # If we have local results, enhance the summary with them
    if local_results and ENABLE_RESULT_SUMMARIZATION:
        combined_results["summary"] = await generate_combined_summary(
            query, 
            web_results.get("results", []), 
            local_results
        )
    
    return combined_results

async def generate_combined_summary(query: str, web_results: List[Dict[str, Any]], 
                                   local_results: List[Dict[str, Any]]) -> str:
    """Generate a summary that combines web and local data results."""
    if not AZURE_OPENAI_API_KEY or not web_results and not local_results:
        return ""
        
    try:
        # Prepare context from web results
        web_context = "\n\n".join([f"Web Result {i+1}:\n{result.get('snippet', '')}" 
                                  for i, result in enumerate(web_results[:3])])
        
        # Prepare context from local results
        local_context = "\n\n".join([f"Local Document {i+1} ({result['file_id']}):\n{result['content'][:1000]}..." 
                                    for i, result in enumerate(local_results)])
        
        # Combine contexts
        combined_context = f"{web_context}\n\n{local_context}"
        
        client = openai.AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes search results."},
                {"role": "user", "content": f"Summarize the following information to answer the query: '{query}'\n\n{combined_context}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating combined summary: {e}")
        return ""

async def search_with_local_data(query: str, conversation_history: Optional[List[Dict[str, str]]] = None, 
                               max_results: int = 10) -> Dict[str, Any]:
    """
    Main search interface that combines web search with local data.
    This function should be used as the primary search interface.
    
    Args:
        query: The search query
        conversation_history: Optional conversation history for context
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing search results and metadata
    """
    return await enhanced_search_with_local_data(query, conversation_history, max_results)

# Initialize vector DB on module load
def _init_module():
    """Initialize the module by loading the vector database."""
    if ENABLE_RAG:
        _load_vector_db()

# Cleanup function to stop background services when the module is unloaded
def _cleanup_module():
    """Clean up resources when the module is unloaded."""
    if hasattr(data_manager, 'stop_file_monitoring'):
        data_manager.stop_file_monitoring()
    if hasattr(data_manager, 'stop_background_embedding'):
        data_manager.stop_background_embedding()

# Register cleanup handler for graceful shutdown
import atexit
atexit.register(_cleanup_module)

# For testing
if __name__ == "__main__":
    # Initialize the module
    _init_module()
    
    async def test_search():
        # Test the enhanced search with local data
        result = await search_with_local_data("latest developments in quantum computing", max_results=5)
        print("Search results:")
        print(f"Query: {result['query']}")
        print(f"Reformulated query: {result['reformulated_query']}")
        print(f"Sources: {result['sources']}")
        print(f"Summary: {result['summary']}")
        print(f"Web results: {len(result['web_results'])}")
        print(f"Local results: {len(result['local_results'])}")
    
    asyncio.run(test_search())
else:
    # Initialize the module when imported
    _init_module()
