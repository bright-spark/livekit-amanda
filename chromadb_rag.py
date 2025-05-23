"""
ChromaDB-based RAG implementation for LiveKit Amanda.

This module provides a vector database implementation using ChromaDB,
a free and open-source vector database that's easy to use and deploy.
"""

import os
import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import asyncio

# Import ChromaDB
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Import environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
DATA_DIR = os.environ.get("ENHANCED_SEARCH_RAG_CACHE_DIR", os.path.expanduser("~/.enhanced_search_rag_cache"))
ENABLE_RAG = os.environ.get("ENHANCED_SEARCH_ENABLE_RAG", "true").lower() in ("true", "1", "yes")
RAG_MAX_ENTRIES = int(os.environ.get("ENHANCED_SEARCH_RAG_MAX_ENTRIES", "1000"))
RAG_MIN_SIMILARITY = float(os.environ.get("ENHANCED_SEARCH_RAG_MIN_SIMILARITY", "0.75"))

# Azure OpenAI configuration - using model router
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "model-router")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_VERSION")

# Create the data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# ChromaDB client and collection
_client = None
_collection = None

class AzureOpenAIEmbeddingFunction:
    """Custom embedding function for Azure OpenAI with model router."""
    
    def __init__(self, api_key, endpoint, deployment, api_version):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment = deployment  # This should be 'model-router'
        self.api_version = api_version
        
        # Import Azure OpenAI client
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
    
    def __call__(self, input):
        """Generate embeddings for input text(s) using Azure model router.
        
        Args:
            input: A string or list of strings to generate embeddings for
            
        Returns:
            A list of embeddings, one for each input string
        """
        if not input:
            return []
        
        try:
            # Convert to list if single string
            if isinstance(input, str):
                input = [input]
            
            # Truncate texts if too long
            truncated_texts = [text[:8000] for text in input]
            
            # Get embeddings using model router
            response = self.client.embeddings.create(
                input=truncated_texts,
                model=self.deployment  # Model router will select the appropriate embedding model
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return empty embeddings as fallback
            return [[0.0] * 1536] * len(input)

def _init_chroma_client():
    """Initialize the ChromaDB client and collection."""
    global _client, _collection
    
    if not ENABLE_RAG:
        return
    
    try:
        # Create ChromaDB client with persistent storage
        chroma_path = os.path.join(DATA_DIR, "chroma")
        os.makedirs(chroma_path, exist_ok=True)
        
        _client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get embedding function
        embedding_func = _get_embedding_function()
        
        # Check if collection exists
        collection_name = "search_results"
        collection_exists = False
        
        try:
            # List all collections
            collections = _client.list_collections()
            for collection in collections:
                if collection.name == collection_name:
                    collection_exists = True
                    break
            
            if collection_exists:
                # Get existing collection
                _collection = _client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_func
                )
                logger.info(f"Loaded existing ChromaDB collection with {_collection.count()} entries")
            else:
                # Create new collection
                _collection = _client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_func,
                    metadata={"description": "Search results for RAG"}
                )
                logger.info("Created new ChromaDB collection")
        except Exception as e:
            logger.error(f"Error accessing collection: {e}")
            # Try to create a new collection with a unique name as fallback
            try:
                import uuid
                unique_name = f"search_results_{uuid.uuid4().hex[:8]}"
                _collection = _client.create_collection(
                    name=unique_name,
                    embedding_function=embedding_func,
                    metadata={"description": "Search results for RAG (fallback)"}
                )
                logger.info(f"Created fallback ChromaDB collection: {unique_name}")
            except Exception as fallback_error:
                logger.error(f"Error creating fallback collection: {fallback_error}")
                _collection = None
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        _client = None
        _collection = None

def _get_embedding_function():
    """Get the embedding function for ChromaDB using Azure model router."""
    # Try to use Azure OpenAI with model router
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        try:
            return AzureOpenAIEmbeddingFunction(
                api_key=AZURE_OPENAI_API_KEY,
                endpoint=AZURE_OPENAI_ENDPOINT,
                deployment=AZURE_OPENAI_DEPLOYMENT,  # Should be 'model-router'
                api_version=AZURE_OPENAI_API_VERSION
            )
        except Exception as e:
            logger.warning(f"Error creating Azure OpenAI embedding function: {e}")
    
    # Fall back to default embedding function from ChromaDB
    # This uses the default embedding model from sentence-transformers
    try:
        # Use the built-in default embedding function which has the correct interface
        return embedding_functions.DefaultEmbeddingFunction()
    except Exception as e:
        logger.error(f"Error creating default embedding function: {e}")
        
        # As a last resort, create a simple embedding function that returns random vectors
        # This is just for testing and should not be used in production
        class FallbackEmbeddingFunction:
            def __call__(self, input):
                import random
                if isinstance(input, str):
                    input = [input]
                return [[random.uniform(-1, 1) for _ in range(768)] for _ in range(len(input))]
        
        return FallbackEmbeddingFunction()

async def add_to_rag_cache(query: str, search_results: str) -> Optional[str]:
    """Add search results to the RAG cache.
    
    Args:
        query: The search query
        search_results: The search results
        
    Returns:
        ID of the added entry, or None if not added
    """
    if not ENABLE_RAG:
        return None
    
    try:
        global _client, _collection
        # Initialize ChromaDB if not initialized
        if _client is None or _collection is None:
            _init_chroma_client()
        
        # Create entry with unique ID
        entry_id = hashlib.md5(f"{query}:{time.time()}".encode()).hexdigest()
        
        # Add to ChromaDB
        _collection.add(
            ids=[entry_id],
            documents=[search_results],
            metadatas=[{
                "query": query,
                "timestamp": str(time.time()),
                "is_valid": "true"
            }]
        )
        
        logger.info(f"Added search results for '{query}' to RAG cache with ID: {entry_id}")
        return entry_id
    except Exception as e:
        logger.error(f"Error adding to RAG cache: {e}")
        return None

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
        global _client, _collection
        # Initialize ChromaDB if not initialized
        if _client is None or _collection is None:
            _init_chroma_client()
        
        # Delete the entry
        _collection.delete(ids=[entry_id])
        
        logger.info(f"Invalidated RAG entry with ID: {entry_id}")
        return True
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
    if not ENABLE_RAG:
        return 0
    
    try:
        global _client, _collection
        # Initialize ChromaDB if not initialized
        if _client is None or _collection is None:
            _init_chroma_client()
        
        # Query ChromaDB for similar entries
        results = _collection.query(
            query_texts=[query],
            n_results=100,  # Get up to 100 results to filter
            include=["metadatas", "documents", "distances"]
        )
        
        # Filter by similarity threshold
        # ChromaDB returns distances, not similarities, so we need to convert
        # Distance in ChromaDB is 1 - cosine_similarity
        ids_to_delete = []
        for i, distance in enumerate(results.get("distances", [[]])[0]):
            similarity = 1.0 - distance
            if similarity >= similarity_threshold:
                ids_to_delete.append(results["ids"][0][i])
        
        # Delete entries
        if ids_to_delete:
            _collection.delete(ids=ids_to_delete)
            
        logger.info(f"Invalidated {len(ids_to_delete)} RAG entries matching query: '{query}'")
        return len(ids_to_delete)
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
    if not ENABLE_RAG:
        return []
    
    try:
        global _client, _collection
        # Initialize ChromaDB if not initialized
        if _client is None or _collection is None:
            _init_chroma_client()
        
        # Query ChromaDB
        results = _collection.query(
            query_texts=[query],
            n_results=max_results,
            include=["metadatas", "documents", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results.get("ids", [[]])[0])):
            entry_id = results["ids"][0][i]
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 - distance  # Convert distance to similarity
            
            # Skip entries with low similarity
            if similarity < RAG_MIN_SIMILARITY:
                continue
            
            formatted_results.append({
                "id": entry_id,
                "query": metadata.get("query", ""),
                "results": document,
                "timestamp": float(metadata.get("timestamp", 0)),
                "similarity": similarity,
                "is_valid": metadata.get("is_valid", "true") == "true"
            })
        
        logger.info(f"Retrieved {len(formatted_results)} relevant entries from RAG cache for '{query}'")
        return formatted_results
    except Exception as e:
        logger.error(f"Error retrieving from RAG cache: {e}")
        return []

# Initialize ChromaDB on module load
_init_chroma_client()

# Cleanup function to stop background services when the module is unloaded
def _cleanup_module():
    """Clean up resources when the module is unloaded."""
    global _client
    if _client is not None:
        try:
            _client = None
            logger.info("ChromaDB client closed")
        except Exception as e:
            logger.error(f"Error closing ChromaDB client: {e}")

# Register cleanup handler for graceful shutdown
import atexit
atexit.register(_cleanup_module)

# For testing
if __name__ == "__main__":
    async def test_rag():
        # Test adding to RAG cache
        entry_id = await add_to_rag_cache(
            "quantum computing advancements",
            "Recent advancements in quantum computing include improvements in qubit stability and error correction."
        )
        print(f"Added entry with ID: {entry_id}")
        
        # Test retrieving from RAG cache
        results = await retrieve_from_rag_cache("quantum computing research")
        print(f"Retrieved {len(results)} results")
        for result in results:
            print(f"ID: {result['id']}")
            print(f"Query: {result['query']}")
            print(f"Results: {result['results']}")
            print(f"Similarity: {result['similarity']}")
        
        # Test invalidating entry
        if entry_id:
            success = await invalidate_rag_entry(entry_id)
            print(f"Invalidated entry: {success}")
    
    asyncio.run(test_rag())
