#!/usr/bin/env python3
"""
Brave Search Quality API RAG Integration using LangChain.

This module implements the flow: 
1. Quality query result / query result → Memory cache (quality data)
2. Quality result → Persistent cache (quality enriched data)
3. Persistent cache → Vector store (quality enriched data)
4. @data directory → Vector store (RAG data)

It leverages LangChain tools and libraries for implementation, ensuring search results 
are enriched with relevant local data while maintaining system performance.
The implementation uses parallel processing and chunking to minimize resource usage.
"""

import os
import time
import json
import asyncio
import logging
import hashlib
from typing import List, Tuple, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent

# LangChain imports
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool, Tool
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

# Brave Search Quality API imports
from brave_search_quality_api import get_quality_api, high_quality_web_search
from brave_search_persistent_cache import get_persistent_cache
from enhanced_search import data_manager, add_to_rag_cache

# Import RAG and vector store components
from enhanced_search import (
    add_to_rag_cache,
    retrieve_from_rag_cache,
    invalidate_rag_entry,
    invalidate_rag_entries_by_query,
    _cosine_similarity,
    DataManager,
    data_manager
)

# Import sentence-transformers for local embeddings
from sentence_transformers import SentenceTransformer

# Import environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brave_search_quality_rag")

# Configuration from environment variables
ENABLE_RAG = os.environ.get("ENHANCED_SEARCH_ENABLE_RAG", "true").lower() in ("true", "1", "yes")
ENABLE_BACKGROUND_EMBEDDING = os.environ.get("ENABLE_BACKGROUND_EMBEDDING", "true").lower() in ("true", "1", "yes")
ENABLE_REALTIME_EMBEDDING = os.environ.get("ENABLE_REALTIME_MONITORING", "true").lower() in ("true", "1", "yes")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
# Maximum number of workers for embedding operations
MAX_EMBEDDING_WORKERS = int(os.environ.get("MAX_EMBEDDING_WORKERS", "4").split('#')[0].strip())
# Size of embedding chunks for processing
EMBEDDING_CHUNK_SIZE = int(os.environ.get("EMBEDDING_CHUNK_SIZE", "10").split('#')[0].strip())
# Minimum similarity threshold for RAG results
MIN_SIMILARITY_THRESHOLD = float(os.environ.get("ENHANCED_SEARCH_RAG_MIN_SIMILARITY", "0.75").split('#')[0].strip())

# Embedding model configuration
# Using a small, modern embedding model from Hugging Face
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # Small, modern, efficient model with good performance

# Model cache directory
MODEL_CACHE_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Global embedding model instance
_embedding_model = None

# Function to get or create the embedding model
def get_embedding_model():
    """Get the embedding model, loading it from cache if available."""
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
        
    try:
        logger.info(f"Loading embedding model from cache: {EMBEDDING_MODEL_NAME}")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=MODEL_CACHE_DIR)
        logger.info(f"Successfully loaded embedding model: {EMBEDDING_MODEL_NAME}")
        return _embedding_model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None

async def _get_embedding(text: str) -> List[float]:
    """Get embedding for a text using the cached local embedding model.
    
    Args:
        text: The text to embed
        
    Returns:
        Text embedding as a list of floats
    """
    try:
        # Truncate text if too long (bge-small-en-v1.5 has a context length of 512 tokens)
        # We'll use a conservative character limit
        if len(text) > 2000:
            text = text[:2000]
        
        # Get the embedding model (loads from cache if available)
        model = get_embedding_model()
        if model is None:
            # Return a random vector as fallback if model couldn't be loaded
            import random
            return [random.uniform(-1, 1) for _ in range(384)]  # bge-small-en-v1.5 has 384 dimensions
        
        # Generate embedding
        # We use asyncio.to_thread to run the CPU-intensive embedding generation in a separate thread
        # This prevents blocking the event loop
        import asyncio
        embedding = await asyncio.to_thread(model.encode, text)
        
        # Convert numpy array to list
        return embedding.tolist()
    
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * 384  # bge-small-en-v1.5 has 384 dimensions

# Queue for background embedding tasks
embedding_queue = Queue()
embedding_workers = []
embedding_stop_event = asyncio.Event()

# File change event queue for data directory monitoring
file_change_queue = Queue()
file_watcher_stop_event = asyncio.Event()


class DataDirectoryWatcher(FileSystemEventHandler):
    """
    Watches the data directory for file changes and updates the RAG vector store accordingly.
    Monitors additions, modifications, and deletions of files.
    """
    
    def __init__(self, data_dir: str, vector_store_rag, text_splitter, file_change_queue: Queue):
        """
        Initialize the data directory watcher.
        
        Args:
            data_dir: Path to the data directory
            vector_store_rag: RAG vector store to update
            text_splitter: Text splitter for processing documents
            file_change_queue: Queue for file change events
        """
        self.data_dir = data_dir
        self.vector_store_rag = vector_store_rag
        self.text_splitter = text_splitter
        self.file_change_queue = file_change_queue
        self.file_hashes = {}  # Track file hashes to detect content changes
        self.observer = None
        
        # Initialize file hashes for existing files
        self._initialize_file_hashes()
        
        logger.info(f"Initialized DataDirectoryWatcher for {data_dir}")
    
    def _initialize_file_hashes(self):
        """
        Initialize file hashes for existing files in the data directory.
        """
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if self._is_supported_file(file):
                    file_path = os.path.join(root, file)
                    try:
                        self.file_hashes[file_path] = self._get_file_hash(file_path)
                    except Exception as e:
                        logger.error(f"Error hashing file {file_path}: {e}")
    
    def _is_supported_file(self, filename: str) -> bool:
        """
        Check if the file is supported for RAG integration.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        extensions = {".txt", ".md", ".json", ".csv", ".html", ".xml", ".py", ".js"}
        return any(filename.endswith(ext) for ext in extensions)
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Get the hash of a file to detect changes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash of the file
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(65536)  # Read in 64k chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def start(self):
        """
        Start watching the data directory.
        """
        if self.observer is None:
            self.observer = Observer()
            self.observer.schedule(self, self.data_dir, recursive=True)
            self.observer.start()
            logger.info(f"Started watching data directory: {self.data_dir}")
    
    def stop(self):
        """
        Stop watching the data directory.
        """
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info(f"Stopped watching data directory: {self.data_dir}")
    
    def on_created(self, event):
        """
        Handle file creation events.
        
        Args:
            event: File created event
        """
        if not event.is_directory and self._is_supported_file(event.src_path):
            logger.info(f"File created: {event.src_path}")
            self.file_hashes[event.src_path] = self._get_file_hash(event.src_path)
            self.file_change_queue.put(("add", event.src_path))
    
    def on_modified(self, event):
        """
        Handle file modification events.
        
        Args:
            event: File modified event
        """
        if not event.is_directory and self._is_supported_file(event.src_path):
            # Check if the content has actually changed using hash
            new_hash = self._get_file_hash(event.src_path)
            old_hash = self.file_hashes.get(event.src_path)
            
            if old_hash != new_hash:
                logger.info(f"File modified: {event.src_path}")
                self.file_hashes[event.src_path] = new_hash
                self.file_change_queue.put(("update", event.src_path))
    
    def on_deleted(self, event):
        """
        Handle file deletion events.
        
        Args:
            event: File deleted event
        """
        if not event.is_directory and event.src_path in self.file_hashes:
            logger.info(f"File deleted: {event.src_path}")
            del self.file_hashes[event.src_path]
            self.file_change_queue.put(("delete", event.src_path))

class BraveSearchQualityRAGIntegration:
    """
    Implements the flow: 
    1. Quality query result / query result → Memory cache
    2. Quality result → Persistent cache
    3. Persistent cache → Vector store
    4. @data directory → Vector store
    """
    
    def __init__(self):
        """Initialize the integration with LangChain components."""
        # Initialize Brave Search components
        self.quality_api = get_quality_api()  # For quality results
        self.persistent_cache = get_persistent_cache()  # For persistent caching
        
        # Initialize LangChain components
        
        # 1. Memory cache for quality data (in-memory using LangChain's ConversationBufferMemory)
        self.memory_cache = ConversationBufferMemory(memory_key="search_history", return_messages=True)
        self.memory_cache_ttl = 3600  # 1 hour in seconds
        
        # 2. Embedding model for vector operations - using local all-MiniLM-L6-v2 model
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 3. Vector store for quality enriched data and RAG data
        # Create separate collections for quality data and RAG data
        self.vector_store_quality = Chroma(
            collection_name="quality_enriched_data",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(DATA_DIR, "vector_stores/quality")
        )
        
        self.vector_store_rag = Chroma(
            collection_name="rag_data",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(DATA_DIR, "vector_stores/rag")
        )
        
        # 4. Text splitter for processing documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # 5. Contextual compression for better retrieval
        self.retriever_quality = ContextualCompressionRetriever(
            base_compressor=EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=MIN_SIMILARITY_THRESHOLD
            ),
            base_retriever=self.vector_store_quality.as_retriever()
        )
        
        self.retriever_rag = ContextualCompressionRetriever(
            base_compressor=EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=MIN_SIMILARITY_THRESHOLD
            ),
            base_retriever=self.vector_store_rag.as_retriever()
        )
        
        # Keep reference to data manager for processing @data directory
        self.data_manager = data_manager
        
        # Track embedding process
        self.embedding_in_progress = False
        
        # Initialize file watcher for data directory
        self.file_watcher = None
        self.file_change_processor_running = False
        self.file_change_processor_thread = None
        
        # Ensure data directories exist
        os.makedirs(os.path.join(DATA_DIR, "vector_stores/quality"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "vector_stores/rag"), exist_ok=True)
        
        logger.info(f"Initialized Brave Search Quality RAG Integration with LangChain components")
        logger.info(f"Data directory: {DATA_DIR}")
        
        # Start background embedding if enabled
        if ENABLE_BACKGROUND_EMBEDDING:
            self.start_background_embedding()
            
        # Start file watcher for data directory
        self.start_file_watcher()
    
    async def search_with_rag(self, context, query: str, num_results: int = 5) -> str:
        """
        Perform a search using the flow: 
        1. Quality query result / query result → Memory cache
        2. Quality result → Persistent cache
        3. Persistent cache → Vector store
        
        Args:
            context: The run context
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Formatted search results enriched with local data
        """
        # Step 1: Check memory cache first (fastest)
        memory_cache_key = f"memory:{query}:{num_results}"
        current_time = time.time()
        
        if memory_cache_key in self.memory_cache:
            cache_entry = self.memory_cache[memory_cache_key]
            # Check if the cache entry is still valid
            if current_time - cache_entry["timestamp"] < self.memory_cache_ttl:
                logger.info(f"Using memory cache for query: {query}")
                web_results = cache_entry["results"]
                
                # Get local data to combine with memory cache results
                local_results = []
                if ENABLE_RAG:
                    local_results = await self._get_relevant_local_data(query)
                    
                # Combine and return immediately
                return await self._combine_results(query, web_results, local_results)
        
        # Step 2: Check persistent cache if not in memory cache
        persistent_cache_key = f"quality_search:{query}:{num_results}"
        cached_results = None
        try:
            cached_results = await asyncio.to_thread(self.persistent_cache.get, persistent_cache_key)
            if cached_results:
                logger.info(f"Using persistent cache for query: {query}")
                
                # Update memory cache with results from persistent cache
                self.memory_cache[memory_cache_key] = {
                    "results": cached_results,
                    "timestamp": current_time
                }
                
                # Schedule async task to update vector store from persistent cache
                if ENABLE_RAG and ENABLE_BACKGROUND_EMBEDDING:
                    asyncio.create_task(self._sync_persistent_to_vector_store(query, cached_results))
                    
                # Get local data to combine with persistent cache results
                local_results = []
                if ENABLE_RAG:
                    local_results = await self._get_relevant_local_data(query)
                    
                # Combine and return
                return await self._combine_results(query, cached_results, local_results)
        except Exception as e:
            logger.error(f"Error checking persistent cache: {e}")
        
        # Step 3: If not in any cache, perform high-quality web search
        try:
            # Perform high-quality web search
            web_results = await high_quality_web_search(context, query, num_results)
            
            if web_results:
                # Update memory cache with fresh results
                self.memory_cache[memory_cache_key] = {
                    "results": web_results,
                    "timestamp": current_time
                }
                
                # Store in persistent cache
                try:
                    await asyncio.to_thread(
                        self.persistent_cache.store,
                        persistent_cache_key,
                        web_results,
                        metadata={"query": query, "num_results": num_results}
                    )
                    logger.debug(f"Stored high-quality results in persistent cache for query: {query}")
                    
                    # Schedule async task to update vector store from persistent cache
                    if ENABLE_RAG and ENABLE_BACKGROUND_EMBEDDING:
                        asyncio.create_task(self._sync_persistent_to_vector_store(query, web_results))
                except Exception as e:
                    logger.error(f"Error storing in persistent cache: {e}")
        except Exception as e:
            logger.error(f"Error performing high-quality web search: {e}")
            web_results = f"Error performing search: {str(e)}"
        
        # Get local data to combine with web results
        local_results = []
        if ENABLE_RAG:
            local_results = await self._get_relevant_local_data(query)
        
        # Combine web results with local data
        combined_results = await self._combine_results(query, web_results, local_results)
        
        return combined_results
    
    async def _get_relevant_local_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Get relevant local data from both vector stores using LangChain components.
        This retrieves data from both quality enriched data and RAG data vector stores.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant local data items
        """
        try:
            # Use LangChain retrievers to get relevant documents
            # First, get documents from quality vector store
            quality_docs = await asyncio.to_thread(
                self.retriever_quality.get_relevant_documents,
                query
            )
            
            # Then, get documents from RAG vector store
            rag_docs = await asyncio.to_thread(
                self.retriever_rag.get_relevant_documents,
                query
            )
            
            # Combine results and convert to the expected format
            results = []
            
            # Process quality documents
            for i, doc in enumerate(quality_docs):
                results.append({
                    "source": doc.metadata.get("source", "quality_vector_store"),
                    "title": doc.metadata.get("title", f"Quality Result {i+1}"),
                    "url": doc.metadata.get("url", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Process RAG documents
            for i, doc in enumerate(rag_docs):
                results.append({
                    "source": doc.metadata.get("source", "rag_vector_store"),
                    "title": doc.metadata.get("title", f"Local Data {i+1}"),
                    "url": doc.metadata.get("url", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.debug(f"Retrieved {len(quality_docs)} quality documents and {len(rag_docs)} RAG documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error getting relevant local data: {e}")
            return []
    
    async def _sync_persistent_to_vector_store(self, query: str, search_results: str) -> None:
        """
        Sync data from persistent cache to vector store using LangChain components.
        This implements the flow: Persistent cache → Vector store (quality enriched data)
        
        Args:
            query: Search query
            search_results: Search results to add to vector store
        """
        try:
            logger.debug(f"Syncing persistent cache to vector store for query: {query}")
            
            # Create documents from search results
            # Split the search results into chunks for better retrieval
            documents = []
            try:
                # Parse search results if they're in JSON format
                if isinstance(search_results, str) and search_results.strip().startswith('{'):
                    results_data = json.loads(search_results)
                    
                    # Process each search result as a separate document
                    if 'results' in results_data and isinstance(results_data['results'], list):
                        for i, result in enumerate(results_data['results']):
                            # Extract title and content
                            title = result.get('title', '')
                            content = result.get('description', '')
                            url = result.get('url', '')
                            
                            # Create a document with metadata
                            doc = Document(
                                page_content=f"{title}\n\n{content}",
                                metadata={
                                    "source": "brave_quality_search",
                                    "query": query,
                                    "url": url,
                                    "title": title,
                                    "result_index": i
                                }
                            )
                            documents.append(doc)
                else:
                    # Handle plain text results
                    chunks = self.text_splitter.split_text(search_results)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": "brave_quality_search",
                                "query": query,
                                "chunk_index": i
                            }
                        )
                        documents.append(doc)
            except json.JSONDecodeError:
                # Handle as plain text if not valid JSON
                chunks = self.text_splitter.split_text(search_results)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": "brave_quality_search",
                            "query": query,
                            "chunk_index": i
                        }
                    )
                    documents.append(doc)
            
            # Add documents to the quality vector store
            if documents:
                await asyncio.to_thread(
                    self.vector_store_quality.add_documents,
                    documents
                )
                logger.debug(f"Added {len(documents)} documents to quality vector store for query: {query}")
            else:
                logger.warning(f"No documents created from search results for query: {query}")
                
        except Exception as e:
            logger.error(f"Error syncing persistent cache to vector store: {e}")
    
    async def _add_to_rag_cache_async(self, query: str, search_results: str) -> None:
        """
        Add search results to the RAG cache asynchronously.
        This method is kept for backward compatibility.
        
        Args:
            query: Search query
            search_results: Search results to add
        """
        try:
            # Use the new method for syncing to vector store
            await self._sync_persistent_to_vector_store(query, search_results)
        except Exception as e:
            logger.error(f"Error adding to RAG cache: {e}")
    
    async def _combine_results(self, query: str, web_results: str, local_results: List[Dict[str, Any]]) -> str:
        """
        Combine web search results with local data.
        
        Args:
            query: Search query
            web_results: Web search results
            local_results: Local data results
            
        Returns:
            Combined results as a formatted string
        """
        if not local_results:
            return web_results
        
        # Format local results
        local_results_formatted = "\n\n--- RELEVANT LOCAL DATA ---\n\n"
        for i, result in enumerate(local_results, 1):
            local_results_formatted += f"{i}. {result.get('title', 'Untitled Document')}\n"
            local_results_formatted += f"   Source: {result.get('file_id', 'Unknown')}\n"
            content = result.get('content', '')
            # Truncate content if too long
            if len(content) > 500:
                content = content[:500] + "..."
            local_results_formatted += f"   {content}\n\n"
        
        # Combine with web results
        combined_results = f"{web_results}\n\n{local_results_formatted}"
        
        return combined_results
    
    def start_background_embedding(self) -> None:
        """Start background embedding workers."""
        if self.embedding_in_progress:
            logger.warning("Background embedding already in progress")
            return
        
        logger.info(f"Starting background embedding with {MAX_EMBEDDING_WORKERS} workers")
        self.embedding_in_progress = True
        embedding_stop_event.clear()
        
        # Start worker threads
        for i in range(MAX_EMBEDDING_WORKERS):
            worker = threading.Thread(
                target=self._embedding_worker,
                name=f"embedding-worker-{i}",
                daemon=True
            )
            worker.start()
            embedding_workers.append(worker)
    
    def stop_background_embedding(self) -> None:
        """Stop background embedding workers."""
        if not self.embedding_in_progress:
            return
        
        logger.info("Stopping background embedding workers")
        embedding_stop_event.set()
        
        # Wait for workers to finish
        for worker in embedding_workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
        
        embedding_workers.clear()
        self.embedding_in_progress = False
    
    def _embedding_worker(self) -> None:
        """Worker thread for processing embedding tasks."""
        logger.info(f"Embedding worker started: {threading.current_thread().name}")
        
        # Process items in chunks to minimize memory usage
        chunk = []
        
        while not embedding_stop_event.is_set():
            try:
                # Get item from queue with timeout
                try:
                    item = embedding_queue.get(timeout=1.0)
                    chunk.append(item)
                    embedding_queue.task_done()
                except queue.Empty:
                    # Process any remaining items in chunk
                    if chunk:
                        self._process_embedding_chunk(chunk)
                        chunk = []
                    continue
                
                # Process chunk when it reaches the desired size
                if len(chunk) >= EMBEDDING_CHUNK_SIZE:
                    self._process_embedding_chunk(chunk)
                    chunk = []
                
            except Exception as e:
                logger.error(f"Error in embedding worker: {e}")
                # Clear the problematic chunk and continue
                chunk = []
        
        # Process any remaining items before exiting
        if chunk:
            self._process_embedding_chunk(chunk)
        
        logger.info(f"Embedding worker stopped: {threading.current_thread().name}")
    
    def _process_embedding_chunk(self, chunk: List[Tuple[str, str]]) -> None:
        """
        Process a chunk of embedding tasks by embedding the persistent cache.
        
        Args:
            chunk: List of (query, search_results) tuples
        """
        logger.debug(f"Processing embedding chunk of size {len(chunk)}")
        
        # Step 1: Generate embeddings for all queries in the chunk
        # This reduces the number of API calls to the embedding service
        query_embeddings = {}
        
        try:
            # Use ThreadPoolExecutor to generate embeddings in parallel
            with ThreadPoolExecutor(max_workers=min(MAX_EMBEDDING_WORKERS, len(chunk))) as executor:
                # Map queries to embedding tasks
                future_to_query = {executor.submit(_get_embedding, query): query for query, _ in chunk}
                
                # Collect results as they complete
                for future in future_to_query:
                    query = future_to_query[future]
                    try:
                        embedding = future.result()
                        query_embeddings[query] = embedding
                    except Exception as e:
                        logger.error(f"Error generating embedding for query '{query}': {e}")
        except Exception as e:
            logger.error(f"Error in parallel embedding generation: {e}")
        
        # Step 2: Add search results to RAG cache with their embeddings
        try:
            # Use ThreadPoolExecutor to process items in parallel
            with ThreadPoolExecutor(max_workers=min(MAX_EMBEDDING_WORKERS, len(chunk))) as executor:
                # Submit all tasks with pre-computed embeddings
                futures = []
                for query, search_results in chunk:
                    embedding = query_embeddings.get(query)
                    if embedding is not None:
                        futures.append(
                            executor.submit(
                                add_to_rag_cache, 
                                query, 
                                search_results, 
                                embedding=embedding
                            )
                        )
                    else:
                        logger.warning(f"No embedding available for query '{query}', skipping RAG cache update")
                
                # Wait for all tasks to complete
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing embedding task: {e}")
        except Exception as e:
            logger.error(f"Error in parallel RAG cache update: {e}")
            
        # Log memory usage after processing
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.debug(f"Memory usage after processing chunk: {memory_info.rss / 1024 / 1024:.2f} MB")
        except ImportError:
            pass  # psutil not available
    
    async def process_data_directory(self) -> None:
        """Process all files in the data directory for RAG integration using LangChain."""
        if not os.path.exists(DATA_DIR):
            logger.warning(f"Data directory does not exist: {DATA_DIR}")
            return
        
        logger.info(f"Processing data directory: {DATA_DIR}")
        
        try:
            # Use LangChain's DirectoryLoader to load documents from the data directory
            # This implements the flow: @data directory → Vector store (RAG data)
            loader = DirectoryLoader(
                DATA_DIR,
                glob="**/*.{txt,md,json,csv,html,xml,py,js}",  # Load common file types
                loader_cls=TextLoader,
                show_progress=True
            )
            
            # Load documents
            documents = await asyncio.to_thread(loader.load)
            logger.info(f"Loaded {len(documents)} documents from data directory")
            
            if not documents:
                logger.warning(f"No documents found in data directory: {DATA_DIR}")
                return
            
            # Split documents into chunks for better retrieval
            texts = await asyncio.to_thread(self.text_splitter.split_documents, documents)
            logger.info(f"Split into {len(texts)} text chunks")
            
            # Add documents to the RAG vector store
            await asyncio.to_thread(self.vector_store_rag.add_documents, texts)
            logger.info(f"Added {len(texts)} documents to RAG vector store")
            
            # Persist the vector store
            await asyncio.to_thread(self.vector_store_rag.persist)
            logger.info("Persisted RAG vector store")
            
        except Exception as e:
            logger.error(f"Error processing data directory: {e}")
            
            # Fallback to using the DataManager if available
            if hasattr(self.data_manager, 'process_directory'):
                logger.info("Falling back to DataManager for processing data directory")
                await asyncio.to_thread(self.data_manager.process_directory, DATA_DIR)
            else:
                logger.warning("DataManager does not have process_directory method")
    
    def start_file_watcher(self):
        """
        Start watching the data directory for file changes.
        """
        if self.file_watcher is None:
            try:
                # Initialize and start the file watcher
                self.file_watcher = DataDirectoryWatcher(
                    DATA_DIR,
                    self.vector_store_rag,
                    self.text_splitter,
                    file_change_queue
                )
                self.file_watcher.start()
                
                # Start the file change processor thread
                if not self.file_change_processor_running:
                    self.file_change_processor_running = True
                    self.file_change_processor_thread = Thread(
                        target=self._file_change_processor,
                        daemon=True
                    )
                    self.file_change_processor_thread.start()
                    logger.info("Started file change processor thread")
            except Exception as e:
                logger.error(f"Error starting file watcher: {e}")
    
    def stop_file_watcher(self):
        """
        Stop watching the data directory for file changes.
        """
        # Stop the file watcher
        if self.file_watcher is not None:
            try:
                self.file_watcher.stop()
                self.file_watcher = None
                logger.info("Stopped file watcher")
            except Exception as e:
                logger.error(f"Error stopping file watcher: {e}")
        
        # Stop the file change processor thread
        if self.file_change_processor_running:
            try:
                self.file_change_processor_running = False
                file_watcher_stop_event.set()
                if self.file_change_processor_thread:
                    self.file_change_processor_thread.join(timeout=5)
                logger.info("Stopped file change processor thread")
            except Exception as e:
                logger.error(f"Error stopping file change processor thread: {e}")
    
    def _file_change_processor(self):
        """
        Process file change events from the queue.
        This runs in a separate thread to handle file changes asynchronously.
        """
        logger.info("File change processor started")
        
        while self.file_change_processor_running and not file_watcher_stop_event.is_set():
            try:
                # Get a file change event from the queue with a timeout
                try:
                    change_type, file_path = file_change_queue.get(timeout=1)
                except Queue.Empty:
                    continue
                
                logger.debug(f"Processing file change: {change_type} - {file_path}")
                
                # Handle different types of file changes
                if change_type in ("add", "update"):
                    self._process_file_addition_or_update(file_path)
                elif change_type == "delete":
                    self._process_file_deletion(file_path)
                
                # Mark the task as done
                file_change_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing file change: {e}")
        
        logger.info("File change processor stopped")
    
    def _process_file_addition_or_update(self, file_path):
        """
        Process a file addition or update event.
        
        Args:
            file_path: Path to the added or updated file
        """
        try:
            # Load the file content
            loader = TextLoader(file_path)
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content loaded from file: {file_path}")
                return
            
            # Get the relative path from the data directory for metadata
            rel_path = os.path.relpath(file_path, DATA_DIR)
            
            # Add file metadata to documents
            for doc in documents:
                doc.metadata["source_file"] = rel_path
                doc.metadata["last_updated"] = time.time()
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # First, remove any existing documents from this file
            self._remove_documents_by_source(rel_path)
            
            # Then, add the new or updated documents
            self.vector_store_rag.add_documents(texts)
            logger.info(f"Added/updated {len(texts)} chunks from file: {rel_path}")
            
            # Persist the vector store
            self.vector_store_rag.persist()
            
        except Exception as e:
            logger.error(f"Error processing file addition/update for {file_path}: {e}")
    
    def _process_file_deletion(self, file_path):
        """
        Process a file deletion event.
        
        Args:
            file_path: Path to the deleted file
        """
        try:
            # Get the relative path from the data directory
            rel_path = os.path.relpath(file_path, DATA_DIR)
            
            # Remove documents from this source
            self._remove_documents_by_source(rel_path)
            logger.info(f"Removed documents from deleted file: {rel_path}")
            
            # Persist the vector store
            self.vector_store_rag.persist()
            
        except Exception as e:
            logger.error(f"Error processing file deletion for {file_path}: {e}")
    
    def _remove_documents_by_source(self, source_path):
        """
        Remove documents from the vector store by source path.
        
        Args:
            source_path: Relative path of the source file
        """
        try:
            # Get all document IDs from the vector store
            all_ids = self.vector_store_rag.get()
            
            if not all_ids:
                return
            
            # Find IDs of documents from the specified source
            ids_to_remove = []
            for i, metadata in enumerate(all_ids["metadatas"]):
                if metadata.get("source_file") == source_path:
                    ids_to_remove.append(all_ids["ids"][i])
            
            if ids_to_remove:
                # Remove the documents
                self.vector_store_rag.delete(ids_to_remove)
                logger.debug(f"Removed {len(ids_to_remove)} documents from source: {source_path}")
            
        except Exception as e:
            logger.error(f"Error removing documents by source {source_path}: {e}")
    
    async def close(self) -> None:
        """Close the integration and release resources."""
        logger.info("Closing Brave Search Quality RAG Integration with LangChain components")
        
        # Stop background embedding
        self.stop_background_embedding()
        
        # Stop file watcher
        self.stop_file_watcher()
        
        # Persist vector stores before closing
        try:
            await asyncio.to_thread(self.vector_store_quality.persist)
            logger.debug("Persisted quality vector store")
            
            await asyncio.to_thread(self.vector_store_rag.persist)
            logger.debug("Persisted RAG vector store")
        except Exception as e:
            logger.error(f"Error persisting vector stores: {e}")
        
        # Close quality API and persistent cache
        try:
            await close_quality_api()
            logger.debug("Closed quality API")
        except Exception as e:
            logger.error(f"Error closing quality API: {e}")

# Singleton instance
_integration = None

def get_integration() -> BraveSearchQualityRAGIntegration:
    """
    Get the singleton instance of the integration.
    
    Returns:
        BraveSearchQualityRAGIntegration instance
    """
    global _integration
    
    if _integration is None:
        _integration = BraveSearchQualityRAGIntegration()
    
    return _integration

async def close_integration() -> None:
    """Close the integration."""
    global _integration
    
    if _integration is not None:
        await _integration.close()
        _integration = None

async def search_with_rag(context, query: str, num_results: int = 5) -> str:
    """
    Perform a search using Brave Search Quality API and enrich with RAG.
    
    Args:
        context: The run context
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Formatted search results enriched with local data
    """
    integration = get_integration()
    return await integration.search_with_rag(context, query, num_results)

async def process_data_directory() -> None:
    """Process all files in the data directory for RAG integration."""
    integration = get_integration()
    await integration.process_data_directory()

# Register cleanup handler for graceful shutdown
import atexit

def _cleanup():
    """Clean up resources when the module is unloaded."""
    if _integration is not None:
        asyncio.run(close_integration())

atexit.register(_cleanup)

# For testing
if __name__ == "__main__":
    async def test_search():
        # Process data directory
        await process_data_directory()
        
        # Test search with RAG
        context = {"session_id": "test_session"}
        results = await search_with_rag(context, "climate change solutions", 5)
        print(results)
        
        # Close integration
        await close_integration()
    
    asyncio.run(test_search())
