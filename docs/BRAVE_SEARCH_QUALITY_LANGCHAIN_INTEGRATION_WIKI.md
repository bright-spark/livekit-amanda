# Brave Search Quality API Integration with LangChain and RAG

## Overview

This document provides comprehensive documentation on the integration of the Brave Search Quality API with LangChain components and Retrieval-Augmented Generation (RAG) in the LiveKit Amanda project. This integration enhances search capabilities by combining high-quality web search results with local data through vector embeddings and retrieval.

## Architecture

The integration follows a multi-level architecture:

1. **Memory Cache Layer** - Fast in-memory caching for frequently accessed quality data
2. **Persistent Cache Layer** - Durable storage for quality-enriched search results
3. **Vector Store Layer** - Semantic search capabilities for both web results and local data
4. **Local Data Integration** - Processing and embedding of local documents for RAG

### Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Query    │────▶│  Memory Cache   │────▶│    Response     │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │                                              ▲
         ▼                                              │
┌─────────────────┐     ┌─────────────────┐     ┌──────┴────────┐
│ Persistent Cache│────▶│  Vector Store   │────▶│ Combined Data │
└────────┬────────┘     │  (Web Results)  │     └───────────────┘
         │              └─────────────────┘             ▲
         ▼                                              │
┌─────────────────┐     ┌─────────────────┐     ┌──────┴────────┐
│ Brave Quality   │────▶│  Quality Data   │────▶│  Enrichment   │
│  Search API     │     │   Processing    │     │    Process    │
└─────────────────┘     └─────────────────┘     └───────────────┘
                                                        ▲
                                                        │
┌─────────────────┐     ┌─────────────────┐     ┌──────┴────────┐
│  Local Data     │────▶│  Vector Store   │────▶│ RAG Retrieval │
│   Directory     │     │   (Local Data)  │     │    Process    │
└─────────────────┘     └─────────────────┘     └───────────────┘
```

## Key Components

### 1. LangChain Integration

The integration leverages several LangChain components:

- **ConversationBufferMemory**: Provides in-memory caching of quality data for fast access
- **Chroma Vector Stores**: Two separate vector stores for quality-enriched data and RAG data
- **AzureOpenAIEmbeddings**: Generates vector representations using Azure OpenAI with model router
- **DirectoryLoader and TextLoader**: Process local files for RAG integration
- **RecursiveCharacterTextSplitter**: Chunks documents for optimal retrieval
- **ContextualCompressionRetriever**: Enhances retrieval with contextual compression

### 2. Data Directory Watcher

A file monitoring system that watches the `@data` directory for changes:

- Detects file additions, modifications, and deletions
- Updates the RAG vector store in real-time based on file changes
- Uses content hashing to detect actual changes in files
- Processes files asynchronously to avoid blocking the main application

### 3. Background Embedding Process

Optimized background processing for embedding operations:

- Processes embeddings in chunks to minimize memory usage
- Runs in parallel to avoid blocking other operations
- Implements efficient memory management
- Prioritizes low overhead on system resources

## Implementation Details

### Initialization

```python
def __init__(self):
    """Initialize the integration with LangChain components."""
    # Initialize Brave Search components
    self.quality_api = get_quality_api()
    self.persistent_cache = get_persistent_cache()
    
    # Initialize LangChain components
    
    # 1. Memory cache for quality data
    self.memory_cache = ConversationBufferMemory(
        memory_key="search_history", 
        return_messages=True
    )
    
    # 2. Embedding model for vector operations
    self.embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "model-router"),
        openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2025-01-01-preview"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", "")
    )
    
    # 3. Vector stores for quality enriched data and RAG data
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
        chunk_size=int(os.environ.get("EMBEDDING_CHUNK_SIZE", "1000")),
        chunk_overlap=200
    )
    
    # 5. Initialize file watcher if real-time embedding is enabled
    if os.environ.get("ENABLE_REALTIME_EMBEDDING", "true").lower() == "true":
        self.file_watcher = DataDirectoryWatcher(
            data_dir=os.environ.get("DATA_DIR", "./data"),
            callback=self._process_file_change
        )
        self.file_watcher.start()
    else:
        self.file_watcher = None
```

### Search with RAG

The search process follows this flow:

1. Check memory cache for the query
2. Check persistent cache if not in memory
3. Perform high-quality web search if not in cache
4. Retrieve relevant local data from RAG vector store
5. Combine web search results with local data
6. Store results in memory and persistent caches
7. Update vector stores with new data

```python
async def search_with_rag(self, context, query: str, num_results: int = 5) -> str:
    """Perform a search using the flow: 
    1. Quality query result / query result → Memory cache
    2. Quality result → Persistent cache
    3. Persistent cache → Vector store (quality enriched data)
    4. @data directory → Vector store (RAG data)
    """
    # Check memory cache first (fastest)
    memory_cache_key = f"memory:{query}:{num_results}"
    memory_result = self._check_memory_cache(query)
    if memory_result:
        return memory_result
    
    # Check persistent cache
    persistent_cache_key = f"quality_search:{query}:{num_results}"
    cached_results = self._check_persistent_cache(persistent_cache_key)
    if cached_results:
        # Store in memory cache for faster access next time
        self._update_memory_cache(query, cached_results)
        return cached_results
    
    # Perform high-quality web search
    web_results = await self._perform_web_search(context, query, num_results)
    
    # Get relevant local data from RAG vector store
    local_data = await self._get_relevant_local_data(query)
    
    # Combine web results with local data
    combined_results = self._combine_results(web_results, local_data)
    
    # Store in memory and persistent caches
    self._update_memory_cache(query, combined_results)
    self._update_persistent_cache(persistent_cache_key, combined_results, query)
    
    # Update vector stores with new data
    await self._sync_persistent_to_vector_store(query, combined_results)
    
    return combined_results
```

### Data Directory Processing

The system processes local data for RAG integration:

```python
async def process_data_directory(self) -> None:
    """Process all files in the data directory for RAG integration using LangChain."""
    data_dir = os.environ.get("DATA_DIR", "./data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Created data directory: {data_dir}")
    
    try:
        # Use LangChain's DirectoryLoader to load documents
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.{txt,md,json,csv,html,xml,py,js,pdf,docx,xlsx}",
            loader_cls=TextLoader,
            show_progress=True
        )
        
        # Load documents
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from data directory")
        
        if not documents:
            logger.info(f"No documents found in data directory: {data_dir}")
            return
        
        # Split documents into chunks for better retrieval
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(texts)} text chunks")
        
        # Add documents to the RAG vector store
        self.vector_store_rag.add_documents(texts)
        logger.info(f"Added {len(texts)} documents to RAG vector store")
        
        # Persist the vector store
        self.vector_store_rag.persist()
        logger.info("Persisted RAG vector store")
        
    except Exception as e:
        logger.error(f"Error processing data directory: {e}")
```

### File Watcher Implementation

The file watcher monitors the data directory for changes:

```python
class DataDirectoryWatcher:
    """Watches the data directory for changes and updates the RAG vector store."""
    
    def __init__(self, data_dir: str, callback: Callable):
        """Initialize the watcher with the data directory and callback function."""
        self.data_dir = data_dir
        self.callback = callback
        self.observer = None
        self.file_hashes = {}
        self._initialize_file_hashes()
    
    def _initialize_file_hashes(self):
        """Initialize the file hashes for all files in the data directory."""
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.file_hashes[file_path] = self._get_file_hash(file_path)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get the hash of a file to detect changes."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def start(self):
        """Start watching the data directory."""
        if self.observer is None:
            self.observer = Observer()
            self.observer.schedule(
                FileSystemEventHandler(self._on_file_change),
                self.data_dir,
                recursive=True
            )
            self.observer.start()
            logger.info(f"Started watching data directory: {self.data_dir}")
    
    def stop(self):
        """Stop watching the data directory."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info(f"Stopped watching data directory: {self.data_dir}")
    
    def _on_file_change(self, event):
        """Handle file change events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Check if the file has actually changed by comparing hashes
        if event.event_type in ["created", "modified"]:
            new_hash = self._get_file_hash(file_path)
            old_hash = self.file_hashes.get(file_path, "")
            
            if new_hash != old_hash:
                self.file_hashes[file_path] = new_hash
                asyncio.create_task(self.callback(file_path, event.event_type))
        
        # Handle deleted files
        elif event.event_type == "deleted":
            if file_path in self.file_hashes:
                del self.file_hashes[file_path]
                asyncio.create_task(self.callback(file_path, "deleted"))
```

## Configuration

The integration can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_RAG` | Enable/disable RAG integration | `true` |
| `ENABLE_REALTIME_EMBEDDING` | Enable/disable real-time file monitoring | `true` |
| `EMBEDDING_CHUNK_SIZE` | Size of chunks for document splitting | `1000` |
| `MIN_SIMILARITY_THRESHOLD` | Minimum similarity threshold for retrieval | `0.75` |
| `DATA_DIR` | Directory for local data files | `./data` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | - |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - |
| `AZURE_OPENAI_VERSION` | Azure OpenAI API version | `2025-01-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI deployment name | `model-router` |

## Usage Examples

### Basic Usage

```python
from brave_search_quality_rag_integration import get_integration, search_with_rag

async def main():
    # Initialize the integration
    integration = get_integration()
    
    # Process the data directory
    await integration.process_data_directory()
    
    # Perform a search with RAG
    context = {"session_id": "example_session"}
    results = await search_with_rag(context, "climate change solutions", 5)
    
    print(results)
```

### Advanced Usage with Custom Configuration

```python
import os
from brave_search_quality_rag_integration import BraveSearchQualityRAGIntegration

# Set custom configuration
os.environ["ENABLE_RAG"] = "true"
os.environ["ENABLE_REALTIME_EMBEDDING"] = "true"
os.environ["EMBEDDING_CHUNK_SIZE"] = "1500"
os.environ["MIN_SIMILARITY_THRESHOLD"] = "0.8"
os.environ["DATA_DIR"] = "./custom_data"

async def main():
    # Create a custom instance
    integration = BraveSearchQualityRAGIntegration()
    
    # Process the data directory
    await integration.process_data_directory()
    
    # Start the file watcher
    integration.file_watcher.start()
    
    # Perform a search with RAG
    context = {"session_id": "custom_session"}
    results = await integration.search_with_rag(context, "renewable energy", 10)
    
    print(results)
    
    # Clean up
    integration.file_watcher.stop()
```

## Performance Considerations

- **Memory Usage**: The embedding process is chunked to minimize memory usage
- **Parallel Processing**: Background embedding runs in parallel to avoid blocking
- **Caching Strategy**: Multi-level caching improves response times
- **File Monitoring**: Uses content hashing to avoid unnecessary processing
- **Asynchronous Operations**: Non-blocking operations for better performance

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Ensure all required packages are installed: `langchain`, `langchain-community`, `langchain-openai`, `chromadb`, `watchdog`

2. **Configuration Issues**:
   - Check that all required environment variables are set correctly
   - Verify Azure OpenAI credentials are valid

3. **File Processing Errors**:
   - Ensure the data directory exists and has appropriate permissions
   - Check file formats are supported by the loaders

4. **Vector Store Issues**:
   - Verify the vector store directories exist and are writable
   - Check embedding model configuration

### Logging

The integration uses Python's logging module for diagnostics:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("brave_search_quality_rag")
```

## Future Enhancements

1. **Additional File Formats**: Support for more document types
2. **Advanced Retrieval**: Implement hybrid search and re-ranking
3. **Metadata Extraction**: Better metadata handling for improved retrieval
4. **Streaming Responses**: Support for streaming search results
5. **Custom Embeddings**: Support for custom embedding models

## Conclusion

The Brave Search Quality API integration with LangChain and RAG provides a powerful search solution that combines high-quality web results with local data. The multi-level architecture ensures optimal performance, while the real-time file monitoring keeps the system up-to-date with the latest information.
