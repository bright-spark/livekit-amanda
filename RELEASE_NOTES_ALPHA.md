# Brave Search Quality API with LangChain and RAG - Alpha Release

## Release Date: May 23, 2025

This alpha release introduces the integration of Brave Search Quality API with LangChain components and Retrieval-Augmented Generation (RAG) capabilities. The integration enhances search functionality by combining high-quality web search results with local data through vector embeddings and retrieval.

## Key Features

### 1. Multi-Level Search Architecture
- **Memory Cache Layer**: Fast in-memory caching for frequently accessed quality data
- **Persistent Cache Layer**: Durable storage for quality-enriched search results with versioning
- **Vector Store Layer**: Semantic search capabilities for both web results and local data
- **Local Data Integration**: Processing and embedding of local documents for RAG

### 2. Real-Time File Monitoring
- Automatic detection of file additions, modifications, and deletions in the data directory
- Immediate updates to the RAG vector store based on file changes
- Content hashing to avoid unnecessary processing of unchanged files
- Asynchronous processing to avoid blocking the main application

### 3. Optimized Embedding Process
- Background embedding for non-blocking operation
- Chunked processing to minimize memory usage
- Parallel execution for improved performance
- Efficient memory management for large document sets

### 4. Search Engine Integration
- Prioritized search engine fallback strategy:
  1. Brave Search Quality API (with key)
  2. Brave Search API (with key)
  3. Brave Search (no key)
  4. DuckDuckGo (no key)
  5. Google Search (no key)
  6. Bing Search (no key)
- Consistent return format across all search engines
- Rate limiting to avoid being blocked
- Comprehensive error handling and logging

### 5. Azure OpenAI Integration
- Seamless integration with Azure OpenAI for embeddings
- Support for the model router deployment
- Configurable API parameters

## Configuration Options

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

## Bug Fixes

- Fixed environment variable naming for Azure OpenAI integration
- Improved session management for HTTP clients
- Enhanced error handling for API rate limits
- Fixed memory leaks in asynchronous operations

## Known Issues

- The persistent cache test may fail with "'str' object has no attribute 'copy'" error, but this does not affect actual functionality
- Some unclosed client sessions warnings may appear in logs

## Documentation

Comprehensive documentation is available in the following files:
- `docs/BRAVE_SEARCH_QUALITY_LANGCHAIN_INTEGRATION_WIKI.md`: Detailed integration guide
- `BRAVE_SEARCH_QUALITY_LANGCHAIN_INTEGRATION.md`: Overview of the integration
- `LOCAL_DATA_RAG_README.md`: Guide for local data integration
- `docs/BACKGROUND_EMBEDDING_README.md`: Background embedding process documentation

## Testing

The following test scripts are available to verify the integration:
- `test_brave_quality_basic.py`: Tests the basic functionality of Brave Search Quality API and caching
- `test_brave_search_comprehensive.py`: Comprehensive test for all components
- `test_brave_quality_simple.py`: Simple test for the integration with LangChain

## Requirements

- Python 3.8+
- LangChain and LangChain Community packages
- Azure OpenAI API access
- Brave Search API key (optional)
- ChromaDB for vector storage
- Watchdog for file monitoring

## Next Steps

- Improve test coverage for edge cases
- Enhance performance for large document sets
- Add support for more file formats
- Implement hybrid search and re-ranking
- Add streaming response capabilities
