# Local Data Integration for RAG

## Overview

The LiveKit Amanda project now supports enhanced RAG (Retrieval Augmented Generation) capabilities by integrating local data from the `/data` directory. This feature allows the system to ground responses in both web search results and local documents. The system now uses LangChain components for improved performance and functionality, including real-time file monitoring to keep the vector store synchronized with the data directory.

## Features

- **Local Data Indexing**: Automatically indexes documents from the `/data` directory
- **Vector Embeddings**: Generates and caches vector embeddings for efficient semantic search
- **Hybrid Search**: Combines web search results with relevant local documents
- **Enhanced Summaries**: Generates comprehensive summaries using both web and local data sources
- **Rich File Format Support**: Processes a wide range of file formats including PDF, DOCX, and Excel
- **Real-time File Monitoring**: Watches the data directory for any changes and automatically updates the vector store
- **LangChain Integration**: Uses LangChain components for improved performance and functionality
- **Multi-level Caching**: Implements a memory cache → persistent cache → vector store flow for optimal performance
- **Chunked Processing**: Processes embeddings in chunks to minimize memory usage

## Configuration

Enable or disable local data integration using the environment variable:

```
ENABLE_LOCAL_DATA=true
```

## Usage

Use the new `search_with_local_data()` function as your primary search interface:

```python
results = await search_with_local_data(
    query="Your search query", 
    conversation_history=conversation_history,
    max_results=10
)
```

The function returns a combined result object containing:
- `web_results`: Results from web search engines
- `local_results`: Relevant documents from the data directory
- `summary`: A combined summary of both web and local results
- `sources`: List of sources used (web search engines and local data)

## Data Directory Structure

Place your documents in the `/data` directory. Supported file formats include:

### Text-based formats
- Text files (.txt)
- Markdown files (.md)
- Python files (.py)
- JavaScript files (.js)
- HTML files (.html)
- CSS files (.css)
- JSON files (.json)
- CSV files (.csv)
- XML files (.xml)
- YAML files (.yaml, .yml)

### Document formats
- PDF files (.pdf)
- Word documents (.docx, .doc)
- Excel spreadsheets (.xlsx, .xls)

## Dependencies

To support all file formats, install the following dependencies:

```bash
pip install PyPDF2 python-docx pandas openpyxl
```

## File Monitoring

The system now includes a real-time file watcher for the `/data` directory that monitors for any changes and automatically updates the vector store:

- **File Additions**: When new files are added to the data directory, they are automatically processed, embedded, and added to the vector store
- **File Updates**: When existing files are modified, the corresponding documents in the vector store are updated with the new content
- **File Deletions**: When files are removed from the data directory, the corresponding documents are removed from the vector store

This ensures that the RAG system always has the most up-to-date information from your local data files without requiring manual reprocessing.

### Implementation Details

- Uses the `watchdog` library to monitor file system events
- Implements content hashing to detect actual content changes (not just file metadata changes)
- Processes file changes incrementally to minimize resource usage
- Supports the same file formats as the data directory processor

## Performance Optimization

The system uses multiple optimization techniques:

1. **Vector Embedding Cache**: Caches vector embeddings to improve performance, stored in `/data/vector_cache.pkl`
2. **Multi-level Caching**: Implements a memory cache → persistent cache → vector store flow for optimal performance
3. **Chunked Processing**: Processes embeddings in chunks to minimize memory usage
4. **Parallel Processing**: Uses thread pools for parallel embedding generation
