# Local Data Integration for RAG

## Overview

The LiveKit Amanda project now supports enhanced RAG (Retrieval Augmented Generation) capabilities by integrating local data from the `/data` directory. This feature allows the system to ground responses in both web search results and local documents.

## Features

- **Local Data Indexing**: Automatically indexes documents from the `/data` directory
- **Vector Embeddings**: Generates and caches vector embeddings for efficient semantic search
- **Hybrid Search**: Combines web search results with relevant local documents
- **Enhanced Summaries**: Generates comprehensive summaries using both web and local data sources
- **Rich File Format Support**: Processes a wide range of file formats including PDF, DOCX, and Excel

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

## Performance Optimization

The system caches vector embeddings to improve performance. The cache is stored in `/data/vector_cache.pkl`.
