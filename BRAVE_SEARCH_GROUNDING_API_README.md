# Brave Search Grounding API Documentation

This document outlines the Brave Search Grounding API implementation, which provides specialized functionality for grounding queries to Brave Search with elastic parallel processing capabilities.

## Overview

The Brave Search Grounding API extends the standard Brave Search functionality by adding:

1. **Dedicated Grounding API Key**: Uses a separate API key specifically for grounding operations
2. **Elastic Parallel Processing**: Supports multiple API keys for concurrent processing
3. **Round-Robin Key Rotation**: Distributes load across available API keys
4. **Per-Key Rate Limiting**: Each API key has its own rate limiting semaphore
5. **Fallback Mechanisms**: Gracefully falls back to web search when AI search is unavailable

## Architecture

The Grounding API system consists of several components:

- **`brave_search_grounding.py`**: Core implementation of the grounding service with elastic parallel processing
- **`brave_search_quality_api.py`**: Integration with the Quality API for grounding operations
- **`brave_search_quality_integration.py`**: Integration layer for using the Grounding API in applications
- **`brave_search_quality_cli.py`**: Command-line interface with grounding commands

## Key Features

### Dedicated API Key for Grounding

The Grounding API uses a dedicated API key (`BRAVE_GROUNDING_API_KEY`) to separate grounding operations from regular search operations. This allows for:

- Better management of API usage and rate limits
- Independent scaling of grounding capabilities
- Clearer monitoring and analytics of grounding-specific usage

### Elastic Parallel Processing

The system implements elastic parallel processing to maximize throughput while respecting Brave Search API's rate limits:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  API Key 1      │     │  API Key 2      │     │  API Key 3      │
│  (Semaphore)    │     │  (Semaphore)    │     │  (Semaphore)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Request Session │     │ Request Session │     │ Request Session │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │  Brave Search   │
                       │      API        │
                       └─────────────────┘
```

Each API key:
- Has its own semaphore limiting to 1 request at a time (per Brave Search's free tier limits)
- Maintains its own session for connection pooling
- Tracks its own rate limiting state

### Round-Robin Key Rotation

The system implements a round-robin key rotation strategy to distribute load across all available API keys:

1. Each new request gets the next available API key in the rotation
2. The rotation index is incremented after each key selection
3. When the index reaches the end of the key list, it wraps around to the beginning

This ensures even distribution of requests across all available keys.

### Multiple API Key Configuration

The system supports multiple API keys through indexed environment variables:

```
# Primary grounding API key
BRAVE_GROUNDING_API_KEY=your_primary_key

# Additional grounding API keys
BRAVE_GROUNDING_API_KEY_1=your_first_key
BRAVE_GROUNDING_API_KEY_2=your_second_key
BRAVE_GROUNDING_API_KEY_3=your_third_key
```

The system will automatically detect and use all available keys.

### Fallback Mechanisms

The Grounding API includes robust fallback mechanisms:

- Falls back to web search when AI search fails
- Adds clear notes in the results when fallbacks occur
- Preserves the original search type in the output for transparency

## Usage

### Command Line Interface

The Grounding API can be used via the command-line interface:

```bash
# Ground a query using web search
python brave_search_quality_cli.py ground web "climate change solutions" --results 3

# Ground a query using AI search (with automatic fallback to web if needed)
python brave_search_quality_cli.py ground ai "benefits of meditation"
```

### Programmatic Usage

The Grounding API can be used programmatically:

```python
from brave_search_quality_api import get_quality_api

async def ground_query_example():
    # Get the quality API
    quality_api = get_quality_api()
    
    # Ground a query using web search
    result = await quality_api.ground_query(
        context={},
        query="climate change solutions",
        search_type="web",
        num_results=3
    )
    
    print(result)
```

## Performance Considerations

- **Scaling**: The system scales linearly with the number of API keys
- **Throughput**: With N API keys, the system can handle N concurrent requests
- **Rate Limiting**: Each key is limited to 1 request per second (free tier)
- **Connection Pooling**: Each key maintains its own connection pool for efficiency

## Configuration

The Grounding API can be configured through environment variables:

```
# API Keys
BRAVE_GROUNDING_API_KEY=your_primary_key
BRAVE_GROUNDING_API_KEY_1=your_first_key
BRAVE_GROUNDING_API_KEY_2=your_second_key

# Rate Limiting
BRAVE_GROUNDING_RATE_LIMIT=1  # Requests per second per key
```

See the `.env.example` file for a complete list of configuration options.
