# Brave Search API Integration

This document provides an overview of the Brave Search API integration implemented in this project, including the caching and rate limiting mechanisms to optimize API usage.

## Overview

The Brave Search API has been integrated to replace web scraping for the following functionalities:
- General web searches
- Locanto job searches
- Indeed job searches

## Components

### Core Components

1. **`brave_search_optimized.py`**
   - Core client with caching and rate limiting
   - Manages API keys and request handling
   - Provides a singleton instance pattern

2. **`brave_search_tools_optimized.py`**
   - Implements general web search functionality
   - Uses the optimized client for caching and rate limiting

3. **`brave_search_indeed_optimized.py`**
   - Implements Indeed job search functionality
   - Extracts job details from search results

4. **`brave_search_locanto_optimized.py`**
   - Implements Locanto search functionality
   - Maintains compatibility with existing Locanto functions

5. **`brave_search_integration.py`**
   - Master integration module
   - Provides a single import point for all optimized functions
   - Includes cache management utilities

## Caching Mechanism

The caching system works as follows:

1. **In-Memory Cache**: Stores search results based on query parameters
2. **Cache Key**: Generated from the search query and other parameters
3. **Expiration**: Cache entries expire after a configurable time period
4. **Statistics**: Cache hit/miss statistics are tracked for monitoring

Benefits:
- Reduces API calls for repeated searches
- Significantly improves response times (15,000x+ faster for cached results)
- Minimizes API usage costs

## Rate Limiting

The rate limiting system:

1. **Request Throttling**: Limits requests to comply with Brave Search API limits
2. **Queuing**: Queues requests that exceed the rate limit
3. **Backoff Strategy**: Implements exponential backoff for failed requests

Benefits:
- Prevents API quota exhaustion
- Ensures reliable operation even under high load
- Avoids API service disruptions

## Usage

### Basic Usage

```python
from brave_search_integration import web_search, indeed_job_search, search_locanto

# General web search
results = await web_search(context, "climate change solutions", num_results=5)

# Indeed job search
job_results = await indeed_job_search("software developer", "Cape Town", max_results=5)

# Locanto search
locanto_results = await search_locanto(context, category_path="personals/men-seeking-men")
```

### Cache Management

```python
from brave_search_integration import get_cache_stats, clear_cache

# Get cache statistics
stats = get_cache_stats()
print(f"Cache size: {stats['size']}, Hit rate: {stats['hit_rate']}")

# Clear the cache if needed
clear_cache()
```

## Configuration

The Brave Search API requires an API key, which should be stored in your environment variables:

```
BRAVE_API_KEY=your_api_key_here
```

You can use a `.env` file with the `python-dotenv` package to manage this environment variable.

## Testing

A test script is provided to verify the functionality and demonstrate the caching benefits:

```bash
python test_brave_optimized.py
```

The test script shows:
- Cache hit/miss statistics
- Performance improvements from caching
- Sample search results

## Performance

Based on testing, the caching mechanism provides significant performance improvements:
- Web Search: ~15,715x faster with cache hits
- Indeed Job Search: ~13,866x faster with cache hits
- Locanto Search: ~20,069x faster with cache hits

## Troubleshooting

Common issues:

1. **Missing API Key**: Ensure `BRAVE_API_KEY` is set in your environment
2. **Rate Limiting**: If you encounter rate limiting errors, the system will automatically retry with backoff
3. **Cache Issues**: Use `clear_cache()` to reset the cache if needed

## Next Steps

Potential improvements:

1. **Persistent Cache**: Implement disk-based caching for persistence between restarts
2. **Advanced Rate Limiting**: Add more sophisticated rate limiting strategies
3. **Result Processing**: Enhance result parsing for more accurate information extraction
