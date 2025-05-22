# LiveKit Amanda Caching and Search API Wiki

This wiki provides detailed documentation for the caching system and Brave Search API integration in the LiveKit Amanda project.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Base Cache Manager](#base-cache-manager)
3. [Specialized Cache Implementations](#specialized-cache-implementations)
   - [Locanto Cache](#locanto-cache)
   - [Indeed Cache](#indeed-cache)
   - [Brave Search Cache](#brave-search-cache)
4. [Cache Configuration and Tuning](#cache-configuration-and-tuning)
5. [Brave Search API Integration](#brave-search-api-integration)
   - [Web Search](#web-search)
   - [AI Search](#ai-search)
6. [Statistics Tracking](#statistics-tracking)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)

## Architecture Overview

The caching system is designed with a layered architecture:

1. **Base Layer**: `cache_manager.py` provides the foundational caching functionality
2. **Specialized Layer**: Service-specific cache implementations extend the base layer
3. **Configuration Layer**: `cache_config.py` provides a unified interface for configuration and tuning
4. **Application Layer**: Brave Search API modules use the caching system

This design allows for:
- Code reuse through inheritance
- Service-specific optimizations
- Centralized configuration
- Independent scaling of different cache types

### Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ brave_search.py│  │brave_web_search│  │brave_ai_search│   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Layer                      │
│                     cache_config.py                         │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Specialized Layer                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │locanto_cache.py│  │indeed_cache.py│  │brave_search_cache│ │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────────┐
│                       Base Layer                            │
│                     cache_manager.py                        │
└─────────────────────────────────────────────────────────────┘
```

## Base Cache Manager

The `CacheManager` class in `cache_manager.py` provides the core caching functionality:

### Features

- **Dual-Layer Caching**: In-memory LRU cache for speed and disk cache for persistence
- **Configurable TTL**: Time-to-live settings for cache entries
- **Cache Key Generation**: Consistent hashing for cache keys
- **Statistics Tracking**: Hit/miss rates, cache size, and other metrics
- **Invalidation Strategies**: Methods to invalidate entries by key, pattern, or expiration

### Key Methods

- `get_cache_key(query, **kwargs)`: Generate a cache key from a query and parameters
- `get(key)`: Retrieve a value from the cache
- `set(key, value, ttl)`: Store a value in the cache with a specified TTL
- `invalidate(key)`: Remove a specific entry from the cache
- `invalidate_pattern(pattern)`: Remove entries matching a pattern
- `invalidate_expired()`: Remove all expired entries
- `get_stats()`: Get cache statistics

### Configuration Options

- `name`: Name of the cache (used for directory naming)
- `cache_dir`: Directory to store cache files
- `memory_cache_size`: Size of the in-memory LRU cache
- `disk_cache_size_limit`: Size limit for disk cache in bytes
- `default_ttl`: Default time-to-live for cache entries
- `enable_memory_cache`: Whether to enable in-memory caching
- `enable_disk_cache`: Whether to enable disk caching

## Specialized Cache Implementations

### Locanto Cache

The `LocantoCache` class in `locanto_cache.py` is optimized for Locanto job searches:

#### Features

- **Job-Specific TTLs**: Longer TTL for job listings (24 hours)
- **Search-Specific TTLs**: Shorter TTL for search results (1 hour)
- **Popular Search Handling**: Extended TTL for popular searches (6 hours)
- **Automatic Invalidation**: Scheduled removal of stale job listings
- **Location-Based Invalidation**: Invalidate entries by location

#### Default Configuration

```python
DEFAULT_CONFIG = {
    "enable_cache": True,
    "enable_memory_cache": True,
    "enable_disk_cache": True,
    "memory_cache_size": 500,  # Number of entries
    "disk_cache_size_limit": 100 * 1024 * 1024,  # 100 MB
    "default_ttl": 3600,  # 1 hour in seconds
    "job_listing_ttl": 86400,  # 24 hours for job listings
    "search_results_ttl": 3600,  # 1 hour for search results
    "popular_search_ttl": 21600,  # 6 hours for popular searches
    "auto_invalidation_interval": 86400,  # 24 hours
    "max_age_for_jobs": 30,  # Maximum age in days for job listings
}
```

### Indeed Cache

The `IndeedCache` class in `indeed_cache.py` is optimized for Indeed job searches:

#### Features

- **Shorter TTLs**: Faster refresh rate for Indeed job listings (12 hours)
- **Search-Specific TTLs**: Very short TTL for search results (30 minutes)
- **Popular Search Handling**: Extended TTL for popular searches (3 hours)
- **Automatic Invalidation**: More frequent scheduled removal of stale job listings
- **Salary-Based Invalidation**: Invalidate entries by salary range

#### Default Configuration

```python
DEFAULT_CONFIG = {
    "enable_cache": True,
    "enable_memory_cache": True,
    "enable_disk_cache": True,
    "memory_cache_size": 1000,  # Number of entries
    "disk_cache_size_limit": 200 * 1024 * 1024,  # 200 MB
    "default_ttl": 3600,  # 1 hour in seconds
    "job_listing_ttl": 43200,  # 12 hours for job listings
    "search_results_ttl": 1800,  # 30 minutes for search results
    "popular_search_ttl": 10800,  # 3 hours for popular searches
    "auto_invalidation_interval": 43200,  # 12 hours
    "max_age_for_jobs": 14,  # Maximum age in days for job listings
}
```

### Brave Search Cache

The `BraveSearchCache` class in `brave_search_cache.py` handles both AI and Web searches:

#### Features

- **Dual Cache Managers**: Separate cache managers for AI and Web searches
- **Content-Aware TTLs**: Different TTLs based on query type and content
- **News Detection**: Shorter TTL for news-related queries (30 minutes)
- **Factual Query Detection**: Longer TTL for factual queries (1 week)
- **Result-Based TTL Adjustment**: TTL varies based on number of results

#### Default Configuration

```python
DEFAULT_CONFIG = {
    "enable_cache": True,
    "enable_memory_cache": True,
    "enable_disk_cache": True,
    
    # Web search cache configuration
    "web_memory_cache_size": 2000,  # Number of entries
    "web_disk_cache_size_limit": 300 * 1024 * 1024,  # 300 MB
    "web_default_ttl": 3600,  # 1 hour in seconds
    "web_news_ttl": 1800,  # 30 minutes for news searches
    "web_popular_search_ttl": 21600,  # 6 hours for popular searches
    "web_rare_search_ttl": 604800,  # 1 week for rare searches
    
    # AI search cache configuration
    "ai_memory_cache_size": 1000,  # Number of entries
    "ai_disk_cache_size_limit": 200 * 1024 * 1024,  # 200 MB
    "ai_default_ttl": 86400,  # 24 hours in seconds
    "ai_popular_search_ttl": 43200,  # 12 hours for popular searches
    "ai_factual_search_ttl": 604800,  # 1 week for factual searches
    
    # Common configuration
    "auto_invalidation_interval": 86400,  # 24 hours
    "max_age_for_searches": 7,  # Maximum age in days for search results
}
```

## Cache Configuration and Tuning

The `CacheConfig` class in `cache_config.py` provides a unified interface for configuring and tuning all caches:

### Features

- **Global Configuration**: Settings that apply to all caches
- **Service-Specific Configuration**: Override settings for individual services
- **Auto-Tuning**: Automatic adjustment of cache parameters based on usage patterns
- **Statistics Collection**: Periodic collection and logging of cache statistics
- **Cache Cleanup**: Scheduled cleanup of all caches

### Global Configuration Options

```python
DEFAULT_GLOBAL_CONFIG = {
    "enable_all_caches": True,
    "enable_memory_caches": True,
    "enable_disk_caches": True,
    "global_disk_cache_dir": None,  # If None, uses default location
    "global_disk_cache_size_limit": 1024 * 1024 * 1024,  # 1 GB total limit
    "auto_tuning_enabled": True,
    "auto_tuning_interval": 86400,  # 24 hours
    "stats_collection_enabled": True,
    "stats_collection_interval": 3600,  # 1 hour
    "cache_cleanup_interval": 86400,  # 24 hours
    "memory_cache_allocation": {
        "locanto": 0.2,  # 20% of memory cache
        "indeed": 0.3,   # 30% of memory cache
        "brave_web": 0.3,  # 30% of memory cache
        "brave_ai": 0.2   # 20% of memory cache
    },
    "disk_cache_allocation": {
        "locanto": 0.2,  # 20% of disk cache
        "indeed": 0.3,   # 30% of disk cache
        "brave_web": 0.3,  # 30% of disk cache
        "brave_ai": 0.2   # 20% of disk cache
    }
}
```

### Auto-Tuning

The auto-tuning system adjusts cache parameters based on usage patterns:

1. **Memory Cache Size**: Increased if hit rate is low
2. **TTL Values**: Adjusted based on hit rates and query patterns
3. **Disk Cache Allocation**: Rebalanced based on usage distribution

## Brave Search API Integration

The Brave Search API integration consists of two main components:

1. **Web Search**: Implemented in `brave_web_search.py`
2. **AI Search**: Implemented in `brave_ai_search.py`

A unified interface is provided in `brave_search.py` that routes queries to the appropriate search type.

### Web Search

The Web Search implementation provides:

- **Rate Limiting**: Configurable rate limiting (1 request per second for free tier)
- **Caching**: Integration with the caching system
- **Result Formatting**: Structured formatting of search results
- **Statistics Tracking**: Integration with the statistics tracking system

#### Key Methods

- `web_search(context, query, num_results)`: Perform a web search
- `format_search_results(results, num_results)`: Format search results into a readable string
- `get_brave_web_search_client()`: Get or create a singleton instance of the search client

### AI Search

The AI Search implementation provides:

- **Rate Limiting**: Configurable rate limiting (1 request per second for free tier)
- **Caching**: Integration with the caching system
- **Result Formatting**: AI-specific formatting of search results
- **Statistics Tracking**: Integration with the statistics tracking system

#### Key Methods

- `ai_search(context, query)`: Perform an AI search
- `format_ai_search_results(results)`: Format AI search results into a readable string
- `get_brave_ai_search_client()`: Get or create a singleton instance of the search client

### Unified Interface

The unified interface in `brave_search.py` provides:

- **Search Routing**: Route queries to the appropriate search type
- **Fallback Handling**: Handle cases where a search type is unavailable
- **Statistics Access**: Access to statistics for both search types

#### Key Methods

- `search(context, query, search_type, num_results)`: Unified search function
- `web_search(context, query, num_results)`: Convenience method for web search
- `ai_search(context, query)`: Convenience method for AI search
- `get_search_stats()`: Get statistics about API usage

## Statistics Tracking

The statistics tracking system is implemented in `brave_search_stats.py`:

### Features

- **Request Counting**: Track the number of requests by search type
- **Cache Hit/Miss Tracking**: Monitor cache performance
- **Response Time Tracking**: Measure API response times
- **Error Rate Tracking**: Monitor API errors
- **Rate Limiting Tracking**: Track rate limiting events

### Key Methods

- `record_request(query, response_time, search_type, ...)`: Record statistics for a request
- `get_stats()`: Get the statistics object
- `get_stats_report()`: Get a formatted report of statistics

### CLI Interface

A command-line interface is provided in `brave_stats_cli.py` for viewing statistics:

```bash
python brave_stats_cli.py --summary  # Show summary statistics
python brave_stats_cli.py --detailed  # Show detailed statistics
python brave_stats_cli.py --reset  # Reset statistics
```

## Usage Examples

### Basic Usage

```python
# Import the unified interface
from brave_search import search, web_search, ai_search

# Perform a web search
web_results = await web_search(context, "climate change solutions", num_results=5)

# Perform an AI search
ai_results = await ai_search(context, "explain quantum computing")

# Use the unified interface with explicit search type
results = await search(context, "renewable energy", search_type="web", num_results=5)
```

### Cache Usage

```python
# Import cache implementations
from locanto_cache import get_locanto_cache
from indeed_cache import get_indeed_cache
from brave_search_cache import get_brave_search_cache

# Get cache instances
locanto_cache = get_locanto_cache()
indeed_cache = get_indeed_cache()
brave_cache = get_brave_search_cache()

# Use the caches
job_key = locanto_cache.get_job_cache_key("job123")
cached_job = await locanto_cache.get(job_key)

# Get cache statistics
locanto_stats = locanto_cache.get_stats()
indeed_stats = indeed_cache.get_stats()
brave_stats = brave_cache.get_stats()
```

### Unified Cache Configuration

```python
# Import the unified configuration
from cache_config import get_cache_config

# Get the cache configuration
cache_config = get_cache_config()

# Get statistics from all caches
all_stats = cache_config.get_all_stats()

# Clean up all caches
await cache_config.cleanup_all_caches()

# Invalidate entries by keyword
await cache_config.invalidate_by_keyword("python")
```

## Troubleshooting

### Common Issues

#### Cache Not Working

1. **Check Configuration**: Ensure caching is enabled in the configuration
2. **Check Disk Space**: Ensure sufficient disk space for the cache
3. **Check Permissions**: Ensure the application has write permissions to the cache directory

#### API Rate Limiting

1. **Check Rate Limits**: Verify the rate limits in the configuration
2. **Check API Keys**: Ensure API keys are valid and have sufficient quota
3. **Monitor Statistics**: Check the statistics for rate limiting events

#### Memory Usage

1. **Adjust Cache Sizes**: Reduce memory cache sizes if memory usage is too high
2. **Enable Disk Cache**: Use disk cache instead of memory cache for large datasets
3. **Adjust TTLs**: Shorter TTLs will reduce cache size over time

### Debugging

To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Memory Optimization

- **Adjust Cache Sizes**: Set appropriate memory cache sizes based on available RAM
- **Use Disk Cache**: Enable disk cache for persistence with lower memory usage
- **Selective Caching**: Cache only frequently accessed data

### Speed Optimization

- **Increase Memory Cache**: Larger memory cache improves hit rates
- **Adjust TTLs**: Longer TTLs reduce API calls but may return stale data
- **Preload Cache**: Preload cache with common queries during startup

### API Usage Optimization

- **Adjust Rate Limits**: Set appropriate rate limits based on API tier
- **Balance Search Types**: Allocate quota between Web and AI searches
- **Monitor Statistics**: Regularly review statistics to identify optimization opportunities

---

This wiki is a living document and will be updated as the caching system and Brave Search API integration evolve.

Last updated: May 22, 2025
