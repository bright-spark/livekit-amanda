# Brave Search Quality API Documentation

This document outlines the Brave Search Quality API implementation, which provides enhanced search capabilities with high-quality data processing and persistent caching.

## Overview

The Brave Search Quality API extends the standard Brave Search functionality by adding:

1. **Enhanced Data Quality Processing**: Automatically assesses and improves the quality of search results before storage
2. **Persistent Caching**: Stores high-quality results in a sophisticated persistence layer with versioning
3. **Quality Improvement Strategies**: Tries different query variations to find the best possible results
4. **Adaptive Search**: Intelligently selects the best search strategy based on the query type
5. **Combined Search**: Merges web and AI search results for comprehensive answers

## Architecture

The Quality API system consists of several components:

- **`brave_search_quality_api.py`**: Core API for retrieving and storing high-quality search results
- **`brave_search_persistent_cache.py`**: Specialized persistent cache with data quality processing
- **`brave_search_quality_integration.py`**: Integration layer for using the Quality API in applications
- **`brave_search_quality_cli.py`**: Command-line interface for testing and demonstrating the API
- **`test_brave_search_quality_api.py`**: Comprehensive test suite for the Quality API

## Key Features

### High-Quality Web Search

The Quality API enhances web search results by:

- Assessing result quality based on multiple factors (completeness, diversity, etc.)
- Enriching results with additional metadata and context
- Formatting results with explicit grounding instructions for LLMs
- Automatically refreshing stale results while preserving high-quality cached data

```python
from brave_search_quality_integration import quality_web_search

# Perform a high-quality web search
results = await quality_web_search(context, "climate change research", num_results=5)
```

### High-Quality AI Search

For AI-generated answers, the Quality API:

- Evaluates answer quality based on completeness, confidence, and supporting evidence
- Enriches answers with additional context and sources
- Formats results with clear attribution and confidence indicators
- Preserves high-quality answers while refreshing outdated information

```python
from brave_search_quality_integration import quality_ai_search

# Perform a high-quality AI search
results = await quality_ai_search(context, "explain quantum computing")
```

### Combined and Adaptive Search

The integration layer provides advanced search strategies:

- **Combined Search**: Merges web and AI search results for comprehensive answers
- **Adaptive Search**: Automatically selects the best search strategy based on query analysis

```python
from brave_search_quality_integration import combined_quality_search, adaptive_quality_search

# Perform a combined search (both web and AI)
combined_results = await combined_quality_search(context, "benefits of meditation")

# Perform an adaptive search (automatically chooses the best strategy)
adaptive_results = await adaptive_quality_search(context, "how to bake sourdough bread")
```

### Quality Improvement

The API can attempt to improve search quality by trying different query variations:

```python
from brave_search_quality_api import improve_search_quality

# Try to improve search quality for a query
improved_results = await improve_search_quality(context, "machine learning", search_type="web")
```

## Configuration

The Quality API is highly configurable through a configuration dictionary:

```python
config = {
    "enable_quality_api": True,
    "quality_threshold": 0.8,  # Higher threshold for quality API
    "cache_ttl": 1209600,  # 2 weeks for high-quality results
    "refresh_threshold": 604800,  # 1 week before refreshing
    "max_retries": 3,  # Maximum retries for quality improvement
    "enrichment_enabled": True,
    "fallback_to_regular": True,  # Fallback to regular search if no high-quality result
    "parallel_processing": True,  # Process in parallel for faster results
}

from brave_search_quality_integration import get_quality_integration

# Initialize the integration with custom configuration
integration = get_quality_integration(config)
```

## Persistent Cache

The Quality API uses a sophisticated persistent cache with:

- SQLite storage backend for reliable persistence
- Data quality assessment and enrichment
- Automatic versioning of cached entries
- Configurable TTL and invalidation strategies
- Automatic cleanup of expired entries
- Performance optimization and statistics tracking

## Command-Line Interface

The `brave_search_quality_cli.py` script provides a convenient way to test and demonstrate the Quality API:

```bash
# Perform a high-quality web search
python brave_search_quality_cli.py web "climate change latest research"

# Perform a high-quality AI search
python brave_search_quality_cli.py ai "explain quantum computing"

# Perform a combined search (both web and AI)
python brave_search_quality_cli.py combined "benefits of meditation"

# Perform an adaptive search (automatically chooses the best strategy)
python brave_search_quality_cli.py adaptive "how to bake sourdough bread"

# Improve search quality by trying different query variations
python brave_search_quality_cli.py web "climate change" --improve

# Get statistics about the quality API
python brave_search_quality_cli.py stats
```

## Testing

The Quality API includes a comprehensive test suite in `test_brave_search_quality_api.py`:

```bash
# Run the tests
python test_brave_search_quality_api.py
```

## Integration with LiveKit Amanda

The Quality API is designed to integrate seamlessly with the LiveKit Amanda project:

1. **Cache Configuration**: Uses the global cache configuration system
2. **Statistics Tracking**: Integrates with the existing statistics tracking system
3. **Fallback Mechanisms**: Gracefully falls back to standard search when needed
4. **Singleton Pattern**: Ensures consistent state and efficient resource usage

## Performance Considerations

- **Caching Strategy**: High-quality results are cached for longer periods (2 weeks by default)
- **Refresh Threshold**: Cached results are refreshed after a configurable period (1 week by default)
- **Parallel Processing**: Web and AI searches can be performed in parallel for faster combined results
- **Quality Assessment**: Quality assessment adds minimal overhead while significantly improving result quality

## Security and Privacy

- **API Keys**: Managed through environment variables to avoid hardcoding sensitive information
- **Data Storage**: All cached data is stored locally, with no external dependencies for persistence
- **Configurable Storage**: Storage backend and location are configurable

## Future Enhancements

Planned enhancements for the Quality API include:

1. **Advanced Query Understanding**: Better analysis of query intent for improved adaptive search
2. **Multi-Modal Search**: Integration with image and video search capabilities
3. **Federated Search**: Combining results from multiple search providers
4. **User Feedback Loop**: Incorporating user feedback to improve quality assessment
5. **Enhanced Data Enrichment**: More sophisticated data enrichment techniques

## Troubleshooting

Common issues and solutions:

- **Missing API Keys**: Ensure that the Brave Search API keys are properly configured in the environment variables
- **Cache Performance**: If cache performance degrades, try running the `optimize` method on the persistent cache
- **Memory Usage**: For large deployments, monitor memory usage and adjust cache size limits accordingly
- **Rate Limiting**: Be aware of the Brave Search API rate limits when performing multiple searches

## API Reference

For detailed API documentation, refer to the docstrings in the source code:

- `brave_search_quality_api.py`: Core API functionality
- `brave_search_persistent_cache.py`: Persistent cache implementation
- `brave_search_quality_integration.py`: Integration layer for applications

## Contributing

Contributions to the Quality API are welcome! Please follow these guidelines:

1. Write tests for new functionality
2. Maintain backward compatibility
3. Document new features and configuration options
4. Follow the existing code style and naming conventions
