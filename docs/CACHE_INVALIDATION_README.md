# Cache Invalidation and Real-time Monitoring

This document explains how to use the cache invalidation and real-time monitoring features in the LiveKit Amanda project.

## Overview

The LiveKit Amanda project now includes comprehensive cache invalidation functionality for both the RAG system and the Brave Search persistent store, along with real-time monitoring of the data directory for the RAG system.

These features ensure data quality and accuracy by providing mechanisms to remove incorrect or outdated information from both the RAG system and the persistent cache.

## Configuration

The following environment variables control the cache invalidation and real-time monitoring features:

```
# Real-time monitoring
ENABLE_REALTIME_MONITORING=true

# Cache invalidation
ALLOW_CACHE_INVALIDATION=true
AUTO_INVALIDATE_STALE_ENTRIES=true
MAX_CACHE_ENTRY_AGE_DAYS=30
```

## RAG Cache Invalidation

### Invalidate a Specific RAG Entry

To invalidate a specific entry in the RAG cache by its ID:

```python
from enhanced_search import invalidate_rag_entry

# Invalidate a specific RAG entry
async def invalidate_entry(entry_id: str):
    success = await invalidate_rag_entry(entry_id)
    if success:
        print(f"Successfully invalidated RAG entry: {entry_id}")
    else:
        print(f"Failed to invalidate RAG entry: {entry_id}")
```

### Invalidate RAG Entries by Query

To invalidate entries in the RAG cache that match a specific query with high similarity:

```python
from enhanced_search import invalidate_rag_entries_by_query

# Invalidate RAG entries by query
async def invalidate_entries(query: str, similarity_threshold: float = 0.9):
    count = await invalidate_rag_entries_by_query(query, similarity_threshold)
    print(f"Invalidated {count} RAG entries matching query: '{query}'")
```

### Invalidate Local Data

To invalidate a specific document in the local data store:

```python
from enhanced_search import invalidate_local_data

# Invalidate a local document
async def invalidate_document(file_id: str):
    success = await invalidate_local_data(file_id)
    if success:
        print(f"Successfully invalidated document: {file_id}")
    else:
        print(f"Failed to invalidate document: {file_id}")
```

## Brave Search Cache Invalidation

### Invalidate Brave Search Cache by Query

To invalidate entries in the Brave Search cache that match a specific query:

```python
from enhanced_search import invalidate_brave_search_cache

# Invalidate Brave Search cache by query
async def invalidate_brave_cache_by_query(query: str):
    success = await invalidate_brave_search_cache(query=query)
    if success:
        print(f"Successfully invalidated Brave Search cache entries for query: '{query}'")
    else:
        print(f"Failed to invalidate Brave Search cache entries for query: '{query}'")
```

### Invalidate Brave Search Cache by URL

To invalidate entries in the Brave Search cache that contain a specific URL:

```python
from enhanced_search import invalidate_brave_search_cache

# Invalidate Brave Search cache by URL
async def invalidate_brave_cache_by_url(url: str):
    success = await invalidate_brave_search_cache(url=url)
    if success:
        print(f"Successfully invalidated Brave Search cache entries containing URL: '{url}'")
    else:
        print(f"Failed to invalidate Brave Search cache entries containing URL: '{url}'")
```

## Real-time Monitoring

The real-time monitoring feature automatically watches the data directory for changes and updates the vector cache accordingly. This happens in the background and requires no explicit action from the user.

### How It Works

1. When a new file is added to the data directory, it is automatically processed and indexed.
2. When a file is modified, its content is re-processed and its embedding is updated.
3. When a file is deleted, it is automatically removed from the index.
4. When a file is moved, it is treated as a deletion followed by a creation.

### Testing Real-time Monitoring

You can test the real-time monitoring feature using the provided test script:

```bash
python test_rag_cache_invalidation.py --test-monitoring
```

This will:
1. Create a test document in the data directory
2. Search for related content
3. Modify the document and observe the changes in search results
4. Create another document and search for it
5. Delete the first document and verify it's removed from the index

## Example Usage

A complete example script (`test_rag_cache_invalidation.py`) is provided to demonstrate the cache invalidation and real-time monitoring features.

To run the full test suite:

```bash
python test_rag_cache_invalidation.py
```

To run specific tests:

```bash
# Test search with local data
python test_rag_cache_invalidation.py --search "quantum computing"

# Create a test document
python test_rag_cache_invalidation.py --create-doc "This is a test document about AI."

# Invalidate a document
python test_rag_cache_invalidation.py --invalidate-doc "test_doc.txt"

# Invalidate RAG entries by query
python test_rag_cache_invalidation.py --invalidate-rag "climate change"

# Invalidate Brave Search cache by query
python test_rag_cache_invalidation.py --invalidate-brave "climate change"

# Invalidate Brave Search cache by URL
python test_rag_cache_invalidation.py --invalidate-url "example.com"

# Test real-time monitoring
python test_rag_cache_invalidation.py --test-monitoring
```

## Best Practices

1. **Selective Invalidation**: Only invalidate entries that are known to be incorrect or outdated.
2. **Regular Maintenance**: Set up a regular maintenance schedule to review and clean up the cache.
3. **Monitoring**: Monitor the cache size and performance to ensure it's operating efficiently.
4. **Backup**: Before performing large-scale invalidations, consider backing up the cache.
5. **Testing**: Test cache invalidation in a development environment before applying it to production.

## Troubleshooting

If you encounter issues with cache invalidation or real-time monitoring:

1. Check the logs for error messages.
2. Verify that the environment variables are correctly set.
3. Ensure that the data directory exists and is writable.
4. Check that the watchdog package is installed (`pip install watchdog`).
5. Restart the application to reinitialize the cache and file monitoring.
