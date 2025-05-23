# Background Embedding Processing

This document explains the background embedding processing feature in the LiveKit Amanda project.

## Overview

The LiveKit Amanda project now supports background processing for embedding generation. This feature allows the system to process and embed documents in the background without blocking the agent or other tools, providing a more responsive user experience. The system now integrates with LangChain components for improved performance and functionality, including chunked processing to minimize memory usage.

Key features include:
- Parallel processing of embeddings using a thread pool
- Real-time progress reporting with throughput metrics
- Ability to interrupt and resume processing
- Persistent state tracking for recovery after restarts
- LangChain integration for improved vector storage and retrieval
- Chunked processing to minimize memory usage
- Real-time file monitoring to detect changes in the data directory
- Multi-level caching strategy for optimal performance

## Configuration

The following environment variables control the background embedding processing:

```
# Enable/disable background embedding
ENABLE_BACKGROUND_EMBEDDING=true

# Enable/disable RAG functionality
ENABLE_RAG=true

# Enable/disable real-time embedding
ENABLE_REALTIME_EMBEDDING=true

# Maximum number of worker threads for embedding generation
MAX_EMBEDDING_WORKERS=4

# Size of embedding chunks for processing
EMBEDDING_CHUNK_SIZE=10

# Minimum similarity threshold for retrieval
MIN_SIMILARITY_THRESHOLD=0.75

# Progress reporting interval in percentage
EMBEDDING_PROGRESS_INTERVAL=5.0

# Directory for local data files
DATA_DIR=/path/to/data
```

## How It Works

1. **File Discovery**: When new files are added to the data directory, they are detected by the file system monitor.
2. **Metadata Processing**: File metadata is processed immediately, making the content available for search.
3. **Embedding Queue**: Files are added to a queue for background embedding generation.
4. **Parallel Processing**: A thread pool processes the queue, generating embeddings in parallel.
5. **Progress Tracking**: The system tracks progress and reports metrics at regular intervals.
6. **State Persistence**: The embedding state is saved to disk, allowing for recovery after restarts.

## Embedding Monitor Tool

A command-line tool (`embedding_monitor.py`) is provided to monitor and manage the background embedding process:

```bash
# Show current embedding progress
python embedding_monitor.py

# Continuously monitor embedding progress
python embedding_monitor.py monitor --continuous

# Show embedding statistics
python embedding_monitor.py stats

# List all data files
python embedding_monitor.py list

# List only pending files
python embedding_monitor.py list --pending

# Force reprocessing of files matching a pattern
python embedding_monitor.py reprocess "pattern"
```

## Progress Metrics

The system reports the following metrics:
- **Percentage Complete**: Percentage of files that have been processed
- **Throughput Rate**: Number of files processed per second
- **Time Elapsed**: Total time spent on embedding generation
- **Estimated Time Remaining**: Projected time to complete all pending files

## Benefits

1. **Non-blocking Operation**: The agent and search functionality remain responsive while embeddings are generated.
2. **Efficient Resource Usage**: Parallel processing makes efficient use of system resources.
3. **Resilience**: The system can recover from interruptions and continue processing.
4. **Visibility**: Real-time progress reporting provides visibility into the embedding process.

## LangChain Integration

The system now integrates with LangChain components for improved vector storage and retrieval:

1. **Vector Stores**: Uses LangChain's `Chroma` vector stores for quality enriched data and RAG data
2. **Memory Cache**: Uses LangChain's `ConversationBufferMemory` for in-memory caching of quality data
3. **Embeddings**: Uses LangChain's `OpenAIEmbeddings` for generating vector representations
4. **Document Loaders**: Uses LangChain's `DirectoryLoader` and `TextLoader` for processing local files
5. **Text Splitter**: Uses LangChain's `RecursiveCharacterTextSplitter` for chunking documents
6. **Retrievers**: Uses LangChain's `ContextualCompressionRetriever` with `EmbeddingsFilter` for better retrieval

### Data Flow

The integration implements the following data flow:

1. **Quality query result / query result → Memory cache (quality data)**
   - Search queries and their results are first stored in a high-performance in-memory cache
   - Uses LangChain's `ConversationBufferMemory` for efficient storage and retrieval

2. **Quality result → Persistent cache (quality enriched data)**
   - High-quality search results are stored in a persistent cache for longer retention
   - Includes metadata about quality scores, timestamps, and refresh thresholds

3. **Persistent cache → Vector store (quality enriched data)**
   - Data from the persistent cache is embedded and stored in a vector database
   - Uses LangChain's `Chroma` vector store with semantic search capabilities

4. **@data directory → Vector store (RAG data)**
   - Local data files are processed, embedded, and stored in a separate vector collection
   - Supports multiple file formats (txt, md, json, csv, html, xml, py, js)

## Implementation Details

### Components

1. **EmbeddingProgressTracker**: Tracks and reports embedding progress.
2. **ThreadPoolExecutor**: Manages a pool of worker threads for parallel processing.
3. **Embedding Queue**: Stores files waiting for embedding generation.
4. **State Persistence**: Saves and loads embedding state to/from disk.
5. **DataDirectoryWatcher**: Monitors the data directory for file changes and updates the vector store.

### Process Flow

#### Background Embedding Process
1. When a new file is detected, its metadata is processed immediately.
2. The file is added to the embedding queue and the embedding state.
3. Worker threads pick up files from the queue and generate embeddings in chunks to minimize memory usage.
4. As embeddings are generated, they are added to the vector store.
5. Progress is tracked and reported at regular intervals.
6. The embedding state is updated to reflect completed files.

#### File Monitoring Process
1. The file watcher monitors the data directory for any changes (additions, modifications, deletions).
2. When a change is detected, it is added to the file change queue.
3. The file change processor thread retrieves events from the queue.
4. For additions/updates: The file is loaded, split into chunks, and added to the vector store.
5. For deletions: The corresponding documents are removed from the vector store.
6. The vector store is persisted to save changes.

## Best Practices

1. **Resource Management**: Adjust `MAX_EMBEDDING_WORKERS` based on your system's capabilities.
2. **Monitoring**: Use the embedding monitor tool to track progress and identify issues.
3. **Graceful Shutdown**: Allow the system to complete current embedding tasks before shutdown.
4. **Regular Backups**: Periodically back up the vector cache and embedding state.

## Troubleshooting

If you encounter issues with background embedding:

1. Check the logs for error messages.
2. Verify that the environment variables are correctly set.
3. Ensure that the data directory exists and is writable.
4. Use the embedding monitor tool to check the status of the embedding process.
5. Try reprocessing problematic files using the `reprocess` command.
