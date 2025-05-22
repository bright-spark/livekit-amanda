# Background Embedding Processing

This document explains the background embedding processing feature in the LiveKit Amanda project.

## Overview

The LiveKit Amanda project now supports background processing for embedding generation. This feature allows the system to process and embed documents in the background without blocking the agent or other tools, providing a more responsive user experience.

Key features include:
- Parallel processing of embeddings using a thread pool
- Real-time progress reporting with throughput metrics
- Ability to interrupt and resume processing
- Persistent state tracking for recovery after restarts

## Configuration

The following environment variables control the background embedding processing:

```
# Enable/disable background embedding
ENABLE_BACKGROUND_EMBEDDING=true

# Maximum number of worker threads for embedding generation
MAX_EMBEDDING_WORKERS=4

# Progress reporting interval in percentage
EMBEDDING_PROGRESS_INTERVAL=5.0
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

## Implementation Details

### Components

1. **EmbeddingProgressTracker**: Tracks and reports embedding progress.
2. **ThreadPoolExecutor**: Manages a pool of worker threads for parallel processing.
3. **Embedding Queue**: Stores files waiting for embedding generation.
4. **State Persistence**: Saves and loads embedding state to/from disk.

### Process Flow

1. When a new file is detected, its metadata is processed immediately.
2. The file is added to the embedding queue and the embedding state.
3. Worker threads pick up files from the queue and generate embeddings.
4. As embeddings are generated, they are added to the vector store.
5. Progress is tracked and reported at regular intervals.
6. The embedding state is updated to reflect completed files.

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
