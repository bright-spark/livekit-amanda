#!/usr/bin/env python3
"""
Test script for Brave Search Quality API integration with LangChain and RAG.
This script tests the full data flow:
1. Quality query result / query result → Memory cache (quality data)
2. Quality result → Persistent cache (quality enriched data)
3. Persistent cache → Vector store (quality enriched data)
4. @data directory → Vector store (RAG data)
"""

import os
import asyncio
import logging
from pprint import pprint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_brave_search_quality_rag")

# Import the integration components
from brave_search_quality_rag_integration import (
    get_integration,
    process_data_directory,
    search_with_rag,
    close_integration
)

async def test_data_directory_processing():
    """Test processing the data directory."""
    logger.info("Testing data directory processing...")
    await process_data_directory()
    logger.info("Data directory processing completed")

async def test_search_with_rag():
    """Test searching with RAG."""
    logger.info("Testing search with RAG...")
    
    # Create a test context
    context = {"session_id": "test_session"}
    
    # Perform a search
    query = "climate change solutions"
    logger.info(f"Searching for: {query}")
    results = await search_with_rag(context, query, num_results=3)
    
    # Print a preview of the results
    preview = results[:300] + "..." if len(results) > 300 else results
    logger.info(f"Search results preview: {preview}")
    
    return results

async def test_memory_cache():
    """Test the memory cache."""
    logger.info("Testing memory cache...")
    
    # Get the integration instance
    integration = get_integration()
    
    # Check if memory cache is initialized
    logger.info(f"Memory cache initialized: {integration.memory_cache is not None}")
    
    # Print memory cache info
    memory_info = str(integration.memory_cache)
    preview = memory_info[:100] + "..." if len(memory_info) > 100 else memory_info
    logger.info(f"Memory cache: {preview}")

async def test_vector_stores():
    """Test the vector stores."""
    logger.info("Testing vector stores...")
    
    # Get the integration instance
    integration = get_integration()
    
    # Check if vector stores are initialized
    logger.info(f"Quality vector store initialized: {integration.vector_store_quality is not None}")
    logger.info(f"RAG vector store initialized: {integration.vector_store_rag is not None}")
    
    # Try to get documents from the vector stores
    try:
        quality_docs = integration.vector_store_quality.get()
        if quality_docs:
            logger.info(f"Quality vector store contains {len(quality_docs.get('ids', []))} documents")
        else:
            logger.info("Quality vector store is empty")
    except Exception as e:
        logger.error(f"Error accessing quality vector store: {e}")
    
    try:
        rag_docs = integration.vector_store_rag.get()
        if rag_docs:
            logger.info(f"RAG vector store contains {len(rag_docs.get('ids', []))} documents")
        else:
            logger.info("RAG vector store is empty")
    except Exception as e:
        logger.error(f"Error accessing RAG vector store: {e}")

async def test_file_watcher():
    """Test the file watcher."""
    logger.info("Testing file watcher...")
    
    # Get the integration instance
    integration = get_integration()
    
    # Check if file watcher is running
    logger.info(f"File watcher running: {integration.file_watcher is not None}")
    
    if integration.file_watcher:
        logger.info(f"Watching directory: {integration.file_watcher.data_dir}")
        logger.info(f"Number of tracked files: {len(integration.file_watcher.file_hashes)}")

async def main():
    """Run all tests."""
    try:
        logger.info("Starting tests for Brave Search Quality API integration with LangChain and RAG")
        
        # Test data directory processing
        await test_data_directory_processing()
        
        # Test search with RAG
        await test_search_with_rag()
        
        # Test memory cache
        await test_memory_cache()
        
        # Test vector stores
        await test_vector_stores()
        
        # Test file watcher
        await test_file_watcher()
        
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
    finally:
        # Close the integration
        await close_integration()
        logger.info("Integration closed")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
