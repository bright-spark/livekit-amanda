#!/usr/bin/env python3
"""
Test script for RAG cache invalidation and real-time monitoring.

This script demonstrates:
1. Adding documents to the data directory
2. Searching with local data integration
3. Invalidating specific entries in the RAG cache
4. Observing real-time updates to the data directory
"""

import os
import sys
import time
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_rag_cache")

# Import our enhanced search module
from enhanced_search import (
    search_with_local_data,
    invalidate_rag_entry,
    invalidate_rag_entries_by_query,
    invalidate_local_data,
    invalidate_brave_search_cache,
    data_manager
)

async def test_search(query: str) -> Dict[str, Any]:
    """Perform a search with local data integration."""
    logger.info(f"Searching for: '{query}'")
    result = await search_with_local_data(query, max_results=5)
    
    logger.info(f"Search results:")
    logger.info(f"Query: {result['query']}")
    logger.info(f"Reformulated query: {result.get('reformulated_query', result['query'])}")
    logger.info(f"Sources: {result.get('sources', [])}")
    logger.info(f"Web results: {len(result.get('web_results', []))}")
    logger.info(f"Local results: {len(result.get('local_results', []))}")
    
    return result

async def create_test_document(content: str, filename: str = None) -> str:
    """Create a test document in the data directory."""
    if filename is None:
        filename = f"test_doc_{int(time.time())}.txt"
    
    file_path = os.path.join(data_manager.DATA_DIR, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Created test document: {file_path}")
    
    # Wait a moment for the file watcher to process the new file
    await asyncio.sleep(3)
    
    return os.path.relpath(file_path, data_manager.DATA_DIR)

async def test_invalidate_document(file_id: str) -> bool:
    """Test invalidating a document."""
    logger.info(f"Invalidating document: {file_id}")
    result = await invalidate_local_data(file_id)
    logger.info(f"Document invalidation result: {result}")
    return result

async def test_invalidate_rag_by_query(query: str, similarity_threshold: float = 0.9) -> int:
    """Test invalidating RAG entries by query."""
    logger.info(f"Invalidating RAG entries matching query: '{query}' with threshold: {similarity_threshold}")
    count = await invalidate_rag_entries_by_query(query, similarity_threshold)
    logger.info(f"Invalidated {count} RAG entries")
    return count

async def test_invalidate_brave_cache(query: str = None, url: str = None) -> bool:
    """Test invalidating Brave Search cache entries."""
    if query and url:
        logger.info(f"Invalidating Brave Search cache entries matching query: '{query}' and URL: '{url}'")
    elif query:
        logger.info(f"Invalidating Brave Search cache entries matching query: '{query}'")
    elif url:
        logger.info(f"Invalidating Brave Search cache entries containing URL: '{url}'")
    else:
        logger.info("No query or URL provided for Brave Search cache invalidation")
        return False
    
    result = await invalidate_brave_search_cache(query, url)
    logger.info(f"Brave Search cache invalidation result: {result}")
    return result

async def test_real_time_monitoring() -> None:
    """Test real-time monitoring of the data directory."""
    logger.info("Testing real-time monitoring of the data directory...")
    
    # Create a test document
    doc1_id = await create_test_document("This is a test document about artificial intelligence and machine learning.")
    
    # Search for related content
    await test_search("artificial intelligence")
    
    # Modify the document
    file_path = os.path.join(data_manager.DATA_DIR, doc1_id)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write("\n\nAI systems can now perform tasks that typically require human intelligence.")
    
    logger.info(f"Modified document: {file_path}")
    
    # Wait for the file watcher to process the modification
    await asyncio.sleep(3)
    
    # Search again to see the updated content
    await test_search("artificial intelligence tasks")
    
    # Create another document
    doc2_id = await create_test_document("Quantum computing uses quantum mechanics to perform computations.", "quantum_computing.txt")
    
    # Search for the new document
    await test_search("quantum computing")
    
    # Delete the first document
    os.remove(file_path)
    logger.info(f"Deleted document: {file_path}")
    
    # Wait for the file watcher to process the deletion
    await asyncio.sleep(3)
    
    # Search again to verify the document was removed
    result = await test_search("artificial intelligence")
    
    # Check if the deleted document is still in the results
    for local_result in result.get('local_results', []):
        if local_result.get('file_id') == doc1_id:
            logger.error(f"Document {doc1_id} was not properly removed from the index!")
        else:
            logger.info(f"Document {doc1_id} was successfully removed from the index.")
    
    # Clean up
    os.remove(os.path.join(data_manager.DATA_DIR, doc2_id))
    logger.info("Real-time monitoring test completed.")

async def main() -> None:
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test RAG cache invalidation and real-time monitoring")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--create-doc", type=str, help="Create a test document with the given content")
    parser.add_argument("--invalidate-doc", type=str, help="Invalidate a document by its ID")
    parser.add_argument("--invalidate-rag", type=str, help="Invalidate RAG entries by query")
    parser.add_argument("--invalidate-brave", type=str, help="Invalidate Brave Search cache entries by query")
    parser.add_argument("--invalidate-url", type=str, help="Invalidate Brave Search cache entries by URL")
    parser.add_argument("--test-monitoring", action="store_true", help="Test real-time monitoring")
    
    args = parser.parse_args()
    
    if args.search:
        await test_search(args.search)
    
    if args.create_doc:
        await create_test_document(args.create_doc)
    
    if args.invalidate_doc:
        await test_invalidate_document(args.invalidate_doc)
    
    if args.invalidate_rag:
        await test_invalidate_rag_by_query(args.invalidate_rag)
    
    if args.invalidate_brave:
        await test_invalidate_brave_cache(args.invalidate_brave, args.invalidate_url)
    
    if args.test_monitoring:
        await test_real_time_monitoring()
    
    # If no arguments were provided, run the full test suite
    if not any(vars(args).values()):
        logger.info("Running full test suite...")
        
        # Test search
        await test_search("latest developments in quantum computing")
        
        # Test document creation and invalidation
        doc_id = await create_test_document("This is a test document about climate change and its effects on the environment.")
        await test_search("climate change")
        await test_invalidate_document(doc_id)
        
        # Test RAG invalidation
        await test_invalidate_rag_by_query("climate change")
        
        # Test Brave Search cache invalidation
        await test_invalidate_brave_cache("climate change")
        
        # Test real-time monitoring
        await test_real_time_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
