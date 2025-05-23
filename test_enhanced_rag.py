"""
Test script for the enhanced RAG search functionality using ChromaDB.

This script tests the RAG functionality by adding search results to the cache,
retrieving them, and verifying that the vector database is working correctly.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set environment variables for testing if not already set
if "ENHANCED_SEARCH_ENABLE_RAG" not in os.environ:
    os.environ["ENHANCED_SEARCH_ENABLE_RAG"] = "true"
if "ENHANCED_SEARCH_RAG_CACHE_DIR" not in os.environ:
    os.environ["ENHANCED_SEARCH_RAG_CACHE_DIR"] = os.path.expanduser("~/.enhanced_search_rag_cache")
if "ENHANCED_SEARCH_RAG_MAX_ENTRIES" not in os.environ:
    os.environ["ENHANCED_SEARCH_RAG_MAX_ENTRIES"] = "1000"
if "ENHANCED_SEARCH_RAG_MIN_SIMILARITY" not in os.environ:
    os.environ["ENHANCED_SEARCH_RAG_MIN_SIMILARITY"] = "0.75"

# Import the modules to test
try:
    from chromadb_rag import (
        add_to_rag_cache,
        retrieve_from_rag_cache,
        invalidate_rag_entry,
        invalidate_rag_entries_by_query
    )
    logger.info("Using ChromaDB RAG implementation")
except ImportError:
    logger.error("ChromaDB RAG implementation not available")
    from enhanced_search import (
        add_to_rag_cache,
        retrieve_from_rag_cache,
        invalidate_rag_entry,
        invalidate_rag_entries_by_query
    )
    logger.info("Using fallback RAG implementation")

async def test_rag_functionality():
    """Test the RAG functionality."""
    logger.info("Starting RAG functionality test")
    
    # Test data
    test_queries = [
        "quantum computing advancements",
        "artificial intelligence ethics",
        "climate change solutions",
        "space exploration technologies",
        "renewable energy innovations"
    ]
    
    test_results = [
        "Recent advancements in quantum computing include improvements in qubit stability, error correction, and quantum algorithms. IBM and Google have made significant progress in increasing qubit counts and reducing error rates.",
        "AI ethics concerns include bias in algorithms, privacy issues, job displacement, and the need for transparent and explainable AI systems. Organizations are developing ethical guidelines for AI development and deployment.",
        "Climate change solutions include renewable energy adoption, carbon capture technologies, sustainable agriculture practices, and policy changes to reduce greenhouse gas emissions.",
        "Space exploration technologies include reusable rockets, small satellites, space habitats, and propulsion systems for deep space missions. Companies like SpaceX and Blue Origin are leading private sector space initiatives.",
        "Renewable energy innovations include improved solar panel efficiency, advanced battery storage systems, floating wind turbines, and smart grid technologies for better energy distribution and management."
    ]
    
    # Add test data to RAG cache
    entry_ids = []
    for i, (query, result) in enumerate(zip(test_queries, test_results)):
        logger.info(f"Adding test data {i+1}/{len(test_queries)}: {query}")
        entry_id = await add_to_rag_cache(query, result)
        if entry_id:
            entry_ids.append(entry_id)
            logger.info(f"Added entry with ID: {entry_id}")
        else:
            logger.warning(f"Failed to add entry for query: {query}")
    
    # Test retrieval
    for i, query in enumerate(test_queries):
        logger.info(f"Testing retrieval {i+1}/{len(test_queries)}: {query}")
        results = await retrieve_from_rag_cache(query, max_results=3)
        logger.info(f"Retrieved {len(results)} results for '{query}'")
        for j, result in enumerate(results):
            logger.info(f"  Result {j+1}: Similarity={result['similarity']:.4f}")
            logger.info(f"    Query: {result['query']}")
            logger.info(f"    Results: {result['results'][:100]}...")
    
    # Test semantic retrieval (should retrieve related entries)
    semantic_queries = [
        "latest quantum computing research",
        "ethical considerations in AI development",
        "solutions to global warming",
        "Mars mission technology",
        "solar and wind power advances"
    ]
    
    for i, query in enumerate(semantic_queries):
        logger.info(f"Testing semantic retrieval {i+1}/{len(semantic_queries)}: {query}")
        results = await retrieve_from_rag_cache(query, max_results=3)
        logger.info(f"Retrieved {len(results)} results for '{query}'")
        for j, result in enumerate(results):
            logger.info(f"  Result {j+1}: Similarity={result['similarity']:.4f}")
            logger.info(f"    Query: {result['query']}")
            logger.info(f"    Results: {result['results'][:100]}...")
    
    # Test invalidation of a specific entry
    if entry_ids:
        entry_id_to_invalidate = entry_ids[0]
        logger.info(f"Testing invalidation of entry: {entry_id_to_invalidate}")
        success = await invalidate_rag_entry(entry_id_to_invalidate)
        logger.info(f"Invalidation {'succeeded' if success else 'failed'}")
        
        # Verify the entry was invalidated
        results = await retrieve_from_rag_cache(test_queries[0], max_results=1)
        if results and results[0]['id'] == entry_id_to_invalidate:
            logger.warning("Entry was not properly invalidated")
        else:
            logger.info("Entry was successfully invalidated")
    
    # Test invalidation by query similarity
    query_to_invalidate = test_queries[1]
    logger.info(f"Testing invalidation by query similarity: {query_to_invalidate}")
    count = await invalidate_rag_entries_by_query(query_to_invalidate, similarity_threshold=0.9)
    logger.info(f"Invalidated {count} entries")
    
    # Verify the entries were invalidated
    results = await retrieve_from_rag_cache(query_to_invalidate, max_results=1)
    if results and results[0]['query'] == query_to_invalidate:
        logger.warning("Entries were not properly invalidated by query")
    else:
        logger.info("Entries were successfully invalidated by query")
    
    logger.info("RAG functionality test completed")

if __name__ == "__main__":
    asyncio.run(test_rag_functionality())
