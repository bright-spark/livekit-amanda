"""
Example script demonstrating the enhanced search with RAG functionality.

This script shows how to use the enhanced search with RAG to perform searches
and retrieve relevant information from both web search and the RAG cache.
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

# Import the enhanced search module
from enhanced_search import search_with_local_data, reformulate_query

async def main():
    """Run the example search with RAG."""
    # Example search queries
    queries = [
        "latest advancements in quantum computing",
        "ethical considerations in AI development",
        "climate change solutions for urban areas",
        "space exploration technologies for Mars missions",
        "renewable energy innovations for residential use"
    ]
    
    # Perform searches
    for query in queries:
        logger.info(f"Searching for: {query}")
        
        # Reformulate the query for better results
        reformulated_query = await reformulate_query(query)
        logger.info(f"Reformulated query: {reformulated_query}")
        
        # Perform the search with RAG
        results = await search_with_local_data(query, max_results=3)
        
        # Display results
        logger.info(f"Found {len(results.get('web_results', []))} web results")
        logger.info(f"Found {len(results.get('local_results', []))} local results")
        
        if 'summary' in results and results['summary']:
            logger.info(f"Summary: {results['summary']}")
        
        # Display web results
        for i, result in enumerate(results.get('web_results', [])[:3]):
            logger.info(f"Web Result {i+1}: {result.get('title', 'No title')}")
            logger.info(f"  URL: {result.get('url', 'No URL')}")
            snippet = result.get('snippet', 'No snippet')
            logger.info(f"  Snippet: {snippet[:100]}...")
        
        # Display RAG results if available
        for i, result in enumerate(results.get('rag_results', [])[:3]):
            logger.info(f"RAG Result {i+1}: {result.get('query', 'No query')}")
            logger.info(f"  Similarity: {result.get('similarity', 0):.4f}")
            content = result.get('results', 'No content')
            logger.info(f"  Content: {content[:100]}...")
        
        logger.info("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
