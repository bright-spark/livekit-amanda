#!/usr/bin/env python3
"""
Script to run the embedding process for all files in the data directory.
"""

import asyncio
import logging
from brave_search_quality_rag_integration import process_data_directory, close_integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_embedding")

async def main():
    """Run the embedding process."""
    logger.info("Starting embedding process for all files in the data directory")
    
    try:
        # Process all files in the data directory
        await process_data_directory()
        logger.info("Embedding process completed successfully")
    except Exception as e:
        logger.error(f"Error during embedding process: {e}")
    finally:
        # Close the integration
        await close_integration()

if __name__ == "__main__":
    asyncio.run(main())
