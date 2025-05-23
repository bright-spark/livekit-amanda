#!/usr/bin/env python3
"""
Script to check if all files in the data directory have been embedded.
"""

import os
import pickle
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("check_embedding_status")

def check_embedding_status(data_dir):
    """
    Check if all files in the data directory have been embedded.
    
    Args:
        data_dir: Path to the data directory
    """
    logger.info(f"Checking embedding status for data directory: {data_dir}")
    
    # Get all files in the data directory
    data_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            # Skip the embedding state file itself
            if file == "embedding_state.pkl":
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, data_dir)
            data_files.append(rel_path)
    
    logger.info(f"Found {len(data_files)} files in data directory")
    
    # Check if embedding state file exists
    embedding_state_path = os.path.join(data_dir, "embedding_state.pkl")
    if not os.path.exists(embedding_state_path):
        logger.warning("Embedding state file not found. No files have been embedded.")
        return False
    
    # Load the embedding state
    try:
        with open(embedding_state_path, "rb") as f:
            embedding_state = pickle.load(f)
        
        if not isinstance(embedding_state, dict):
            logger.warning(f"Embedding state is not a dictionary: {type(embedding_state)}")
            return False
            
        logger.info(f"Loaded embedding state with {len(embedding_state)} entries")
        
        # Check which files have been embedded
        embedded_files = set(embedding_state.keys())
        all_files = set(data_files)
        
        not_embedded = all_files - embedded_files
        
        if not_embedded:
            logger.warning(f"Found {len(not_embedded)} files that have not been embedded:")
            for file in sorted(not_embedded):
                logger.warning(f"  - {file}")
            return False
        else:
            logger.info("All files have been embedded!")
            return True
            
    except Exception as e:
        logger.error(f"Error loading embedding state: {e}")
        return False

if __name__ == "__main__":
    # Use the data directory from the command line or default to the data directory in the current directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
    check_embedding_status(data_dir)
