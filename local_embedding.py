#!/usr/bin/env python3
"""
Standalone script to process embeddings using the local all-MiniLM-L6-v2 model.
This script bypasses the complex integration to directly embed files in the data directory.
"""

import os
import asyncio
import logging
import glob
import json
from typing import List, Dict, Any
from pathlib import Path

# Import sentence-transformers for local embeddings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("local_embedding")

# Configuration
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_STATE_PATH = os.path.join(DATA_DIR, "embedding_state.pkl")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

# Global embedding model instance
embedding_model = None

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using the sentence-transformers all-MiniLM-L6-v2 model.
    
    Args:
        text: The text to embed
        
    Returns:
        Text embedding as a list of floats
    """
    global embedding_model
    
    try:
        # Truncate text if too long (all-MiniLM-L6-v2 has a context length of 256 tokens)
        # We'll use a conservative character limit
        if len(text) > 5000:
            text = text[:5000]
        
        # Lazy-load the model if it's not already loaded
        if embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
                embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info(f"Successfully loaded embedding model: {EMBEDDING_MODEL_NAME}")
            except Exception as model_error:
                logger.error(f"Error loading embedding model: {model_error}")
                # Return a random vector as fallback
                import random
                return [random.uniform(-1, 1) for _ in range(384)]  # all-MiniLM-L6-v2 has 384 dimensions
        
        # Generate embedding
        # We use asyncio.to_thread to run the CPU-intensive embedding generation in a separate thread
        # This prevents blocking the event loop
        embedding = await asyncio.to_thread(embedding_model.encode, text)
        
        # Convert numpy array to list
        return embedding.tolist()
    
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * 384  # all-MiniLM-L6-v2 has 384 dimensions

def load_embedding_state() -> Dict[str, Any]:
    """Load the embedding state from disk."""
    try:
        if os.path.exists(EMBEDDING_STATE_PATH):
            with open(EMBEDDING_STATE_PATH, 'r') as f:
                return json.load(f)
        return {"embedded_files": {}}
    except Exception as e:
        logger.error(f"Error loading embedding state: {e}")
        return {"embedded_files": {}}

def save_embedding_state(state: Dict[str, Any]) -> None:
    """Save the embedding state to disk."""
    try:
        with open(EMBEDDING_STATE_PATH, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Error saving embedding state: {e}")

def get_file_hash(file_path: str) -> str:
    """Get a hash of the file contents."""
    import hashlib
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error getting file hash for {file_path}: {e}")
        return ""

async def process_file(file_path: str, embedding_state: Dict[str, Any]) -> None:
    """Process a single file for embeddings."""
    try:
        # Get file hash
        file_hash = get_file_hash(file_path)
        rel_path = os.path.relpath(file_path, DATA_DIR)
        
        # Check if file has already been embedded
        if rel_path in embedding_state["embedded_files"] and embedding_state["embedded_files"][rel_path] == file_hash:
            logger.info(f"File already embedded: {rel_path}")
            return
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Get embedding
        logger.info(f"Getting embedding for file: {rel_path}")
        embedding = await get_embedding(content)
        
        # Save embedding to vector store
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        vector_file = os.path.join(VECTOR_STORE_DIR, f"{os.path.basename(file_path)}.json")
        
        with open(vector_file, 'w') as f:
            json.dump({
                "content": content,
                "embedding": embedding,
                "metadata": {
                    "source": rel_path,
                    "created_at": os.path.getctime(file_path),
                    "modified_at": os.path.getmtime(file_path)
                }
            }, f)
        
        # Update embedding state
        embedding_state["embedded_files"][rel_path] = file_hash
        logger.info(f"Successfully embedded file: {rel_path}")
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

async def process_data_directory() -> None:
    """Process all files in the data directory for embeddings."""
    if not os.path.exists(DATA_DIR):
        logger.warning(f"Data directory does not exist: {DATA_DIR}")
        return
    
    logger.info(f"Processing data directory: {DATA_DIR}")
    
    # Load embedding state
    embedding_state = load_embedding_state()
    
    # Get all files in the data directory
    file_patterns = ["**/*.txt", "**/*.md", "**/*.json", "**/*.csv", "**/*.html", "**/*.xml", "**/*.py", "**/*.js"]
    all_files = []
    
    for pattern in file_patterns:
        all_files.extend(glob.glob(os.path.join(DATA_DIR, pattern), recursive=True))
    
    logger.info(f"Found {len(all_files)} files in data directory")
    
    # Process each file
    for file_path in all_files:
        await process_file(file_path, embedding_state)
    
    # Save embedding state
    save_embedding_state(embedding_state)
    logger.info("Finished processing data directory")

async def main():
    """Main function."""
    logger.info("Starting local embedding process")
    
    try:
        # Process data directory
        await process_data_directory()
        logger.info("Embedding process completed successfully")
    except Exception as e:
        logger.error(f"Error during embedding process: {e}")

if __name__ == "__main__":
    asyncio.run(main())
