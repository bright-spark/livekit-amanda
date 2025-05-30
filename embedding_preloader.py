"""
Embedding model preloader for LiveKit Amanda.

This module provides functionality to pre-download and cache embedding models
to ensure they're available locally for future use without requiring internet access.
"""

import os
import logging
from typing import Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Embedding model configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Small, modern, efficient model with good performance

# Model cache directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_CACHE_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Global embedding model instance
_embedding_model = None

def preload_embedding_model() -> bool:
    """
    Pre-download the embedding model to ensure it's available locally.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global _embedding_model
    
    try:
        logger.info(f"Pre-downloading embedding model: {EMBEDDING_MODEL_NAME}")
        start_time = time.time()
        
        # Set environment variable to force CPU usage and avoid device-related errors
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["USE_TORCH"] = "0"
        
        # Import here to avoid circular imports
        from sentence_transformers import SentenceTransformer
        
        # Force CPU mode by setting device explicitly
        logger.info("Using CPU device for embedding model (forced for compatibility)")
        
        # Load the model with explicit cache folder
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, 
            cache_folder=MODEL_CACHE_DIR,
            device="cpu"  # Force CPU mode
        )
        
        # Test the model with a simple embedding to ensure it works
        _ = _embedding_model.encode("Test sentence for model verification")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully pre-downloaded embedding model: {EMBEDDING_MODEL_NAME} in {elapsed_time:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error pre-downloading embedding model: {e}")
        _embedding_model = None
        return False

def get_cached_model():
    """
    Get the cached embedding model instance.
    
    Returns:
        The cached embedding model or None if not loaded
    """
    global _embedding_model
    return _embedding_model

# For testing
if __name__ == "__main__":
    success = preload_embedding_model()
    print(f"Model preloading {'successful' if success else 'failed'}")
