"""
Brave Search Persistent Cache with Enhanced Data Quality.

This module provides a specialized persistent cache for Brave Search API results with:
1. Enhanced data quality processing before storage
2. Sophisticated persistence strategy with versioning
3. Data enrichment capabilities
4. Separate API for high-quality data retrieval
5. Configurable storage backends
"""

import os
import time
import json
import logging
import asyncio
import hashlib
import sqlite3
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta
import aiofiles
import aiofiles.os
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
from asyncio import Semaphore

# Load environment variables
load_dotenv()

# Get API key for persistent cache
PERSISTENT_CACHE_API_KEY = os.getenv("BRAVE_PERSISTENT_CACHE_API_KEY")
# Fallback to web search API key if persistent cache API key is not set
if not PERSISTENT_CACHE_API_KEY:
    PERSISTENT_CACHE_API_KEY = os.getenv("BRAVE_WEB_SEARCH_API_KEY")
    if PERSISTENT_CACHE_API_KEY:
        logging.warning("BRAVE_PERSISTENT_CACHE_API_KEY not set, using BRAVE_WEB_SEARCH_API_KEY as fallback")
    else:
        logging.warning("No API key found for persistent cache. Some features may not work properly.")

# Check if API key is available
HAS_API_KEY = bool(PERSISTENT_CACHE_API_KEY)

# Get rate limit for persistent cache API
try:
    PERSISTENT_CACHE_RATE_LIMIT = int(os.getenv("BRAVE_PERSISTENT_CACHE_RATE_LIMIT", "1"))
    if PERSISTENT_CACHE_RATE_LIMIT <= 0:
        PERSISTENT_CACHE_RATE_LIMIT = 1
        logging.warning("Invalid BRAVE_PERSISTENT_CACHE_RATE_LIMIT value. Using default: 1")
    logging.info(f"Persistent cache rate limit set to: {PERSISTENT_CACHE_RATE_LIMIT} requests per second")
except (ValueError, TypeError):
    PERSISTENT_CACHE_RATE_LIMIT = 1
    logging.warning("Invalid BRAVE_PERSISTENT_CACHE_RATE_LIMIT value. Using default: 1")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brave_persistent_cache")

# Default configuration
DEFAULT_CONFIG = {
    "enable_cache": True,
    "db_path": None,  # If None, uses default location
    "default_ttl": 604800,  # 1 week in seconds
    "max_storage_size": 1024 * 1024 * 1024,  # 1 GB
    "quality_threshold": 0.7,  # Minimum quality score to store in persistent cache
    "enrichment_enabled": True,
    "versioning_enabled": True,
    "compression_enabled": True,
    "auto_cleanup_interval": 86400,  # 24 hours
    "backup_interval": 604800,  # 1 week
    "max_versions_per_entry": 3,
    "storage_backend": "sqlite",  # 'sqlite', 'json', or 'hybrid'
}

class DataQualityProcessor:
    """Processes data to enhance quality before storage."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data quality processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.quality_threshold = config.get("quality_threshold", 0.7)
        self.enrichment_enabled = config.get("enrichment_enabled", True)
        logger.info(f"Initialized DataQualityProcessor with quality threshold: {self.quality_threshold}")
    
    def assess_quality(self, data: Dict[str, Any], search_type: str) -> float:
        """Assess the quality of search results.
        
        Args:
            data: Search result data
            search_type: Type of search ('web' or 'ai')
            
        Returns:
            Quality score between 0 and 1
        """
        if not data:
            return 0.0
        
        score = 0.0
        
        if search_type == "web":
            # For web search, assess based on number of results, descriptions, etc.
            if "web" in data and "results" in data["web"]:
                results = data["web"]["results"]
                num_results = len(results)
                
                # Base score on number of results (up to 10)
                score += min(num_results / 10, 1.0) * 0.3
                
                # Check for complete descriptions
                complete_descriptions = sum(1 for r in results if r.get("description", "").strip())
                if num_results > 0:
                    score += (complete_descriptions / num_results) * 0.2
                
                # Check for diverse domains
                domains = set(r.get("domain", "") for r in results)
                if num_results > 0:
                    score += min(len(domains) / num_results, 1.0) * 0.2
                
                # Check for news results
                if "news" in data and "results" in data["news"] and data["news"]["results"]:
                    score += 0.15
                
                # Check for featured snippets
                if "featured_snippet" in data["web"] and data["web"]["featured_snippet"]:
                    score += 0.15
            
        elif search_type == "ai":
            # For AI search, assess based on generated answer, sources, etc.
            if "generated_answer" in data:
                gen_answer = data["generated_answer"]
                
                # Check for answer
                if "answer" in gen_answer and gen_answer["answer"].strip():
                    answer_length = len(gen_answer["answer"])
                    # Score based on answer length (up to 1000 chars)
                    score += min(answer_length / 1000, 1.0) * 0.4
                
                # Check for supporting points
                if "points" in gen_answer and gen_answer["points"]:
                    points_count = len(gen_answer["points"])
                    score += min(points_count / 5, 1.0) * 0.3
                
                # Check for sources
                if "sources" in gen_answer and gen_answer["sources"]:
                    sources_count = len(gen_answer["sources"])
                    score += min(sources_count / 3, 1.0) * 0.3
        
        logger.debug(f"Quality assessment for {search_type} search: {score:.2f}")
        return score
    
    async def enrich_data(self, data: Dict[str, Any], search_type: str) -> Dict[str, Any]:
        """Enrich data to improve quality.
        
        Args:
            data: Search result data
            search_type: Type of search ('web' or 'ai')
            
        Returns:
            Enriched data
        """
        if not self.enrichment_enabled or not data:
            return data
        
        # Create a copy to avoid modifying the original
        enriched_data = json.loads(json.dumps(data))
        
        # Add metadata
        enriched_data["_metadata"] = {
            "enriched_at": datetime.now().isoformat(),
            "search_type": search_type,
            "quality_score": self.assess_quality(data, search_type),
            "version": 1
        }
        
        if search_type == "web":
            # For web search, enhance result descriptions, extract key information, etc.
            if "web" in enriched_data and "results" in enriched_data["web"]:
                for result in enriched_data["web"]["results"]:
                    # Ensure description is not empty
                    if not result.get("description"):
                        result["description"] = result.get("title", "No description available")
                    
                    # Add timestamp for when this result was enriched
                    result["_enriched_at"] = datetime.now().isoformat()
        
        elif search_type == "ai":
            # For AI search, enhance the generated answer, add metadata, etc.
            if "generated_answer" in enriched_data:
                gen_answer = enriched_data["generated_answer"]
                
                # Add timestamp for when this answer was enriched
                gen_answer["_enriched_at"] = datetime.now().isoformat()
                
                # Add confidence score based on sources and points
                sources_count = len(gen_answer.get("sources", []))
                points_count = len(gen_answer.get("points", []))
                
                confidence_score = min((sources_count * 0.2) + (points_count * 0.1), 1.0)
                gen_answer["_confidence_score"] = confidence_score
        
        logger.debug(f"Enriched {search_type} search data")
        return enriched_data
    
    async def process_data(self, data: Dict[str, Any], search_type: str) -> Tuple[Dict[str, Any], float]:
        """Process data to assess quality and enrich if needed.
        
        Args:
            data: Search result data
            search_type: Type of search ('web' or 'ai')
            
        Returns:
            Tuple of (processed_data, quality_score)
        """
        # Assess quality
        quality_score = self.assess_quality(data, search_type)
        
        # Enrich data if quality is above threshold
        if quality_score >= self.quality_threshold:
            processed_data = await self.enrich_data(data, search_type)
        else:
            # Just add basic metadata without full enrichment
            processed_data = data.copy()
            processed_data["_metadata"] = {
                "processed_at": datetime.now().isoformat(),
                "search_type": search_type,
                "quality_score": quality_score,
                "version": 1
            }
        
        return processed_data, quality_score

class SQLiteStorage:
    """SQLite storage backend for persistent cache."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the SQLite storage.
        
        Args:
            db_path: Path to the SQLite database file
        """
        if db_path is None:
            home_dir = os.path.expanduser("~")
            db_dir = os.path.join(home_dir, ".cache", "livekit-amanda")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "brave_persistent_cache.db")
        
        self.db_path = db_path
        self._initialize_db()
        logger.info(f"Initialized SQLite storage at {db_path}")
    
    def _initialize_db(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cache table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            data BLOB,
            search_type TEXT,
            quality_score REAL,
            created_at TIMESTAMP,
            expires_at TIMESTAMP,
            version INTEGER,
            is_compressed INTEGER
        )
        ''')
        
        # Create versions table for versioning
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            data BLOB,
            version INTEGER,
            created_at TIMESTAMP,
            FOREIGN KEY (key) REFERENCES cache(key) ON DELETE CASCADE
        )
        ''')
        
        # Create index on key and search_type
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_key ON cache(key)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_search_type ON cache(search_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache(expires_at)')
        
        conn.commit()
        conn.close()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from the storage.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT data, is_compressed FROM cache WHERE key = ? AND expires_at > ?',
            (key, datetime.now().timestamp())
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data_blob, is_compressed = result
            
            # Decompress if needed
            if is_compressed:
                import zlib
                data_blob = zlib.decompress(data_blob)
            
            # Deserialize JSON
            return json.loads(data_blob)
        
        return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int, 
                 search_type: str, quality_score: float, 
                 version: int = 1, compress: bool = False) -> None:
        """Set a value in the storage.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds
            search_type: Type of search ('web' or 'ai')
            quality_score: Quality score of the data
            version: Version number
            compress: Whether to compress the data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize to JSON
        data_blob = json.dumps(value)
        
        # Compress if enabled
        is_compressed = 0
        if compress:
            import zlib
            data_blob = zlib.compress(data_blob.encode())
            is_compressed = 1
        else:
            data_blob = data_blob.encode()
        
        now = datetime.now().timestamp()
        expires_at = now + ttl
        
        # Check if key exists
        cursor.execute('SELECT version FROM cache WHERE key = ?', (key,))
        existing = cursor.fetchone()
        
        if existing:
            # Store the current version in versions table
            cursor.execute(
                'INSERT INTO versions (key, data, version, created_at) VALUES (?, ?, ?, ?)',
                (key, data_blob, existing[0], now)
            )
            
            # Update the cache entry
            cursor.execute(
                '''
                UPDATE cache 
                SET data = ?, search_type = ?, quality_score = ?, 
                    created_at = ?, expires_at = ?, version = ?, is_compressed = ?
                WHERE key = ?
                ''',
                (data_blob, search_type, quality_score, now, expires_at, version, is_compressed, key)
            )
        else:
            # Insert new cache entry
            cursor.execute(
                '''
                INSERT INTO cache 
                (key, data, search_type, quality_score, created_at, expires_at, version, is_compressed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (key, data_blob, search_type, quality_score, now, expires_at, version, is_compressed)
            )
        
        # Cleanup old versions
        cursor.execute(
            '''
            DELETE FROM versions 
            WHERE key = ? AND id NOT IN (
                SELECT id FROM versions 
                WHERE key = ? 
                ORDER BY created_at DESC 
                LIMIT 2
            )
            ''',
            (key, key)
        )
        
        conn.commit()
        conn.close()
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the storage.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM versions WHERE key = ?', (key,))
        cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return deleted
    
    async def delete_by_pattern(self, pattern: str) -> int:
        """Delete values matching a pattern from the storage.
        
        Args:
            pattern: Pattern to match against keys
            
        Returns:
            Number of deleted entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT key FROM cache WHERE key LIKE ?', (f'%{pattern}%',))
        keys = [row[0] for row in cursor.fetchall()]
        
        count = 0
        for key in keys:
            cursor.execute('DELETE FROM versions WHERE key = ?', (key,))
            cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
            count += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return count
    
    async def delete_expired(self) -> int:
        """Delete expired entries from the storage.
        
        Returns:
            Number of deleted entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().timestamp()
        cursor.execute('SELECT key FROM cache WHERE expires_at <= ?', (now,))
        keys = [row[0] for row in cursor.fetchall()]
        
        count = 0
        for key in keys:
            cursor.execute('DELETE FROM versions WHERE key = ?', (key,))
            cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
            count += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return count
    
    async def get_all_keys(self) -> List[str]:
        """Get all keys from the storage.
        
        Returns:
            List of keys
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT key FROM cache')
        keys = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return keys
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total entries
        cursor.execute('SELECT COUNT(*) FROM cache')
        total_entries = cursor.fetchone()[0]
        
        # Get entries by search type
        cursor.execute('SELECT search_type, COUNT(*) FROM cache GROUP BY search_type')
        entries_by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get average quality score
        cursor.execute('SELECT AVG(quality_score) FROM cache')
        avg_quality = cursor.fetchone()[0] or 0
        
        # Get storage size
        cursor.execute('SELECT SUM(LENGTH(data)) FROM cache')
        cache_size = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT SUM(LENGTH(data)) FROM versions')
        versions_size = cursor.fetchone()[0] or 0
        
        # Get expired entries
        now = datetime.now().timestamp()
        cursor.execute('SELECT COUNT(*) FROM cache WHERE expires_at <= ?', (now,))
        expired_entries = cursor.fetchone()[0]
        
        # Get version statistics
        cursor.execute('SELECT AVG(version) FROM cache')
        avg_version = cursor.fetchone()[0] or 1
        
        cursor.execute('SELECT COUNT(*) FROM versions')
        total_versions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_entries": total_entries,
            "entries_by_type": entries_by_type,
            "avg_quality_score": avg_quality,
            "cache_size_bytes": cache_size,
            "versions_size_bytes": versions_size,
            "total_size_bytes": cache_size + versions_size,
            "expired_entries": expired_entries,
            "avg_version": avg_version,
            "total_versions": total_versions
        }
    
    async def backup(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the storage.
        
        Args:
            backup_path: Path to store the backup. If None, uses default location.
            
        Returns:
            Path to the backup file
        """
        if backup_path is None:
            backup_dir = os.path.dirname(self.db_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"brave_persistent_cache_backup_{timestamp}.db")
        
        # Create a copy of the database
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        logger.info(f"Created backup at {backup_path}")
        return backup_path
    
    async def optimize(self) -> None:
        """Optimize the storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Run VACUUM to reclaim space
        cursor.execute('VACUUM')
        
        # Run ANALYZE to update statistics
        cursor.execute('ANALYZE')
        
        conn.commit()
        conn.close()
        
        logger.info("Optimized SQLite storage")

class BraveSearchPersistentCache:
    """Persistent cache for Brave Search API with enhanced data quality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the persistent cache.
        
        Args:
            config: Configuration dictionary. If None, uses default configuration.
        """
        # Merge provided config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Initialize components
        self.quality_processor = DataQualityProcessor(self.config)
        
        # Initialize storage backend
        storage_backend = self.config.get("storage_backend", "sqlite")
        if storage_backend == "sqlite":
            self.storage = SQLiteStorage(self.config.get("db_path"))
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
        
        # Set up auto-cleanup if enabled
        auto_cleanup_interval = self.config.get("auto_cleanup_interval")
        if auto_cleanup_interval:
            self._setup_auto_cleanup(auto_cleanup_interval)
        
        # Set up auto-backup if enabled
        backup_interval = self.config.get("backup_interval")
        if backup_interval:
            self._setup_auto_backup(backup_interval)
            
        # Initialize API client session
        self.session = None
        self.api_key = PERSISTENT_CACHE_API_KEY
        self.has_api_key = HAS_API_KEY
        
        # Set up rate limiting
        self.rate_limit = PERSISTENT_CACHE_RATE_LIMIT
        self.rate_limit_semaphore = Semaphore(1)  # Allow only one request at a time
        self.last_request_time = 0
        
        logger.info(f"Initialized BraveSearchPersistentCache with config: {self.config}")
    
    def _setup_auto_cleanup(self, interval: int):
        """Set up automatic cleanup.
        
        Args:
            interval: Cleanup interval in seconds
        """
        async def _auto_cleanup():
            while True:
                try:
                    # Sleep first
                    await asyncio.sleep(interval)
                    
                    # Perform cleanup
                    count = await self.cleanup()
                    
                    # Optimize storage
                    await self.storage.optimize()
                    
                    logger.info(f"Auto-cleanup removed {count} expired entries")
                except Exception as e:
                    logger.error(f"Error during auto-cleanup: {e}")
        
        # Start the auto-cleanup task
        asyncio.create_task(_auto_cleanup())
        logger.info(f"Set up auto-cleanup with interval: {interval} seconds")
    
    def _setup_auto_backup(self, interval: int):
        """Set up automatic backup.
        
        Args:
            interval: Backup interval in seconds
        """
        async def _auto_backup():
            while True:
                try:
                    # Sleep first
                    await asyncio.sleep(interval)
                    
                    # Perform backup
                    backup_path = await self.storage.backup()
                    
                    logger.info(f"Auto-backup created at {backup_path}")
                except Exception as e:
                    logger.error(f"Error during auto-backup: {e}")
        
        # Start the auto-backup task
        asyncio.create_task(_auto_backup())
        logger.info(f"Set up auto-backup with interval: {interval} seconds")
    
    def get_cache_key(self, query: str, search_type: str, **kwargs) -> str:
        """Generate a cache key.
        
        Args:
            query: Search query
            search_type: Type of search ('web' or 'ai')
            **kwargs: Additional parameters
            
        Returns:
            Cache key
        """
        # Create a dictionary with all parameters
        key_dict = {
            "query": query.lower().strip(),
            "search_type": search_type,
            **kwargs
        }
        
        # Convert to a sorted string representation for consistent hashing
        key_str = json.dumps(key_dict, sort_keys=True)
        
        # Generate a hash
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"brave_{search_type}_{key_hash}"
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        if not self.config.get("enable_cache", True):
            return None
        
        return await self.storage.get(key)
    
    async def set(self, key: str, value: Dict[str, Any], search_type: str, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache after quality processing.
        
        Args:
            key: Cache key
            value: Value to cache
            search_type: Type of search ('web' or 'ai')
            ttl: Time-to-live in seconds. If None, uses default_ttl.
            
        Returns:
            True if stored, False if rejected due to quality
        """
        if not self.config.get("enable_cache", True):
            return False
        
        # Process data for quality and enrichment
        processed_data, quality_score = await self.quality_processor.process_data(value, search_type)
        
        # Only store if quality is above threshold
        if quality_score >= self.config.get("quality_threshold", 0.7):
            # Use provided TTL or default
            ttl = ttl if ttl is not None else self.config.get("default_ttl", 604800)
            
            # Get version from metadata or default to 1
            version = processed_data.get("_metadata", {}).get("version", 1)
            
            # Store with compression if enabled
            compression_enabled = self.config.get("compression_enabled", True)
            
            await self.storage.set(
                key=key,
                value=processed_data,
                ttl=ttl,
                search_type=search_type,
                quality_score=quality_score,
                version=version,
                compress=compression_enabled
            )
            
            logger.info(f"Stored high-quality {search_type} search result with score {quality_score:.2f}")
            return True
        else:
            logger.info(f"Rejected low-quality {search_type} search result with score {quality_score:.2f}")
            return False
    
    async def get_high_quality_result(self, query: str, search_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get a high-quality result from the persistent cache.
        
        Args:
            query: Search query
            search_type: Type of search ('web' or 'ai')
            **kwargs: Additional parameters
            
        Returns:
            High-quality cached result if found, None otherwise
        """
        key = self.get_cache_key(query, search_type, **kwargs)
        result = await self.get(key)
        
        if result:
            # Check if it has quality metadata
            quality_score = result.get("_metadata", {}).get("quality_score", 0)
            
            # Only return if it meets the current quality threshold
            if quality_score >= self.config.get("quality_threshold", 0.7):
                logger.info(f"Found high-quality {search_type} search result with score {quality_score:.2f}")
                return result
            else:
                logger.info(f"Found cached result but quality {quality_score:.2f} below threshold")
                return None
        
        return None
    
    async def store_high_quality_result(self, query: str, data: Dict[str, Any], search_type: str, 
                                      ttl: Optional[int] = None, **kwargs) -> bool:
        """Store a high-quality result in the persistent cache.
        
        Args:
            query: Search query
            data: Search result data
            search_type: Type of search ('web' or 'ai')
            ttl: Time-to-live in seconds. If None, uses default_ttl.
            **kwargs: Additional parameters
            
        Returns:
            True if stored, False if rejected due to quality
        """
        key = self.get_cache_key(query, search_type, **kwargs)
        return await self.set(key, data, search_type, ttl)
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if invalidated, False otherwise
        """
        return await self.storage.delete(key)
    
    async def invalidate_by_query(self, query: str, search_type: str, **kwargs) -> bool:
        """Invalidate a specific query result.
        
        Args:
            query: Search query
            search_type: Type of search ('web' or 'ai')
            **kwargs: Additional parameters
            
        Returns:
            True if invalidated, False otherwise
        """
        key = self.get_cache_key(query, search_type, **kwargs)
        return await self.invalidate(key)
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            Number of invalidated entries
        """
        return await self.storage.delete_by_pattern(pattern)
    
    async def invalidate_by_search_type(self, search_type: str) -> int:
        """Invalidate all cache entries for a specific search type.
        
        Args:
            search_type: Type of search ('web' or 'ai')
            
        Returns:
            Number of invalidated entries
        """
        return await self.storage.delete_by_pattern(f"brave_{search_type}_")
    
    async def cleanup(self) -> int:
        """Clean up expired entries.
        
        Returns:
            Number of cleaned up entries
        """
        return await self.storage.delete_expired()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        storage_stats = await self.storage.get_stats()
        
        stats = {
            "config": self.config,
            "storage": storage_stats,
            "timestamp": datetime.now().timestamp()
        }
        
        return stats
    
    async def optimize(self) -> None:
        """Optimize the cache storage."""
        await self.storage.optimize()
    
    async def backup(self) -> str:
        """Create a backup of the cache.
        
        Returns:
            Path to the backup file
        """
        return await self.storage.backup()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp client session.
        
        Returns:
            aiohttp.ClientSession: The client session
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API request to the Brave Search API using the persistent cache API key.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response as a dictionary
        """
        if not self.has_api_key:
            raise ValueError("No API key available for persistent cache")
        
        # Base URL for Brave Search API
        base_url = "https://api.search.brave.com/res/v1"
        url = f"{base_url}/{endpoint}"
        
        # Add API key to headers
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        # Get or create session
        session = await self._get_session()
        
        # Apply rate limiting
        async with self.rate_limit_semaphore:
            # Calculate time to wait based on rate limit
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            wait_time = max(0, (1.0 / self.rate_limit) - time_since_last_request)
            
            if wait_time > 0:
                logger.info(f"Persistent cache rate limiting: waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            
            # Update last request time
            self.last_request_time = time.time()
            
            try:
                # Make the request
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        try:
                            # Try to parse the response as JSON
                            return await response.json()
                        except aiohttp.ContentTypeError:
                            # If response is not JSON, return it as text in an error dict
                            error_text = await response.text()
                            logger.error(f"API returned non-JSON response: {error_text[:100]}...")
                            return {"error": "Invalid JSON response", "details": error_text[:500]}
                    else:
                        try:
                            # Try to get error text, but handle if it's not text
                            error_text = await response.text()
                        except Exception:
                            error_text = "<Unable to read error text>"
                            
                        logger.error(f"API request failed: {response.status} - {error_text[:100]}")
                        return {"error": f"API error: {response.status}", "details": error_text[:500]}
            except Exception as e:
                logger.error(f"Error making API request: {e}")
                return {"error": f"Request error: {str(e)}"}
    
    async def fetch_fresh_web_results(self, query: str, **kwargs) -> Dict[str, Any]:
        """Fetch fresh web search results using the persistent cache API key.
        
        Args:
            query: Search query
            **kwargs: Additional parameters for the search
            
        Returns:
            Search results
        """
        # Prepare parameters
        params = {
            "q": query,
            **kwargs
        }
        
        # Make the request
        return await self._make_api_request("web/search", params)
    
    async def fetch_fresh_ai_results(self, query: str, **kwargs) -> Dict[str, Any]:
        """Fetch fresh AI search results using the persistent cache API key.
        
        Args:
            query: Search query
            **kwargs: Additional parameters for the search
            
        Returns:
            AI search results
        """
        # Prepare parameters
        params = {
            "q": query,
            **kwargs
        }
        
        # Make the API request
        # The correct endpoint for AI search is 'ai/search'
        return await self._make_api_request("ai/search", params)
    
    async def refresh_high_quality_result(self, query: str, search_type: str, **kwargs) -> Dict[str, Any]:
        """Refresh a high-quality search result using the persistent cache API key.
        
        This method fetches fresh results using the dedicated API key, processes them,
        and stores them in the cache, then returns the processed result.
        
        Args:
            query: Search query
            search_type: Type of search ("web" or "ai")
            **kwargs: Additional parameters for the search
            
        Returns:
            Refreshed high-quality search result
        """
        logger.info(f"Refreshing high-quality {search_type} search result for query: {query}")
        
        if not self.has_api_key:
            logger.warning("No dedicated API key available for refresh. Using fallback method.")
            # Use the regular store_high_quality_result method as fallback
            # This will use the regular API keys from the main API client
            return await self.get_high_quality_result(query, search_type, force_refresh=True, **kwargs)
        
        # Fetch fresh results using the dedicated API key
        if search_type == "web":
            # For web search, get count parameter from kwargs or use default
            count = kwargs.get("count", 10)
            # If num_results is specified, use it to determine count
            if "num_results" in kwargs:
                count = min(kwargs["num_results"] + 5, 20)  # Request extra results for better quality
                
            raw_results = await self.fetch_fresh_web_results(query, count=count, **kwargs)
        elif search_type == "ai":
            raw_results = await self.fetch_fresh_ai_results(query, **kwargs)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
        
        if "error" in raw_results:
            logger.error(f"Error refreshing {search_type} search: {raw_results['error']}")
            return raw_results
        
        # Process the results
        processed_data, quality_score = await self.quality_processor.process_data(raw_results, search_type)
        
        # Add metadata
        processed_data["_metadata"] = {
            "query": query,
            "search_type": search_type,
            "quality_score": quality_score,
            "enriched_at": datetime.now().isoformat()
        }
        
        # Store in cache
        key = self.get_cache_key(query, search_type, **kwargs)
        await self.set(key, processed_data, search_type, self.config["cache_ttl"])
        
        logger.info(f"Successfully refreshed {search_type} search result with quality score: {quality_score:.2f}")
        return processed_data
    
    async def close(self) -> None:
        """Close the cache and release resources."""
        # Close the API client session if it exists
        if self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None
        
        logger.info("Closed BraveSearchPersistentCache")

# Singleton instance
_persistent_cache = None

def get_persistent_cache(config: Optional[Dict[str, Any]] = None) -> BraveSearchPersistentCache:
    """Get or create the persistent cache.
    
    Args:
        config: Configuration dictionary. If None, uses default configuration.
        
    Returns:
        BraveSearchPersistentCache instance
    """
    global _persistent_cache
    
    if _persistent_cache is None:
        _persistent_cache = BraveSearchPersistentCache(config=config)
    
    return _persistent_cache

async def close_persistent_cache() -> None:
    """Close the persistent cache."""
    global _persistent_cache
    
    if _persistent_cache is not None:
        await _persistent_cache.close()
        _persistent_cache = None
