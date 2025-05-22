"""
Brave Search API Statistics Tracking Module

This module tracks and persists statistics about Brave Search API usage, including:
- Request counts
- Cache hit/miss rates
- Performance metrics
- Rate limiting data
- Error rates
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import threading
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BraveSearchStats:
    """Statistics tracker for Brave Search API usage."""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'BraveSearchStats':
        """Get the singleton instance of BraveSearchStats."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = BraveSearchStats()
        return cls._instance
    
    def __init__(self):
        """Initialize the statistics tracker."""
        self.stats_dir = Path(os.path.expanduser("~")) / ".brave_search_stats"
        self.db_path = self.stats_dir / "brave_search_stats.db"
        
        # Create stats directory if it doesn't exist
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Initialize the database
        self._init_db()
        
        # In-memory counters for the current session
        self.session_stats = {
            "start_time": time.time(),
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_response_time": 0,
            "slowest_request": 0,
            "fastest_request": float('inf'),
            "rate_limit_delays": 0,
            "total_delay_time": 0,
        }
        
        logging.info(f"Brave Search statistics tracking initialized at {self.stats_dir}")
    
    def _init_db(self):
        """Initialize the SQLite database for storing statistics."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create requests table with search_type column
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                search_type TEXT,  -- 'web' or 'ai'
                response_time REAL,
                cache_hit INTEGER,
                error INTEGER,
                rate_limited INTEGER,
                delay_time REAL,
                status_code INTEGER,
                result_count INTEGER
            )
            ''')
            
            # Create daily stats table with separate columns for web and AI search
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                web_requests INTEGER,
                web_cache_hits INTEGER,
                web_cache_misses INTEGER,
                web_errors INTEGER,
                web_avg_response_time REAL,
                web_total_delay_time REAL,
                web_rate_limit_delays INTEGER,
                ai_requests INTEGER,
                ai_cache_hits INTEGER,
                ai_cache_misses INTEGER,
                ai_errors INTEGER,
                ai_avg_response_time REAL,
                ai_total_delay_time REAL,
                ai_rate_limit_delays INTEGER,
                total_requests INTEGER
            )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Brave Search statistics database initialized")
        except Exception as e:
            logging.error(f"Error initializing statistics database: {e}")
    
    def record_request(self, 
                      query: str, 
                      response_time: float, 
                      search_type: str = "web",  # 'web' or 'ai'
                      cache_hit: bool = False, 
                      error: bool = False,
                      rate_limited: bool = False,
                      delay_time: float = 0.0,
                      status_code: int = 200,
                      result_count: int = 0):
        """Record statistics for a single API request.
        
        Args:
            query: The search query
            response_time: Time taken to get the response in seconds
            cache_hit: Whether the result was served from cache
            error: Whether an error occurred
            rate_limited: Whether rate limiting was applied
            delay_time: Time spent waiting due to rate limiting
            status_code: HTTP status code of the response
            result_count: Number of results returned
        """
        # Validate search_type
        if search_type not in ["web", "ai"]:
            search_type = "web"  # Default to web if invalid
            
        # Update session stats - both total and per search type
        self.session_stats["requests"] += 1
        self.session_stats[f"{search_type}_requests"] = self.session_stats.get(f"{search_type}_requests", 0) + 1
        self.session_stats["total_response_time"] += response_time
        self.session_stats[f"{search_type}_total_response_time"] = self.session_stats.get(f"{search_type}_total_response_time", 0) + response_time
        
        if cache_hit:
            self.session_stats["cache_hits"] += 1
            self.session_stats[f"{search_type}_cache_hits"] = self.session_stats.get(f"{search_type}_cache_hits", 0) + 1
        else:
            self.session_stats["cache_misses"] += 1
            self.session_stats[f"{search_type}_cache_misses"] = self.session_stats.get(f"{search_type}_cache_misses", 0) + 1
        
        if error:
            self.session_stats["errors"] += 1
            self.session_stats[f"{search_type}_errors"] = self.session_stats.get(f"{search_type}_errors", 0) + 1
        
        if rate_limited:
            self.session_stats["rate_limit_delays"] += 1
            self.session_stats[f"{search_type}_rate_limit_delays"] = self.session_stats.get(f"{search_type}_rate_limit_delays", 0) + 1
            self.session_stats["total_delay_time"] += delay_time
            self.session_stats[f"{search_type}_total_delay_time"] = self.session_stats.get(f"{search_type}_total_delay_time", 0) + delay_time
        
        # Update fastest/slowest request times
        if response_time > self.session_stats["slowest_request"]:
            self.session_stats["slowest_request"] = response_time
        
        if response_time < self.session_stats["fastest_request"]:
            self.session_stats["fastest_request"] = response_time
        
        # Record in database
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Add request record with search_type
            cursor.execute('''
            INSERT INTO requests (
                timestamp, query, search_type, response_time, cache_hit, 
                error, rate_limited, delay_time, status_code, result_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                query,
                search_type,
                response_time,
                1 if cache_hit else 0,
                1 if error else 0,
                1 if rate_limited else 0,
                delay_time,
                status_code,
                result_count
            ))
            
            # Update daily stats
            today = datetime.now().date().isoformat()
            
            # Check if we have an entry for today
            cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (today,))
            daily_record = cursor.fetchone()
            
            if daily_record:
                # Update existing record based on search_type
                if search_type == "web":
                    cursor.execute('''
                    UPDATE daily_stats SET
                        web_requests = web_requests + 1,
                        web_cache_hits = web_cache_hits + ?,
                        web_cache_misses = web_cache_misses + ?,
                        web_errors = web_errors + ?,
                        web_avg_response_time = (web_avg_response_time * web_requests + ?) / (web_requests + 1),
                        web_total_delay_time = web_total_delay_time + ?,
                        web_rate_limit_delays = web_rate_limit_delays + ?,
                        total_requests = total_requests + 1
                    WHERE date = ?
                    ''', (
                        1 if cache_hit else 0,
                        0 if cache_hit else 1,
                        1 if error else 0,
                        response_time,
                        delay_time,
                        1 if rate_limited else 0,
                        today
                    ))
                else:  # ai search
                    cursor.execute('''
                    UPDATE daily_stats SET
                        ai_requests = ai_requests + 1,
                        ai_cache_hits = ai_cache_hits + ?,
                        ai_cache_misses = ai_cache_misses + ?,
                        ai_errors = ai_errors + ?,
                        ai_avg_response_time = (ai_avg_response_time * ai_requests + ?) / (ai_requests + 1),
                        ai_total_delay_time = ai_total_delay_time + ?,
                        ai_rate_limit_delays = ai_rate_limit_delays + ?,
                        total_requests = total_requests + 1
                    WHERE date = ?
                    ''', (
                        1 if cache_hit else 0,
                        0 if cache_hit else 1,
                        1 if error else 0,
                        response_time,
                        delay_time,
                        1 if rate_limited else 0,
                        today
                    ))
            else:
                # Create new record for today with zeros for all counters
                if search_type == "web":
                    cursor.execute('''
                    INSERT INTO daily_stats (
                        date, web_requests, web_cache_hits, web_cache_misses, 
                        web_errors, web_avg_response_time, web_total_delay_time, web_rate_limit_delays,
                        ai_requests, ai_cache_hits, ai_cache_misses, 
                        ai_errors, ai_avg_response_time, ai_total_delay_time, ai_rate_limit_delays,
                        total_requests
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        today,
                        1,  # web_requests
                        1 if cache_hit else 0,  # web_cache_hits
                        0 if cache_hit else 1,  # web_cache_misses
                        1 if error else 0,  # web_errors
                        response_time,  # web_avg_response_time
                        delay_time,  # web_total_delay_time
                        1 if rate_limited else 0,  # web_rate_limit_delays
                        0,  # ai_requests
                        0,  # ai_cache_hits
                        0,  # ai_cache_misses
                        0,  # ai_errors
                        0,  # ai_avg_response_time
                        0,  # ai_total_delay_time
                        0,  # ai_rate_limit_delays
                        1   # total_requests
                    ))
                else:  # ai search
                    cursor.execute('''
                    INSERT INTO daily_stats (
                        date, web_requests, web_cache_hits, web_cache_misses, 
                        web_errors, web_avg_response_time, web_total_delay_time, web_rate_limit_delays,
                        ai_requests, ai_cache_hits, ai_cache_misses, 
                        ai_errors, ai_avg_response_time, ai_total_delay_time, ai_rate_limit_delays,
                        total_requests
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        today,
                        0,  # web_requests
                        0,  # web_cache_hits
                        0,  # web_cache_misses
                        0,  # web_errors
                        0,  # web_avg_response_time
                        0,  # web_total_delay_time
                        0,  # web_rate_limit_delays
                        1,  # ai_requests
                        1 if cache_hit else 0,  # ai_cache_hits
                        0 if cache_hit else 1,  # ai_cache_misses
                        1 if error else 0,  # ai_errors
                        response_time,  # ai_avg_response_time
                        delay_time,  # ai_total_delay_time
                        1 if rate_limited else 0,  # ai_rate_limit_delays
                        1   # total_requests
                    ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error recording request statistics: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session.
        
        Returns:
            Dict containing session statistics
        """
        stats = self.session_stats.copy()
        
        # Calculate derived metrics
        duration = time.time() - stats["start_time"]
        stats["duration"] = duration
        
        if stats["requests"] > 0:
            stats["avg_response_time"] = stats["total_response_time"] / stats["requests"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["requests"] * 100
            stats["error_rate"] = stats["errors"] / stats["requests"] * 100
        else:
            stats["avg_response_time"] = 0
            stats["cache_hit_rate"] = 0
            stats["error_rate"] = 0
        
        if stats["fastest_request"] == float('inf'):
            stats["fastest_request"] = 0
        
        # Calculate requests per minute
        if duration > 0:
            stats["requests_per_minute"] = stats["requests"] / (duration / 60)
        else:
            stats["requests_per_minute"] = 0
        
        return stats
    
    def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily statistics for the specified number of days.
        
        Args:
            days: Number of days to retrieve statistics for
            
        Returns:
            List of daily statistics
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get the last 'days' days of stats
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days-1)
            
            cursor.execute('''
            SELECT * FROM daily_stats 
            WHERE date >= ? AND date <= ?
            ORDER BY date DESC
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dicts
            result = []
            for row in rows:
                result.append({
                    "date": row[0],
                    "requests": row[1],
                    "cache_hits": row[2],
                    "cache_misses": row[3],
                    "errors": row[4],
                    "avg_response_time": row[5],
                    "total_delay_time": row[6],
                    "rate_limit_delays": row[7],
                    "cache_hit_rate": (row[2] / row[1] * 100) if row[1] > 0 else 0,
                    "error_rate": (row[4] / row[1] * 100) if row[1] > 0 else 0
                })
            
            return result
        except Exception as e:
            logging.error(f"Error retrieving daily statistics: {e}")
            return []
    
    def get_popular_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most popular search queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of (query, count) tuples
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT query, COUNT(*) as count
            FROM requests
            GROUP BY query
            ORDER BY count DESC
            LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [(row[0], row[1]) for row in rows]
        except Exception as e:
            logging.error(f"Error retrieving popular queries: {e}")
            return []
    
    def get_performance_stats(self, search_type: str = None) -> Dict[str, Any]:
        """Get performance statistics for API requests.
        
        Args:
            search_type: Optional filter for 'web' or 'ai' search types. If None, returns overall stats.
            
        Returns:
            Dict containing performance statistics
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Build the query based on search_type filter
            where_clause = ""
            if search_type in ["web", "ai"]:
                where_clause = f"WHERE search_type = '{search_type}'"
            
            # Get performance stats
            cursor.execute(f'''
            SELECT 
                COUNT(*) as total_requests,
                AVG(response_time) as avg_response_time,
                MIN(response_time) as min_response_time,
                MAX(response_time) as max_response_time,
                SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as errors,
                SUM(CASE WHEN rate_limited = 1 THEN 1 ELSE 0 END) as rate_limited_requests,
                AVG(CASE WHEN rate_limited = 1 THEN delay_time ELSE 0 END) as avg_delay_time
            FROM requests
            {where_clause}
            ''')
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                total_requests = row[0]
                return {
                    "total_requests": total_requests,
                    "avg_response_time": row[1] if row[1] is not None else 0,
                    "min_response_time": row[2] if row[2] is not None else 0,
                    "max_response_time": row[3] if row[3] is not None else 0,
                    "cache_hits": row[4],
                    "cache_hit_rate": (row[4] / total_requests * 100) if total_requests > 0 else 0,
                    "errors": row[5],
                    "error_rate": (row[5] / total_requests * 100) if total_requests > 0 else 0,
                    "rate_limited_requests": row[6],
                    "rate_limited_rate": (row[6] / total_requests * 100) if total_requests > 0 else 0,
                    "avg_delay_time": row[7] if row[7] is not None else 0
                }
            return {}
        except Exception as e:
            logging.error(f"Error retrieving performance statistics: {e}")
            return {}
    
    def export_stats(self, format: str = "json") -> str:
        """Export statistics to a file.
        
        Args:
            format: Export format (currently only 'json' is supported)
            
        Returns:
            Path to the exported file
        """
        if format.lower() != "json":
            raise ValueError("Only JSON format is currently supported")
        
        try:
            # Gather all stats
            export_data = {
                "session": self.get_session_stats(),
                "daily": self.get_daily_stats(30),  # Last 30 days
                "performance": self.get_performance_stats(),
                "popular_queries": self.get_popular_queries(20),  # Top 20 queries
                "export_time": datetime.now().isoformat()
            }
            
            # Create export filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.stats_dir / f"brave_search_stats_{timestamp}.json"
            
            # Write to file
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)
            
            logging.info(f"Statistics exported to {export_path}")
            return str(export_path)
        except Exception as e:
            logging.error(f"Error exporting statistics: {e}")
            return ""

# Convenience function to get the stats instance
def get_stats() -> BraveSearchStats:
    """Get the singleton instance of BraveSearchStats."""
    return BraveSearchStats.get_instance()

# Function to record a request (for easy integration)
def record_request(query: str, 
                  response_time: float, 
                  search_type: str = "web",  # 'web' or 'ai'
                  cache_hit: bool = False, 
                  error: bool = False,
                  rate_limited: bool = False,
                  delay_time: float = 0.0,
                  status_code: int = 200,
                  result_count: int = 0):
    """Record statistics for a single API request."""
    stats = get_stats()
    stats.record_request(
        query=query,
        response_time=response_time,
        search_type=search_type,
        cache_hit=cache_hit,
        error=error,
        rate_limited=rate_limited,
        delay_time=delay_time,
        status_code=status_code,
        result_count=result_count
    )

# Function to get a formatted stats report
def get_stats_report() -> str:
    """Get a formatted report of Brave Search API statistics."""
    stats = get_stats()
    session_stats = stats.get_session_stats()
    overall_stats = stats.get_performance_stats()
    web_stats = stats.get_performance_stats("web")
    ai_stats = stats.get_performance_stats("ai")
    daily_stats = stats.get_daily_stats(7)  # Last 7 days
    popular_queries = stats.get_popular_queries(5)  # Top 5 queries
    
    report = []
    report.append("=== Brave Search API Statistics Report ===")
    report.append("")
    
    # Session stats
    report.append("--- Current Session ---")
    report.append(f"Duration: {timedelta(seconds=int(session_stats.get('duration', 0)))}")
    report.append(f"Requests: {session_stats.get('requests', 0)}")
    report.append(f"Requests per minute: {session_stats.get('requests_per_minute', 0):.2f}")
    report.append(f"Cache hit rate: {session_stats.get('cache_hit_rate', 0):.2f}%")
    report.append(f"Average response time: {session_stats.get('avg_response_time', 0)*1000:.2f}ms")
    report.append(f"Fastest request: {session_stats.get('fastest_request', 0)*1000:.2f}ms")
    report.append(f"Slowest request: {session_stats.get('slowest_request', 0)*1000:.2f}ms")
    report.append(f"Rate limit delays: {session_stats.get('rate_limit_delays', 0)}")
    report.append(f"Total delay time: {session_stats.get('total_delay_time', 0):.2f}s")
    report.append("")
    
    # Overall performance
    report.append("--- Overall Performance ---")
    report.append(f"Total requests: {overall_stats.get('total_requests', 0)}")
    report.append(f"Cache hit rate: {overall_stats.get('cache_hit_rate', 0):.2f}%")
    report.append(f"Average response time: {overall_stats.get('avg_response_time', 0)*1000:.2f}ms")
    report.append(f"Minimum response time: {overall_stats.get('min_response_time', 0)*1000:.2f}ms")
    report.append(f"Maximum response time: {overall_stats.get('max_response_time', 0)*1000:.2f}ms")
    report.append(f"Error rate: {overall_stats.get('error_rate', 0):.2f}%")
    report.append(f"Rate limited requests: {overall_stats.get('rate_limited_rate', 0):.2f}%")
    report.append("")
    
    # Web Search performance
    report.append("--- Web Search Performance ---")
    report.append(f"Total requests: {web_stats.get('total_requests', 0)}")
    report.append(f"Cache hit rate: {web_stats.get('cache_hit_rate', 0):.2f}%")
    report.append(f"Average response time: {web_stats.get('avg_response_time', 0)*1000:.2f}ms")
    report.append(f"Error rate: {web_stats.get('error_rate', 0):.2f}%")
    report.append(f"Rate limited requests: {web_stats.get('rate_limited_rate', 0):.2f}%")
    report.append("")
    
    # AI Search performance
    report.append("--- AI Search Performance ---")
    report.append(f"Total requests: {ai_stats.get('total_requests', 0)}")
    report.append(f"Cache hit rate: {ai_stats.get('cache_hit_rate', 0):.2f}%")
    report.append(f"Average response time: {ai_stats.get('avg_response_time', 0)*1000:.2f}ms")
    report.append(f"Error rate: {ai_stats.get('error_rate', 0):.2f}%")
    report.append(f"Rate limited requests: {ai_stats.get('rate_limited_rate', 0):.2f}%")
    report.append("")
    
    # Daily stats
    report.append("--- Daily Statistics (Last 7 Days) ---")
    for day in daily_stats:
        report.append(f"{day['date']}: {day['requests']} requests, {day['cache_hit_rate']:.2f}% cache hits, {day['avg_response_time']*1000:.2f}ms avg")
    report.append("")
    
    # Popular queries
    report.append("--- Popular Queries ---")
    for query, count in popular_queries:
        report.append(f"{query}: {count} requests")
    
    return "\n".join(report)

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Brave Search API Statistics")
    parser.add_argument("--report", action="store_true", help="Print statistics report")
    parser.add_argument("--export", action="store_true", help="Export statistics to JSON")
    
    args = parser.parse_args()
    
    if args.report:
        print(get_stats_report())
    
    if args.export:
        export_path = get_stats().export_stats()
        print(f"Statistics exported to {export_path}")
    
    if not args.report and not args.export:
        print(get_stats_report())
