#!/usr/bin/env python3
"""
Brave Search Statistics CLI Tool

This command-line tool allows you to view and manage statistics for the Brave Search API.
It provides insights into API usage, performance, caching efficiency, and more.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from tabulate import tabulate
from typing import Dict, Any, List

# Try to import the statistics modules
try:
    from brave_search_stats import get_stats, get_stats_report
    from brave_search_free_tier import get_cache_stats
    HAS_STATS_MODULES = True
except ImportError:
    HAS_STATS_MODULES = False
    print("Warning: Could not import statistics modules. Make sure they are available.")

def format_size(size_bytes: int) -> str:
    """Format a size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def format_time(seconds: float) -> str:
    """Format a time in seconds to a human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {seconds:.2f}s"

async def show_summary():
    """Show a summary of Brave Search API statistics."""
    if not HAS_STATS_MODULES:
        print("Error: Statistics modules not available.")
        return
    
    try:
        # Get cache statistics
        cache_stats = await get_cache_stats()
        
        # Get API usage statistics
        stats = get_stats()
        if not stats:
            print("Error: Could not get statistics.")
            return
        
        session_stats = stats.get_session_stats()
        performance_stats = stats.get_performance_stats()
        
        # Print summary
        print("\n=== Brave Search API Statistics Summary ===\n")
        
        # Cache statistics
        print("--- Cache Statistics ---")
        if "enabled" in cache_stats:
            print(f"Cache enabled: {cache_stats.get('enabled', False)}")
            print(f"Persistence enabled: {cache_stats.get('persistence', False)}")
            print(f"Rate limit: {cache_stats.get('rate_limit', 1)} requests per second")
        
        if "hit_count" in cache_stats:
            print(f"Cache hits: {cache_stats.get('hit_count', 0)}")
            print(f"Cache misses: {cache_stats.get('miss_count', 0)}")
            print(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.2f}%")
            print(f"Memory cache size: {cache_stats.get('memory_cache_size', 0)} items")
            
            if "memory_usage_mb" in cache_stats:
                print(f"Memory usage: {cache_stats.get('memory_usage_mb', 0):.2f} MB")
            
            if "disk_usage_mb" in cache_stats:
                print(f"Disk usage: {cache_stats.get('disk_usage_mb', 0):.2f} MB")
        
        print()
        
        # Session statistics
        print("--- Current Session ---")
        print(f"Duration: {timedelta(seconds=int(session_stats.get('duration', 0)))}")
        print(f"Requests: {session_stats.get('requests', 0)}")
        print(f"Requests per minute: {session_stats.get('requests_per_minute', 0):.2f}")
        print(f"Cache hit rate: {session_stats.get('cache_hit_rate', 0):.2f}%")
        print(f"Average response time: {format_time(session_stats.get('avg_response_time', 0))}")
        print(f"Fastest request: {format_time(session_stats.get('fastest_request', 0))}")
        print(f"Slowest request: {format_time(session_stats.get('slowest_request', 0))}")
        print(f"Rate limit delays: {session_stats.get('rate_limit_delays', 0)}")
        print(f"Total delay time: {format_time(session_stats.get('total_delay_time', 0))}")
        print()
        
        # Overall performance
        print("--- Overall Performance ---")
        print(f"Total requests: {performance_stats.get('total_requests', 0)}")
        print(f"Cache hit rate: {performance_stats.get('cache_hit_rate', 0):.2f}%")
        print(f"Average response time: {format_time(performance_stats.get('avg_response_time', 0))}")
        print(f"Minimum response time: {format_time(performance_stats.get('min_response_time', 0))}")
        print(f"Maximum response time: {format_time(performance_stats.get('max_response_time', 0))}")
        print(f"Error rate: {performance_stats.get('error_rate', 0):.2f}%")
        print(f"Rate limited requests: {performance_stats.get('rate_limited_rate', 0):.2f}%")
        print()
        
        # Popular queries
        print("--- Popular Queries ---")
        popular_queries = stats.get_popular_queries(5)
        if popular_queries:
            table_data = [(query, count) for query, count in popular_queries]
            print(tabulate(table_data, headers=["Query", "Count"], tablefmt="simple"))
        else:
            print("No query data available.")
        print()
        
    except Exception as e:
        print(f"Error: {e}")

async def show_detailed_report():
    """Show a detailed report of Brave Search API statistics."""
    if not HAS_STATS_MODULES:
        print("Error: Statistics modules not available.")
        return
    
    try:
        # Get the full statistics report
        report = get_stats_report()
        print(report)
    except Exception as e:
        print(f"Error: {e}")

async def export_statistics(format_type: str, output_file: str = None):
    """Export statistics to a file.
    
    Args:
        format_type: Export format (json or csv)
        output_file: Output file path (optional)
    """
    if not HAS_STATS_MODULES:
        print("Error: Statistics modules not available.")
        return
    
    try:
        stats = get_stats()
        if not stats:
            print("Error: Could not get statistics.")
            return
        
        # Gather all stats
        export_data = {
            "session": stats.get_session_stats(),
            "daily": stats.get_daily_stats(30),  # Last 30 days
            "performance": stats.get_performance_stats(),
            "popular_queries": stats.get_popular_queries(20),  # Top 20 queries
            "export_time": datetime.now().isoformat()
        }
        
        # Get cache statistics
        cache_stats = await get_cache_stats()
        export_data["cache"] = cache_stats
        
        # Create export filename with timestamp if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"brave_search_stats_{timestamp}.{format_type}"
        
        # Export based on format
        if format_type.lower() == "json":
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"Statistics exported to {output_file} in JSON format")
        elif format_type.lower() == "csv":
            # Create multiple CSV files for different data types
            base_name = os.path.splitext(output_file)[0]
            
            # Session stats
            session_file = f"{base_name}_session.csv"
            with open(session_file, "w") as f:
                f.write("Metric,Value\n")
                for key, value in export_data["session"].items():
                    f.write(f"{key},{value}\n")
            
            # Daily stats
            daily_file = f"{base_name}_daily.csv"
            with open(daily_file, "w") as f:
                if export_data["daily"]:
                    # Get headers from first item
                    headers = list(export_data["daily"][0].keys())
                    f.write(",".join(headers) + "\n")
                    
                    # Write data
                    for day in export_data["daily"]:
                        f.write(",".join(str(day.get(h, "")) for h in headers) + "\n")
            
            # Popular queries
            queries_file = f"{base_name}_queries.csv"
            with open(queries_file, "w") as f:
                f.write("Query,Count\n")
                for query, count in export_data["popular_queries"]:
                    f.write(f"\"{query}\",{count}\n")
            
            print(f"Statistics exported to multiple CSV files with base name {base_name}")
        else:
            print(f"Unsupported format: {format_type}")
    except Exception as e:
        print(f"Error exporting statistics: {e}")

async def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(description="Brave Search Statistics CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show a summary of statistics")
    
    # Detailed report command
    report_parser = subparsers.add_parser("report", help="Show a detailed report of statistics")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export statistics to a file")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format")
    export_parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not HAS_STATS_MODULES:
        print("Error: Statistics modules not available. Make sure they are properly installed.")
        return
    
    if args.command == "summary":
        await show_summary()
    elif args.command == "report":
        await show_detailed_report()
    elif args.command == "export":
        await export_statistics(args.format, args.output)
    else:
        # Default to summary if no command is provided
        await show_summary()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
