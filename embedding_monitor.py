#!/usr/bin/env python3
"""
Embedding Monitor Tool

This tool provides a command-line interface for monitoring and managing
the background embedding process for the LiveKit Amanda project.

Features:
- Monitor embedding progress
- View embedding statistics
- Pause/resume embedding process
- Force reprocessing of specific files
"""

import os
import sys
import time
import json
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding_monitor")

# Import our enhanced search module
from enhanced_search import (
    data_manager,
    invalidate_local_data,
    ENABLE_BACKGROUND_EMBEDDING,
    ENABLE_LOCAL_DATA,
    DATA_DIR
)

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def format_size(bytes_size: int) -> str:
    """Format bytes into a human-readable size string."""
    if bytes_size < 1024:
        return f"{bytes_size} bytes"
    elif bytes_size < 1024 * 1024:
        kb = bytes_size / 1024
        return f"{kb:.1f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        mb = bytes_size / (1024 * 1024)
        return f"{mb:.1f} MB"
    else:
        gb = bytes_size / (1024 * 1024 * 1024)
        return f"{gb:.1f} GB"

async def show_progress(continuous: bool = False, interval: float = 1.0):
    """Show the current embedding progress."""
    if not ENABLE_BACKGROUND_EMBEDDING or not ENABLE_LOCAL_DATA:
        print("Background embedding is disabled. Enable it in the .env file.")
        return
    
    if not continuous:
        # Single progress report
        progress = data_manager.get_embedding_progress()
        
        print("\nEmbedding Progress:")
        print(f"  Total files: {progress['total_files']}")
        print(f"  Processed files: {progress['processed_files']}")
        print(f"  Completion: {progress['percentage']:.1f}%")
        print(f"  Elapsed time: {format_time(progress['elapsed_time'])}")
        print(f"  Throughput: {progress['throughput']:.2f} files/sec")
        print(f"  Status: {'Paused' if progress['paused'] else 'Running'}")
        
        # Show queue status
        queue_size = data_manager.embedding_queue.qsize()
        print(f"  Queue size: {queue_size} files")
        
        return
    
    # Continuous monitoring
    try:
        print("Starting continuous monitoring (press Ctrl+C to stop)...")
        while True:
            progress = data_manager.get_embedding_progress()
            
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("\nEmbedding Progress Monitor")
            print("=" * 50)
            print(f"  Total files: {progress['total_files']}")
            print(f"  Processed files: {progress['processed_files']}")
            print(f"  Completion: {progress['percentage']:.1f}%")
            print(f"  Elapsed time: {format_time(progress['elapsed_time'])}")
            print(f"  Throughput: {progress['throughput']:.2f} files/sec")
            print(f"  Status: {'Paused' if progress['paused'] else 'Running'}")
            
            # Show queue status
            queue_size = data_manager.embedding_queue.qsize()
            print(f"  Queue size: {queue_size} files")
            
            # Show progress bar
            bar_width = 40
            filled_width = int(progress['percentage'] / 100 * bar_width)
            bar = '█' * filled_width + '░' * (bar_width - filled_width)
            print(f"  [{bar}] {progress['percentage']:.1f}%")
            
            # Show ETA
            if progress['throughput'] > 0 and progress['percentage'] < 100:
                remaining_files = progress['total_files'] - progress['processed_files']
                eta_seconds = remaining_files / progress['throughput']
                print(f"  Estimated time remaining: {format_time(eta_seconds)}")
            
            print("\nPress Ctrl+C to stop monitoring")
            
            # Wait for the next update
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

async def show_statistics():
    """Show statistics about the embedding process and data files."""
    if not ENABLE_LOCAL_DATA:
        print("Local data integration is disabled. Enable it in the .env file.")
        return
    
    # Get statistics about data files
    total_files = len(data_manager.data_files)
    embedded_files = len(data_manager.embeddings)
    pending_files = total_files - embedded_files
    
    # Calculate total size of data files
    total_size = 0
    file_types = {}
    
    for file_id, file_data in data_manager.data_files.items():
        content = file_data.get('content', '')
        total_size += len(content.encode('utf-8'))
        
        # Count file types
        file_type = file_data.get('type', '').lower()
        if file_type:
            file_types[file_type] = file_types.get(file_type, 0) + 1
    
    print("\nEmbedding Statistics:")
    print(f"  Total files: {total_files}")
    print(f"  Embedded files: {embedded_files}")
    print(f"  Pending files: {pending_files}")
    print(f"  Embedding completion: {(embedded_files / total_files * 100) if total_files > 0 else 0:.1f}%")
    print(f"  Total data size: {format_size(total_size)}")
    
    print("\nFile Types:")
    for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {file_type}: {count} files")
    
    # Show embedding queue status if background embedding is enabled
    if ENABLE_BACKGROUND_EMBEDDING:
        queue_size = data_manager.embedding_queue.qsize()
        print(f"\nEmbedding Queue Size: {queue_size} files")
        
        # Show embedding state
        state_count = len(data_manager.embedding_state)
        print(f"Embedding State: {state_count} pending files")

async def list_files(show_embedded: bool = True, show_pending: bool = True):
    """List all data files with their embedding status."""
    if not ENABLE_LOCAL_DATA:
        print("Local data integration is disabled. Enable it in the .env file.")
        return
    
    print("\nData Files:")
    print(f"{'Status':<10} {'Size':<10} {'Type':<10} {'File':<50}")
    print("-" * 80)
    
    for file_id, file_data in sorted(data_manager.data_files.items()):
        status = "Embedded" if file_id in data_manager.embeddings else "Pending"
        
        # Skip based on filters
        if status == "Embedded" and not show_embedded:
            continue
        if status == "Pending" and not show_pending:
            continue
        
        content = file_data.get('content', '')
        size = len(content.encode('utf-8'))
        file_type = file_data.get('type', '').lower()
        
        print(f"{status:<10} {format_size(size):<10} {file_type:<10} {file_id:<50}")

async def force_reprocess(file_pattern: str):
    """Force reprocessing of files matching the pattern."""
    if not ENABLE_LOCAL_DATA:
        print("Local data integration is disabled. Enable it in the .env file.")
        return
    
    # Find files matching the pattern
    matching_files = []
    for file_id in data_manager.data_files.keys():
        if file_pattern in file_id:
            matching_files.append(file_id)
    
    if not matching_files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(matching_files)} files matching pattern: {file_pattern}")
    print("Files to reprocess:")
    for file_id in matching_files:
        print(f"  {file_id}")
    
    confirm = input("Do you want to reprocess these files? (y/n): ")
    if confirm.lower() != 'y':
        print("Reprocessing cancelled.")
        return
    
    # Reprocess each file
    for file_id in matching_files:
        try:
            # Invalidate the file first
            await invalidate_local_data(file_id)
            
            # Get the full path
            file_path = os.path.join(DATA_DIR, file_id)
            
            # Reprocess the file
            if os.path.exists(file_path):
                result = data_manager.process_file(file_path)
                if result:
                    print(f"Successfully reprocessed: {file_id}")
                else:
                    print(f"Failed to reprocess: {file_id}")
            else:
                print(f"File not found: {file_id}")
        except Exception as e:
            print(f"Error reprocessing {file_id}: {e}")
    
    print("Reprocessing complete.")

async def main():
    """Main function to run the embedding monitor tool."""
    parser = argparse.ArgumentParser(description="Embedding Monitor Tool")
    
    # Command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor embedding progress")
    monitor_parser.add_argument("--continuous", "-c", action="store_true", help="Continuously monitor progress")
    monitor_parser.add_argument("--interval", "-i", type=float, default=1.0, help="Update interval in seconds for continuous monitoring")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show embedding statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List data files")
    list_parser.add_argument("--embedded", "-e", action="store_true", help="Show only embedded files")
    list_parser.add_argument("--pending", "-p", action="store_true", help="Show only pending files")
    
    # Reprocess command
    reprocess_parser = subparsers.add_parser("reprocess", help="Force reprocessing of files")
    reprocess_parser.add_argument("pattern", help="File pattern to match for reprocessing")
    
    args = parser.parse_args()
    
    # Default to monitor if no command is specified
    if not args.command:
        await show_progress(continuous=False)
        return
    
    # Execute the specified command
    if args.command == "monitor":
        await show_progress(continuous=args.continuous, interval=args.interval)
    elif args.command == "stats":
        await show_statistics()
    elif args.command == "list":
        # If neither flag is specified, show both
        if not args.embedded and not args.pending:
            show_embedded = True
            show_pending = True
        else:
            show_embedded = args.embedded
            show_pending = args.pending
        
        await list_files(show_embedded=show_embedded, show_pending=show_pending)
    elif args.command == "reprocess":
        await force_reprocess(args.pattern)

if __name__ == "__main__":
    asyncio.run(main())
