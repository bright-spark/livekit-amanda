#!/usr/bin/env python3
"""
Brave Search Grounding CLI.

This CLI tool allows you to test and demonstrate the Brave Search Grounding API
with various search types and options.
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import Brave Search Grounding
from brave_search_grounding import (
    get_grounding_service,
    close_grounding_service,
    ground_query
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brave_search_grounding_cli")

def setup_argparse() -> argparse.ArgumentParser:
    """Set up argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Brave Search Grounding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ground a query using web search
  python brave_search_grounding_cli.py web "climate change latest research"
  
  # Ground a query using AI search
  python brave_search_grounding_cli.py ai "explain quantum computing"
  
  # Save grounding results to a file
  python brave_search_grounding_cli.py web "renewable energy" --output results.txt
  
  # Get results in JSON format
  python brave_search_grounding_cli.py web "machine learning" --format json
"""
    )
    
    # Create subparsers for different search types
    subparsers = parser.add_subparsers(dest="search_type", help="Search type")
    
    # Web search parser
    web_parser = subparsers.add_parser("web", help="Ground using web search")
    web_parser.add_argument("query", help="Search query")
    web_parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    web_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    web_parser.add_argument("--output", help="Output file path")
    
    # AI search parser
    ai_parser = subparsers.add_parser("ai", help="Ground using AI search")
    ai_parser.add_argument("query", help="Search query")
    ai_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    ai_parser.add_argument("--output", help="Output file path")
    
    return parser

async def run_web_grounding(args) -> None:
    """Run web grounding.
    
    Args:
        args: Command line arguments
    """
    print(f"Grounding query using web search: {args.query}")
    print(f"Number of results: {args.results}")
    
    # Perform the grounding
    results = await ground_query(args.query, "web", args.results)
    
    # Handle output
    if args.format == "json":
        # For JSON format, we need to parse the text results
        result_data = {
            "query": args.query,
            "search_type": "web",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result_data, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result_data, indent=2))
    else:
        # For text format, just print the results
        if args.output:
            with open(args.output, "w") as f:
                f.write(results)
            print(f"Results saved to {args.output}")
        else:
            print("\n" + results)

async def run_ai_grounding(args) -> None:
    """Run AI grounding.
    
    Args:
        args: Command line arguments
    """
    print(f"Grounding query using AI search: {args.query}")
    
    # Perform the grounding
    results = await ground_query(args.query, "ai")
    
    # Handle output
    if args.format == "json":
        # For JSON format, we need to parse the text results
        result_data = {
            "query": args.query,
            "search_type": "ai",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result_data, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result_data, indent=2))
    else:
        # For text format, just print the results
        if args.output:
            with open(args.output, "w") as f:
                f.write(results)
            print(f"Results saved to {args.output}")
        else:
            print("\n" + results)

async def main() -> None:
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.search_type is None:
        parser.print_help()
        return
    
    try:
        if args.search_type == "web":
            await run_web_grounding(args)
        elif args.search_type == "ai":
            await run_ai_grounding(args)
        else:
            print(f"Unsupported search type: {args.search_type}")
    finally:
        # Always close the grounding service
        await close_grounding_service()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
