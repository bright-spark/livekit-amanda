#!/usr/bin/env python3
"""
Brave Search Quality API Command Line Interface.

This CLI tool allows you to test and demonstrate the Brave Search Quality API
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

# Import Brave Search Quality Integration
from brave_search_quality_integration import (
    get_quality_integration,
    close_quality_integration,
    quality_web_search,
    quality_ai_search,
    combined_quality_search,
    adaptive_quality_search
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brave_search_quality_cli")

def setup_argparse() -> argparse.ArgumentParser:
    """Set up argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Brave Search Quality API Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Perform a high-quality web search
  python brave_search_quality_cli.py web "climate change latest research"
  
  # Perform a high-quality AI search
  python brave_search_quality_cli.py ai "explain quantum computing"
  
  # Perform a combined search (both web and AI)
  python brave_search_quality_cli.py combined "benefits of meditation"
  
  # Perform an adaptive search (automatically chooses the best strategy)
  python brave_search_quality_cli.py adaptive "how to bake sourdough bread"
  
  # Ground a query using a dedicated API key
  python brave_search_quality_cli.py ground web "latest research on climate change"
  python brave_search_quality_cli.py ground ai "explain quantum computing"
  
  # Get statistics about the quality API
  python brave_search_quality_cli.py stats
"""
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Web search parser
    web_parser = subparsers.add_parser("web", help="Perform a high-quality web search")
    web_parser.add_argument("query", help="Search query")
    web_parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    web_parser.add_argument("--improve", action="store_true", help="Try to improve search quality")
    web_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    web_parser.add_argument("--output", help="Output file path")
    
    # AI search parser
    ai_parser = subparsers.add_parser("ai", help="Perform a high-quality AI search")
    ai_parser.add_argument("query", help="Search query")
    ai_parser.add_argument("--improve", action="store_true", help="Try to improve search quality")
    ai_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    ai_parser.add_argument("--output", help="Output file path")
    
    # Combined search parser
    combined_parser = subparsers.add_parser("combined", help="Perform a combined search (web + AI)")
    combined_parser.add_argument("query", help="Search query")
    combined_parser.add_argument("--results", type=int, default=5, help="Number of web results to return")
    combined_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    combined_parser.add_argument("--output", help="Output file path")
    
    # Adaptive search parser
    adaptive_parser = subparsers.add_parser("adaptive", help="Perform an adaptive search (auto-select best strategy)")
    adaptive_parser.add_argument("query", help="Search query")
    adaptive_parser.add_argument("--results", type=int, default=5, help="Number of web results to return")
    adaptive_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    adaptive_parser.add_argument("--output", help="Output file path")
    
    # Ground search parser
    ground_parser = subparsers.add_parser("ground", help="Ground a query using a dedicated API key")
    ground_subparsers = ground_parser.add_subparsers(dest="search_type", help="Search type for grounding")
    
    # Ground web search parser
    ground_web_parser = ground_subparsers.add_parser("web", help="Ground using web search")
    ground_web_parser.add_argument("query", help="Search query")
    ground_web_parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    ground_web_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    ground_web_parser.add_argument("--output", help="Output file path")
    
    # Ground AI search parser
    ground_ai_parser = ground_subparsers.add_parser("ai", help="Ground using AI search")
    ground_ai_parser.add_argument("query", help="Search query")
    ground_ai_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    ground_ai_parser.add_argument("--output", help="Output file path")
    
    # Stats parser
    stats_parser = subparsers.add_parser("stats", help="Get statistics about the quality API")
    stats_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    stats_parser.add_argument("--output", help="Output file path")

    return parser

async def run_web_search(args) -> None:
    """Run a high-quality web search.
    
    Args:
        args: Command line arguments
    """
    # Create a mock context
    context = {"session_id": "cli_session"}
    
    # Get the integration
    integration = get_quality_integration()
    
    print(f"Performing high-quality web search for: {args.query}")
    print(f"Number of results: {args.results}")
    if args.improve:
        print("Attempting to improve search quality...")
    
    # Perform the search
    results = await integration.search_with_quality(
        context, args.query, "web", args.results, args.improve
    )
    
    # Handle output
    if args.format == "json":
        # For JSON format, we need to parse the text results
        # This is a simplified version, in a real implementation
        # you would return structured data from the API
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

async def run_ai_search(args) -> None:
    """Run a high-quality AI search.
    
    Args:
        args: Command line arguments
    """
    # Create a mock context
    context = {"session_id": "cli_session"}
    
    # Get the integration
    integration = get_quality_integration()
    
    print(f"Performing high-quality AI search for: {args.query}")
    if args.improve:
        print("Attempting to improve search quality...")
    
    # Perform the search
    results = await integration.search_with_quality(
        context, args.query, "ai", improve_quality=args.improve
    )
    
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

async def run_combined_search(args) -> None:
    """Run a combined search (both web and AI).
    
    Args:
        args: Command line arguments
    """
    # Create a mock context
    context = {"session_id": "cli_session"}
    
    # Get the integration
    integration = get_quality_integration()
    
    print(f"Performing combined search for: {args.query}")
    print(f"Number of web results: {args.results}")
    
    # Perform the search
    results = await integration.combined_search(
        context, args.query, args.results
    )
    
    # Handle output
    if args.format == "json":
        # For JSON format, we need to parse the text results
        result_data = {
            "query": args.query,
            "search_type": "combined",
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

async def run_adaptive_search(args) -> None:
    """Run an adaptive search (automatically chooses the best strategy).
    
    Args:
        args: Command line arguments
    """
    # Create a mock context
    context = {"session_id": "cli_session"}
    
    # Get the integration
    integration = get_quality_integration()
    
    print(f"Performing adaptive search for: {args.query}")
    print(f"Number of web results (if needed): {args.results}")
    
    # Perform the search
    results = await integration.adaptive_search(
        context, args.query, args.results
    )
    
    # Handle output
    if args.format == "json":
        # For JSON format, we need to parse the text results
        result_data = {
            "query": args.query,
            "search_type": "adaptive",
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

async def run_stats(args) -> None:
    """Get statistics about the quality API.
    
    Args:
        args: Command line arguments
    """
    # Get the integration
    integration = get_quality_integration()
    
    print("Getting statistics about the quality API...")
    
    # Get statistics
    stats = await integration.get_stats()
    
    # Handle output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {args.output}")
    else:
        print(json.dumps(stats, indent=2))

async def run_ground_search(args) -> None:
    """Run grounding search using a dedicated API key.
    
    Args:
        args: Command line arguments
    """
    # Create a mock context
    context = {"session_id": "cli_session"}
    
    # Get the integration
    integration = get_quality_integration()
    
    print(f"Grounding query using {args.search_type} search: {args.query}")
    if args.search_type == "web":
        print(f"Number of results: {args.results}")
    
    # Perform the grounding search
    if args.search_type == "web":
        results = await integration.ground_query(context, args.query, "web", args.results)
    else:
        results = await integration.ground_query(context, args.query, "ai")
    
    # Handle output
    if args.format == "json":
        # For JSON format, we need to parse the text results
        result_data = {
            "query": args.query,
            "search_type": f"ground_{args.search_type}",
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
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "web":
            await run_web_search(args)
        elif args.command == "ai":
            await run_ai_search(args)
        elif args.command == "combined":
            await run_combined_search(args)
        elif args.command == "adaptive":
            await run_adaptive_search(args)
        elif args.command == "ground":
            if not args.search_type:
                print("Error: You must specify a search type (web or ai) for grounding")
                return
            await run_ground_search(args)
        elif args.command == "stats":
            await run_stats(args)
        else:
            print(f"Unknown command: {args.command}")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
    finally:
        # Always close the integration
        await close_quality_integration()

if __name__ == "__main__":
    asyncio.run(main())
