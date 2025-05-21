#!/usr/bin/env python3
"""
Script to search for job listings on Locanto using the LocantoBrowserScraper.
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional

# Import the LocantoBrowserScraper from the project
from locanto_browser_scraper import LocantoBrowserScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def search_jobs(
    query: Optional[str] = None,
    location: Optional[str] = None,
    max_pages: int = 2,
    use_proxy: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for job listings on Locanto.
    
    Args:
        query: Search query string (job title, keywords, etc.)
        location: Location to search in
        max_pages: Maximum number of pages to scrape
        use_proxy: Whether to use proxy for scraping
        
    Returns:
        List of job listing dictionaries
    """
    logger.info(f"Searching for jobs with query: '{query}' in location: '{location}'")
    
    # Initialize the scraper with proxy support
    scraper = LocantoBrowserScraper(use_proxy=use_proxy)
    
    try:
        # Search for job listings in the Jobs section
        listings = await scraper.search_listings(
            query=query,
            location=location,
            max_pages=max_pages,
            section="J",  # "J" is the section code for Jobs
            fetch_details=True,
            sort="date"  # Sort by date to get newest listings first
        )
        
        logger.info(f"Found {len(listings)} job listings")
        return listings
        
    except Exception as e:
        logger.error(f"Error searching for jobs: {str(e)}")
        return []
    finally:
        # Close the scraper
        await scraper.close()

def format_job_listing(listing: Dict[str, Any]) -> str:
    """Format a job listing for display."""
    result = []
    result.append(f"Title: {listing.get('title', 'N/A')}")
    result.append(f"Location: {listing.get('location', 'N/A')}")
    
    if 'category' in listing:
        result.append(f"Category: {listing['category']}")
    
    if 'description' in listing:
        # Truncate description if too long
        desc = listing['description']
        if len(desc) > 200:
            desc = desc[:197] + "..."
        result.append(f"Description: {desc}")
    
    if 'url' in listing:
        result.append(f"URL: {listing['url']}")
    
    if 'timestamp' in listing:
        from datetime import datetime
        date_str = datetime.fromtimestamp(listing['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        result.append(f"Posted: {date_str}")
    
    return "\n".join(result)

async def main():
    """Main function to run the job search."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search for job listings on Locanto')
    parser.add_argument('--query', '-q', type=str, help='Search query (job title, keywords)')
    parser.add_argument('--location', '-l', type=str, help='Location to search in')
    parser.add_argument('--max-pages', '-p', type=int, default=2, help='Maximum number of pages to scrape')
    parser.add_argument('--no-proxy', action='store_true', help='Disable proxy usage')
    parser.add_argument('--output', '-o', type=str, help='Output file for JSON results')
    
    args = parser.parse_args()
    
    # Search for job listings
    listings = await search_jobs(
        query=args.query,
        location=args.location,
        max_pages=args.max_pages,
        use_proxy=not args.no_proxy
    )
    
    # Display results
    if listings:
        print(f"\nFound {len(listings)} job listings:\n")
        for i, listing in enumerate(listings, 1):
            print(f"--- Job Listing {i} ---")
            print(format_job_listing(listing))
            print()
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(listings, f, indent=2)
            print(f"Results saved to {args.output}")
    else:
        print("No job listings found.")

if __name__ == "__main__":
    asyncio.run(main())
