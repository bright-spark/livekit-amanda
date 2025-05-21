#!/usr/bin/env python3
"""
FastAPI application for LiveKit Amanda project.
This exposes the functionality of the project as a REST API.
"""
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import project modules
from locanto_browser_scraper import LocantoBrowserScraper, clean_url, add_proxy_to_url
from indeed import search_indeed_jobs
from tools import get_weather

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LiveKit Amanda API",
    description="API for LiveKit Amanda project with web scraping capabilities and more",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define Pydantic models for request/response
class JobSearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Search query string (job title, keywords, etc.)")
    location: Optional[str] = Field(None, description="Location to search in")
    max_pages: int = Field(1, description="Maximum number of pages to scrape", ge=1, le=5)
    use_proxy: bool = Field(True, description="Whether to use proxy for scraping")

class JobListing(BaseModel):
    title: str = Field(..., description="Job title")
    location: Optional[str] = Field(None, description="Job location")
    description: Optional[str] = Field(None, description="Job description")
    url: Optional[str] = Field(None, description="URL to the job listing")
    category: Optional[str] = Field(None, description="Job category")
    timestamp: Optional[int] = Field(None, description="Timestamp when the job was posted or scraped")
    images: Optional[List[str]] = Field(None, description="List of image URLs")

class JobSearchResponse(BaseModel):
    count: int = Field(..., description="Number of job listings found")
    listings: List[JobListing] = Field(..., description="List of job listings")

class WeatherRequest(BaseModel):
    location: str = Field(..., description="Location to get weather for")
    units: str = Field("metric", description="Units for temperature (metric or imperial)")

class ProxyRequest(BaseModel):
    url: str = Field(..., description="URL to proxy")
    use_proxy: bool = Field(True, description="Whether to use proxy")

class ProxyResponse(BaseModel):
    original_url: str = Field(..., description="Original URL")
    proxied_url: str = Field(..., description="Proxied URL")
    cleaned_url: str = Field(..., description="Cleaned URL")

# Background task for cleaning up browser instances
async def cleanup_browser(scraper: LocantoBrowserScraper):
    """Background task to close browser after request is complete."""
    await asyncio.sleep(5)  # Give time for response to be sent
    await scraper.close()

# API endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LiveKit Amanda API",
        "version": "1.0.0",
        "description": "API for LiveKit Amanda project with web scraping capabilities and more",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "POST /jobs/locanto": "Search for jobs on Locanto",
            "POST /jobs/indeed": "Search for jobs on Indeed",
            "POST /weather": "Get weather information",
            "POST /proxy": "Proxy a URL"
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": asyncio.get_event_loop().time()}

@app.post("/jobs/locanto", response_model=JobSearchResponse, tags=["Jobs"])
async def search_locanto_jobs(
    request: JobSearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Search for job listings on Locanto.
    
    This endpoint uses the LocantoBrowserScraper to search for job listings on Locanto.
    """
    try:
        # Initialize the scraper with proxy support
        scraper = LocantoBrowserScraper(use_proxy=request.use_proxy)
        
        # Add cleanup task
        background_tasks.add_task(cleanup_browser, scraper)
        
        # Search for job listings in the Jobs section
        listings = await scraper.search_listings(
            query=request.query,
            location=request.location,
            max_pages=request.max_pages,
            section="J",  # "J" is the section code for Jobs
            fetch_details=True,
            sort="date"  # Sort by date to get newest listings first
        )
        
        logger.info(f"Found {len(listings)} job listings")
        
        # Convert to response model
        return JobSearchResponse(
            count=len(listings),
            listings=listings
        )
        
    except Exception as e:
        logger.error(f"Error searching for jobs on Locanto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching for jobs: {str(e)}")

@app.post("/jobs/indeed", response_model=JobSearchResponse, tags=["Jobs"])
async def search_indeed_jobs_endpoint(request: JobSearchRequest):
    """
    Search for job listings on Indeed.
    
    This endpoint uses the Indeed scraper to search for job listings.
    """
    try:
        # Search for job listings on Indeed
        listings = await search_indeed_jobs(
            query=request.query,
            location=request.location,
            max_pages=request.max_pages,
            use_proxy=request.use_proxy
        )
        
        logger.info(f"Found {len(listings)} job listings on Indeed")
        
        # Convert to response model
        return JobSearchResponse(
            count=len(listings),
            listings=listings
        )
        
    except Exception as e:
        logger.error(f"Error searching for jobs on Indeed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching for jobs: {str(e)}")

@app.post("/weather", tags=["Weather"])
async def get_weather_endpoint(request: WeatherRequest):
    """
    Get weather information for a location.
    
    This endpoint uses the OpenWeather API to get weather information.
    """
    try:
        # Get weather information
        # The original get_weather function expects a RunContext, but we'll adapt it for FastAPI
        # by passing only the location parameter
        weather_data = await get_weather(None, request.location)
        return {"location": request.location, "weather": weather_data}
        
    except Exception as e:
        logger.error(f"Error getting weather: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting weather: {str(e)}")

@app.post("/proxy", response_model=ProxyResponse, tags=["Utilities"])
async def proxy_url(request: ProxyRequest):
    """
    Proxy a URL through the configured proxy service.
    
    This endpoint uses the proxy configuration to proxy a URL.
    """
    try:
        # Clean the URL
        cleaned_url = clean_url(request.url)
        
        # Add proxy if requested
        proxied_url = add_proxy_to_url(cleaned_url, use_proxy=request.use_proxy)
        
        return ProxyResponse(
            original_url=request.url,
            cleaned_url=cleaned_url,
            proxied_url=proxied_url
        )
        
    except Exception as e:
        logger.error(f"Error proxying URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error proxying URL: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app with uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
