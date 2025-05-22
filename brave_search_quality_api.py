"""
Brave Search Quality API.

This module provides an API for retrieving and storing high-quality search results
using the persistent cache with enhanced data quality processing.
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime

# Import persistent cache
from brave_search_persistent_cache import (
    get_persistent_cache, 
    close_persistent_cache,
    BraveSearchPersistentCache
)

# Import grounding service
from brave_search_grounding import (
    get_grounding_service,
    close_grounding_service,
    ground_query
)

# Import Brave Search API modules
try:
    from brave_web_search import web_search as brave_web_search, get_brave_web_search_client
    from brave_ai_search import ai_search as brave_ai_search, get_brave_ai_search_client
    HAS_BRAVE_SEARCH = True
except ImportError:
    HAS_BRAVE_SEARCH = False
    
    # Define dummy functions for when Brave Search is not available
    async def brave_web_search(*args, **kwargs):
        return "Web search functionality not available"
    
    async def brave_ai_search(*args, **kwargs):
        return "AI search functionality not available"
    
    async def get_brave_web_search_client(*args, **kwargs):
        return None
    
    async def get_brave_ai_search_client(*args, **kwargs):
        return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brave_search_quality_api")

# Default configuration
DEFAULT_CONFIG = {
    "enable_quality_api": True,
    "quality_threshold": 0.45,  # More practical threshold for quality API (was 0.8)
    "cache_ttl": 1209600,  # 2 weeks for high-quality results
    "refresh_threshold": 604800,  # 1 week before refreshing
    "max_retries": 3,  # Maximum retries for quality improvement
    "enrichment_enabled": True,
    "fallback_to_regular": True,  # Fallback to regular search if no high-quality result
    "parallel_processing": True,  # Process in parallel for faster results
}

class BraveSearchQualityAPI:
    """API for retrieving and storing high-quality search results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the quality API.
        
        Args:
            config: Configuration dictionary. If None, uses default configuration.
        """
        # Merge provided config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Get persistent cache instance
        cache_config = {
            "quality_threshold": self.config["quality_threshold"],
            "enrichment_enabled": self.config["enrichment_enabled"],
        }
        self.persistent_cache = get_persistent_cache(cache_config)
        
        logger.info(f"Initialized BraveSearchQualityAPI with config: {self.config}")
    
    async def get_high_quality_web_search(self, context, query: str, 
                                        num_results: int = 5, 
                                        force_refresh: bool = False,
                                        **kwargs) -> Dict[str, Any]:
        """Get high-quality web search results.
        
        Args:
            context: The run context
            query: Search query
            num_results: Number of results to return
            force_refresh: Whether to force a refresh from the API
            **kwargs: Additional parameters for the search
            
        Returns:
            High-quality web search results
        """
        if not self.config["enable_quality_api"]:
            # If quality API is disabled, fall back to regular search
            logger.info("Quality API disabled, using regular web search")
            return await brave_web_search(context, query, num_results)
        
        # Add num_results to kwargs for cache key generation
        search_kwargs = {"num_results": num_results, **kwargs}
        
        # If forcing refresh, use the persistent cache's dedicated API key
        if force_refresh:
            logger.info("Forcing refresh of high-quality web search result using dedicated API key")
            return await self.persistent_cache.refresh_high_quality_result(
                query=query,
                search_type="web",
                **search_kwargs
            )
        
        # Try to get from persistent cache first
        cached_result = await self.persistent_cache.get_high_quality_result(
            query=query,
            search_type="web",
            **search_kwargs
        )
        
        if cached_result:
            # Check if result is still fresh
            metadata = cached_result.get("_metadata", {})
            enriched_at = metadata.get("enriched_at")
            
            if enriched_at:
                try:
                    enriched_time = datetime.fromisoformat(enriched_at)
                    age = (datetime.now() - enriched_time).total_seconds()
                    
                    if age < self.config["refresh_threshold"]:
                        logger.info(f"Using high-quality cached web search result (age: {age:.1f}s)")
                        return cached_result
                    else:
                        logger.info(f"Cached result too old ({age:.1f}s), refreshing with dedicated API key")
                        # Use the persistent cache's dedicated API key to refresh
                        return await self.persistent_cache.refresh_high_quality_result(
                            query=query,
                            search_type="web",
                            **search_kwargs
                        )
                except (ValueError, TypeError):
                    logger.warning("Invalid enriched_at timestamp in cached result")
        
        # If we get here, we need to fetch a new result
        logger.info("No cached result found, fetching new high-quality web search result")
        
        # Try to use the persistent cache's dedicated API key first
        if self.persistent_cache.has_api_key:
            logger.info("Using dedicated persistent cache API key for web search")
            result = await self.persistent_cache.fetch_fresh_web_results(
                query=query,
                count=min(num_results + 5, 20),  # Request extra results for better quality
                **kwargs
            )
            
            if "error" not in result:
                # Process and store the result
                processed_data, quality_score = await self.persistent_cache.quality_processor.process_data(result, "web")
                
                # Add metadata
                processed_data["_metadata"] = {
                    "query": query,
                    "search_type": "web",
                    "quality_score": quality_score,
                    "enriched_at": datetime.now().isoformat()
                }
                
                # Store in persistent cache
                key = self.persistent_cache.get_cache_key(query, "web", **search_kwargs)
                await self.persistent_cache.set(key, processed_data, "web", self.config["cache_ttl"])
                
                logger.info(f"Successfully fetched and stored web search result with quality score: {quality_score:.2f}")
                return processed_data
            else:
                logger.warning(f"Error using dedicated API key: {result.get('error')}. Falling back to regular API key.")
        
        # Fallback to regular web search client
        web_client = await get_brave_web_search_client()
        if not web_client:
            logger.error("Failed to get web search client")
            return {"error": "Failed to get web search client"}
        
        # Perform the search with raw results
        raw_results = await web_client.search(
            query=query,
            count=min(num_results + 5, 20),  # Request extra results for better quality
            **kwargs
        )
        
        if "error" in raw_results:
            logger.error(f"Error in web search: {raw_results['error']}")
            
            if self.config["fallback_to_regular"]:
                logger.info("Falling back to regular web search")
                return await brave_web_search(context, query, num_results)
            else:
                return raw_results
        
        # Store in persistent cache and get processed version
        stored = await self.persistent_cache.store_high_quality_result(
            query=query,
            data=raw_results,
            search_type="web",
            ttl=self.config["cache_ttl"],
            **search_kwargs
        )
        
        if stored:
            # Get the processed version from cache
            processed_result = await self.persistent_cache.get_high_quality_result(
                query=query,
                search_type="web",
                **search_kwargs
            )
            
            if processed_result:
                logger.info("Successfully stored and retrieved high-quality web search result")
                return processed_result
        
        # If storage failed or retrieval failed, return the raw results
        logger.info("Using raw web search results (storage or retrieval failed)")
        return raw_results
    
    async def get_high_quality_ai_search(self, context, query: str,
                                       force_refresh: bool = False,
                                       **kwargs) -> Dict[str, Any]:
        """Get high-quality AI search results.
        
        Args:
            context: The run context
            query: Search query
            force_refresh: Whether to force a refresh from the API
            **kwargs: Additional parameters for the search
            
        Returns:
            High-quality AI search results
        """
        if not self.config["enable_quality_api"]:
            # If quality API is disabled, fall back to regular search
            logger.info("Quality API disabled, using regular AI search")
            return await brave_ai_search(context, query)
        
        # If forcing refresh, use the persistent cache's dedicated API key
        if force_refresh:
            logger.info("Forcing refresh of high-quality AI search result using dedicated API key")
            return await self.persistent_cache.refresh_high_quality_result(
                query=query,
                search_type="ai",
                **kwargs
            )
        
        # Try to get from persistent cache first
        cached_result = await self.persistent_cache.get_high_quality_result(
            query=query,
            search_type="ai",
            **kwargs
        )
        
        if cached_result:
            # Check if result is still fresh
            metadata = cached_result.get("_metadata", {})
            enriched_at = metadata.get("enriched_at")
            
            if enriched_at:
                try:
                    enriched_time = datetime.fromisoformat(enriched_at)
                    age = (datetime.now() - enriched_time).total_seconds()
                    
                    if age < self.config["refresh_threshold"]:
                        logger.info(f"Using high-quality cached AI search result (age: {age:.1f}s)")
                        return cached_result
                    else:
                        logger.info(f"Cached result too old ({age:.1f}s), refreshing with dedicated API key")
                        # Use the persistent cache's dedicated API key to refresh
                        return await self.persistent_cache.refresh_high_quality_result(
                            query=query,
                            search_type="ai",
                            **kwargs
                        )
                except (ValueError, TypeError):
                    logger.warning("Invalid enriched_at timestamp in cached result")
        
        # If we get here, we need to fetch a new result
        logger.info("No cached result found, fetching new high-quality AI search result")
        
        # Try to use the persistent cache's dedicated API key first
        if self.persistent_cache.has_api_key:
            logger.info("Using dedicated persistent cache API key for AI search")
            result = await self.persistent_cache.fetch_fresh_ai_results(
                query=query,
                **kwargs
            )
            
            if "error" not in result:
                # Process and store the result
                processed_data, quality_score = await self.persistent_cache.quality_processor.process_data(result, "ai")
                
                # Add metadata
                processed_data["_metadata"] = {
                    "query": query,
                    "search_type": "ai",
                    "quality_score": quality_score,
                    "enriched_at": datetime.now().isoformat()
                }
                
                # Store in persistent cache
                key = self.persistent_cache.get_cache_key(query, "ai", **kwargs)
                await self.persistent_cache.set(key, processed_data, "ai", self.config["cache_ttl"])
                
                logger.info(f"Successfully fetched and stored AI search result with quality score: {quality_score:.2f}")
                return processed_data
            else:
                logger.warning(f"Error using dedicated API key: {result.get('error')}. Falling back to regular API key.")
        
        # Fallback to regular AI search client
        ai_client = await get_brave_ai_search_client()
        if not ai_client:
            logger.error("Failed to get AI search client")
            return {"error": "Failed to get AI search client"}
        
        # Perform the search with raw results
        raw_results = await ai_client.ai_search(
            query=query,
            **kwargs
        )
        
        if "error" in raw_results:
            logger.error(f"Error in AI search: {raw_results['error']}")
            
            if self.config["fallback_to_regular"]:
                logger.info("Falling back to regular AI search")
                try:
                    return await brave_ai_search(context, query)
                except Exception as e:
                    logger.error(f"[TOOL] high_quality_ai_search exception: {e}")
                    # Return a properly formatted error response
                    return {"error": f"AI search failed: {str(e)}", "results": []}
            else:
                # Ensure we return a properly formatted response even on error
                if not isinstance(raw_results, dict):
                    return {"error": str(raw_results), "results": []}
                return raw_results
        
        # Store in persistent cache and get processed version
        stored = await self.persistent_cache.store_high_quality_result(
            query=query,
            data=raw_results,
            search_type="ai",
            ttl=self.config["cache_ttl"],
            **kwargs
        )
        
        if stored:
            # Get the processed version from cache
            processed_result = await self.persistent_cache.get_high_quality_result(
                query=query,
                search_type="ai",
                **kwargs
            )
            
            if processed_result:
                logger.info("Successfully stored and retrieved high-quality AI search result")
                return processed_result
        
        # If storage failed or retrieval failed, return the raw results
        logger.info("Using raw AI search results (storage or retrieval failed)")
        return raw_results
    
    async def improve_search_quality(self, context, query: str, search_type: str,
                                   max_retries: Optional[int] = None,
                                   **kwargs) -> Dict[str, Any]:
        """Attempt to improve search quality by trying different query variations.
        
        Args:
            context: The run context
            query: Search query
            search_type: Type of search ('web' or 'ai')
            max_retries: Maximum number of retries. If None, uses config value.
            **kwargs: Additional parameters for the search
            
        Returns:
            Highest quality search results
        """
        if max_retries is None:
            max_retries = self.config["max_retries"]
        
        # Try the original query first
        if search_type == "web":
            original_result = await self.get_high_quality_web_search(
                context, query, force_refresh=True, **kwargs
            )
        else:
            original_result = await self.get_high_quality_ai_search(
                context, query, force_refresh=True, **kwargs
            )
        
        # Check if quality is already good enough
        original_quality = original_result.get("_metadata", {}).get("quality_score", 0)
        if original_quality >= self.config["quality_threshold"]:
            logger.info(f"Original query already has good quality: {original_quality:.2f}")
            return original_result
        
        # Generate query variations
        variations = self._generate_query_variations(query)
        
        # Try each variation
        best_result = original_result
        best_quality = original_quality
        
        for i, variation in enumerate(variations[:max_retries]):
            logger.info(f"Trying query variation {i+1}/{min(max_retries, len(variations))}: {variation}")
            
            if search_type == "web":
                result = await self.get_high_quality_web_search(
                    context, variation, force_refresh=True, **kwargs
                )
            else:
                result = await self.get_high_quality_ai_search(
                    context, variation, force_refresh=True, **kwargs
                )
            
            quality = result.get("_metadata", {}).get("quality_score", 0)
            logger.info(f"Variation quality: {quality:.2f}")
            
            if quality > best_quality:
                best_result = result
                best_quality = quality
                
                # If we've reached a good enough quality, stop
                if best_quality >= self.config["quality_threshold"]:
                    logger.info(f"Found good quality result: {best_quality:.2f}")
                    break
        
        logger.info(f"Best quality achieved: {best_quality:.2f}")
        return best_result
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate variations of a query to improve search quality.
        
        Args:
            query: Original search query
            
        Returns:
            List of query variations
        """
        variations = []
        
        # Add more specific variations
        variations.append(f"{query} detailed information")
        variations.append(f"{query} comprehensive guide")
        
        # Add variations with different phrasing
        if query.lower().startswith("how to"):
            variations.append(query.lower().replace("how to", "steps for"))
            variations.append(query.lower().replace("how to", "guide for"))
        
        if query.lower().startswith("what is"):
            variations.append(query.lower().replace("what is", "definition of"))
            variations.append(query.lower().replace("what is", "explain"))
        
        # Add variations with more context
        if len(query.split()) < 5:
            variations.append(f"{query} explained in detail")
            variations.append(f"{query} with examples")
        
        return variations
    
    async def format_high_quality_web_results(self, results: Dict[str, Any], num_results: int = 5) -> str:
        """Format high-quality web search results into a readable string.
        
        Args:
            results: High-quality web search results
            num_results: Number of results to include
            
        Returns:
            Formatted string of search results
        """
        if "error" in results:
            return f"Search error: {results['error']}"
        
        # Get current timestamp for grounding
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract metadata
        metadata = results.get("_metadata", {})
        quality_score = metadata.get("quality_score", 0)
        
        # Ensure we have a valid enriched_at timestamp
        enriched_at = metadata.get("enriched_at", None)
        if not enriched_at or enriched_at == "unknown":
            enriched_at = current_time
        
        # Extract the query and ensure it's not empty
        query_obj = results.get("query", {})
        if isinstance(query_obj, dict):
            query = query_obj.get("query", "")
        else:
            query = str(query_obj) if query_obj else ""
            
        # If query is still empty, try to get it from metadata
        if not query:
            query = metadata.get("query", "")
            
        # If still empty, use a placeholder
        if not query:
            query = "[No query specified]"
        
        # Start with a grounding header
        formatted = f"""[HIGH-QUALITY WEB SEARCH GROUNDING INFORMATION]
- Query: '{query}'
- Results retrieved: {enriched_at}
- Current time: {current_time}
- Quality score: {quality_score:.2f}
- Search API: Brave Web Search (Enhanced Quality)

"""
        
        # Extract web results
        if "web" in results and "results" in results["web"]:
            web_results = results["web"]["results"]
            formatted += f"Found {len(web_results)} results:\n\n"
            
            # Format each result
            for idx, result in enumerate(web_results[:num_results], 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                description = result.get("description", "")
                
                # Extract age if available
                age = result.get("age", "")
                date_info = ""
                if age:
                    date_info = f" [Published: {age}]"
                
                # Extract and format domain information for source credibility
                domain = result.get("domain", "")
                domain_info = f" [Source: {domain}]" if domain else ""
                
                formatted += f"{idx}. {title}{date_info}{domain_info}\n   {url}\n"
                if description:
                    formatted += f"   {description}\n"
                formatted += "\n"
        else:
            formatted += "No web search results found.\n\n"
        
        # Extract and include featured snippet if available
        if "web" in results and "featured_snippet" in results["web"] and results["web"]["featured_snippet"]:
            snippet = results["web"]["featured_snippet"]
            formatted += "\n[FEATURED SNIPPET]\n"
            formatted += f"{snippet.get('title', 'No title')}\n"
            formatted += f"{snippet.get('description', 'No description')}\n"
            if "url" in snippet:
                formatted += f"Source: {snippet['url']}\n"
            formatted += "\n"
        
        # Extract and include news results if available
        if "news" in results and "results" in results["news"] and results["news"]["results"]:
            news_results = results["news"]["results"]
            formatted += "\n[RECENT NEWS RESULTS - Time-sensitive information]\n"
            for i, news in enumerate(news_results[:3], 1):  # Limit to top 3 news items
                news_title = news.get("title", "")
                news_url = news.get("url", "")
                news_description = news.get("description", "")
                news_age = news.get("age", "")
                news_source = news.get("source", "")
                
                formatted += f"News {i}: {news_title}"
                if news_age:
                    formatted += f" [Published: {news_age}]"
                if news_source:
                    formatted += f" [Source: {news_source}]"
                formatted += f"\n   {news_url}\n"
                if news_description:
                    formatted += f"   {news_description}\n"
                formatted += "\n"
        
        # Add a timestamp footer for grounding
        formatted += f"\n[End of high-quality search results. Retrieved at {enriched_at}, current time: {current_time}]\n"
        formatted += "Remember to consider the recency of information when answering time-sensitive questions."
        
        return formatted
    
    async def format_high_quality_ai_results(self, results: Dict[str, Any]) -> str:
        """Format high-quality AI search results into a readable string.
        
        Args:
            results: High-quality AI search results
            
        Returns:
            Formatted string of AI search results
        """
        if "error" in results:
            return f"AI Search error: {results['error']}"
        
        # Get current timestamp for grounding
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract metadata
        metadata = results.get("_metadata", {})
        quality_score = metadata.get("quality_score", 0)
        
        # Ensure we have a valid enriched_at timestamp
        enriched_at = metadata.get("enriched_at", None)
        if not enriched_at or enriched_at == "unknown":
            enriched_at = current_time
        
        # Extract the query and ensure it's not empty
        query_obj = results.get("query", {})
        if isinstance(query_obj, dict):
            query = query_obj.get("query", "")
        else:
            query = str(query_obj) if query_obj else ""
            
        # If query is still empty, try to get it from metadata
        if not query:
            query = metadata.get("query", "")
            
        # If still empty, use a placeholder
        if not query:
            query = "[No query specified]"
        
        # Start with a grounding header
        formatted = f"""[HIGH-QUALITY AI SEARCH GROUNDING INFORMATION]
- Query: '{query}'
- Results retrieved: {enriched_at}
- Current time: {current_time}
- Quality score: {quality_score:.2f}
- Search API: Brave AI Search (Enhanced Quality)

"""
        
        # Extract the generated answer
        if "generated_answer" in results:
            gen_answer = results["generated_answer"]
            
            # Add confidence score if available
            confidence_score = gen_answer.get("_confidence_score", None)
            if confidence_score is not None:
                formatted += f"AI ANSWER (Confidence: {confidence_score:.2f}):\n"
            else:
                formatted += "AI ANSWER:\n"
            
            # Add the main answer
            if "answer" in gen_answer:
                formatted += f"{gen_answer['answer']}\n\n"
            
            # Add supporting points if available
            if "points" in gen_answer and gen_answer["points"]:
                formatted += "SUPPORTING POINTS:\n"
                for i, point in enumerate(gen_answer["points"], 1):
                    formatted += f"{i}. {point}\n"
                formatted += "\n"
            
            # Add sources if available
            if "sources" in gen_answer and gen_answer["sources"]:
                formatted += "SOURCES:\n"
                for i, source in enumerate(gen_answer["sources"], 1):
                    title = source.get("title", "No title")
                    url = source.get("url", "")
                    formatted += f"{i}. {title}\n   {url}\n"
                formatted += "\n"
        else:
            formatted += "No AI-generated answer available for this query.\n\n"
        
        # Add a footer
        formatted += f"[End of high-quality AI search results. Retrieved at {enriched_at}, current time: {current_time}]\n"
        formatted += "This is an AI-generated response based on the latest information available to Brave Search."
        
        return formatted
    
    async def combined_search(self, context, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        """Perform a combined search using both web and AI.
        
        Args:
            context: The run context
            query: Search query
            num_results: Number of web results to return
            **kwargs: Additional parameters for the search
            
        Returns:
            Combined search results
        """
        try:
            # Initialize combined results with default structure
            combined = {
                "web_results": [],
                "ai_answer": None,
                "query": query,
                "_metadata": {
                    "web_quality_score": 0,
                    "ai_quality_score": 0,
                    "combined_at": datetime.now().isoformat()
                }
            }
            
            # Get web search results
            web_results = await self.get_high_quality_web_search(
                context, query, num_results=num_results, **kwargs
            )
            
            # Add web results if available
            if isinstance(web_results, dict):
                if "web" in web_results and "results" in web_results["web"]:
                    combined["web_results"] = web_results["web"]["results"]
                    if "_metadata" in web_results and "quality_score" in web_results["_metadata"]:
                        combined["_metadata"]["web_quality_score"] = web_results["_metadata"]["quality_score"]
                elif "results" in web_results:
                    # Handle case where results are directly in the root
                    combined["web_results"] = web_results["results"]
                elif "error" in web_results:
                    logger.warning(f"Web search error: {web_results.get('error')}")
            
            # Get AI search results
            ai_results = await self.get_high_quality_ai_search(
                context, query, **kwargs
            )
            
            # Add AI results if available
            if isinstance(ai_results, dict):
                if "ai" in ai_results and "results" in ai_results["ai"] and ai_results["ai"]["results"]:
                    combined["ai_answer"] = ai_results["ai"]["results"][0] if ai_results["ai"]["results"] else None
                    if "_metadata" in ai_results and "quality_score" in ai_results["_metadata"]:
                        combined["_metadata"]["ai_quality_score"] = ai_results["_metadata"]["quality_score"]
                elif "results" in ai_results and ai_results["results"]:
                    # Handle case where results are directly in the root
                    combined["ai_answer"] = ai_results["results"][0] if ai_results["results"] else None
                elif "error" in ai_results:
                    logger.warning(f"AI search error: {ai_results.get('error')}")
            
            return combined
        except Exception as e:
            logger.error(f"Error in combined_search: {e}")
            # Return a properly formatted response even on error
            return {
                "web_results": [],
                "ai_answer": None,
                "query": query,
                "error": f"Combined search failed: {str(e)}",
                "_metadata": {
                    "combined_at": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
    
    async def high_quality_web_search(self, context, query: str, num_results: int = 5) -> str:
        """Perform a high-quality web search and return formatted results.
        
        Args:
            context: The run context
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Formatted high-quality web search results
        """
        start_time = time.time()
        logger.info(f"[TOOL] high_quality_web_search called for query: {query}")
        
        try:
            # Get high-quality web search results
            results = await self.get_high_quality_web_search(context, query, num_results=num_results)
            
            # Format the results
            formatted_results = await self.format_high_quality_web_results(results, num_results)
            
            # Add explicit grounding instructions for the LLM
            grounding_header = (
                """[GROUNDING INSTRUCTIONS FOR LLM]
"""
                """When answering the user's question, use these high-quality search results as your primary source of information.
"""
                """These results have been enhanced for accuracy and relevance.
"""
                """Pay special attention to the publication dates and source credibility information.
"""
                """If the search results don't contain relevant information to answer the user's question, clearly state this limitation.
"""
                """[END OF GROUNDING INSTRUCTIONS]

"""
            )
            
            # Insert the grounding header at the beginning of the results
            formatted_results = grounding_header + formatted_results
            
            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"[PERFORMANCE] high_quality_web_search completed in {elapsed:.4f}s")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"[TOOL] high_quality_web_search exception: {e}")
            return f"I couldn't find high-quality results for '{query}'. Try a different query."
    
    async def high_quality_ai_search(self, context, query: str) -> str:
        """Perform a high-quality AI search and return formatted results.
        
        Args:
            context: The run context
            query: Search query
            
        Returns:
            Formatted high-quality AI search results
        """
        start_time = time.time()
        logger.info(f"[TOOL] high_quality_ai_search called for query: {query}")
        
        try:
            # Get high-quality AI search results
            results = await self.get_high_quality_ai_search(context, query)
            
            # Format the results
            formatted_results = await self.format_high_quality_ai_results(results)
            
            # Add explicit grounding instructions for the LLM
            grounding_header = (
                """[GROUNDING INSTRUCTIONS FOR LLM]
"""
                """When answering the user's question, use this high-quality AI-generated response as your primary source of information.
"""
                """This response has been enhanced for accuracy and relevance.
"""
                """Pay special attention to the supporting points and sources provided.
"""
                """If the AI-generated response doesn't contain relevant information to answer the user's question, clearly state this limitation.
"""
                """[END OF GROUNDING INSTRUCTIONS]

"""
            )
            
            # Insert the grounding header at the beginning of the results
            formatted_results = grounding_header + formatted_results
            
            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"[PERFORMANCE] high_quality_ai_search completed in {elapsed:.4f}s")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"[TOOL] high_quality_ai_search exception: {e}")
            return f"I couldn't find a high-quality AI-generated answer for '{query}'. Try a different query."
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the quality API and persistent cache.
        
        Returns:
            Dictionary with statistics
        """
        cache_stats = await self.persistent_cache.get_stats()
        
        stats = {
            "config": self.config,
            "persistent_cache": cache_stats,
            "timestamp": datetime.now().timestamp()
        }
        
        return stats
    
    async def ground_query(self, context, query: str, search_type: str = "web", num_results: int = 5, **kwargs) -> str:
        """Ground a query using the Brave Search API with a dedicated grounding API key.
        
        Args:
            context: Context for the query
            query: Search query
            search_type: Type of search ("web" or "ai")
            num_results: Number of results to return
            **kwargs: Additional parameters for the search
            
        Returns:
            Formatted grounding results as a string
        """
        logger.info(f"[TOOL] ground_query called for query: {query} using {search_type} search")
        
        # Create a grounding client
        grounding_client = await get_grounding_service()
        
        # Add grounding header
        grounding_header = """[GROUNDING INSTRUCTIONS FOR LLM]
Use these grounding search results as your primary source of information for answering the user's question.
These results have been specifically retrieved for grounding purposes using a dedicated API key.
Pay special attention to the publication dates and source credibility information.
If the grounding results don't contain relevant information to answer the user's question, clearly state this limitation.
[END OF GROUNDING INSTRUCTIONS]

"""
        
        try:
            # Make the grounding request
            results = await grounding_client.ground_query(query, search_type, num_results, **kwargs)
            
            # Format the results
            result = await grounding_client.format_grounding_results(results, search_type, query)
            
            return grounding_header + result
        except Exception as e:
            logger.error(f"[TOOL] ground_query exception: {e}")
            error_message = f"Grounding error: {str(e)}"
            return grounding_header + error_message
    
    async def close(self) -> None:
        """Close the quality API and release resources."""
        # Close the persistent cache
        await close_persistent_cache()
        
        # Close the grounding service
        await close_grounding_service()
        
        # Close any open client sessions from Brave Search API
        if HAS_BRAVE_SEARCH:
            try:
                # Try to close web search client
                web_client = await get_brave_web_search_client()
                if hasattr(web_client, 'close') and callable(web_client.close):
                    await web_client.close()
                elif hasattr(web_client, 'session') and hasattr(web_client.session, 'close'):
                    await web_client.session.close()
                
                # Try to close AI search client
                ai_client = await get_brave_ai_search_client()
                if hasattr(ai_client, 'close') and callable(ai_client.close):
                    await ai_client.close()
                elif hasattr(ai_client, 'session') and hasattr(ai_client.session, 'close'):
                    await ai_client.session.close()
            except Exception as e:
                logger.warning(f"Error closing client sessions: {e}")
        
        logger.info("Closed BraveSearchQualityAPI")


# Singleton instance
_quality_api = None

def get_quality_api(config: Optional[Dict[str, Any]] = None) -> BraveSearchQualityAPI:
    """Get or create the quality API.
    
    Args:
        config: Configuration dictionary. If None, uses default configuration.
        
    Returns:
        BraveSearchQualityAPI instance
    """
    global _quality_api
    
    if _quality_api is None:
        _quality_api = BraveSearchQualityAPI(config=config)
    
    return _quality_api

async def close_quality_api() -> None:
    """Close the quality API."""
    global _quality_api
    
    if _quality_api is not None:
        await _quality_api.close()
        _quality_api = None

# Convenience functions for use in tools
async def high_quality_web_search(context, query: str, num_results: int = 5) -> str:
    """Perform a high-quality web search and return formatted results.
    
    Args:
        context: The run context
        query: Search query
        num_results: Number of results to return
            
    Returns:
        Formatted high-quality web search results
    """
    start_time = time.time()
    logger.info(f"[TOOL] high_quality_web_search called for query: {query}")
    
    try:
        # Get high-quality web search results
        results = await self.get_high_quality_web_search(context, query, num_results=num_results)
        
        # Format the results
        formatted_results = await self.format_high_quality_web_results(results, num_results)
        
        # Calculate performance
        elapsed = time.time() - start_time
        logger.info(f"[PERFORMANCE] high_quality_web_search completed in {elapsed:.4f}s")
        
        return formatted_results
    except Exception as e:
        logger.error(f"[TOOL] high_quality_web_search exception: {e}")
        return f"I couldn't find high-quality web search results for '{query}'. Try a different query."

# Singleton instance
_quality_api = None

def get_quality_api(config: Optional[Dict[str, Any]] = None) -> BraveSearchQualityAPI:
    """Get or create the quality API.
    
    Args:
        config: Configuration dictionary. If None, uses default configuration.
        
    Returns:
        BraveSearchQualityAPI instance
    """
    global _quality_api
    
    if _quality_api is None:
        _quality_api = BraveSearchQualityAPI(config=config)
    
    return _quality_api

async def close_quality_api() -> None:
    """Close the quality API."""
    global _quality_api
    
    if _quality_api is not None:
        await _quality_api.close()
        _quality_api = None

# Convenience functions for use in tools
async def high_quality_web_search(context, query: str, num_results: int = 5) -> str:
    """Perform a high-quality web search.
    
    Args:
        context: The run context
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Formatted high-quality web search results
    """
    api = get_quality_api()
    return await api.high_quality_web_search(context, query, num_results)

async def high_quality_ai_search(context, query: str) -> str:
    """Perform a high-quality AI search.
    
    Args:
        context: The run context
        query: Search query
        
    Returns:
        Formatted high-quality AI search results
    """
    api = get_quality_api()
    return await api.high_quality_ai_search(context, query)

async def improve_search_quality(context, query: str, search_type: str = "web") -> str:
    """Attempt to improve search quality by trying different query variations.
    
    Args:
        context: The run context
        query: Search query
        search_type: Type of search ('web' or 'ai')
        
    Returns:
        Formatted high-quality search results
    """
    api = get_quality_api()
    results = await api.improve_search_quality(context, query, search_type)
    
    if search_type == "web":
        return await api.format_high_quality_web_results(results)
    else:
        return await api.format_high_quality_ai_results(results)
