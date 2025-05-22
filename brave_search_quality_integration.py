"""
Brave Search Quality API Integration.

This module demonstrates how to integrate the Brave Search Quality API
with other components of the LiveKit Amanda project.
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Import Brave Search Quality API
from brave_search_quality_api import (
    get_quality_api,
    close_quality_api,
    high_quality_web_search,
    high_quality_ai_search,
    improve_search_quality,
    ground_query as api_ground_query
)

# Import cache configuration
from cache_config import get_cache_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brave_search_quality_integration")

class BraveSearchQualityIntegration:
    """Integration class for Brave Search Quality API."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integration.
        
        Args:
            config: Configuration dictionary. If None, uses default configuration.
        """
        # Create a default quality API configuration
        quality_config = {
            "enable_quality_api": True,
            "quality_threshold": 0.8,
            "cache_ttl": 1209600,  # 2 weeks
            "refresh_threshold": 604800,  # 1 week
            "max_retries": 3,
            "enrichment_enabled": True,
            "fallback_to_regular": True,
            "parallel_processing": True,
        }
        
        # Try to get cache configuration if available
        try:
            cache_config = get_cache_config()
            if hasattr(cache_config, 'config') and isinstance(cache_config.config, dict):
                # Extract values from the config dictionary
                config_dict = cache_config.config
                if "brave_search_quality_threshold" in config_dict:
                    quality_config["quality_threshold"] = config_dict["brave_search_quality_threshold"]
                if "brave_search_quality_ttl" in config_dict:
                    quality_config["cache_ttl"] = config_dict["brave_search_quality_ttl"]
                if "brave_search_quality_refresh" in config_dict:
                    quality_config["refresh_threshold"] = config_dict["brave_search_quality_refresh"]
                if "brave_search_quality_max_retries" in config_dict:
                    quality_config["max_retries"] = config_dict["brave_search_quality_max_retries"]
                if "brave_search_quality_enrichment" in config_dict:
                    quality_config["enrichment_enabled"] = config_dict["brave_search_quality_enrichment"]
                if "brave_search_quality_fallback" in config_dict:
                    quality_config["fallback_to_regular"] = config_dict["brave_search_quality_fallback"]
                if "brave_search_quality_parallel" in config_dict:
                    quality_config["parallel_processing"] = config_dict["brave_search_quality_parallel"]
        except Exception as e:
            logger.warning(f"Could not load cache configuration: {e}. Using default values.")
            # Continue with default values
        
        # Override with provided config if any
        if config:
            quality_config.update(config)
        
        # Initialize the quality API
        self.api = get_quality_api(quality_config)
        
        logger.info(f"Initialized BraveSearchQualityIntegration with config: {quality_config}")
    
    async def search_with_quality(self, context, query: str, 
                               search_type: str = "web",
                               num_results: int = 5,
                               improve_quality: bool = False,
                               **kwargs) -> str:
        """Perform a search with quality enhancement.
        
        Args:
            context: The run context
            query: Search query
            search_type: Type of search ('web' or 'ai')
            num_results: Number of results to return (for web search)
            improve_quality: Whether to try improving quality
            **kwargs: Additional parameters for the search
            
        Returns:
            Formatted search results
        """
        start_time = time.time()
        logger.info(f"Performing {search_type} search with quality enhancement for: {query}")
        
        try:
            if improve_quality:
                # Try to improve quality by trying different query variations
                logger.info(f"Attempting to improve quality for query: {query}")
                results = await improve_search_quality(context, query, search_type)
                
                # Log performance
                elapsed = time.time() - start_time
                logger.info(f"Quality-improved {search_type} search completed in {elapsed:.4f}s")
                
                return results
            
            if search_type == "web":
                # Perform high-quality web search
                results = await high_quality_web_search(context, query, num_results)
            else:
                # Perform high-quality AI search
                results = await high_quality_ai_search(context, query)
            
            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"High-quality {search_type} search completed in {elapsed:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_with_quality: {e}")
            return f"Error performing high-quality {search_type} search: {str(e)}"
    
    async def combined_search(self, context, query: str, num_results: int = 5) -> str:
        """Perform both web and AI searches and combine the results.
        
        Args:
            context: The run context
            query: Search query
            num_results: Number of results to return for web search
            
        Returns:
            Combined formatted search results
        """
        start_time = time.time()
        logger.info(f"Performing combined search for: {query}")
        
        try:
            # Initialize results
            web_results = None
            ai_results = None
            web_error = None
            ai_error = None
            
            # Run web search
            try:
                web_results = await high_quality_web_search(context, query, num_results)
                logger.info("Web search completed successfully")
            except Exception as e:
                web_error = str(e)
                logger.error(f"Web search failed: {e}")
            
            # Run AI search
            try:
                ai_results = await high_quality_ai_search(context, query)
                logger.info("AI search completed successfully")
            except Exception as e:
                ai_error = str(e)
                logger.error(f"AI search failed: {e}")
            
            # Check if both searches failed
            if web_results is None and ai_results is None:
                return f"Error performing combined search: Web search error: {web_error}, AI search error: {ai_error}"
            
            # Start building the combined result
            combined = f"""
[COMBINED HIGH-QUALITY SEARCH RESULTS]
Query: '{query}'
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Add AI results if available
            if ai_results is not None and "I couldn't find" not in ai_results:
                try:
                    ai_content = ai_results.split('[HIGH-QUALITY AI SEARCH GROUNDING INFORMATION]')[1].strip()
                    combined += f"\n[AI-GENERATED ANSWER]\n{ai_content}\n"
                except (IndexError, AttributeError) as e:
                    logger.error(f"Error parsing AI results: {e}")
                    combined += f"\n[AI-GENERATED ANSWER]\nError processing AI search results.\n"
            else:
                combined += f"\n[AI-GENERATED ANSWER]\nNo AI-generated answer available for this query.\n"
            
            # Add web results if available
            if web_results is not None and "I couldn't find" not in web_results:
                try:
                    web_content = web_results.split('[HIGH-QUALITY WEB SEARCH GROUNDING INFORMATION]')[1].strip()
                    combined += f"\n[WEB SEARCH RESULTS]\n{web_content}\n"
                except (IndexError, AttributeError) as e:
                    logger.error(f"Error parsing web results: {e}")
                    combined += f"\n[WEB SEARCH RESULTS]\nError processing web search results.\n"
            else:
                combined += f"\n[WEB SEARCH RESULTS]\nNo web search results available for this query.\n"
            
            # Add grounding instructions
            combined += """
[GROUNDING INSTRUCTIONS FOR LLM]
When answering the user's question, use the available search results as your sources of information.
If both AI and web results are available, consider both sources, with web results providing more detailed information.
If only one type of result is available, rely on that information while acknowledging its limitations.
If there are contradictions between sources, acknowledge them and explain the different perspectives.
[END OF GROUNDING INSTRUCTIONS]
"""
            
            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"Combined search completed in {elapsed:.4f}s")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in combined_search: {e}")
            return f"Error performing combined search: {str(e)}"
    
    async def ground_query(self, context, query: str, search_type: str = "web", num_results: int = 5, **kwargs) -> str:
        """Ground a query using a dedicated API key for grounding purposes.
        
        Args:
            context: The run context
            query: Search query
            search_type: Type of search ("web" or "ai")
            num_results: Number of results for web search
            **kwargs: Additional parameters for the search
            
        Returns:
            Formatted grounding results
        """
        start_time = time.time()
        logger.info(f"Grounding query: {query} using {search_type} search")
        
        try:
            # Get the quality API
            quality_api = get_quality_api()
            
            # Use the dedicated grounding API
            results = await quality_api.ground_query(context, query, search_type, num_results, **kwargs)
            
            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"Grounding completed in {elapsed:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ground_query: {e}")
            return f"Error grounding query: {str(e)}"
    
    async def adaptive_search(self, context, query: str, num_results: int = 5) -> str:
        """Adaptively choose the best search strategy based on the query.
        
        Args:
            context: The run context
            query: Search query
            num_results: Number of results to return for web search
            
        Returns:
            Formatted search results using the best strategy
        """
        start_time = time.time()
        logger.info(f"Performing adaptive search for: {query}")
        
        try:
            # Analyze the query to determine the best search strategy
            query_lower = query.lower()
            
            # Queries that benefit from AI search
            ai_keywords = [
                "explain", "how to", "what is", "why does", "summarize",
                "difference between", "compare", "pros and cons"
            ]
            
            # Queries that benefit from web search
            web_keywords = [
                "latest", "news", "recent", "update", "today", "yesterday",
                "download", "buy", "price", "where to", "near me"
            ]
            
            # Determine the search strategy
            use_ai = any(keyword in query_lower for keyword in ai_keywords)
            use_web = any(keyword in query_lower for keyword in web_keywords)
            
            if use_ai and not use_web:
                # AI search is likely to be more helpful
                logger.info(f"Using AI search for query: {query}")
                results = await high_quality_ai_search(context, query)
                search_type = "AI"
            elif use_web and not use_ai:
                # Web search is likely to be more helpful
                logger.info(f"Using web search for query: {query}")
                results = await high_quality_web_search(context, query, num_results)
                search_type = "web"
            else:
                # Use combined search for balanced queries
                logger.info(f"Using combined search for query: {query}")
                results = await self.combined_search(context, query, num_results)
                search_type = "combined"
            
            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"Adaptive search ({search_type}) completed in {elapsed:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in adaptive_search: {e}")
            return f"Error performing adaptive search: {str(e)}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the quality API integration.
        
        Returns:
            Dictionary with statistics
        """
        api_stats = await self.api.get_stats()
        
        stats = {
            "integration": {
                "timestamp": datetime.now().timestamp(),
                "version": "1.0.0"
            },
            "quality_api": api_stats
        }
        
        return stats
    
    async def close(self) -> None:
        """Close the integration and release resources."""
        await close_quality_api()
        logger.info("Closed BraveSearchQualityIntegration")

# Singleton instance
_integration = None

def get_quality_integration(config: Optional[Dict[str, Any]] = None) -> BraveSearchQualityIntegration:
    """Get or create the quality integration.
    
    Args:
        config: Configuration dictionary. If None, uses default configuration.
        
    Returns:
        BraveSearchQualityIntegration instance
    """
    global _integration
    
    if _integration is None:
        _integration = BraveSearchQualityIntegration(config=config)
    
    return _integration

async def close_quality_integration() -> None:
    """Close the quality integration."""
    global _integration
    
    if _integration is not None:
        await _integration.close()
        _integration = None

# Convenience functions for tools
async def quality_web_search(context, query: str, num_results: int = 5) -> str:
    """Perform a high-quality web search.
    
    Args:
        context: The run context
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Formatted high-quality web search results
    """
    integration = get_quality_integration()
    return await integration.search_with_quality(context, query, "web", num_results)

async def quality_ai_search(context, query: str) -> str:
    """Perform a high-quality AI search.
    
    Args:
        context: The run context
        query: Search query
        
    Returns:
        Formatted high-quality AI search results
    """
    integration = get_quality_integration()
    return await integration.search_with_quality(context, query, "ai")

async def combined_quality_search(context, query: str, num_results: int = 5) -> str:
    """Perform a combined high-quality search (both web and AI).
    
    Args:
        context: The run context
        query: Search query
        num_results: Number of results to return for web search
        
    Returns:
        Combined formatted high-quality search results
    """
    integration = get_quality_integration()
    return await integration.combined_search(context, query, num_results)

async def adaptive_quality_search(context, query: str, num_results: int = 5) -> str:
    """Perform an adaptive high-quality search, choosing the best strategy.
    
    Args:
        context: The run context
        query: Search query
        num_results: Number of results to return for web search
        
    Returns:
        Formatted high-quality search results using the best strategy
    """
    integration = get_quality_integration()
    return await integration.adaptive_search(context, query, num_results)

# Example usage
async def main():
    """Example usage of the Brave Search Quality Integration."""
    # Create a mock context
    context = {"session_id": "test_session"}
    
    # Get the integration
    integration = get_quality_integration()
    
    # Perform a high-quality web search
    web_results = await integration.search_with_quality(
        context, "climate change latest research", "web", 5
    )
    print("\n=== HIGH-QUALITY WEB SEARCH RESULTS ===")
    print(web_results[:500] + "...\n")
    
    # Perform a high-quality AI search
    ai_results = await integration.search_with_quality(
        context, "explain quantum computing", "ai"
    )
    print("\n=== HIGH-QUALITY AI SEARCH RESULTS ===")
    print(ai_results[:500] + "...\n")
    
    # Perform a combined search
    combined_results = await integration.combined_search(
        context, "benefits of meditation"
    )
    print("\n=== COMBINED SEARCH RESULTS ===")
    print(combined_results[:500] + "...\n")
    
    # Perform an adaptive search
    adaptive_results = await integration.adaptive_search(
        context, "how to bake sourdough bread"
    )
    print("\n=== ADAPTIVE SEARCH RESULTS ===")
    print(adaptive_results[:500] + "...\n")
    
    # Get statistics
    stats = await integration.get_stats()
    print("\n=== STATISTICS ===")
    print(json.dumps(stats, indent=2))
    
    # Close the integration
    await close_quality_integration()

if __name__ == "__main__":
    asyncio.run(main())
