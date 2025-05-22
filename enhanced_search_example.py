"""
Enhanced Search Example for LiveKit Amanda

This example demonstrates how to use the enhanced_search module with Azure OpenAI
to provide better search results with RAG capabilities.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import RunContext

# Import the enhanced search module
from enhanced_search import enhanced_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_enhanced_search():
    """Test the enhanced search functionality with various options."""
    
    # Basic search
    print("\n=== Basic Enhanced Search ===")
    query = "latest advancements in quantum computing"
    results = await enhanced_search(query, num_results=3)
    print(f"Results for '{query}':\n{results[:500]}...\n")
    
    # Search with conversation context
    print("\n=== Enhanced Search with Conversation Context ===")
    conversation_history = [
        {"role": "user", "content": "Tell me about quantum computing"},
        {"role": "assistant", "content": "Quantum computing uses quantum bits or qubits..."},
        {"role": "user", "content": "What are the latest advancements?"}
    ]
    results = await enhanced_search("latest advancements", num_results=3, conversation_history=conversation_history)
    print(f"Results with conversation context:\n{results[:500]}...\n")
    
    # Search with RAG
    print("\n=== Enhanced Search with RAG ===")
    # First search to populate the RAG cache
    await enhanced_search("quantum computing applications in cryptography", num_results=3)
    # Second search that should benefit from RAG
    results = await enhanced_search("quantum cryptography", num_results=3)
    print(f"Results with RAG:\n{results[:500]}...\n")
    
    # Search without RAG for comparison
    print("\n=== Enhanced Search without RAG ===")
    results = await enhanced_search("quantum cryptography", num_results=3, use_rag=False)
    print(f"Results without RAG:\n{results[:500]}...\n")

async def example_agent_integration(context: RunContext, query: str, use_conversation_history: bool = True):
    """Example of how to integrate enhanced search with the agent.
    
    Args:
        context: The run context for the tool
        query: The search query
        use_conversation_history: Whether to use conversation history for context
        
    Returns:
        Enhanced search results
    """
    try:
        # Get conversation history from the context if available and requested
        conversation_history = None
        if use_conversation_history and hasattr(context, 'chat_context'):
            # Extract the last few messages from the chat context
            chat_context = getattr(context, 'chat_context', None)
            if chat_context and hasattr(chat_context, 'messages'):
                messages = getattr(chat_context, 'messages', [])
                # Convert to the format expected by enhanced_search
                conversation_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages[-5:]  # Use the last 5 messages
                ]
        
        # Perform the enhanced search
        results = await enhanced_search(
            query=query,
            num_results=5,
            conversation_history=conversation_history
        )
        
        return results
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        return f"Error performing enhanced search for '{query}': {str(e)}"

# Example of how to register the enhanced search as a tool in your agent
def register_enhanced_search_tool(agent):
    """Register the enhanced search tool with the agent.
    
    Args:
        agent: The LiveKit agent to register the tool with
    """
    from livekit.agents import function_tool
    
    @function_tool
    async def enhanced_web_search(context: RunContext, query: str, use_conversation_history: bool = True):
        """Perform an enhanced web search with query reformulation, multi-source aggregation, and RAG.
        
        This tool provides better search results by:
        1. Reformulating the query based on conversation context
        2. Searching across multiple search engines
        3. Using RAG (Retrieval-Augmented Generation) to enhance results with previously cached information
        4. Summarizing and organizing the results
        
        Args:
            context: The run context for the tool
            query: The search query
            use_conversation_history: Whether to use conversation history for context
            
        Returns:
            Enhanced search results
        """
        return await example_agent_integration(context, query, use_conversation_history)
    
    # Register the tool with the agent
    agent.register_tool(enhanced_web_search)
    logger.info("Registered enhanced_web_search tool with the agent")

if __name__ == "__main__":
    asyncio.run(test_enhanced_search())
