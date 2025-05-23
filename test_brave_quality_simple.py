#!/usr/bin/env python3
"""
Simple test script for Brave Search Quality API integration with LangChain and RAG.
This script tests the basic functionality without relying on the enhanced_search module.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_brave_quality_simple")

# Import LangChain components
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Import Brave Search components
from brave_search_quality_api import get_quality_api, high_quality_web_search
from brave_search_persistent_cache import get_persistent_cache

# Configuration
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
ENABLE_RAG = True
ENABLE_BACKGROUND_EMBEDDING = True
MIN_SIMILARITY_THRESHOLD = 0.75

class SimpleQualityRAGTest:
    """Simple test class for Brave Search Quality API with LangChain and RAG."""
    
    def __init__(self):
        """Initialize test components."""
        logger.info("Initializing test components")
        
        # Initialize Brave Search components
        self.quality_api = get_quality_api()
        self.persistent_cache = get_persistent_cache()
        
        # Initialize LangChain components
        
        # 1. Memory cache for quality data
        self.memory_cache = ConversationBufferMemory(memory_key="search_history", return_messages=True)
        
        # 2. Embedding model for vector operations using Azure OpenAI with model router
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "model-router"),
            openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", "")
        )
        
        # 3. Vector stores for quality enriched data and RAG data
        os.makedirs(os.path.join(DATA_DIR, "vector_stores/quality"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "vector_stores/rag"), exist_ok=True)
        
        self.vector_store_quality = Chroma(
            collection_name="quality_enriched_data",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(DATA_DIR, "vector_stores/quality")
        )
        
        self.vector_store_rag = Chroma(
            collection_name="rag_data",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(DATA_DIR, "vector_stores/rag")
        )
        
        # 4. Text splitter for processing documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        logger.info("Test components initialized")
    
    async def process_data_directory(self):
        """Process the data directory for RAG."""
        logger.info(f"Processing data directory: {DATA_DIR}")
        
        if not os.path.exists(DATA_DIR):
            logger.warning(f"Data directory does not exist: {DATA_DIR}")
            return
        
        try:
            # Use LangChain's DirectoryLoader to load documents
            loader = DirectoryLoader(
                DATA_DIR,
                glob="**/*.{txt,md,json,csv,html,xml,py,js}",
                loader_cls=TextLoader,
                show_progress=True
            )
            
            # Load documents
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from data directory")
            
            if not documents:
                logger.warning(f"No documents found in data directory: {DATA_DIR}")
                return
            
            # Split documents into chunks for better retrieval
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(texts)} text chunks")
            
            # Add documents to the RAG vector store
            self.vector_store_rag.add_documents(texts)
            logger.info(f"Added {len(texts)} documents to RAG vector store")
            
            # Persist the vector store
            self.vector_store_rag.persist()
            logger.info("Persisted RAG vector store")
            
        except Exception as e:
            logger.error(f"Error processing data directory: {e}")
    
    async def search_with_rag(self, context: Dict[str, Any], query: str, num_results: int = 5) -> str:
        """
        Perform a search with RAG integration.
        
        Args:
            context: The run context
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Formatted search results enriched with local data
        """
        logger.info(f"Searching for: {query}")
        
        # Step 1: Check memory cache first (fastest)
        memory_cache_key = f"memory:{query}:{num_results}"
        memory_messages = self.memory_cache.chat_memory.messages
        
        for message in memory_messages:
            if query in message.content:
                logger.info(f"Found in memory cache: {query}")
                return message.content
        
        # Step 2: Check persistent cache
        persistent_cache_key = f"quality_search:{query}:{num_results}"
        cached_results = None
        try:
            cached_results = self.persistent_cache.get(persistent_cache_key)
            if cached_results:
                logger.info(f"Using persistent cache for query: {query}")
                
                # Store in memory cache for faster access next time
                self.memory_cache.chat_memory.add_user_message(query)
                self.memory_cache.chat_memory.add_ai_message(cached_results)
                
                return cached_results
        except Exception as e:
            logger.error(f"Error checking persistent cache: {e}")
        
        # Step 3: Perform high-quality web search
        try:
            web_results = await high_quality_web_search(context, query, num_results)
            
            if web_results:
                # Store in memory cache
                self.memory_cache.chat_memory.add_user_message(query)
                self.memory_cache.chat_memory.add_ai_message(web_results)
                
                # Store in persistent cache
                try:
                    self.persistent_cache.store(
                        persistent_cache_key,
                        web_results,
                        metadata={"query": query, "num_results": num_results}
                    )
                    logger.info(f"Stored in persistent cache: {query}")
                except Exception as e:
                    logger.error(f"Error storing in persistent cache: {e}")
                
                # Add to vector store
                try:
                    # Create documents from search results
                    documents = []
                    chunks = self.text_splitter.split_text(web_results)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": "brave_quality_search",
                                "query": query,
                                "chunk_index": i
                            }
                        )
                        documents.append(doc)
                    
                    # Add to vector store
                    if documents:
                        self.vector_store_quality.add_documents(documents)
                        logger.info(f"Added {len(documents)} documents to vector store")
                except Exception as e:
                    logger.error(f"Error adding to vector store: {e}")
                
                return web_results
            else:
                logger.warning(f"No results found for query: {query}")
                return "No results found."
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return f"Error performing search: {str(e)}"
    
    async def get_relevant_local_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Get relevant local data from the RAG vector store.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant local data items
        """
        logger.info(f"Getting relevant local data for: {query}")
        
        try:
            # Get relevant documents from RAG vector store
            docs = self.vector_store_rag.similarity_search(query, k=5)
            
            if docs:
                logger.info(f"Found {len(docs)} relevant documents")
                
                # Convert to the expected format
                results = []
                for i, doc in enumerate(docs):
                    results.append({
                        "source": doc.metadata.get("source", "local_data"),
                        "title": doc.metadata.get("title", f"Local Document {i+1}"),
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                
                return results
            else:
                logger.info("No relevant local data found")
                return []
        except Exception as e:
            logger.error(f"Error getting relevant local data: {e}")
            return []
    
    async def close(self):
        """Close and clean up resources."""
        logger.info("Closing test components")
        
        try:
            # Persist vector stores
            self.vector_store_quality.persist()
            self.vector_store_rag.persist()
            logger.info("Vector stores persisted")
        except Exception as e:
            logger.error(f"Error persisting vector stores: {e}")

async def main():
    """Run the test."""
    test = SimpleQualityRAGTest()
    
    try:
        # Process data directory
        await test.process_data_directory()
        
        # Perform a search
        context = {"session_id": "test_session"}
        results = await test.search_with_rag(context, "climate change solutions", 3)
        
        # Print results
        logger.info("Search results:")
        print("\n" + "-" * 80)
        print(results[:500] + "..." if len(results) > 500 else results)
        print("-" * 80 + "\n")
        
        # Get relevant local data
        local_data = await test.get_relevant_local_data("climate change solutions")
        
        # Print local data
        logger.info(f"Found {len(local_data)} relevant local data items")
        for i, item in enumerate(local_data):
            print(f"Local Data {i+1}: {item['title']}")
            print(item['content'][:200] + "..." if len(item['content']) > 200 else item['content'])
            print("-" * 40)
    
    finally:
        # Close test components
        await test.close()

if __name__ == "__main__":
    asyncio.run(main())
