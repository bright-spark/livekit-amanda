async def load_enhanced_search(self):
    """Load enhanced search with RAG capabilities."""
    try:
        logger.info("Loading enhanced search with RAG")
        
        # Import the enhanced search module
        try:
            from enhanced_search import setup_enhanced_search, data_manager, get_embedding_model
            logger.info("Imported enhanced search module")
            
            # Pre-load the embedding model to ensure it's available locally
            logger.info("Pre-loading embedding model from cache...")
            model = get_embedding_model()
            if model is not None:
                logger.info("Successfully loaded embedding model from cache")
            else:
                logger.warning("Could not load embedding model from cache, will attempt to load when needed")
        except ImportError as e:
            logger.error(f"Error importing enhanced search module: {e}")
            return
        
        # Set up enhanced search
        enhanced_search_tools = await setup_enhanced_search(self.agent)
        
        # Import and set up Brave Search Quality RAG integration if available
        try:
            from brave_search_quality_rag_integration import get_integration, process_data_directory
            logger.info("Imported Brave Search Quality RAG integration")
            
            # Initialize the integration
            integration = get_integration()
            logger.info("Initialized Brave Search Quality RAG integration")
            
            # Process data directory in the background
            asyncio.create_task(process_data_directory())
            logger.info("Started processing data directory for RAG integration")
        except ImportError as e:
            logger.warning(f"Brave Search Quality RAG integration not available: {e}")
        
        if enhanced_search_tools:
            logger.info(f"Added {len(enhanced_search_tools)} enhanced search tools")
            
            # Start background embedding if enabled
            if self.has_background_embedding and not self.background_embedding_started:
                logger.info("Starting background embedding process")
                # Start background embedding in a non-blocking way
                asyncio.create_task(data_manager.start_background_embedding())
                self.background_embedding_started = True
                self.loaded_tools["background_embedding"] = True
        else:
            logger.warning("No enhanced search tools were added")
            
        self.enhanced_search_loaded = True
        self.tool_sources.append("Enhanced Search")
        
    except Exception as e:
        logger.error(f"Error loading enhanced search: {e}")
        logger.warning("Enhanced search could not be loaded")

    # Update instructions to reflect enhanced search
    if self.has_enhanced_search:
        await self.update_agent_instructions("Enhanced search with RAG loaded: You can now search through local knowledge base")
    else:
        await self.update_agent_instructions("Enhanced search with RAG is not enabled")
        
    # Mark enhanced search as loaded
    self.enhanced_search_loaded = True
    self.tool_sources.append("Enhanced Search with RAG")
    logger.info("Enhanced search with RAG loaded successfully")
    
    # Load MCP tools next if enabled
    if self.has_mcp_client:
        asyncio.create_task(self.load_mcp_tools())
    else:
        # Finalize the agent configuration
        await self.finalize_agent_configuration()
