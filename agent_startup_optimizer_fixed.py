"""
Fixed version of the register_locanto_search method for the agent_startup_optimizer.py file.
"""

async def register_locanto_search(self):
    """Register Locanto search tool with the agent."""
    try:
        # Try to import the fixed locanto search function first
        try:
            from locanto_fixed import search_locanto_fixed as search_locanto
            logger.info("Added Fixed Locanto Search tool to agent")
        except ImportError:
            # Fall back to the original implementation if the fixed version is not available
            from locanto import search_locanto
            logger.info("Added Locanto Search tool to agent (original version)")
        
        # Register the tool with the agent using our duplicate-prevention method
        added_tools = await self.update_agent_tools([search_locanto])
        
        if added_tools:
            # Notify console
            await self.update_agent_instructions(f"Tool added: search_locanto - Search for listings on Locanto")
            self.loaded_tools["search_locanto"] = True
            logger.info("Registered Locanto search tool")
        else:
            logger.info("Locanto Search tool already registered")
    except Exception as e:
        logger.error(f"Error adding Locanto Search tool: {e}")
        logger.warning("Could not add Locanto Search tool to agent")
