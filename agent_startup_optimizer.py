#!/usr/bin/env python3
"""Agent Startup Optimizer

This module optimizes the startup sequence of the LiveKit Amanda agent
to ensure it becomes responsive as quickly as possible, loading local
tools first before connecting to external MCP servers.
"""

import os
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent_startup")

# Import LiveKit agent components
from livekit.agents import Agent, AgentSession, JobContext, RunContext, function_tool

# Import necessary modules for tools
from dotenv import load_dotenv
load_dotenv()  # Ensure environment variables are loaded

class AgentStartupOptimizer:
    """
    Optimizes the startup sequence of the LiveKit Amanda agent to ensure
    it becomes responsive as quickly as possible, loading local tools first
    and then MCP servers.
    """
    
    def __init__(self):
        """Initialize the optimizer with default values."""
        self.tool_sources = []
        self.local_tools_loaded = False
        self.mcp_tools_loaded = False
        self.agent = None
        self.ctx = None
        self.session = None
        
        # Track loading states
        self.basic_tools_loaded = False
        self.search_tools_loaded = False
        self.enhanced_search_loaded = False
        self.background_embedding_started = False
        
        # Tool collections
        self.web_search_tools = []
        self.time_date_tools = []
        self.utility_tools = []
        self.job_search_tools = []
        
        # Import flags
        self.has_brave_search = os.environ.get("BRAVE_SEARCH_ENABLE", "").lower() in ("true", "1", "yes")
        self.has_bing_search = os.environ.get("BING_SEARCH_ENABLE", "").lower() in ("true", "1", "yes")
        self.has_duckduckgo = os.environ.get("DUCKDUCKGO_SEARCH_ENABLE", "").lower() in ("true", "1", "yes")
        self.has_google_search = os.environ.get("GOOGLE_SEARCH_ENABLE", "").lower() in ("true", "1", "yes")
        self.has_locanto = os.environ.get("LOCANTO_ENABLE", "").lower() in ("true", "1", "yes")
        self.has_indeed = os.environ.get("INDEED_ENABLE", "").lower() in ("true", "1", "yes")
        self.has_wikipedia = os.environ.get("WIKIPEDIA_ENABLE", "").lower() in ("true", "1", "yes")
        self.has_local_tools = os.environ.get("LOCAL_TOOLS", "").lower() in ("true", "1", "yes")
        self.has_mcp_client = os.environ.get("MCP_CLIENT", "").lower() in ("true", "1", "yes")
        self.has_openweather = os.environ.get("OPENWEATHER_ENABLE", "").lower() in ("true", "1", "yes")
        self.has_enhanced_search = os.environ.get("ENHANCED_SEARCH_ENABLE_RAG", "").lower() in ("true", "1", "yes")
        self.has_background_embedding = os.environ.get("ENABLE_BACKGROUND_EMBEDDING", "").lower() in ("true", "1", "yes")
        
        logger.info(f"Agent startup optimizer initialized with feature flags:")
        logger.info(f"  Brave Search: {self.has_brave_search}")
        logger.info(f"  Bing Search: {self.has_bing_search}")
        logger.info(f"  DuckDuckGo: {self.has_duckduckgo}")
        logger.info(f"  Google Search: {self.has_google_search}")
        logger.info(f"  Locanto: {self.has_locanto}")
        logger.info(f"  Indeed: {self.has_indeed}")
        logger.info(f"  Wikipedia: {self.has_wikipedia}")
        logger.info(f"  Local Tools: {self.has_local_tools}")
        logger.info(f"  MCP Client: {self.has_mcp_client}")
        logger.info(f"  OpenWeather: {self.has_openweather}")
        logger.info(f"  Enhanced Search RAG: {self.has_enhanced_search}")
        logger.info(f"  Background Embedding: {self.has_background_embedding}")
        
    async def initialize_agent(self, agent: Agent, ctx: JobContext):
        """
        Initialize the agent with optimized startup sequence.
        
        Args:
            agent: The LiveKit agent to optimize
            ctx: The job context for the agent
        """
        self.agent = agent
        self.ctx = ctx
        
        logger.info("Starting optimized agent initialization sequence")
        
        # Step 1: Make the agent responsive with minimal instructions
        await self.make_agent_responsive()
        
        # Step 2: Load basic tools in the background
        asyncio.create_task(self.load_basic_tools())
        
        return self.agent
    
    async def make_agent_responsive(self):
        """
        Make the agent responsive as quickly as possible with minimal instructions.
        """
        logger.info("Making agent responsive with minimal configuration")
        
        # The agent already has initial instructions from the entrypoint function
        # So we just need to connect to the room and start the session
        
        # Connect to the room to make the agent responsive
        await self.ctx.connect()
        
        # Start the agent session
        self.session = AgentSession()
        await self.session.start(agent=self.agent, room=self.ctx.room)
        
        logger.info("Agent is now responsive with minimal configuration")
        
    async def load_basic_tools(self):
        """
        Load basic tools that don't require external services.
        """
        logger.info("Loading basic tools")
        
        # Add basic time and date tools
        await self.register_time_date_tools()
        
        # Update instructions to reflect basic tools
        await self.update_agent_instructions("Basic tools loaded")
        
        # Mark basic tools as loaded
        self.basic_tools_loaded = True
        self.tool_sources.append("Basic Tools")
        
        # Load search tools next
        asyncio.create_task(self.load_search_tools())
        
    async def register_time_date_tools(self):
        """
        Register time and date tools with the agent.
        """
        @function_tool
        async def get_current_time(context: RunContext, timezone: str = "UTC") -> str:
            """Get the current time in the specified timezone."""
            try:
                import pytz
                from datetime import datetime
                
                # Get the timezone
                tz = pytz.timezone(timezone)
                
                # Get the current time in the specified timezone
                now = datetime.now(tz)
                
                # Format the time
                formatted_time = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
                
                return f"The current time in {timezone} is {formatted_time}"
            except Exception as e:
                return f"Error getting time: {str(e)}"
        
        @function_tool
        async def get_current_date(context: RunContext, timezone: str = "UTC") -> str:
            """Get the current date in the specified timezone."""
            try:
                import pytz
                from datetime import datetime
                
                # Get the timezone
                tz = pytz.timezone(timezone)
                
                # Get the current date in the specified timezone
                current_date = datetime.now(tz)
                
                # Format the date
                formatted_date = current_date.strftime("%Y-%m-%d %A")
                
                return f"The current date in {timezone} is {formatted_date}"
            except Exception as e:
                return f"Error getting current date: {str(e)}"
        
        # Register the tools with the agent
        self.time_date_tools = [get_current_time, get_current_date]
        
        # Register tools with the agent using the proper update_tools method
        try:
            # Get current tools
            current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
            
            # Add our new tools
            all_tools = list(current_tools) + self.time_date_tools
            
            # Update the agent's tools
            await self.agent.update_tools(all_tools)
            
            tool_names = [tool.__name__ for tool in self.time_date_tools]
            logger.info(f"Added time and date tools to agent: {', '.join(tool_names)}")
            
            # Update agent instructions with specific tool names
            await self.update_agent_instructions(f"Basic tools loaded: {', '.join(tool_names)}")
        except Exception as e:
            logger.error(f"Error adding time and date tools: {e}")
            logger.warning("Could not add time and date tools to agent")
        
    async def load_search_tools(self):
        """
        Load search tools based on enabled providers.
        """
        logger.info("Loading search tools")
        
        # Load search tools based on enabled providers
        if self.has_brave_search:
            await self.register_brave_search()
        
        if self.has_bing_search:
            await self.register_bing_search()
        
        if self.has_duckduckgo:
            await self.register_duckduckgo_search()
        
        if self.has_google_search:
            await self.register_google_search()
        
        if self.has_wikipedia:
            await self.register_wikipedia_search()
        
        # Register the combined web_search tool
        await self.register_web_search()
        
        # Update instructions to reflect search tools
        tool_names = [tool.__name__ for tool in self.web_search_tools]
        if tool_names:
            await self.update_agent_instructions(f"Search tools loaded: {', '.join(tool_names)}")
        else:
            await self.update_agent_instructions("No search tools were loaded")
        
        # Mark search tools as loaded
        self.search_tools_loaded = True
        self.tool_sources.append("Search Tools")
        
        # Load job search tools next
        asyncio.create_task(self.load_job_search_tools())
        
    async def register_brave_search(self):
        """Register Brave Search tool with the agent."""
        try:
            # Import the brave search module
            from brave_search_api import brave_web_search
            
            @function_tool
            async def brave_search(context: RunContext, query: str, num_results: int = 5) -> str:
                """Search the web using Brave Search API specifically."""
                try:
                    results = await brave_web_search(query, num_results)
                    return results
                except Exception as e:
                    return f"Error searching with Brave: {str(e)}"
            
            # Register the tool with the agent using the proper update_tools method
            try:
                # Get current tools
                current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
                
                # Add our new tool
                all_tools = list(current_tools) + [brave_search]
                
                # Update the agent's tools
                await self.agent.update_tools(all_tools)
                
                logger.info("Added Brave Search tool to agent")
                # Notify console
                await self.update_agent_instructions(f"Tool added: brave_search - Search the web using Brave Search API specifically")
            except Exception as e:
                logger.error(f"Error adding Brave Search tool: {e}")
                logger.warning("Could not add Brave Search tool to agent")
            
            self.web_search_tools.append(brave_search)
            
            logger.info("Registered Brave Search tool")
        except ImportError:
            logger.warning("Brave Search API not available")
            
    async def register_bing_search(self):
        """Register Bing Search tool with the agent."""
        try:
            # Import the bing search module
            from bing_search import bing_search as bing_search_func
            
            @function_tool
            async def bing_search(context: RunContext, query: str, num_results: int = 5) -> str:
                """Search the web using Bing specifically."""
                try:
                    results = await bing_search_func(context, query, num_results)
                    return results
                except Exception as e:
                    return f"Error searching with Bing: {str(e)}"
            
            # Register the tool with the agent using the proper update_tools method
            try:
                # Get current tools
                current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
                
                # Add our new tool
                all_tools = list(current_tools) + [bing_search]
                
                # Update the agent's tools
                await self.agent.update_tools(all_tools)
                
                logger.info("Added Bing Search tool to agent")
                # Notify console
                await self.update_agent_instructions(f"Tool added: bing_search - Search the web using Bing specifically")
            except Exception as e:
                logger.error(f"Error adding Bing Search tool: {e}")
                logger.warning("Could not add Bing Search tool to agent")
            
            self.web_search_tools.append(bing_search)
            
            logger.info("Registered Bing Search tool")
        except ImportError:
            logger.warning("Bing Search not available")
            
    async def register_duckduckgo_search(self):
        """Register DuckDuckGo Search tool with the agent."""
        try:
            # Import the duckduckgo search module
            from duckduckgo_search import duckduckgo_search as ddg_search_func
            
            @function_tool
            async def duckduckgo_search(context: RunContext, query: str, num_results: int = 5) -> str:
                """Search the web using DuckDuckGo specifically."""
                try:
                    results = await ddg_search_func(context, query, num_results)
                    return results
                except Exception as e:
                    return f"Error searching with DuckDuckGo: {str(e)}"
            
            # Register the tool with the agent using the proper update_tools method
            try:
                # Get current tools
                current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
                
                # Add our new tool
                all_tools = list(current_tools) + [duckduckgo_search]
                
                # Update the agent's tools
                await self.agent.update_tools(all_tools)
                
                logger.info("Added DuckDuckGo Search tool to agent")
                # Notify console
                await self.update_agent_instructions(f"Tool added: duckduckgo_search - Search the web using DuckDuckGo specifically")
            except Exception as e:
                logger.error(f"Error adding DuckDuckGo Search tool: {e}")
                logger.warning("Could not add DuckDuckGo Search tool to agent")
            
            self.web_search_tools.append(duckduckgo_search)
            
            logger.info("Registered DuckDuckGo Search tool")
        except ImportError:
            logger.warning("DuckDuckGo Search not available")
            
    async def register_google_search(self):
        """Register Google Search tool with the agent."""
        try:
            # Import the google search module
            from googlesearch.googlesearch import google_search as google_search_func
            
            @function_tool
            async def google_search(context: RunContext, query: str, num_results: int = 5) -> str:
                """Search the web using Google specifically."""
                try:
                    results = await google_search_func(context, query, num_results)
                    return results
                except Exception as e:
                    return f"Error searching with Google: {str(e)}"
            
            # Register the tool with the agent using the proper update_tools method
            try:
                # Get current tools
                current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
                
                # Add our new tool
                all_tools = list(current_tools) + [google_search]
                
                # Update the agent's tools
                await self.agent.update_tools(all_tools)
                
                logger.info("Added Google Search tool to agent")
                # Notify console
                await self.update_agent_instructions(f"Tool added: google_search - Search the web using Google specifically")
            except Exception as e:
                logger.error(f"Error adding Google Search tool: {e}")
                logger.warning("Could not add Google Search tool to agent")
            
            self.web_search_tools.append(google_search)
            
            logger.info("Registered Google Search tool")
        except ImportError:
            logger.warning("Google Search not available")
            
    async def register_wikipedia_search(self):
        """Register Wikipedia Search tool with the agent."""
        try:
            # Import the wikipedia module
            import wikipediaapi
            
            @function_tool
            async def wikipedia_search(context: RunContext, query: str) -> str:
                """Search Wikipedia for information on a topic."""
                try:
                    wiki = wikipediaapi.Wikipedia('en')
                    page = wiki.page(query)
                    
                    if page.exists():
                        # Get a summary (first 500 characters)
                        summary = page.summary[0:500] + "..." if len(page.summary) > 500 else page.summary
                        return f"Wikipedia: {query}\n\n{summary}\n\nFull article: {page.fullurl}"
                    else:
                        return f"No Wikipedia article found for '{query}'"
                except Exception as e:
                    return f"Error searching Wikipedia: {str(e)}"
            
            # Register the tool with the agent using the proper update_tools method
            try:
                # Get current tools
                current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
                
                # Add our new tool
                all_tools = list(current_tools) + [wikipedia_search]
                
                # Update the agent's tools
                await self.agent.update_tools(all_tools)
                
                logger.info("Added Wikipedia Search tool to agent")
                # Notify console
                await self.update_agent_instructions(f"Tool added: wikipedia_search - Search Wikipedia for information on a topic")
            except Exception as e:
                logger.error(f"Error adding Wikipedia Search tool: {e}")
                logger.warning("Could not add Wikipedia Search tool to agent")
            
            self.web_search_tools.append(wikipedia_search)
            
            logger.info("Registered Wikipedia Search tool")
        except ImportError:
            logger.warning("Wikipedia API not available")
            
    async def register_web_search(self):
        """Register combined web search tool with the agent."""
        @function_tool
        async def web_search(context: RunContext, query: str, num_results: int = 5) -> str:
            """Search the web using the best available search engine."""
            # Prioritize search engines based on availability
            if self.has_brave_search:
                try:
                    from brave_search_api import brave_web_search
                    results = await brave_web_search(query, num_results)
                    return results
                except Exception:
                    pass
            
            if self.has_bing_search:
                try:
                    from bing_search import bing_search
                    results = await bing_search(context, query, num_results)
                    return results
                except Exception:
                    pass
            
            if self.has_duckduckgo:
                try:
                    from duckduckgo_search import duckduckgo_search
                    results = await duckduckgo_search(context, query, num_results)
                    return results
                except Exception:
                    pass
            
            if self.has_google_search:
                try:
                    from googlesearch.googlesearch import google_search
                    results = await google_search(context, query, num_results)
                    return results
                except Exception:
                    pass
            
            # Fallback to a simple message if no search engines are available
            return "No search engines are currently available. Please try again later."
        
        # Register the tool with the agent using the proper update_tools method
        try:
            # Get current tools
            current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
            
            # Add our new tool
            all_tools = list(current_tools) + [web_search]
            
            # Update the agent's tools
            await self.agent.update_tools(all_tools)
            
            logger.info("Added Web Search tool to agent")
            # Notify console
            await self.update_agent_instructions(f"Tool added: web_search - Search the web using the best available search engine")
        except Exception as e:
            logger.error(f"Error adding Web Search tool: {e}")
            logger.warning("Could not add Web Search tool to agent")
        
        self.web_search_tools.append(web_search)
        
        logger.info("Registered combined web_search tool")
    
    async def load_job_search_tools(self):
        """Load job search tools if enabled."""
        logger.info("Loading job search tools")
        
        if self.has_indeed:
            await self.register_indeed_search()
        
        if self.has_locanto:
            await self.register_locanto_search()
        
        # Update instructions to reflect job search tools
        tool_names = [tool.__name__ for tool in self.job_search_tools]
        if tool_names:
            await self.update_agent_instructions(f"Job search tools loaded: {', '.join(tool_names)}")
        else:
            await self.update_agent_instructions("No job search tools were loaded")
        
        # Mark job search tools as loaded
        self.job_search_tools_loaded = True
        self.tool_sources.append("Job Search Tools")
        
        # Load enhanced search next if enabled
        if self.has_enhanced_search:
            asyncio.create_task(self.load_enhanced_search())
        else:
            # Otherwise, load MCP tools if enabled
            if self.has_mcp_client:
                asyncio.create_task(self.load_mcp_tools())
            else:
                # Finalize the agent configuration
                await self.finalize_agent_configuration()
        
    async def register_indeed_search(self):
        """Register Indeed job search tool with the agent."""
        try:
            # Import the indeed module
            from brave_search_indeed_optimized import indeed_job_search as indeed_search_func
            
            @function_tool
            async def indeed_job_search(context: RunContext, query: str = "customer service", location: str = "Johannesburg, Gauteng") -> str:
                """Search for jobs on Indeed using Brave Search API or scraping."""
                try:
                    results = await indeed_search_func(context, query, location)
                    return results
                except Exception as e:
                    return f"Error searching Indeed: {str(e)}"
            
            # Register the tool with the agent using the proper update_tools method
            try:
                # Get current tools
                current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
                
                # Add our new tool
                all_tools = list(current_tools) + [indeed_job_search]
                
                # Update the agent's tools
                await self.agent.update_tools(all_tools)
                
                logger.info("Added Indeed Job Search tool to agent")
                # Notify console
                await self.update_agent_instructions(f"Tool added: indeed_job_search - Search for jobs on Indeed")
            except Exception as e:
                logger.error(f"Error adding Indeed Job Search tool: {e}")
                logger.warning("Could not add Indeed Job Search tool to agent")
            
            self.job_search_tools.append(indeed_job_search)
            
            logger.info("Registered Indeed job search tool")
        except ImportError:
            logger.warning("Indeed job search not available")
            
    async def register_locanto_search(self):
        """Register Locanto search tool with the agent."""
        try:
            # Import the locanto module
            from brave_search_locanto_optimized import search_locanto as locanto_search_func
            
            @function_tool
            async def search_locanto(context: RunContext, category_path='personals/men-seeking-men', location='western-cape', max_pages=3, return_url=False) -> str:
                """Search Locanto for listings in a specific category and location."""
                try:
                    results = await locanto_search_func(context, category_path, location, max_pages, return_url)
                    return results
                except Exception as e:
                    return f"Error searching Locanto: {str(e)}"
            
            # Register the tool with the agent using the proper update_tools method
            try:
                # Get current tools
                current_tools = self.agent.tools if hasattr(self.agent, 'tools') else []
                
                # Add our new tool
                all_tools = list(current_tools) + [search_locanto]
                
                # Update the agent's tools
                await self.agent.update_tools(all_tools)
                
                logger.info("Added Locanto Search tool to agent")
                # Notify console
                await self.update_agent_instructions(f"Tool added: search_locanto - Search for listings on Locanto")
            except Exception as e:
                logger.error(f"Error adding Locanto Search tool: {e}")
                logger.warning("Could not add Locanto Search tool to agent")
            
            self.job_search_tools.append(search_locanto)
            
            logger.info("Registered Locanto search tool")
        except ImportError:
            logger.warning("Locanto search not available")
            
    async def load_enhanced_search(self):
        """Load enhanced search with RAG if enabled."""
        logger.info("Loading enhanced search with RAG")
        
        try:
            # Import enhanced search module
            from enhanced_search import setup_enhanced_search, data_manager
            
            # Setup enhanced search
            enhanced_search_tools = await setup_enhanced_search(self.agent)
            
            # Start background embedding if enabled
            if self.has_background_embedding and not self.background_embedding_started:
                # Start background embedding process
                logger.info("Starting background embedding process")
                asyncio.create_task(data_manager.start_background_embedding())
                self.background_embedding_started = True
            
            # Update instructions to reflect enhanced search
            if self.has_enhanced_search:
                await self.update_agent_instructions("Enhanced search with RAG loaded: You can now search through local knowledge base")
            else:
                await self.update_agent_instructions("Enhanced search with RAG is not enabled")
            
            # Mark enhanced search as loaded
            self.enhanced_search_loaded = True
            self.tool_sources.append("Enhanced Search with RAG")
            
            logger.info("Enhanced search with RAG loaded successfully")
        except ImportError:
            logger.warning("Enhanced search with RAG not available")
        
        # Load MCP tools next if enabled
        if self.has_mcp_client:
            asyncio.create_task(self.load_mcp_tools())
        else:
            # Finalize the agent configuration
            await self.finalize_agent_configuration()
        
    async def load_mcp_tools(self):
        """Load MCP tools after local tools are loaded."""
        logger.info("Loading MCP tools")
        
        try:
            # Import MCP modules
            from mcp_client import MCPServerSse, MCPToolsIntegration
            
            # Get Zapier MCP URL from environment
            zapier_mcp_url = os.environ.get("ZAPIER_MCP_URL")
            
            if zapier_mcp_url:
                try:
                    # Initialize MCP server connection
                    mcp_server = MCPServerSse(
                        params={"url": zapier_mcp_url},
                        cache_tools_list=True,
                        name="Zapier MCP Server"
                    )
                    
                    # Connect to MCP server
                    await mcp_server.connect()
                    
                    # Get available tools
                    available_tools = await mcp_server.list_tools()
                    logger.info(f"Found {len(available_tools)} tools on Zapier MCP server")
                    
                    if available_tools:
                        # Register MCP tools with the agent
                        mcp_tools = await MCPToolsIntegration.register_with_agent(
                            agent=self.agent,
                            mcp_servers=[mcp_server],
                            convert_schemas_to_strict=True,
                            auto_connect=False
                        )
                        
                        # Track tool sources
                        self.tool_sources.append("Zapier MCP")
                        
                        # Mark MCP tools as loaded
                        self.mcp_tools_loaded = True
                        
                        # Log registered tools
                        tool_names = [getattr(t, '__name__', str(t)) for t in mcp_tools]
                        logger.info(f"Registered MCP tool names: {', '.join(tool_names)}")
                    else:
                        logger.warning("No tools found on Zapier MCP server")
                except Exception as e:
                    logger.error(f"Failed to register MCP tools: {e}")
                    logger.warning("MCP tools will not be available")
            else:
                logger.warning("ZAPIER_MCP_URL not found in environment variables")
        except ImportError:
            logger.warning("MCP client modules not available")
        
        # Update agent instructions with all tools
        await self.update_agent_instructions("MCP tools loaded")
        
        # Finalize the agent configuration
        await self.finalize_agent_configuration()
        
    async def update_agent_instructions(self, status_message):
        """
        Update the agent's instructions to reflect current capabilities.
        
        Args:
            status_message: A message indicating what has been loaded
        """
        # Log the status message
        logger.info(f"Agent status update: {status_message}")
        
        # Send a message to the console to indicate tool availability
        # We can't directly send messages to the console, so we'll just log it
        # The user will see the loaded tools in the log output
        
        # Update the agent's instructions with the current capabilities
        current_date = datetime.now().strftime("%Y-%m-%d")
        tool_sources_str = ", ".join(self.tool_sources)
        
        updated_instructions = f"""
            You are Amanda, an advanced AI assistant.
            Your primary goal is to be helpful, informative, and efficient in your responses.
            
            It is now {current_date}.
            
            Available tool sources: {tool_sources_str}
            
            Status: {status_message}
        """
        
        # Update the agent's instructions
        try:
            await self.agent.update_instructions(updated_instructions)
        except Exception as e:
            logger.error(f"Error updating agent instructions: {e}")
        
        # Build the full instructions
        instructions = f"""
            You are Amanda, an advanced AI assistant with access to a comprehensive set of tools and capabilities.
            Your primary goal is to be helpful, informative, and efficient in your responses.
            
            It is now {current_date}.
        """
        
        # Add Brave Search info if available
        if self.has_brave_search:
            instructions += """
            You are using the optimized Brave Search API for web searches, which is highly efficient with caching.
            """
        
        # Add search instructions if search tools are loaded
        if self.search_tools_loaded:
            instructions += """
            IMPORTANT: When asked to search for information or look something up, ALWAYS use the web_search tool by default.
            The web_search tool automatically prioritizes Brave Search for the best results.
            """
            
            # Add provider-specific search tools info
            available_search_providers = []
            if self.has_brave_search:
                available_search_providers.append("brave_search: Uses Brave Search API specifically (preferred for most searches)")
            if self.has_bing_search:
                available_search_providers.append("bing_search: Uses Bing search specifically")
            if self.has_duckduckgo:
                available_search_providers.append("duckduckgo_search: Uses DuckDuckGo search specifically")
            if self.has_google_search:
                available_search_providers.append("google_search: Uses Google search specifically (use only when explicitly requested)")
            
            if available_search_providers:
                instructions += """
                You also have access to provider-specific search tools that can be used when explicitly requested:
                - """
                instructions += "\n                - ".join(available_search_providers)
        
        # Add tool sources info
        if self.tool_sources:
            instructions += f"""
            
            You have access to tools from the following sources: {tool_sources_str}
            """
        
        # Add information about available tools based on what's loaded
        available_tools = []
        
        if self.basic_tools_loaded:
            available_tools.append("Time and date tools (get_current_time, get_current_date)")
        
        if self.search_tools_loaded:
            available_tools.append("Web search tools (web_search, brave_search, bing_search, duckduckgo_search, google_search, wikipedia_search)")
        
        if self.has_openweather:
            available_tools.append("Weather tools (get_weather)")
        
        if len(self.job_search_tools) > 0:
            job_tools = []
            if self.has_indeed:
                job_tools.append("indeed_job_search")
            if self.has_locanto:
                job_tools.append("search_locanto")
            if job_tools:
                available_tools.append(f"Job search tools ({', '.join(job_tools)})")
        
        if self.enhanced_search_loaded:
            available_tools.append("Enhanced search with RAG for local documents")
        
        if available_tools:
            instructions += """
            
            Available tools include:
            - """
            instructions += "\n            - ".join(available_tools)
        
        if self.mcp_tools_loaded:
            instructions += """
            
            External MCP tools are now available for use.
            """
        
        # Add status message if provided
        if status_message:
            instructions += f"""
            
            SYSTEM UPDATE: {status_message}
            """
        
        await self.agent.update_instructions(instructions)
        logger.info(f"Updated agent instructions: {status_message if status_message else 'General update'}")
        
    async def finalize_agent_configuration(self):
        """Finalize the agent configuration after all tools are loaded."""
        logger.info("Finalizing agent configuration")
        
        # Update instructions with final configuration
        await self.update_agent_instructions("All tools loaded and ready")
        
        # Log completion of startup sequence
        logger.info(f"Agent startup sequence completed with tools from: {', '.join(self.tool_sources)}")
        
        # Log status of background embedding if applicable
        if self.has_background_embedding and self.background_embedding_started:
            logger.info("Background embedding process is running")
        
        # Log any tools that failed to load
        failed_tools = []
        if self.has_brave_search and "Brave Search" not in str(self.web_search_tools):
            failed_tools.append("Brave Search")
        if self.has_bing_search and "Bing Search" not in str(self.web_search_tools):
            failed_tools.append("Bing Search")
        if self.has_duckduckgo and "DuckDuckGo Search" not in str(self.web_search_tools):
            failed_tools.append("DuckDuckGo Search")
        if self.has_google_search and "Google Search" not in str(self.web_search_tools):
            failed_tools.append("Google Search")
        if self.has_indeed and len([t for t in self.job_search_tools if "indeed" in str(t).lower()]) == 0:
            failed_tools.append("Indeed Job Search")
        if self.has_locanto and len([t for t in self.job_search_tools if "locanto" in str(t).lower()]) == 0:
            failed_tools.append("Locanto Search")
        
        if failed_tools:
            logger.warning(f"The following tools failed to load: {', '.join(failed_tools)}")
        
        # Log completion message
        logger.info("Agent is fully operational with all available tools loaded")

# Function to use in the agent's entrypoint
async def optimized_entrypoint(ctx: JobContext):
    """
    Optimized entrypoint function for the LiveKit Amanda agent.
    
    Args:
        ctx: The job context for the agent
    """
    # Create the agent with minimal initial instructions
    current_date = datetime.now().strftime("%Y-%m-%d")
    initial_instructions = f"""
        You are Amanda, an advanced AI assistant.
        Your primary goal is to be helpful, informative, and efficient in your responses.
        
        It is now {current_date}.
        
        I'm currently loading tools and capabilities. I'll be fully operational shortly.
        I can already respond to general questions and requests.
    """
    
    # Import necessary components for voice capabilities
    from livekit.plugins import azure, openai, silero
    
    # Create the agent with voice capabilities
    agent = Agent(
        instructions=initial_instructions,
        stt=azure.STT(
            speech_key=os.environ["AZURE_STT_API_KEY"],
            speech_region=os.environ["AZURE_STT_REGION"]
        ),
        llm=openai.LLM.with_azure(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.environ["AZURE_OPENAI_VERSION"],
            temperature=0.3,  # Lower temperature for more focused and consistent responses
            parallel_tool_calls=True,
        ),
        tts=azure.TTS(
            speech_key=os.environ["AZURE_TTS_API_KEY"],
            speech_region=os.environ["AZURE_TTS_REGION"]
        ),
        vad=silero.VAD.load(
            activation_threshold=0.7,  # Slightly lower threshold for better voice detection
            min_speech_duration=0.2,  # Shorter minimum speech duration (in seconds)
            min_silence_duration=0.2,  # Shorter silence duration for quicker response (in seconds)
            sample_rate=16000,  # Explicitly set sample rate
            force_cpu=True,  # Use CPU for better stability
            prefix_padding_duration=0.3  # Add small padding before speech
        ),
        allow_interruptions=True
    )
    
    # Create the optimizer
    optimizer = AgentStartupOptimizer()
    
    # Initialize the agent with optimized startup sequence
    await optimizer.initialize_agent(agent, ctx)
    
    # The agent is now responsive and will load additional tools in the background
    logger.info("Agent is now responsive and loading additional tools in the background")
    
    # Return the agent to prevent it from being garbage collected
    return agent
