# LiveKit Agent with MCP Tools

A voice assistant application built using the LiveKit Agents framework, capable of using Multimodal Control Protocol (MCP) tools to interact with external services.

## Medium Article

- [Integrating Zapier MCP with your AI Assistant](https://xthemadgenius.medium.com/integrating-zapier-mcp-with-your-ai-assistant-38e081e3a5b7)

## Features

- Voice-based interaction with a helpful AI assistant
- Integration with MCP tools from external servers
- Speech-to-text using Azure
- Natural language processing using Azure OpenAI
- Text-to-speech using Azure
- Voice activity detection using Silero

## Prerequisites

- Python 3.9+
- API keys for Azure OpenAI and Azure Speech
- MCP server endpoint

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/livekit-examples/basic-mcp.git
   cd basic-mcp
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys and configuration:
   ```
   ZAPIER_MCP_URL=your_mcp_server_url
   AZURE_OPENAI_VERSION=your_azure_openai_version
   AZURE_OPENAI_DEPLOYMENT=your_azure_openai_deployment
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_STT_REGION=your_azure_stt_region
   AZURE_STT_API_KEY=your_azure_stt_api_key
   AZURE_TTS_REGION=your_azure_tts_region
   AZURE_TTS_API_KEY=your_azure_tts_api_key
   ```

## Usage

Run the agent with the LiveKit CLI:

```
python agent.py console
```

The agent will connect to the specified LiveKit room and start listening for voice commands.

## Project Structure

- `agent.py`: Main agent implementation and entrypoint
- `mcp_client/`: Package for MCP server integration
  - `server.py`: MCP server connection handlers
  - `agent_tools.py`: Integration of MCP tools with LiveKit agents
  - `util.py`: Utility functions for MCP client

## Acknowledgements

- [LiveKit](https://livekit.io/) for the underlying real-time communication infrastructure
- [Azure](https://azure.microsoft.com/) for openai, speech-to-text and text-to-speech
- [Silero](https://github.com/snakers4/silero-vad) for Voice Activity Detection