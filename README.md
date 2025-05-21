# LiveKit Amanda - Advanced Voice Assistant & API

An advanced voice assistant application built using the LiveKit Agents framework, capable of using Multimodal Control Protocol (MCP) tools to interact with external services. The assistant includes specialized functionality for web scraping, job search capabilities, and is now available as a FastAPI web service.

## Medium Article

- [Integrating Zapier MCP with your AI Assistant](https://xthemadgenius.medium.com/integrating-zapier-mcp-with-your-ai-assistant-38e081e3a5b7)

## Features

- Voice-based interaction with a helpful AI assistant
- Integration with MCP tools from external servers
- Web scraping capabilities with proxy support for Locanto and other sites
- Job search capabilities with Indeed integration
- Speech-to-text using Azure
- Natural language processing primarily using Azure OpenAI (with optional support for other LLM providers)
- Text-to-speech using Azure
- Voice activity detection using Silero
- Configurable proxy support for web scraping
- REST API with FastAPI for programmatic access to all features

## API Service

The project now includes a FastAPI web service that exposes the following endpoints:

- `/jobs/locanto`: Search for job listings on Locanto
- `/jobs/indeed`: Search for job listings on Indeed
- `/weather`: Get weather information for a location
- `/proxy`: Proxy a URL through the configured proxy service

The API includes automatic Swagger documentation at `/docs` and ReDoc at `/redoc`.

## Prerequisites

- Python 3.9+
- API keys for Azure OpenAI
- API keys for Azure Speech Services (STT and TTS)
- Proxy service for web scraping (configurable via environment variables)
- LiveKit API credentials
- Optional: API keys for additional LLM providers (Cerebras, RedBuilder, Ollama)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/livekit-amanda.git
   cd livekit-amanda
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys and configuration (see `.env.example` for a template):
   ```
   # LiveKit Configuration
   LIVEKIT_API_KEY="your_livekit_api_key"
   LIVEKIT_API_SECRET="your_livekit_api_secret"
   LIVEKIT_URL="your_livekit_url"
   
   # Azure OpenAI Configuration
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_DEPLOYMENT=your_azure_openai_deployment
   AZURE_OPENAI_VERSION=your_azure_openai_version
   
   # Azure Speech Services
   AZURE_STT_API_KEY=your_azure_stt_api_key
   AZURE_STT_REGION=your_azure_stt_region
   AZURE_TTS_API_KEY=your_azure_tts_api_key
   AZURE_TTS_REGION=your_azure_tts_region
   
   # Proxy Configuration
   PROXY_URL=your_proxy_url
   
   # OpenWeather Configuration
   OPENWEATHER_API_KEY=your_openweather_api_key
   
   # Zapier MCP Configuration
   ZAPIER_MCP_URL=your_zapier_mcp_url
   
   # Optional: Additional LLM Providers
   # Uncomment and configure as needed
   
   # RedBuilder Configuration
   #REDBUILDER_KEY=your_redbuilder_key
   #REDBUILDER_ENDPOINT=your_redbuilder_endpoint
   #REDBUILDER_VERSION=your_redbuilder_version
   #REDBUILDER_DEPLOYMENT=your_redbuilder_deployment
   
   # Ollama Configuration
   #OLLAMA_BASE_URL=your_ollama_base_url
   #OLLAMA_MODEL=your_ollama_model
   
   # Cerebras Configuration
   #CEREBRAS_API_KEY=your_cerebras_api_key
   #CEREBRAS_BASE_URL=your_cerebras_base_url
   
   # Google Configuration
   #GOOGLE_API_KEY=your_google_api_key
   #GOOGLE_APPLICATION_CREDENTIALS=your_google_application_credentials
   
   # OpenAI Configuration
   #OPENAI_API_KEY=your_openai_api_key
   #OPENAI_BASE_URL=your_openai_base_url
   ```

## Usage

### Running the Voice Assistant

To start the voice assistant in console mode:

```bash
python agent.py console
```

This will start the voice assistant in your terminal with audio input/output capabilities.

### Web Scraping with Proxy

The application uses a configurable proxy for web scraping to avoid rate limiting and IP blocking. The proxy URL is set via the `PROXY_URL` environment variable.

- Default proxy: `https://please.untaint.us/?url=`
- You can change this by modifying the `PROXY_URL` value in your `.env` file

Example of how the proxy is used in the code:

```python
# The proxy URL is loaded from environment variables
PROXY_PREFIX = os.environ.get("PROXY_URL", "https://please.untaint.us/?url=")

# When making web requests, URLs are processed through the proxy
proxied_url = f"{PROXY_PREFIX}{original_url}"
```

The proxy configuration is used primarily in the `locanto_browser_scraper.py` file for web scraping operations.

### Available Commands

The voice assistant supports various commands, including:

- Web search queries
- Job searches via Indeed and Locanto
- General knowledge questions
- Time and weather information

### Running the FastAPI Application

To run the FastAPI application locally:

```bash
uvicorn app:app --reload
```

This will start the API server on http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

#### Using Docker

You can also run the application using Docker:

```bash
# Build the Docker image
docker build -t livekit-amanda .

# Run the container
docker run -p 8000:8000 --env-file .env livekit-amanda
```

Or using Docker Compose:

```bash
docker-compose up -d
```

#### API Examples

**Search for jobs on Locanto:**

```bash
curl -X POST http://localhost:8000/jobs/locanto \
  -H "Content-Type: application/json" \
  -d '{"query":"software developer", "location":"New York", "max_pages":1}'
```

**Get weather information:**

```bash
curl -X POST http://localhost:8000/weather \
  -H "Content-Type: application/json" \
  -d '{"location":"London"}'
```

## Troubleshooting

### Proxy Issues

If you encounter issues with the proxy:

1. Check that your `PROXY_URL` is correctly set in the `.env` file
2. Verify that the proxy service is operational
3. Try an alternative proxy service by updating the `PROXY_URL` value

### API Connection Errors

If you see API connection errors:

1. Verify that all API keys in your `.env` file are correct and active
2. Check your internet connection
3. Ensure the API services you're using are operational

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Run the agent with the LiveKit CLI:

```
python agent.py console
```

The agent will connect to the specified LiveKit room and start listening for voice commands.

## Project Structure

- `agent.py`: Main agent implementation and entrypoint
- `app.py`: FastAPI application for REST API
- `Dockerfile`: Docker configuration for containerization
- `docker-compose.yml`: Docker Compose configuration for running the application
- `locanto_browser_scraper.py`: Web scraper for Locanto job listings
- `indeed.py`: Web scraper for Indeed job listings
- `tools.py`: Utility functions and tools used by the agent
- `mcp_client/`: Package for MCP server integration
  - `server.py`: MCP server connection handlers
  - `agent_tools.py`: Integration of MCP tools with LiveKit agents
  - `util.py`: Utility functions for MCP client

## Acknowledgements

- [LiveKit](https://livekit.io/) for the underlying real-time communication infrastructure
- [Azure](https://azure.microsoft.com/) for openai, speech-to-text and text-to-speech
- [Silero](https://github.com/snakers4/silero-vad) for Voice Activity Detection