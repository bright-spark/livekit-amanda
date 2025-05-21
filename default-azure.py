from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

import os
import asyncio
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero, openai, azure

class AIVoiceAssistant:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIVoiceAssistant, cls).__new__(cls)
            cls._instance.vad = None
            cls._instance.assistant = None
        return cls._instance

    def initialize_vad(self, proc: JobProcess):
        """Initialize Voice Activity Detection"""
        if self.vad is None:
            proc.userdata["vad"] = silero.VAD.load()
            self.vad = proc.userdata["vad"]

    def create_initial_context(self):
        """Create initial chat context"""
        return ChatContext(
            messages=[
                ChatMessage(
                    role="system",
                    content=("You are a voice assistant. Pretend we're having a human conversation, no special formatting or headings, just natural speech."),
                )
            ]
        )

    def setup_assistant(self, vad):
        """Setup voice assistant with all required components"""
        if self.assistant is None:
            self.assistant = VoiceAssistant(
                vad=vad,
                stt=azure.STT(
                    speech_key=os.environ["AZURE_STT_API_KEY"],
                    speech_region=os.environ["AZURE_STT_REGION"]
                ),
                llm=openai.LLM.with_azure(
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                    api_version=os.environ["AZURE_OPENAI_VERSION"],
                    temperature=0.7,
                ),
                tts=azure.TTS(
                    speech_key=os.environ["AZURE_TTS_API_KEY"],
                    speech_region=os.environ["AZURE_TTS_REGION"]
                ),
                chat_ctx=self.create_initial_context(),
            )
        return self.assistant

def prewarm(proc: JobProcess):
    """Prewarm function to initialize VAD"""
    assistant = AIVoiceAssistant()
    assistant.initialize_vad(proc)

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice assistant"""
    assistant = AIVoiceAssistant()
    voice_assistant = assistant.setup_assistant(ctx.proc.userdata["vad"])

    await ctx.connect()
    voice_assistant.start(ctx.room)
    await asyncio.sleep(1)
    await voice_assistant.say("Hi there, how are you doing today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
