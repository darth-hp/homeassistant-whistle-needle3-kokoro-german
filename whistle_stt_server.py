#!/usr/bin/env python3
import asyncio
import logging
import os
import numpy as np
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.event import Event
from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioStop
from wyoming.info import Describe, Info, AsrModel, AsrProgram, Attribution
import needle

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("whistle_wyoming")

SERVER_PORT = 10300
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_FILE = os.path.join(SCRIPT_DIR, "whistle_stt_keywords.txt")

def load_keywords(file_path: str) -> list[str]:
    """Loads keywords line-by-line from an optional text file."""
    if not os.path.isfile(file_path):
        return []
    keywords = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("#"):
                    keywords.append(clean_line)
    except Exception as err:
        _LOGGER.error(f"Error loading keywords from '{file_path}': {err}")
    return keywords

class WhistleHandler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_buffer = bytearray()
        self.rate = 16000
        self.width = 2
        self.channels = 1
        self.language = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            info = Info(
                asr=[
                    AsrProgram(
                        name="Cactus Whistle STT",
                        description="Speech-to-Text",
                        attribution=Attribution(name="Cactus Compute", url="https://explainx.ai"),
                        installed=True,
                        version="1.0.0",
                        models=[
                            AsrModel(
                                name="whistle",
                                description="Whistle Multilingual",
                                attribution=Attribution(name="Cactus Compute", url="https://explainx.ai"),
                                installed=True,
                                languages=["de", "en", "fr", "es", "it", "nl", "pl"],
                                version="1.0.0",
                            )
                        ]
                    )
                ]
            )
            await self.write_event(info.event())
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.language = transcribe.language
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.rate = chunk.rate
            self.width = chunk.width
            self.channels = chunk.channels
            self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            if self.audio_buffer:
                raw_bytes = bytes(self.audio_buffer)
                self.audio_buffer = bytearray()

                raw_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                audio_float32 = raw_int16.astype(np.float32) / 32768.0

                text = ""
                try:
                    current_keywords = load_keywords(KEYWORDS_FILE)
                    kw_arg = current_keywords if current_keywords else None

                    res = needle.transcribe(
                        audio_float32,
                        language=self.language,
                        keywords=kw_arg
                    )
                    text = res.get("text", "").strip()
                    tokens_per_sec = res.get("tokens_per_second", 0)
                    _LOGGER.info(f"Transcribed [{self.language}]: '{text}' ({tokens_per_sec:.1f} tok/s)")
                except Exception as err:
                    _LOGGER.error(f"Inference error: {err}")

                await self.write_event(Transcript(text=text).event())
            else:
                await self.write_event(Transcript(text="").event())
            return False

        return True

async def main():
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{SERVER_PORT}")
    _LOGGER.info(f"Whistle Wyoming server listening on port {SERVER_PORT}...")
    await server.run(WhistleHandler)

if __name__ == "__main__":
    asyncio.run(main())
