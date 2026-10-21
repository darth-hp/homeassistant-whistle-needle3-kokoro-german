#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import numpy as np

# Suppress non-fatal ONNX affinity warnings on C level
os.environ["ORT_LOGGING_LEVEL"] = "3"

from kokoro_onnx import Kokoro
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kokoro_tts_config.json")
ENV_FILE = os.path.join(SCRIPT_DIR, ".env")

def load_env(env_path: str):
    """Load key-value pairs from a .env file into os.environ."""
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line and not clean_line.startswith("#") and "=" in clean_line:
                key, val = clean_line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("\"'")

load_env(ENV_FILE)

# Add kokoro_german frontend to path
sys.path.append(os.path.join(SCRIPT_DIR, "kokoro_german"))
try:
    from kokoro_german import G2P
    G2P_MODULE = G2P()
except Exception:
    G2P_MODULE = None

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("kokoro_tts")

def load_config() -> dict:
    """Load JSON config with default fallback."""
    if not os.path.isfile(CONFIG_FILE):
        _LOGGER.warning(f"Config '{CONFIG_FILE}' not found. Using default German settings.")
        return {
            "server": {"port": 10200, "default_language": "de"},
            "languages": {
                "de": {
                    "model_path": "models/kokoro/kokoro-martin.onnx",
                    "voices_path": "models/kokoro/voices-martin.npz",
                    "default_voice": "martin",
                    "use_german_g2p": True,
                    "kokoro_lang_code": "de",
                    "ha_voice_description": "German Martin Voice",
                },
                "en": {
                    "model_path": "models/kokoro/kokoro-v1.0.onnx",
                    "voices_path": "models/kokoro/voices-v1.0.bin",
                    "default_voice": "af_bella",
                    "use_german_g2p": False,
                    "kokoro_lang_code": "en-us",
                    "ha_voice_description": "English Bella Voice",
                }
            },
        }
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

TTS_CONFIG = load_config()
SERVER_PORT = TTS_CONFIG.get("server", {}).get("port", 10200)

# Determine active language(s) from .env
ACTIVE_LANGS = [
    lang.strip().lower() for lang in os.environ.get("VOICE_LANG", "de").split(",") if lang.strip()
]
_LOGGER.info(f"Active voice languages configured via .env: {ACTIVE_LANGS}")

# Preload engine instances ONLY for configured active language(s)
ENGINES = {}
for lang, profile in TTS_CONFIG.get("languages", {}).items():
    if lang not in ACTIVE_LANGS:
        continue

    model_abs = os.path.join(SCRIPT_DIR, profile["model_path"])
    voices_abs = os.path.join(SCRIPT_DIR, profile["voices_path"])
    if os.path.isfile(model_abs) and os.path.isfile(voices_abs):
        _LOGGER.info(f"Loading engine for language '{lang}' from {model_abs}...")
        ENGINES[lang] = {
            "engine": Kokoro(model_abs, voices_abs),
            "profile": profile,
        }
    else:
        _LOGGER.warning(f"Assets missing for language '{lang}': {model_abs} or {voices_abs}")

if not ENGINES:
    _LOGGER.critical("No TTS engines could be initialized! Check models and VOICE_LANG setting.")

class KokoroHandler(AsyncEventHandler):
    """Handle Wyoming TTS synthesis requests dynamically based on active languages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            voices_list = []
            for lang, data in ENGINES.items():
                prof = data["profile"]
                voices_list.append(
                    TtsVoice(
                        name=prof["default_voice"],
                        description=prof.get("ha_voice_description", f"{lang} voice"),
                        attribution=Attribution(name="Kokoro", url="https://github.com/hexgrad/kokoro"),
                        installed=True,
                        languages=[lang],
                        version="1.0.0",
                    )
                )

            info = Info(
                tts=[
                    TtsProgram(
                        name="Kokoro-TTS",
                        description="Local natural ONNX TTS service",
                        attribution=Attribution(name="Kokoro", url="https://github.com/hexgrad/kokoro"),
                        installed=True,
                        version="1.0.0",
                        voices=voices_list,
                    )
                ]
            )
            await self.write_event(info.event())
            return True

        if Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            raw_text = synth.text.strip()
            requested_voice = synth.voice.name if synth.voice else ""
            _LOGGER.info(f"Incoming TTS request: '{raw_text}' (requested voice: '{requested_voice}')")

            # Resolve target language based on active engines
            target_lang = ACTIVE_LANGS[0] if ACTIVE_LANGS else "de"
            for lang, data in ENGINES.items():
                if data["profile"]["default_voice"] == requested_voice:
                    target_lang = lang
                    break

            if target_lang not in ENGINES:
                target_lang = list(ENGINES.keys())[0]

            engine_data = ENGINES[target_lang]
            kokoro_inst = engine_data["engine"]
            profile = engine_data["profile"]

            # German G2P frontend if enabled for active profile
            processed_text = raw_text
            if profile.get("use_german_g2p") and G2P_MODULE:
                try:
                    processed_text = G2P_MODULE(raw_text)
                except Exception as err:
                    _LOGGER.warning(f"G2P error: {err}. Using raw text.")

            try:
                samples, sample_rate = kokoro_inst.create(
                    processed_text,
                    voice=profile["default_voice"],
                    speed=1.0,
                    lang=profile.get("kokoro_lang_code", "de"),
                )

                int16_samples = (samples * 32767.0).clip(-32768, 32767).astype(np.int16)
                audio_bytes = int16_samples.tobytes()

                chunk_size = 2048
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await self.write_event(
                        AudioChunk(
                            audio=chunk,
                            rate=sample_rate,
                            width=2,
                            channels=1,
                        ).event()
                    )

            except Exception as err:
                _LOGGER.error(f"TTS inference failure: {err}")

            await self.write_event(AudioStop().event())
            return False

        return True

async def main():
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{SERVER_PORT}")
    _LOGGER.info(f"Kokoro Wyoming TTS server listening on port {SERVER_PORT}...")
    await server.run(KokoroHandler)

if __name__ == "__main__":
    asyncio.run(main())
