#!/usr/bin/env python3
import json
import logging
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
import needle
from text_to_num import alpha2digit
import ha_client

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("needle_bridge")

app = FastAPI(title="Needle Intent Bridge")

# Regular expression mappings to merge split spoken compound words
COMPOUND_REPLACEMENTS = {
    r"\bbürger\s+rollo\b": "Bürorollo",
    r"\bbüro\s+rollo\b": "Bürorollo",
    r"\bbüro\s+rolo\b": "Bürorollo",
    r"\bbüro\s+licht\b": "Bürolicht",
    r"\bküche\s+licht\b": "Küchenlicht",
    r"\bküchen\s+rollo\b": "Küchenrollo",
    r"\besszimmer\s+licht\b": "Esszimmerlicht",
    r"\bwohnzimmer\s+licht\b": "Wohnzimmerlicht",
}

def detect_language(text: str) -> str:
    """Detect German vs English prompt based on indicator words."""
    text_lower = text.lower()
    english_markers = ["turn", "set", "switch", "open", "close", "light", "the", "to", "what", "is", "please"]
    german_markers = ["schalte", "setze", "stelle", "mach", "öffne", "schließe", "das", "die", "der", "auf", "im"]

    en_score = sum(1 for m in english_markers if f" {m} " in f" {text_lower} ")
    de_score = sum(1 for m in german_markers if f" {m} " in f" {text_lower} ")

    return "en" if en_score > de_score else "de"

def clean_input_text(text: str, lang: str = "de") -> str:
    """Normalize written number words and merge split compound device names."""
    try:
        target_lang = lang if lang in ["de", "en", "fr", "es", "it"] else "de"
        normalized = alpha2digit(text, lang=target_lang)
    except Exception as err:
        _LOGGER.warning(f"Failed to normalize number words: {err}")
        normalized = text

    for pattern, replacement in COMPOUND_REPLACEMENTS.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    return normalized

# Symmetric tool schema using 'percentage' for both lights and covers
TOOLS_SCHEMA = [
    {
        "name": "control_light",
        "description": "Control lights, lamps, LEDs. Turn on, turn off, dim, or set brightness percentage level.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "The exact light or lamp name extracted directly from user speech."
                },
                "state": {
                    "type": "string",
                    "enum": ["on", "off"],
                    "description": "Target power state. Must be 'on' when setting percentage."
                },
                "percentage": {
                    "type": "integer",
                    "description": "Brightness percentage from 1 to 100."
                }
            },
            "required": ["device", "state"]
        }
    },
    {
        "name": "control_cover",
        "description": "Control window blinds, shutters, roller covers (Rollo, Rolladen). Open, close, stop, or set position percentage.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Full target cover or blind name extracted directly from user speech."
                },
                "action": {
                    "type": "string",
                    "enum": ["open", "close", "stop"],
                    "description": "Movement action."
                },
                "percentage": {
                    "type": "integer",
                    "description": "Position percentage from 0 to 100."
                }
            },
            "required": ["device"]
        }
    },
    {
        "name": "control_climate",
        "description": "Control heating, AC, cooling, or thermostat target temperature.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "The target thermostat or climate device name extracted directly from user speech."
                },
                "temperature": {
                    "type": "number",
                    "description": "Target temperature in degrees Celsius."
                }
            },
            "required": ["device", "temperature"]
        }
    },
    {
        "name": "get_time",
        "description": "Query the current time, clock, or date.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

NEEDLE_MODEL = needle.Needle(tools=TOOLS_SCHEMA, stateless=True)

class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = ""
    model_config = ConfigDict(extra="allow")

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "needle3"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    model_config = ConfigDict(extra="allow")

def process_prompt(text: str) -> str:
    lang = detect_language(text)
    normalized_text = clean_input_text(text, lang=lang)
    _LOGGER.info(f"Incoming prompt [{lang}]: '{text}' -> Normalized: '{normalized_text}'")

    res = NEEDLE_MODEL.run(query=normalized_text)
    calls = res.get("function_calls", [])

    if not calls:
        return ha_client.get_template("not_understood", lang)

    responses = []
    for c in calls:
        tool_name = c["name"]
        args = c["arguments"]
        _LOGGER.info(f"Executing tool call: {tool_name} with arguments: {args}")

        if tool_name == "control_light":
            device_name = args.get("device", "")
            resp_text = ha_client.dispatch_device_action(device_name, "light", args, lang=lang)
            responses.append(resp_text)

        elif tool_name == "control_cover":
            device_name = args.get("device", "")
            resp_text = ha_client.dispatch_device_action(device_name, "cover", args, lang=lang)
            responses.append(resp_text)

        elif tool_name == "control_climate":
            device_name = args.get("device", "")
            resp_text = ha_client.dispatch_device_action(device_name, "climate", args, lang=lang)
            responses.append(resp_text)

        elif tool_name == "get_time":
            tz_name = os.environ.get("TZ", "Europe/Berlin")
            try:
                now = datetime.now(ZoneInfo(tz_name))
            except Exception:
                now = datetime.now()

            hours_str = str(now.hour)
            minutes_str = f"{now.minute:02d}"

            if now.minute == 0:
                tmpl = ha_client.get_template("time_reply_exact", lang)
                if not tmpl:
                    tmpl = ha_client.get_template("time_reply", lang)
                responses.append(tmpl.format(hours=hours_str, minutes=minutes_str))
            else:
                tmpl = ha_client.get_template("time_reply", lang)
                responses.append(tmpl.format(hours=hours_str, minutes=minutes_str))

    return " ".join(responses)

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "needle3", "object": "model", "created": int(time.time()), "owned_by": "local"}
        ]
    }

async def handle_completion(req: ChatCompletionRequest):
    user_prompt = ""
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content:
            user_prompt = str(msg.content)
            break

    reply = process_prompt(user_prompt)

    if req.stream:
        async def event_generator():
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]
    }

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/v1/v1/chat/completions")
async def chat_endpoint(req: ChatCompletionRequest):
    return await handle_completion(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10400)
