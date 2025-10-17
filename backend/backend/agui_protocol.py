from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

# Minimal subset of AG‑UI event types over SSE, inspired by:
# https://docs.ag-ui.com/quickstart/server

class EventType:
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    WEATHER_CARD = "WEATHER_CARD"
    RESEARCH_CARD = "RESEARCH_CARD"
    TECHNICAL_CARD = "TECHNICAL_CARD"

@dataclass
class RunAgentInput:
    thread_id: str
    run_id: str
    messages: list[Dict[str, Any]]
    tools: Optional[list[Dict[str, Any]]] = None
    # Optional custom metadata
    metadata: Optional[Dict[str, Any]] = None

class EventEncoder:
    """Encodes dict events into SSE lines, or JSON ND for Accept: application/json."""
    def __init__(self, accept: Optional[str] = None):
        self.accept = (accept or "").lower()

    def get_content_type(self) -> str:
        if "application/json" in self.accept:
            return "application/json"
        return "text/event-stream"

    def encode(self, event: Dict[str, Any]) -> str | bytes:
        if self.get_content_type() == "application/json":
            return json.dumps(event) + "\n"
        # SSE format
        return f"event: {event.get('type')}\ndata: {json.dumps(event)}\n\n"

def run_started(thread_id: str, run_id: str) -> Dict[str, Any]:
    return {"type": EventType.RUN_STARTED, "thread_id": thread_id, "run_id": run_id}

def run_finished(thread_id: str, run_id: str) -> Dict[str, Any]:
    return {"type": EventType.RUN_FINISHED, "thread_id": thread_id, "run_id": run_id}

def text_start(message_id: str, agent_name: str | None = None) -> Dict[str, Any]:
    ev = {"type": EventType.TEXT_MESSAGE_START, "message_id": message_id, "role": "assistant"}
    if agent_name:
        ev["agent_name"] = agent_name
    return ev

def text_delta(message_id: str, delta: str) -> Dict[str, Any]:
    return {"type": EventType.TEXT_MESSAGE_CONTENT, "message_id": message_id, "delta": delta}

def text_end(message_id: str) -> Dict[str, Any]:
    return {"type": EventType.TEXT_MESSAGE_END, "message_id": message_id}
