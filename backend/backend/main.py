from __future__ import annotations
import os, uuid, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Disable LiteLLM logging to avoid event loop conflicts
os.environ["LITELLM_LOG"] = "error"
os.environ["LITELLM_DISABLE_LOGGING"] = "true"
os.environ["LITELLM_LOG_LEVEL"] = "ERROR"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "true"
os.environ["LITELLM_DISABLE_STREAMING_LOGGING"] = "true"
os.environ["LITELLM_TURN_OFF_MESSAGE_LOGGING"] = "true"
from .agui_protocol import (
    RunAgentInput, EventEncoder, run_started, run_finished, text_start, text_delta, text_end
)
from .orchestrator import run_sequential, run_collab
from .agents import web_researcher, technical_writer

import httpx

async def geocode_city(city: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            return None
        res = data["results"][0]
        return {"name": res["name"], "lat": res["latitude"], "lon": res["longitude"], "country": res.get("country")}

async def fetch_weather(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        cw = data.get("current_weather", {})
        return {
            "temperature": cw.get("temperature"),
            "windspeed": cw.get("windspeed"),
            "winddirection": cw.get("winddirection"),
            "weathercode": cw.get("weathercode"),
            "time": cw.get("time"),
        }


app = FastAPI(title="ADK × LiteLLM × AG‑UI Demo")

# Add CORS middleware to handle preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunBody(BaseModel):
    prompt: str

@app.post("/api/run/sequential")
async def api_run_seq(body: RunBody):
    data = await run_sequential(body.prompt)
    return JSONResponse(data)

@app.post("/api/run/collab")
async def api_run_col(body: RunBody):
    data = await run_collab(body.prompt)
    return JSONResponse(data)


@app.post("/api/agui/run")
async def agui_run(request: Request):
    payload = await request.json()
    if "thread_id" in payload and "run_id" in payload:
        input_data = RunAgentInput(**payload)
        prompt = " ".join([m.get("content","") for m in input_data.messages if m.get("role") in ("user","system")])
        thread_id = input_data.thread_id
        run_id = input_data.run_id
    else:
        prompt = payload.get("prompt", "")
        thread_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())

    accept = request.headers.get("accept", "")
    encoder = EventEncoder(accept=accept)

    async def gen():
        # lifecycle start
        yield encoder.encode(run_started(thread_id, run_id))

        # --- (Optional) Weather Tool: detect "weather in <city>" ---
        import re
        m = re.search(r"weather\s+in\s+([A-Za-z\-\s]+)", prompt, re.I)
        if m:
            city = m.group(1).strip()
            # TOOL_CALL event
            yield encoder.encode({
                "type": "TOOL_CALL",
                "tool": "WeatherTool",
                "args": {"city": city}
            })
            # perform tool
            geo = await geocode_city(city)
            if geo:
                wx = await fetch_weather(geo["lat"], geo["lon"])
                card = {
                    "type": "WEATHER_CARD",
                    "data": {
                        "location": f'{geo["name"]}, {geo.get("country","")}',
                        **wx
                    }
                }
                # TOOL_RESULT + WEATHER_CARD
                yield encoder.encode({"type": "TOOL_RESULT", "tool": "WeatherTool", "ok": True, "data": card["data"]})
                yield encoder.encode(card)
            else:
                yield encoder.encode({"type": "TOOL_RESULT", "tool": "WeatherTool", "ok": False, "error": f"City not found: {city}"})

        # --- Sequential: Use the proper SequentialAgent execution ---
        # This will run the agents in sequence with proper state passing
        msg_seq = str(uuid.uuid4())
        yield encoder.encode(text_start(msg_seq, agent_name="🔄 Sequential Pipeline"))
        
        # Use the sequential orchestrator which handles state passing between agents
        seq_result = await run_sequential(prompt)
        yield encoder.encode(text_delta(msg_seq, seq_result["output"]))
        yield encoder.encode(text_end(msg_seq))

        # --- Collaboration: orchestrator improves/merges ---
        msg3 = str(uuid.uuid4())
        yield encoder.encode(text_start(msg3, agent_name="🤝 Collaboration Orchestrator"))
        # Use the collab orchestrator via our wrapper to refine final output
        col = await run_collab(prompt + " Improve the result and deduplicate. Keep the best of both agents.")
        yield encoder.encode(text_delta(msg3, col["output"]))
        yield encoder.encode(text_end(msg3))

        # lifecycle end
        yield encoder.encode(run_finished(thread_id, run_id))

    return StreamingResponse(gen(), media_type=encoder.get_content_type())
