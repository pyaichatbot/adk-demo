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
from .orchestrator import run_sequential, run_collab, intelligent_router
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

@app.post("/api/ask")
async def ask_anything(body: RunBody):
    """Unified endpoint that intelligently routes queries"""
    data = await intelligent_router(body.prompt)
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

        # --- Intelligent Router: Analyze user query and route appropriately ---
        router_result = await intelligent_router(prompt)
        route_type = router_result.get("route_type", "GENERAL_ROUTE")
        city = router_result.get("city", "N/A")
        
        
        # Show routing decision
        msg_router = str(uuid.uuid4())
        yield encoder.encode(text_start(msg_router, agent_name="🧠 Intelligent Router"))
        yield encoder.encode(text_delta(msg_router, f"Analyzing your query: '{prompt}'\n\nRouting to: {route_type}"))
        yield encoder.encode(text_end(msg_router))
        
        if route_type == "WEATHER_ROUTE" and city != "N/A":
            # Weather request detected by router
            yield encoder.encode({
                "type": "TOOL_CALL",
                "tool": "WeatherTool",
                "args": {"city": city}
            })
            
            # Perform weather lookup
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
                yield encoder.encode({"type": "TOOL_RESULT", "tool": "WeatherTool", "ok": True, "data": card["data"]})
                yield encoder.encode(card)
                
                # Weather response
                msg_weather = str(uuid.uuid4())
                yield encoder.encode(text_start(msg_weather, agent_name="🌤️ Weather Assistant"))
                weather_response = f"Here's the current weather information for {city}:\n\n"
                weather_response += f"📍 Location: {geo['name']}, {geo.get('country', '')}\n"
                weather_response += f"🌡️ Temperature: {wx.get('temperature', 'N/A')}°C\n"
                weather_response += f"💨 Wind: {wx.get('windspeed', 'N/A')} km/h\n"
                weather_response += f"⏰ Time: {wx.get('time', 'N/A')}\n\n"
                weather_response += "Is there anything specific about the weather you'd like to know more about?"
                yield encoder.encode(text_delta(msg_weather, weather_response))
                yield encoder.encode(text_end(msg_weather))
            else:
                yield encoder.encode({"type": "TOOL_RESULT", "tool": "WeatherTool", "ok": False, "error": f"City not found: {city}"})
                
                msg_error = str(uuid.uuid4())
                yield encoder.encode(text_start(msg_error, agent_name="🌤️ Weather Assistant"))
                yield encoder.encode(text_delta(msg_error, f"Sorry, I couldn't find weather data for {city}. Please try a different city name."))
                yield encoder.encode(text_end(msg_error))
                
        elif route_type == "RESEARCH_ROUTE":
            # Research request - use sequential pipeline (Web Researcher → Technical Writer)
            seq_result = await run_sequential(prompt)
            
            # Emit research summary card if available
            if "research_summary" in seq_result and seq_result["research_summary"]:
                yield encoder.encode({
                    "type": "RESEARCH_CARD",
                    "data": {
                        "content": seq_result["research_summary"],
                        "agent": "Web Researcher"
                    }
                })
            
            # Emit technical writer card if available
            if "technical_summary" in seq_result and seq_result["technical_summary"]:
                yield encoder.encode({
                    "type": "TECHNICAL_CARD", 
                    "data": {
                        "content": seq_result["technical_summary"],
                        "agent": "Technical Writer"
                    }
                })
            
            # Only show final result if we don't have cards, or show a brief summary
            if not (seq_result.get("research_summary") and seq_result.get("technical_summary")):
                # Show final sequential result only if no cards were generated
                msg_seq = str(uuid.uuid4())
                yield encoder.encode(text_start(msg_seq, agent_name="🔄 Sequential Pipeline"))
                yield encoder.encode(text_delta(msg_seq, seq_result["output"]))
                yield encoder.encode(text_end(msg_seq))
            else:
                # Show brief completion message when cards are available
                msg_complete = str(uuid.uuid4())
                yield encoder.encode(text_start(msg_complete, agent_name="✅ Research Complete"))
                yield encoder.encode(text_delta(msg_complete, "Research and analysis completed. See the cards above for detailed findings."))
                yield encoder.encode(text_end(msg_complete))
            
        elif route_type == "COLLABORATION_ROUTE":
            # Collaboration request - use parallel agents for multiple perspectives
            col = await run_collab(prompt)
            
            # Emit collaboration research card if available
            if "collab_research_summary" in col and col["collab_research_summary"]:
                yield encoder.encode({
                    "type": "RESEARCH_CARD",
                    "data": {
                        "content": col["collab_research_summary"],
                        "agent": "Collaboration Researcher"
                    }
                })
            
            # Emit collaboration technical card if available
            if "collab_technical_summary" in col and col["collab_technical_summary"]:
                yield encoder.encode({
                    "type": "TECHNICAL_CARD", 
                    "data": {
                        "content": col["collab_technical_summary"],
                        "agent": "Collaboration Technical Writer"
                    }
                })
            
            # Show final collaboration result
            if not (col.get("collab_research_summary") and col.get("collab_technical_summary")):
                # Show full output if no cards
                msg_collab = str(uuid.uuid4())
                yield encoder.encode(text_start(msg_collab, agent_name="🤝 Collaboration Orchestrator"))
                yield encoder.encode(text_delta(msg_collab, col["output"]))
                yield encoder.encode(text_end(msg_collab))
            else:
                # Show brief completion message when cards are available
                msg_complete = str(uuid.uuid4())
                yield encoder.encode(text_start(msg_complete, agent_name="✅ Collaboration Complete"))
                yield encoder.encode(text_delta(msg_complete, "Multi-perspective analysis completed. See the cards above for detailed insights from different agents."))
                yield encoder.encode(text_end(msg_complete))
            
        else:
            # General query - use a simple LLM response
            msg_general = str(uuid.uuid4())
            yield encoder.encode(text_start(msg_general, agent_name="💬 General Assistant"))
            
            # For now, provide a simple response for general queries
            general_response = f"I understand you're asking: '{prompt}'\n\n"
            general_response += "I can help you with:\n"
            general_response += "• 🌤️ Weather information for any city\n"
            general_response += "• 🔍 Research and analysis of URLs or topics\n"
            general_response += "• 📊 Technical summaries and insights\n\n"
            general_response += "Could you be more specific about what you'd like me to help you with?"
            
            yield encoder.encode(text_delta(msg_general, general_response))
            yield encoder.encode(text_end(msg_general))

        # lifecycle end
        yield encoder.encode(run_finished(thread_id, run_id))

    return StreamingResponse(gen(), media_type=encoder.get_content_type())
