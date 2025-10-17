from __future__ import annotations
import logging
from typing import Dict, Any
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from .agents import sequential_orchestrator, collab_orchestrator, router_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting the orchestrator")
# Setup Runner and Session for Sequential Agent
USER_ID = "user_123"
session_service = InMemorySessionService()

# Create the runner with the sequential orchestrator
sequential_runner = Runner(
    agent=sequential_orchestrator, 
    app_name="Sequential_APP", 
    session_service=session_service

)

# Create the runner with the collaboration orchestrator
collab_runner = Runner(
    agent=collab_orchestrator, 
    app_name="Collab_APP", 
    session_service=session_service
)

async def run_sequential(user_message: str) -> Dict[str, Any]:
    """Run the sequential orchestrator using the correct ADK Runner pattern."""
    import uuid
    
    # Create content from user input
    content = types.Content(
        role="user", 
        parts=[types.Part(text=user_message)]
    )
    # Create a unique session for this run
    session_id = str(uuid.uuid4())
    
    session = await session_service.create_session(
        app_name="Sequential_APP", 
        user_id=USER_ID, 
        session_id=session_id
    )
    
    final_response = "No response received."
    
    # Run the sequential agent
    for event in sequential_runner.run(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    ):
        """ if event.event_type == sequential_runner.ADK_RESPONSE_EVENT_TYPE:
            final_response = event.text """
            
        # Check if this is a text response event
        if hasattr(event, 'text') and event.text:
            final_response = event.text
        elif hasattr(event, 'content') and event.content:
            final_response = str(event.content)
        elif hasattr(event, 'message') and event.message:
            final_response = str(event.message)
            
    session = await session_service.get_session(
        app_name="Sequential_APP", 
        user_id=USER_ID, 
        session_id=session_id
    )
    researcher_agent_summary = session.state.get("research_summary")
    logger.info(f"Researcher agent summary: {researcher_agent_summary}")
    technical_writer_agent_summary = session.state.get("final_output")
    logger.info(f"Technical writer agent summary: {technical_writer_agent_summary}")
    
    return {
        "output": final_response, 
        "trace": "Sequential agent execution completed",
        "research_summary": researcher_agent_summary,
        "technical_summary": technical_writer_agent_summary
    }

async def run_collab(user_message: str) -> Dict[str, Any]:
    """Run the collaboration orchestrator using the correct ADK Runner pattern."""
    import uuid
    
    # Create content from user input
    content = types.Content(
        role="user", 
        parts=[types.Part(text=user_message)]
    )
    
    # Create a unique session for this run
    session_id = str(uuid.uuid4())
    
    session = await session_service.create_session(
        app_name="Collab_APP", 
        user_id=USER_ID, 
        session_id=session_id
    )
    
    final_response = "No response received."
    
    # Run the collaboration agent
    for event in collab_runner.run(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    ):
        # Check if this is a text response event
        if hasattr(event, 'text') and event.text:
            final_response = event.text
        elif hasattr(event, 'content') and event.content:
            final_response = str(event.content)
        elif hasattr(event, 'message') and event.message:
            final_response = str(event.message)
            
    session = await session_service.get_session(
        app_name="Collab_APP", 
        user_id=USER_ID, 
        session_id=session_id
    )
    researcher_agent_summary = session.state.get("collab_research_summary")
    logger.info(f"Collaboration researcher agent summary: {researcher_agent_summary}")
    technical_writer_agent_summary = session.state.get("collab_final_output")
    logger.info(f"Collaboration technical writer agent summary: {technical_writer_agent_summary}")
    
    return {
        "output": final_response, 
        "trace": "Collaboration agent execution completed",
        "collab_research_summary": researcher_agent_summary,
        "collab_technical_summary": technical_writer_agent_summary
    }

# Create the router runner
router_runner = Runner(
    agent=router_agent, 
    app_name="Router_APP", 
    session_service=session_service
)

async def intelligent_router(user_message: str) -> Dict[str, Any]:
    """Intelligent router that analyzes user queries and routes them appropriately."""
    import uuid
    
    # Create content from user input
    content = types.Content(
        role="user", 
        parts=[types.Part(text=user_message)]
    )
    
    # Create a unique session for this run
    session_id = str(uuid.uuid4())
    
    session = await session_service.create_session(
        app_name="Router_APP", 
        user_id=USER_ID, 
        session_id=session_id
    )
    
    # Run the router agent
    for event in router_runner.run(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    ):
        if hasattr(event, 'text') and event.text:
            routing_decision = event.text
            break
        elif hasattr(event, 'content') and event.content:
            routing_decision = str(event.content)
            break
        elif hasattr(event, 'message') and event.message:
            routing_decision = str(event.message)
            break
    else:
        routing_decision = "GENERAL_ROUTE"
    
    # Parse the routing decision (handle ADK response format)
    route_type = routing_decision.strip()
    
    # Handle ADK response format: "parts=[Part(text='WEATHER_ROUTE')] role='model'"
    import re
    if "parts=[Part(" in route_type and "text='" in route_type:
        match = re.search(r"text='([^']+)'", route_type)
        if match:
            route_type = match.group(1)
    elif "role='model'" in route_type:
        # Extract just the route type from the response
        match = re.search(r"text='([^']+)'", route_type)
        if match:
            route_type = match.group(1)
    
    route_type = route_type.strip()
    
    # Additional cleanup for any remaining formatting
    if route_type.startswith("parts=[Part("):
        match = re.search(r"text='([^']+)'", route_type)
        if match:
            route_type = match.group(1)
    
    # Extract city name for weather queries using regex
    import re
    city = "N/A"
    if route_type == "WEATHER_ROUTE":
        # Try to extract city name from the original query
        m = re.search(r"weather\s+(?:in|like\s+in)\s+([A-Za-z\-\s]+)", user_message, re.I)
        if m:
            city = m.group(1).strip()
        else:
            # Fallback: look for common weather patterns
            m = re.search(r"(?:weather|temperature|climate)\s+(?:in|for|at)\s+([A-Za-z\-\s]+)", user_message, re.I)
            if m:
                city = m.group(1).strip()
        
        # Clean up city name (remove trailing punctuation)
        if city != "N/A":
            city = re.sub(r'[?!.,]+$', '', city).strip()
    
    logger.info(f"Router decision: {route_type}, City: {city}")
    logger.info(f"Original routing_decision: {routing_decision}")
    
    return {
        "route_type": route_type,
        "city": city,
        "original_query": user_message,
        "routing_decision": routing_decision
    }
