from __future__ import annotations
from typing import Dict, Any
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from .agents import sequential_orchestrator, collab_orchestrator

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
        # Check if this is a text response event
        if hasattr(event, 'text') and event.text:
            final_response = event.text
        elif hasattr(event, 'content') and event.content:
            final_response = str(event.content)
        elif hasattr(event, 'message') and event.message:
            final_response = str(event.message)
    
    return {"output": final_response, "trace": "Sequential agent execution completed"}

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
    
    return {"output": final_response, "trace": "Collaboration agent execution completed"}
