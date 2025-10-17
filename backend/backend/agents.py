from __future__ import annotations
import os

# Disable LiteLLM logging to avoid event loop conflicts BEFORE any imports
os.environ["LITELLM_LOG"] = "error"
os.environ["LITELLM_DISABLE_LOGGING"] = "true"
os.environ["LITELLM_LOG_LEVEL"] = "ERROR"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "true"
os.environ["LITELLM_DISABLE_STREAMING_LOGGING"] = "true"
os.environ["LITELLM_TURN_OFF_MESSAGE_LOGGING"] = "true"
os.environ["LITELLM_DISABLE_CACHE"] = "true"
os.environ["LITELLM_DISABLE_TOKEN_COUNTER"] = "true"

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
import litellm

# Disable all LiteLLM logging to prevent event loop conflicts
litellm.disable_streaming_logging = True
litellm.turn_off_message_logging = True
litellm.log_level = "ERROR"
litellm.disable_cache = True
litellm.disable_token_counter = True
litellm.disable_end_user_cost_tracking = True
litellm.store_audit_logs = False

# Model config via LiteLLM Proxy (OpenAI-compatible providers)
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "anthropic/claude-3-5-sonnet-20241022")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", ANTHROPIC_API_KEY)                

def _llm():
    # LiteLlm wrapper supports LiteLLM providers directly (per ADK docs)
    return LiteLlm(model=LITELLM_MODEL,
                   api_base=LITELLM_BASE_URL,
                   api_key=LITELLM_API_KEY,
        # Pass authentication headers if needed
        # extra_headers=auth_headers
        # Alternatively, if endpoint uses an API key:
        # api_key="YOUR_ENDPOINT_API_KEY"
        )

# Worker 1: Web Researcher (general purpose)
web_researcher = LlmAgent(
    name="web_researcher",
    model=_llm(),
    instruction=(
        "You are a senior web researcher. If the user gives a URL or topic, "
        "produce a structured summary with key facts, sources (if provided), and caveats. "
        "Prefer concise bullets."
    ),
    output_key="research_summary"  # Store output in state for next agent
)

# Worker 2: Technical Writer
technical_writer = LlmAgent(
    name="technical_writer",
    model=_llm(),
    instruction=(
        "You are a crisp technical writer. Turn the 'research_summary' key in the state into an executive summary "
        "and 3–5 actionable insights for engineering managers. Keep it precise.\n\n"
        "**Research Summary to Process:**\nPlease analyze the research provided by the previous agent and create an executive summary with actionable insights."
    ),
    output_key="final_output"  # Store final output in state
)

# Orchestrator v1 — Sequential pipeline (WebResearcher -> TechnicalWriter)
sequential_orchestrator = SequentialAgent(
    name="sequential_orchestrator",
    sub_agents=[web_researcher, technical_writer],
)

# Create separate instances for collaboration orchestrator
web_researcher_collab = LlmAgent(
    name="web_researcher_collab",
    model=_llm(),
    instruction=(
        "You are a senior web researcher. If the user gives a URL or topic, "
        "produce a structured summary with key facts, sources (if provided), and caveats. "
        "Prefer concise bullets."
    ),
    output_key="collab_research_summary"
)

technical_writer_collab = LlmAgent(
    name="technical_writer_collab",
    model=_llm(),
    instruction=(
        "You are a crisp technical writer. Turn the 'collab_research_summary' key in the state into an executive summary "
        "and 3–5 actionable insights for engineering managers. Keep it precise.\n\n"
        "Please analyze the research provided by the web_researcher_collab agent and create an executive summary with actionable insights."
    ),
    output_key="collab_final_output"
)

# Orchestrator v2 — Parallel collaboration (concurrent execution, merging)
# ParallelAgent executes sub-agents concurrently and merges their results
collab_orchestrator = ParallelAgent(
    name="collab_orchestrator",
    sub_agents=[web_researcher_collab, technical_writer_collab],
)

# Router Agent - Intelligent query routing
router_agent = LlmAgent(
    name="router_agent",
    model=_llm(),
    instruction=(
        "You are a query router. Analyze the user's question and respond with ONLY one of these four options:\n\n"
        "1. If the user asks about WEATHER in any city, respond: WEATHER_ROUTE\n"
        "2. If the user asks to SUMMARIZE, RESEARCH, or ANALYZE URLs/topics, respond: RESEARCH_ROUTE\n"
        "3. If the user asks for COMPARISON, ANALYSIS, or wants MULTIPLE perspectives, respond: COLLABORATION_ROUTE\n"
        "4. For all other questions, respond: GENERAL_ROUTE\n\n"
        "Examples:\n"
        "- 'What's the weather in New York?' → WEATHER_ROUTE\n"
        "- 'Tell me about the weather in London' → WEATHER_ROUTE\n"
        "- 'Weather in Tokyo' → WEATHER_ROUTE\n"
        "- 'Summarize https://ai.google.dev' → RESEARCH_ROUTE\n"
        "- 'Research AI trends' → RESEARCH_ROUTE\n"
        "- 'Compare different AI models' → COLLABORATION_ROUTE\n"
        "- 'Analyze pros and cons of...' → COLLABORATION_ROUTE\n"
        "- 'Hello, how are you?' → GENERAL_ROUTE\n"
        "- 'What can you help with?' → GENERAL_ROUTE\n\n"
        "Respond with ONLY the route type, nothing else."
    ),
    output_key="routing_decision"
)
