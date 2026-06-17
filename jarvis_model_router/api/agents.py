from fastapi import APIRouter, HTTPException

from jarvis_model_router.agents import (
    get_agent_descriptions,
    get_current_agent,
    set_current_agent,
)

router = APIRouter()


@router.get("/v1/agents")
async def list_agents() -> dict:
    """List all available agents with their metadata."""
    return {"agents": get_agent_descriptions()}


@router.get("/v1/agents/current")
async def current_agent() -> dict:
    """Return the currently active agent."""
    agent = get_current_agent()
    return {
        "name": agent.name,
        "display_name": agent.display_name,
        "description": agent.description,
        "tools": agent.get_available_tools(),
    }


@router.post("/v1/agents/switch")
async def switch_agent(name: str) -> dict:
    """Switch to a named agent."""
    try:
        agent = set_current_agent(name)
        return {
            "name": agent.name,
            "display_name": agent.display_name,
            "description": agent.description,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
