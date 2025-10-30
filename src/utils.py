from src.tools import list_all_tools

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import MessageLikeRepresentation, filter_messages, ToolMessage

from typing import List
import os

all_tools: dict = list_all_tools()

async def get_tools(filter_tools: List[str] = None):
    tools = []
    if filter_tools != None:
        for tool_name in filter_tools:
            if tool_name in all_tools:
                tools.append(all_tools[tool_name])
    else:
        tools = list(all_tools.values())
    
    return tools

async def invoke_tool(tool_call, tools_map, config: RunnableConfig, message_prefix: str = ""):
    tool_fn = tools_map.get(tool_call["name"])
    if not tool_fn:
        return ToolMessage(tool_call_id=tool_call["id"], content="Tool doesn't exist!")
    try:
        response = await tool_fn.ainvoke(tool_call["args"])
        return ToolMessage(tool_call_id=tool_call["id"], content=f"{message_prefix}{response}")
    except Exception as e:
        return ToolMessage(tool_call_id=tool_call["id"], content=f"Error: {e}")

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None
    
def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]