from src.states import ResearcherState, ResearcherOutput
from src.prompts import (
    RESEARCHER_PROMPT_STRUCTURED, 
    RESEARCHER_PROMPT_UNSTRUCTURED, 
    COMPRESSSION_PROMPT,
)
from src.utils import invoke_tool, get_tools, get_api_key_for_model
from src.configs import Configuration

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model

from typing import Literal
import asyncio

configurable_model = init_chat_model(
    configurable_fields=("model", "temperature", "max_tokens", "api_key"),
)

prompt_by_agent_categories = {
    "structured_researcher": RESEARCHER_PROMPT_STRUCTURED,
    "unstructured_researcher": RESEARCHER_PROMPT_UNSTRUCTURED,
}

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_category = state.get("researcher_category")
    if researcher_category in configurable.researcher_available_tools.keys():
        tools = await get_tools(configurable.researcher_available_tools[researcher_category])

    research_model_config = {
        **configurable.researcher_llm.get(researcher_category, {}),
        "api_key": get_api_key_for_model(configurable.researcher_llm.get(researcher_category, {}).get("model"), config),
        # "tags": ["langsmith:nostream"]
    }
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_config(research_model_config)
    )


    ## Generate researcher response
    messages = [SystemMessage(content=prompt_by_agent_categories[researcher_category])] + state["researcher_messages"]
    response = await research_model.ainvoke(messages)
    
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        }
    )

async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_category = state.get("researcher_category")
    if researcher_category in configurable.researcher_available_tools.keys():
        tools = await get_tools(configurable.researcher_available_tools[researcher_category])
        tools_map = {tool.name: tool for tool in tools}
    else:
        raise ValueError(f"Unknown researcher category: {researcher_category}")

    recent_message = state["researcher_messages"][-1]
    tool_calls = recent_message.tool_calls
    has_tools = len(tool_calls) > 0

    if not has_tools:
        return Command(
            goto="compress_research",
        )
    
    # Invoke tools
    coros = [invoke_tool(tool_call, tools_map, config) for tool_call in tool_calls]
    tool_responses = await asyncio.gather(*coros)

    # Check if ending conditions are met
    max_allowed_iterations = configurable.researcher_max_iterations.get(researcher_category, None)
    current_iterations = state.get("tool_call_iterations", 0)

    if max_allowed_iterations != None and current_iterations > max_allowed_iterations:
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_responses,
            }
        )
    
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_responses,
        }
    )

    
async def compress_research(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)

    compress_model_config = {
        **configurable.compressor_llm,
        "api_key": get_api_key_for_model(configurable.compressor_llm.get("model"), config),
        # "tags": ["langsmith:nostream"]
    }
    compress_model = (
        configurable_model
        .with_config(compress_model_config)
    )
    messages = state["researcher_messages"] + [HumanMessage(content=COMPRESSSION_PROMPT.format(researcher_category=state.get("researcher_category")))]
    response = await compress_model.ainvoke(messages)

    return {
        "compressed_research": response,
    }

researcher_builder = StateGraph(ResearcherState, output_schema=ResearcherOutput, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)

researcher_builder.set_entry_point("researcher")
researcher_builder.add_edge("compress_research", END)

researcher_subgraph = researcher_builder.compile()