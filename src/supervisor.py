from src.states import SupervisorState
from src.configs import Configuration
from src.tools import StructuredDataResearch, UnstructuredDataResearch, think_tool
from src.prompts import SUPERVISOR_PROMPT
from src.utils import invoke_tool, get_api_key_for_model, get_notes_from_tool_calls
from src.factory_research_agent import researcher_subgraph

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from typing import Literal
import asyncio

configurable_model = init_chat_model(
    configurable_fields=("model", "temperature", "max_tokens", "api_key"),
)

tools = [StructuredDataResearch, UnstructuredDataResearch, think_tool]
tools_map = {tool.name: tool for tool in tools[2:]}

async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    
    supervisor_model_config = {
        **configurable.supervisor_llm,
        "api_key": get_api_key_for_model(configurable.supervisor_llm.get("model"), config),
        # "tags": ["langsmith:nostream"]
    }

    supervisor_model = (
        configurable_model
        .bind_tools(tools)
        .with_config(supervisor_model_config)
    )

    messages = [SystemMessage(SUPERVISOR_PROMPT)] + state.get("supervisor_messages", [])
    response = await supervisor_model.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "supervisor_tools_iterations": state.get("supervisor_tools_iterations", 0) + 1 
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)

    ## Check if has tools or if exceeds allowed maximum iterations
    last_message = state.get("supervisor_messages", [])[-1]
    has_tools = len(last_message.tool_calls) > 0
    exceeds_max_iterations = state.get("supervisor_tool_iterations", 0) > configurable.supervisor_max_iterations

    if not has_tools or exceeds_max_iterations:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(state.get("supervisor_messages", []))
            }
        )

    ## Run the tools
    all_tool_messages = []
    think_tool_calls = [tool for tool in last_message.tool_calls if tool["name"] == "think_tool"]
    sub_agent_calls = [tool for tool in last_message.tool_calls if tool["name"] != "think_tool"]

    ## Run think_tools 
    think_coros = [
        invoke_tool(tool_call, tools_map, config, message_prefix="") for tool_call in think_tool_calls
    ]
    think_responses = await asyncio.gather(*think_coros)
    all_tool_messages.extend(think_responses)

    ## Run research sub-agents
    try:
        ## Trim to fit sub-agent concurrency limit
        trimmed_tools = sub_agent_calls[:configurable.subagent_concurrency_limit]
        overflowed_tools = sub_agent_calls[configurable.subagent_concurrency_limit:]

        structured_data_tools = [tool for tool in trimmed_tools if tool["name"] == "StructuredDataResearch"]
        unstructured_data_tools = [tool for tool in trimmed_tools if tool["name"] == "UnstructuredDataResearch"]
        
        research_agent_coros = []

        ## Handle structured data research agents separately to set their category
        for sub_agent in structured_data_tools:
            research_agent_coros.append(
                researcher_subgraph.ainvoke({
                    "research_topic": sub_agent["args"]["research_topic"],
                    "researcher_category": "structured_researcher",
                    "researcher_messages": [
                        HumanMessage(content=sub_agent["args"]["research_topic"])
                    ]
                }, config=config)
            )
        
        ## Handle unstructured data research agents separately to set their category
        for sub_agent in unstructured_data_tools:
            research_agent_coros.append(
                researcher_subgraph.ainvoke({
                    "research_topic": sub_agent["args"]["research_topic"],
                    "researcher_category": "unstructured_researcher",
                    "researcher_messages": [
                        HumanMessage(content=sub_agent["args"]["research_topic"])
                    ]
                }, config=config)
            )
            
        researcher_responses = await asyncio.gather(*research_agent_coros)

        for response, tool_call in zip(researcher_responses, trimmed_tools):
            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                content=response.get("compressed_research", "Error: No response from sub-agent.")
            )
            all_tool_messages.append(tool_message)

        for tool in overflowed_tools:
            all_tool_messages.append(
                ToolMessage(
                    content=f"Error: Exceeds the max allowed concurrency limit {configurable.subagent_concurreny_limit}. So, it was not run. Follow the limit for future runs.",
                    tool_call_id=tool["id"]
                )
            )

    except Exception as e:
        raise e
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(state.get("supervisor_messages", []))
            }
        )
    
    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": all_tool_messages
        }
    )
    

supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

supervisor_builder.set_entry_point("supervisor")

supervisor_subgraph = supervisor_builder.compile()