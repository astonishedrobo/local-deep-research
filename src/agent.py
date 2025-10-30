from src.supervisor import supervisor_subgraph
from src.states import (
    AgentState,
    ResearchQuestionOutput,
    ClarificationOutput,
    AgentInputState,
)
from src.prompts import (
    FINAL_REPORT_PROMPT, 
    SCOPE_CONVERSATION_PROMPT,
    RESEARCH_QUESTION_GEN_PROMPT,
    SUPERVISOR_PROMPT
)
from src.configs import Configuration
from src.utils import get_api_key_for_model

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    get_buffer_string
)

from typing import Literal

configurable_model = init_chat_model(
    configurable_fields=("model", "temperature", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["generate_research_question", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(
            goto="generate_research_question",
        )
    

    conversation_model_config = {
        **configurable.conversation_llm,
        "api_key": get_api_key_for_model(configurable.conversation_llm.get("model"), config),
        # "tags": ["langsmith:nostream"]
    }
    conversation_model = (
        configurable_model
        .with_structured_output(ClarificationOutput)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(conversation_model_config)
    )

    prompt = SCOPE_CONVERSATION_PROMPT.format(
        messages=get_buffer_string(state["messages"])
    )
    response = await conversation_model.ainvoke([HumanMessage(content=prompt)])

    if response.need_clarification:
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=response.question)]
            }
        )
    else:
        return Command(
            goto="generate_research_question",
            update={
                "messages": [AIMessage(content=response.verification)]
            }
        )
    
async def generate_research_question(state: AgentState, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    configurable = Configuration.from_runnable_config(config)
    report_model_config = {
        **configurable.reporter_llm,
        "api_key": get_api_key_for_model(configurable.reporter_llm.get("model"), config),
        # "tags": ["langsmith:nostream"]
    }
    report_model = (
        configurable_model
        .with_structured_output(ResearchQuestionOutput)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(report_model_config)
    )

    prompt_research_question = RESEARCH_QUESTION_GEN_PROMPT.format(
        messages=get_buffer_string(state.get("messages", []))
    )

    response = await report_model.ainvoke([HumanMessage(content=prompt_research_question)])

    supervisor_system_prompt = SUPERVISOR_PROMPT.format(
        max_concurrent_research_units=configurable.subagent_concurrency_limit,
        max_researcher_iterations=configurable.researcher_max_iterations
    )
    
    return Command(
        goto="supervisor", 
        update={
            "research_topic": response.research_question,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_question)
                ]
            }
        }
    )




async def generate_report(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    report_model_config = {
        **configurable.reporter_llm,
        "api_key": get_api_key_for_model(configurable.reporter_llm.get("model"), config),
        # "tags": ["langsmith:nostream"]
    }
    report_model = (
        configurable_model
        .with_config(report_model_config)
    )

    # TODO: Add token limit exceeded retry logic later.
    final_report_message = FINAL_REPORT_PROMPT.format(
        research_topic=state.get("research_topic"),
        messages=state.get("messages", []),
        findings=findings
    )
    response = await report_model.ainvoke([HumanMessage(content=final_report_message)])

    return {
        "final_report": response.content,
        "messages": [response],
        **cleared_state,
    }

agent_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)

agent_builder.add_node("clarify_with_user", clarify_with_user)
agent_builder.add_node("generate_research_question", generate_research_question)
agent_builder.add_node("supervisor", supervisor_subgraph)
agent_builder.add_node("generate_report", generate_report)

agent_builder.set_entry_point("clarify_with_user")
agent_builder.add_edge("supervisor", "generate_report")
agent_builder.add_edge("generate_report", END)

agent_graph = agent_builder.compile()