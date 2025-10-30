from typing import List
from typing_extensions import Annotated, TypedDict
import operator
from pydantic import BaseModel, Field

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

#### Agent State Definitions ####
class AgentState(MessagesState):
    supervisor_messages: Annotated[List[AnyMessage], override_reducer]
    research_topic: str
    notes: Annotated[List[str], override_reducer] = []
    final_report: str

class ResearcherState(TypedDict):
    researcher_messages: Annotated[List[AnyMessage], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    researcher_category: str

class SupervisorState(TypedDict):
    supervisor_messages: Annotated[List[AnyMessage], override_reducer]
    notes: Annotated[List[str], override_reducer] = []
    research_topic: str
    supervisor_tools_iterations: int = 0
    
#### I/O Schema Definitions ####
class ResearcherOutput(BaseModel):
    compressed_research: str

class ResearchQuestionOutput(BaseModel):
    research_question: str = Field(
        description="Research question to guide the research"
    )

class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class RunBashInput(BaseModel):
    cmd: str = Field(..., description="Shell command to run e.g. 'ls -lR ./'")
    desc: str = Field(..., description="Purpose of the command e.g 'List the files in the directory recursively'") 

class RunPythonInput(BaseModel):
    code: str = Field(..., description="Python code to run e.g 'print(2+2)'")
    desc: str = Field(..., description="Purpose of the code e.g 'Calculate mean rainfall from rainfall_data.csv'")

class KgQueryInput(BaseModel):
    search_query: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (one paragraph/3-4 lines)."
    )

class ClarificationOutput(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope or direct answer for follow up query.",
    )
    verification: str = Field(
        description="Verification message that we will start research after the user has provided the necessary information.",
    )