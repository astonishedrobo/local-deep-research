from langchain_core.runnables import RunnableConfig

from pydantic import BaseModel, Field
from typing import Optional, Any
import os

class Configuration(BaseModel):
    researcher_llm: dict = Field(
        default_factory=lambda: {
            "structured_researcher": {
                "model": "openai:gpt-4.1",
                "temperature": 1,
                "max_tokens": 10000,
            },
            "unstructured_researcher": {
                "model": "openai:gpt-4.1",
                "temperature": 1,
                "max_tokens": 10000,
            },
        },
        description="Configuration for researcher LLMs based on their category.",
    )

    compressor_llm: dict = Field(
        default_factory=lambda: {
            "model": "openai:gpt-4.1",
            "temperature": 1,
            "max_tokens": 8192,
        },
        description="LLM model configuration for compressing research findings.",
    )

    reporter_llm: dict = Field(
        default_factory=lambda: {
            "model": "openai:gpt-4.1",
            "temperature": 1,
            "max_tokens": 10000,
        },
        description="LLM model configuration for generating the research brief.",
    )

    supervisor_llm: dict = Field(
        default_factory=lambda: {
            "model": "openai:gpt-4.1",
            "temperature": 1,
        },
        description="LLM model configuration for the supervisor.",
    )

    conversation_llm: dict = Field(
        default_factory=lambda: {
            "model": "openai:gpt-4.1",
            "temperature": 1,
        },
        description="LLM model configuration for conversations with the user.",
    )

    researcher_available_tools: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "structured_researcher": ["python", "shell", "think_tool"],
            "unstructured_researcher": ["kg", "think_tool"],
            "universal": ["python", "shell", "kg", "think_tool"],
        },
        description="Mapping of researcher categories to their available tools.",
    )

    researcher_max_iterations: dict[str, int] = Field(
        default_factory=lambda: {
            "structured_researcher": 10,
            "unstructured_researcher": 5,
        },
        description="Maximum tool call iterations for each researcher category.",
    )

    supervisor_max_iterations: int = Field(
        default=5,
        description="Maximum research iterations the supervisor can perform.",
    )

    allow_clarification: bool = Field(
        default=True,
        description="Whether to allow clarification with the user before generating the research question.",
    )

    max_structured_output_retries: int = Field(
        default=3,
        description="Maximum retries for structured output parsing.",
    )

    subagent_concurrency_limit: int = Field(
        default=2,
        description="Maximum number of sub-agents to run concurrently.",
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig]) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True