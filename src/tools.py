from src.states import RunBashInput, RunPythonInput, KgQueryInput
from src.kg_search_manager import get_searcher


# from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool

from pydantic import BaseModel, Field
import os
import tempfile
import pathspec
import docker
import shutil
from typing import Type

def list_all_tools():
    return {
        "python": RunPythonCode(),
        "shell": RunBashCommands(),
        "kg": QueryKnowledgeGraph,
        "think_tool": think_tool,
    }


@tool(description="Strategic reflection tool for research planning")
async def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Thinking Reflection: {reflection}"

@tool("QueryKnowledgeGraph", args_schema=KgQueryInput)
async def QueryKnowledgeGraph(search_query: str) -> dict:
    """API to query the knowledge graph in the 'kg' directory."""
    searcher = get_searcher("kg")
    response = await searcher.search(search_query)
    return response

class RunBashCommands(BaseTool):
    name: str = "run_bash_command"
    description: str = "Run any arbitrary bash command in a dockerized sandbox"
    args_schema: Type[BaseModel] = RunBashInput

    def _get_ignore_spec(self, path: str) -> pathspec.PathSpec:
        """Reads .secureignore from the given path and returns a PathSpec object."""
        ignore_file = os.path.join(path, '.secureignore')
        patterns = []
        if os.path.exists(ignore_file):
            with open(ignore_file, 'r') as f:
                patterns = pathspec.GitIgnoreSpec.from_lines(f.readlines())
        return patterns
    
    def _run(self, cmd: str, desc: str) -> str:
        host_root = os.getcwd()
        client = docker.from_env()
        ignore_spec = self._get_ignore_spec(host_root)

        # Use a 'with' statement for a temporary directory that is automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Walk the source directory and copy only the allowed files
            for root, dirs, files in os.walk(host_root, topdown=True):
                # Prune ignored directories from the walk so we don't even look inside them
                # We check the path with a trailing slash to match gitignore behavior for directories
                dirs[:] = [d for d in dirs if not ignore_spec.match_file(
                    os.path.join(os.path.relpath(root, host_root), d).replace('\\','/') + '/'
                )]

                for f in files:
                    # Get the full path relative to the root to check against the ignore spec
                    relative_file_path = os.path.join(os.path.relpath(root, host_root), f)
                    
                    # Use forward slashes for cross-platform compatibility with pathspec
                    if not ignore_spec.match_file(relative_file_path.replace('\\','/')):
                        # This file is NOT ignored, so we copy it
                        source_file_path = os.path.join(root, f)
                        
                        # Calculate the destination directory inside the temp folder
                        dest_dir_in_temp = os.path.join(temp_dir, os.path.relpath(root, host_root))
                        os.makedirs(dest_dir_in_temp, exist_ok=True)
                        
                        # Copy the file, preserving metadata like permissions
                        shutil.copy2(source_file_path, dest_dir_in_temp)

            # --- The Core of the Solution ---
            # Mount the sanitized temporary directory as the workspace.
            # The container will only see what was copied into temp_dir.
            try:
                output = client.containers.run(
                    image="python:3.10-agents",
                    command=["bash", "-lc", cmd],
                    stdout=True, stderr=True,
                    volumes={
                        temp_dir: {"bind": "/workspace", "mode": "ro"}
                    },
                    mem_limit="128m",
                    working_dir="/workspace",
                    network_disabled=True,
                    remove=True
                )
                return f"## {desc}\n ```{output.decode()}\n```"
            except docker.errors.ContainerError as e:
                return f"Command execution failed with error:\n{e.stderr.decode()}"
            except Exception as e:
                return f"An unexpected error occurred: {e}"


class RunPythonCode(BaseTool):
    name: str = "run_python_code"
    description: str = (
        "Run arbitrary Python code in a dockerized sandbox. "
        "If you want to get the output, use the `print` function in your code otherwise the output will be empty. "
    )
    args_schema: Type[BaseModel] = RunPythonInput
    # ----------------------------------------------------------
    # 1. helper: load .secureignore patterns
    # ----------------------------------------------------------
    def _get_ignore_spec(self, path: str) -> pathspec.PathSpec:
        ignore_file = os.path.join(path, ".secureignore")
        if os.path.exists(ignore_file):
            with open(ignore_file, "r", encoding="utf-8") as f:
                return pathspec.GitIgnoreSpec.from_lines(f)
        return pathspec.GitIgnoreSpec([])

    # ----------------------------------------------------------
    # 2. main entry
    # ----------------------------------------------------------
    def _run(self, code: str, desc: str) -> str:
        host_root = os.getcwd()
        client = docker.from_env()
        ignore_spec = self._get_ignore_spec(host_root)

        # Use a temporary directory that is automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Walk the host directory and copy only allowed files
            for root, dirs, files in os.walk(host_root, topdown=True):
                # Prune ignored directories
                dirs[:] = [
                    d for d in dirs
                    if not ignore_spec.match_file(
                        os.path.join(
                            os.path.relpath(root, host_root),
                            d
                        ).replace("\\", "/") + "/"
                    )
                ]

                for f in files:
                    rel_path = os.path.join(
                        os.path.relpath(root, host_root),
                        f
                    ).replace("\\", "/")
                    if ignore_spec.match_file(rel_path):
                        continue

                    src = os.path.join(root, f)
                    dst_dir = os.path.join(temp_dir, os.path.relpath(root, host_root))
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy2(src, dst_dir)

            # --------------------------------------------------
            # 3. run the container
            # --------------------------------------------------
            try:
                output = client.containers.run(
                    image="python:3.10-agents",
                    command=["python3", "-c", code],
                    stdout=True,
                    stderr=True,
                    volumes={temp_dir: {"bind": "/workspace", "mode": "ro"}},
                    working_dir="/workspace",
                    mem_limit="128m",
                    network_disabled=True,
                    remove=True,
                )
                return f"## {desc} \n```{output.decode()}\n```"
            except docker.errors.ContainerError as e:
                return f"Execution failed:\n{e.stderr.decode()}"
            except Exception as e:
                return f"Unexpected error: {e}"

@tool         
class StructuredDataResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

@tool
class UnstructuredDataResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )