"""
Chainlit App for LangGraph Multi-Agent Research System

This app integrates your hierarchical multi-agent system with Chainlit,
displaying tool calls, agent interactions, and research progress.
"""

import chainlit as cl
from typing import Optional, Dict, Any
import os
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from literalai import LiteralClient
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Import your agent
from src.agent import agent_graph
from src.configs import Configuration

# Initialize database tables with correct Chainlit 2.9.0 schema
async def init_database():
    """Initialize SQLite database with Chainlit 2.9.0 schema"""
    engine = create_async_engine("sqlite+aiosqlite:///./chainlit_data.db")
    
    async with engine.begin() as conn:
        # Users table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                "id" TEXT PRIMARY KEY,
                "identifier" TEXT NOT NULL UNIQUE,
                "metadata" TEXT NOT NULL,
                "createdAt" TEXT
            )
        """))
        
        # Threads table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS threads (
                "id" TEXT PRIMARY KEY,
                "createdAt" TEXT,
                "name" TEXT,
                "userId" TEXT,
                "userIdentifier" TEXT,
                "tags" TEXT,
                "metadata" TEXT
            )
        """))
        
        # Steps table with defaultOpen column for Chainlit 2.9.0
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS steps (
                "id" TEXT PRIMARY KEY,
                "name" TEXT NOT NULL,
                "type" TEXT NOT NULL,
                "threadId" TEXT,
                "parentId" TEXT,
                "streaming" INTEGER,
                "waitForAnswer" INTEGER,
                "isError" INTEGER,
                "metadata" TEXT,
                "tags" TEXT,
                "input" TEXT,
                "output" TEXT,
                "createdAt" TEXT,
                "start" TEXT,
                "end" TEXT,
                "generation" TEXT,
                "showInput" TEXT,
                "language" TEXT,
                "defaultOpen" INTEGER
            )
        """))
        
        # Elements table with props column for Chainlit 2.9.0
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS elements (
                "id" TEXT PRIMARY KEY,
                "threadId" TEXT,
                "type" TEXT,
                "url" TEXT,
                "chainlitKey" TEXT,
                "name" TEXT NOT NULL,
                "display" TEXT,
                "objectKey" TEXT,
                "size" TEXT,
                "page" INTEGER,
                "language" TEXT,
                "forId" TEXT,
                "mime" TEXT,
                "props" TEXT
            )
        """))
        
        # Feedbacks table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                "id" TEXT PRIMARY KEY,
                "forId" TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                "value" INTEGER NOT NULL,
                "comment" TEXT
            )
        """))
    
    await engine.dispose()

# Run database initialization
asyncio.run(init_database())

# Setup data persistence for chat history
# Using aiosqlite for async SQLite operations
cl_data_layer = SQLAlchemyDataLayer(
    conninfo="sqlite+aiosqlite:///./chainlit_data.db"
)
cl_data._data_layer = cl_data_layer

# Simple in-memory user storage (replace with database in production)
# Format: {email: {"password": hashed_password, "name": name, "metadata": {...}}}
USER_DATABASE = {}

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """
    Handle password-based authentication with auto-registration.
    
    Args:
        username: User's email address
        password: User's password
    
    Returns:
        User object if authentication succeeds, None otherwise
    """
    if not username or not password:
        return None
    
    if len(password) < 4:
        return None
    
    # Auto-register new users
    if username not in USER_DATABASE:
        name = username.split("@")[0] if "@" in username else username
        
        USER_DATABASE[username] = {
            "password": password,
            "metadata": {
                "name": name,
                "email": username,
                "role": "researcher",
                "is_new": True
            }
        }
        
        print(f"✅ New user auto-registered: {username}")
    
    # Verify password
    if USER_DATABASE[username]["password"] == password:
        user_metadata = USER_DATABASE[username].get("metadata", {})
        return cl.User(
            identifier=username,
            metadata=user_metadata
        )
    else:
        return None

# OAuth callback (disabled but kept for future use)
# Uncomment the decorator and enable OAuth providers in config.toml to use
# @cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, Any],
    default_user: cl.User,
) -> Optional[cl.User]:
    """
    Handle OAuth callback from Google (currently disabled).
    
    To enable:
    1. Uncomment the @cl.oauth_callback decorator above
    2. Uncomment the [[auth.providers]] section in .chainlit/config.toml
    3. Add real OAuth credentials to .env file
    
    Args:
        provider_id: The OAuth provider ID (e.g., "google")
        token: The access token
        raw_user_data: User data from the OAuth provider
        default_user: Default user object from Chainlit
    
    Returns:
        User object or None
    """
    if provider_id == "google":
        # Extract user information from Google OAuth response
        email = raw_user_data.get("email", "")
        name = raw_user_data.get("name", email.split("@")[0] if email else "User")
        picture = raw_user_data.get("picture", "")
        
        # Create user with metadata
        return cl.User(
            identifier=email,
            metadata={
                "name": name,
                "email": email,
                "picture": picture,
                "provider": "google",
                "role": "researcher",
            }
        )
    
    return None


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    # Get authenticated user
    user = cl.user_session.get("user")
    username = user.metadata.get("name", "User") if user else "Guest"
    user_email = user.identifier if user else "guest"
    
    # Check if this is a new user (just registered)
    is_new_user = user and user.metadata.get("is_new", False)
    
    # Clear the is_new flag for next login
    if is_new_user and user_email in USER_DATABASE:
        USER_DATABASE[user_email]["metadata"]["is_new"] = False
    
    # Welcome message
    if is_new_user:
        welcome_text = f"""# 🔬 Welcome to Deep Research, {username}!

## About This System

This is a hierarchical multi-agent research system that can help you:
- 📊 Analyze structured data (CSV files, databases)
- 📄 Research unstructured content (web pages, documents)  
- 🔍 Perform comprehensive multi-source research
- 📈 Generate detailed research reports

---

💾 **Your conversations are automatically saved** to your account.

**Ready to start researching? Ask me anything!**
"""
    else:
        welcome_text = f"""# 🔬 Welcome back, {username}!

This is a hierarchical multi-agent research system ready to help you with:
- 📊 Structured data analysis
- 📄 Unstructured content research
- 🔍 Multi-source research
- 📈 Detailed report generation

💾 Your conversations are automatically saved to your account.

**What would you like to research today?**
"""
    
    await cl.Message(
        content=welcome_text,
        author="System"
    ).send()

    # Store the agent graph in user session
    cl.user_session.set("agent_graph", agent_graph)
    cl.user_session.set("config", {
        "configurable": {
            # Default configuration - you can customize these
            "allow_clarification": True,
            "max_structured_output_retries": 3,
            "supervisor_max_iterations": 10,
            "subagent_concurrency_limit": 3,
            
            # Researcher max iterations per category
            "researcher_max_iterations": {
                "structured_researcher": 10,
                "unstructured_researcher": 5,
            },
            
            # LLM configurations
            "conversation_llm": {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
            },
            "reporter_llm": {
                "model": "gpt-4o",
                "temperature": 0.7,
            },
            "supervisor_llm": {
                "model": "gpt-4o",
                "temperature": 0.7,
            },
            "compressor_llm": {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
            },
            # Researcher LLM per category
            "researcher_llm": {
                "structured_researcher": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.7,
                    "max_tokens": 10000,
                },
                "unstructured_researcher": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.7,
                    "max_tokens": 10000,
                },
            },
        }
    })


@cl.step(type="tool", name="Agent Processing")
async def process_agent_step(event: Dict[str, Any], parent_step: Optional[cl.Step] = None) -> str:
    """Process individual agent events and create steps."""
    
    node_name = event.get("metadata", {}).get("langgraph_node", "unknown")
    
    # Create a step for this node
    step_name = node_name.replace("_", " ").title()
    
    async with cl.Step(
        name=step_name,
        type="llm" if "llm" in node_name.lower() or node_name in ["supervisor", "clarify_with_user", "generate_research_question", "generate_report"] else "tool",
        parent_id=parent_step.id if parent_step else None
    ) as step:
        
        # Extract messages from the event
        if "messages" in event:
            messages = event["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                last_msg = messages[-1]
                
                # Handle AI messages with tool calls
                if isinstance(last_msg, AIMessage):
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        tool_info = []
                        for tc in last_msg.tool_calls:
                            tool_name = tc.get("name", "unknown")
                            tool_args = tc.get("args", {})
                            tool_info.append(f"**🔧 {tool_name}**\n```json\n{tool_args}\n```")
                        
                        step.output = f"**Tool Calls:**\n\n" + "\n\n".join(tool_info)
                    elif last_msg.content:
                        step.output = last_msg.content
                
                # Handle Tool messages (responses from tools)
                elif isinstance(last_msg, ToolMessage):
                    step.output = f"**Tool Response:**\n\n{last_msg.content}"
                
                # Handle regular messages
                elif hasattr(last_msg, 'content'):
                    step.output = last_msg.content
        
        # Handle supervisor-specific messages
        if "supervisor_messages" in event:
            supervisor_msgs = event["supervisor_messages"]
            if isinstance(supervisor_msgs, list) and len(supervisor_msgs) > 0:
                last_msg = supervisor_msgs[-1]
                
                if isinstance(last_msg, AIMessage):
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        tool_info = []
                        for tc in last_msg.tool_calls:
                            tool_name = tc.get("name", "unknown")
                            tool_args = tc.get("args", {})
                            
                            # Show research agent spawning
                            if tool_name in ["StructuredDataResearch", "UnstructuredDataResearch"]:
                                research_topic = tool_args.get("research_topic", "")
                                tool_info.append(
                                    f"**🚀 Spawning {tool_name}**\n"
                                    f"Research Topic: {research_topic[:200]}..."
                                )
                            else:
                                tool_info.append(f"**🔧 {tool_name}**\n```json\n{tool_args}\n```")
                        
                        step.output = f"**Supervisor Decisions:**\n\n" + "\n\n".join(tool_info)
        
        # Handle research notes
        if "notes" in event and event["notes"]:
            notes = event["notes"]
            if notes:
                step.output = f"**📝 Research Notes:**\n\n" + "\n\n".join(notes)
        
        # Handle final report
        if "final_report" in event and event["final_report"]:
            step.output = f"**📊 Final Report:**\n\n{event['final_report']}"
        
        # Handle research topic generation
        if "research_topic" in event and event["research_topic"]:
            step.output = f"**🎯 Research Question Generated:**\n\n{event['research_topic']}"
        
        return step.output or "Processing..."


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and stream agent responses."""
    
    # Get the agent and config from session
    agent_graph: CompiledStateGraph = cl.user_session.get("agent_graph")
    config = cl.user_session.get("config")
    
    # Get or initialize conversation history
    conversation_history = cl.user_session.get("conversation_history", [])
    
    # Add user message to history
    conversation_history.append(HumanMessage(content=message.content))
    cl.user_session.set("conversation_history", conversation_history)
    
    # Create a main processing step
    async with cl.Step(name="🔬 Deep Research Agent", type="llm") as main_step:
        main_step.input = message.content
        
        # Prepare input state
        input_state = {
            "messages": conversation_history
        }
        
        # Create response message
        response_msg = cl.Message(content="", author="Research Agent")
        
        # Stream events from the agent
        try:
            final_state = None
            node_steps = {}  # Track steps for each node
            step_counter = {}  # Counter for multiple calls to same node
            
            async for event in agent_graph.astream(input_state, config=config, stream_mode="updates"):
                # event is a dict with node name as key
                for node_name, node_state in event.items():
                    # Track how many times this node has been called
                    step_counter[node_name] = step_counter.get(node_name, 0) + 1
                    step_key = f"{node_name}_{step_counter[node_name]}"
                    
                    # Create step for this node iteration
                    if step_key not in node_steps:
                        # Format step name with emoji
                        step_icons = {
                            "clarify_with_user": "🧠",
                            "generate_research_question": "📋",
                            "supervisor": "👔",
                            "supervisor_tools": "🔧",
                            "generate_report": "📊"
                        }
                        
                        icon = step_icons.get(node_name, "▶️")
                        step_name = f"{icon} {node_name.replace('_', ' ').title()}"
                        
                        if step_counter[node_name] > 1:
                            step_name += f" (Iteration {step_counter[node_name]})"
                        
                        # Determine step type based on node name
                        if node_name in ["supervisor", "clarify_with_user", "generate_research_question", "generate_report"]:
                            step_type = "llm"
                        elif "tool" in node_name.lower():
                            step_type = "tool"
                        else:
                            step_type = "run"
                        
                        # Create nested step
                        node_step = cl.Step(
                            name=step_name,
                            type=step_type,
                            parent_id=main_step.id
                        )
                        await node_step.__aenter__()
                        node_steps[step_key] = node_step
                    
                    current_step = node_steps[step_key]
                    
                    # Process messages
                    if "messages" in node_state:
                        messages = node_state["messages"]
                        if isinstance(messages, list) and len(messages) > 0:
                            last_msg = messages[-1]
                            
                            if isinstance(last_msg, AIMessage):
                                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                    tool_info = []
                                    for tc in last_msg.tool_calls:
                                        tool_name = tc.get("name", "unknown")
                                        tool_args = tc.get("args", {})
                                        
                                        # Format tool args nicely
                                        import json
                                        args_str = json.dumps(tool_args, indent=2)
                                        tool_info.append(f"🔧 **{tool_name}**\n```json\n{args_str}\n```")
                                    
                                    current_step.output = "\n\n".join(tool_info)
                                    await current_step.update()
                                elif last_msg.content:
                                    current_step.output = last_msg.content
                                    await current_step.update()
                            
                            elif isinstance(last_msg, ToolMessage):
                                # Show truncated tool response
                                content = last_msg.content
                                if len(content) > 1000:
                                    content = content[:1000] + f"\n\n... (truncated, {len(content)} chars total)"
                                current_step.output = f"**✅ Tool Response:**\n\n{content}"
                                await current_step.update()
                    
                    # Handle supervisor messages - IMPORTANT for multi-agent visualization
                    if "supervisor_messages" in node_state:
                        supervisor_msgs = node_state["supervisor_messages"]
                        if isinstance(supervisor_msgs, list) and len(supervisor_msgs) > 0:
                            last_msg = supervisor_msgs[-1]
                            
                            if isinstance(last_msg, AIMessage):
                                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                    tool_info = []
                                    for tc in last_msg.tool_calls:
                                        tool_name = tc.get("name", "unknown")
                                        tool_args = tc.get("args", {})
                                        
                                        if tool_name in ["StructuredDataResearch", "UnstructuredDataResearch"]:
                                            research_topic = tool_args.get("research_topic", "")
                                            agent_type = "📊 Structured Data" if tool_name == "StructuredDataResearch" else "📄 Unstructured Data"
                                            tool_info.append(
                                                f"### 🚀 Spawning Sub-Agent: {agent_type}\n\n"
                                                f"**Research Topic:**\n{research_topic[:300]}..."
                                            )
                                        elif tool_name == "think_tool":
                                            reflection = tool_args.get("reflection", "")
                                            tool_info.append(
                                                f"### 🤔 Strategic Reflection\n\n{reflection}"
                                            )
                                        else:
                                            import json
                                            args_str = json.dumps(tool_args, indent=2)
                                            tool_info.append(f"🔧 **{tool_name}**\n```json\n{args_str}\n```")
                                    
                                    current_step.output = "\n\n".join(tool_info)
                                    await current_step.update()
                            
                            elif isinstance(last_msg, ToolMessage):
                                # Show sub-agent response
                                content = last_msg.content
                                if len(content) > 1000:
                                    content = content[:1000] + f"\n\n... (truncated, {len(content)} chars total)"
                                current_step.output = f"**✅ Sub-Agent Response:**\n\n{content}"
                                await current_step.update()
                    
                    # Handle notes
                    if "notes" in node_state and node_state["notes"]:
                        notes = node_state["notes"]
                        # Convert to list if it's not already (in case it's a dict or other type)
                        if isinstance(notes, dict):
                            notes = list(notes.values())
                        elif not isinstance(notes, list):
                            notes = [str(notes)]
                        
                        # Take first 5 notes
                        notes_to_show = notes[:5] if len(notes) > 5 else notes
                        notes_preview = "\n\n".join([f"{i+1}. {str(note)[:200]}..." for i, note in enumerate(notes_to_show)])
                        current_step.output = f"**📝 Research Notes ({len(notes)} total):**\n\n{notes_preview}"
                        await current_step.update()
                    
                    # Handle final report
                    if "final_report" in node_state and node_state["final_report"]:
                        current_step.output = node_state["final_report"]
                        await current_step.update()
                        response_msg.content = node_state["final_report"]
                    
                    # Handle research topic
                    if "research_topic" in node_state and node_state["research_topic"]:
                        current_step.output = f"🎯 **Research Question Generated:**\n\n{node_state['research_topic']}"
                        await current_step.update()
                    
                    final_state = node_state
            
            # Close all node steps
            for step in node_steps.values():
                await step.__aexit__(None, None, None)
            
            # Get final response
            if final_state:
                if "final_report" in final_state and final_state["final_report"]:
                    response_msg.content = final_state["final_report"]
                elif "messages" in final_state and final_state["messages"]:
                    last_msg = final_state["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        response_msg.content = last_msg.content
                
                # Update conversation history with agent response
                if response_msg.content:
                    conversation_history.append(AIMessage(content=response_msg.content))
                    cl.user_session.set("conversation_history", conversation_history)
            
            main_step.output = response_msg.content or "Research completed!"
        
        except Exception as e:
            import traceback
            error_msg = f"❌ **Error occurred:**\n\n```\n{str(e)}\n\n{traceback.format_exc()}\n```"
            main_step.output = error_msg
            response_msg.content = f"❌ An error occurred during research. Please try again or rephrase your question.\n\nError: {str(e)}"
    
    # Send the final response
    await response_msg.send()


@cl.on_settings_update
async def settings_update(settings: Dict[str, Any]):
    """Handle settings updates for configuration."""
    config = cl.user_session.get("config", {})
    
    # Update configuration based on settings
    if "configurable" not in config:
        config["configurable"] = {}
    
    # You can add UI settings here
    cl.user_session.set("config", config)


if __name__ == "__main__":
    # This allows running the app directly
    import subprocess
    subprocess.run(["chainlit", "run", __file__, "-w"])
