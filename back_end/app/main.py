import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pathlib import Path

def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

# Add the project directories to the system path
if is_docker():
    print("[INFO] Running inside a Docker container")

    # Set the project root
    project_root = "/app"
else:
    print("[INFO] Running locally")

    # Dynamically find the project root
    project_root = Path(__file__).resolve().parents[2]

# Add the project directories to the system path
sys.path.append(os.path.join(project_root, 'back_end'))

# Load environment variables from .env file
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import your actual classes and builder
# Assuming your project structure allows these imports
try:
    from app.app_graph.graph import builder
    from app.app_graph.state import AppAgentState, UserData, Movie # Import necessary state components
    from app.app_graph.configuration import AgentConfiguration
    from app.shared.state import InputState # Assuming InputState is in app.shared.state
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph.state import CompiledStateGraph
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
except ImportError as e:
    print(f"Error importing your modules: {e}")
    print("Please ensure your project structure and imports are correct.")
    print("Expected structure: app/app_graph/graph.py, app/app_graph/state.py, app/app_graph/configuration.py, app/shared/state.py")
    # Exit or handle the error appropriately in a real application
    # For this example, we'll let the FastAPI app start but it won't work correctly

def create_app_agent(with_memory: bool = False) -> CompiledStateGraph:
    """
    Create and compile a App Agent with optional memory capabilities.

    Args:
        with_memory (bool): If True, integrate a memory saver for storing and managing user memories
            during the agent's session. Defaults to False.

    Returns:
        CompiledStateGraph: The compiled App Agent ready for deployment with optional memory
            management.
    """
    # Use the imported builder from your graph.py
    if with_memory:
        # Compile with memory management using MemorySaver
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
    else:
        # Compile without memory management
        graph = builder.compile()

    graph.name = "AppGraph"
    return graph

# Instantiate the LangGraph agent with memory
# In a real application, you might use a persistent checkpointer (e.g., SQL, Redis)
# Ensure the builder is available before creating the agent
if 'builder' in locals():
    agent = create_app_agent(with_memory=True)
else:
    agent = None # Agent creation failed due to import errors

app = FastAPI()

class MessageRequest(BaseModel):
    user_id: str # Use user_id for session/thread management
    message: str

class MessageResponse(BaseModel):
    response: str
    messages: List[Dict[str, Any]] # Include full message history for client to manage

@app.post("/chat")
async def chat_endpoint(request: MessageRequest):
    """
    Receives user messages, invokes the LangGraph agent, and returns the response.
    Uses user_id to manage sessions/threads via the checkpointer.
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent failed to initialize due to import errors.")

    user_id = request.user_id
    user_message_content = request.message

    # Create a HumanMessage from the user's input
    user_message = HumanMessage(content=user_message_content)

    # Prepare the input state for the graph
    # Assuming your InputState expects a list of messages
    input_state = InputState(messages=[user_message])

    # Prepare the configuration for the graph invocation
    # The user_id is passed in the 'configurable' dictionary for the checkpointer
    config = {
        "configurable": {
            "thread_id": user_id, # Use user_id as the thread_id for the checkpointer
            "user_id": user_id # Also pass user_id if your nodes need it directly
        }
        # You might need to add other configuration parameters here
        # based on what your AgentConfiguration expects or what your nodes need
        # For example, if your AgentConfiguration requires specific model names:
        # "configurable": {
        #     "thread_id": user_id,
        #     "user_id": user_id,
        #     "query_model": "your_query_model_name",
        #     "response_model": "your_response_model_name"
        # }
        # However, your graph.py appears to load config from the runnable_config,
        # so passing them in 'configurable' should work if AgentConfiguration.from_runnable_config
        # is implemented to read from there.
    }

    try:
        # Invoke the LangGraph agent asynchronously
        # The agent's state will be loaded/saved automatically by the checkpointer
        # using the provided thread_id in the config.
        response_state: AppAgentState = await agent.ainvoke(input_state, config=config)

        # Extract the last AI message from the response state
        # Or extract whatever part of the state you want to send back
        ai_response = "No AI response generated."
        # Find the last AIMessage in the response state messages
        # Iterate in reverse to get the most recent AI message
        for msg in reversed(response_state.messages):
             if isinstance(msg, AIMessage):
                  ai_response = msg.content
                  break
             elif isinstance(msg, ToolMessage):
                  # Optionally handle ToolMessages if you want to display them
                  # You might want to format this differently for the client
                  ai_response = f"Tool executed: {msg.tool_call_id} - {msg.content}"
                  break
             # Add other message types if needed

        # Prepare the full message history to send back (optional, but useful for client sync)
        # Convert BaseMessage objects to dictionaries for JSON serialization
        message_history_dicts = []
        for msg in response_state.messages:
             # Include type, content, and potentially additional_kwargs
             message_history_dicts.append({
                 "type": msg.__class__.__name__,
                 "content": msg.content,
                 "additional_kwargs": msg.additional_kwargs
                 # Add other relevant attributes from BaseMessage if needed
             })


        return MessageResponse(response=ai_response, messages=message_history_dicts)

    except Exception as e:
        print(f"Error invoking agent: {e}")
        # In a production environment, you might want to log the full traceback
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {e}")

# To run this server:
# 1. Save the code as main.py in the root of your project or appropriate location.
# 2. Ensure your project structure is correct so that imports like app.app_graph.graph work.
# 3. Install necessary libraries: pip install fastapi uvicorn langgraph langchain-core pydantic
#    (You likely already have langgraph and langchain-core from your agent setup)
# 4. Run from your terminal from the project root: uvicorn main:app --reload
#    (Adjust the command if main.py is not in the root)
