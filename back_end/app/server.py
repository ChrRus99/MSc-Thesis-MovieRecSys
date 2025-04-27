from fastapi import FastAPI, Request, status # Import status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import uuid
import asyncio # Import asyncio
from collections import defaultdict # Import defaultdict

import sys
from dotenv import load_dotenv

sys.path.append("D:\\Internship\\recsys\\back_end")
dotenv_path = "D:\\Internship\\recsys\\.env"
load_dotenv(dotenv_path)

from app.app_agent import create_app_agent
# Import movie_graph builder and MemorySaver
from app.app_graph.movie_graph.graph import builder as movie_graph_builder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage # Import ToolMessage
from langgraph.constants import END # Import END

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (enhanced)
SESSIONS: Dict[str, dict] = {}

# --- Graph Initialization ---
# App Graph
app_graph_with_memory = create_app_agent(with_memory=True)

# Movie Graph
movie_checkpointer = MemorySaver()
movie_graph_with_memory = movie_graph_builder.compile(checkpointer=movie_checkpointer)

class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    # messages: List[Dict[str, Any]] # We'll rely on the checkpointer, but might need history for first message check
    user_input: str

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

# --- Server-side adaptation of stream_graph_updates ---
async def run_and_stream_updates_server(graph_to_stream, graph_input, config, graph_name="app", session_data: Dict = None):
    """
    Server-side function to stream graph updates, collect new AI and tool messages grouped by node,
    and update session state.
    Returns a tuple: (list of message groups [ {node_id: str, messages: List[str]} ], bool indicating if END was reached).
    """
    # Use defaultdict to group messages by node_id
    messages_by_node = defaultdict(list)
    added_raw_content_this_turn = set() # Still needed for deduplication within the stream
    switch_point_node = "get_first_user_query"
    reached_end = False

    try:
        async for update in graph_to_stream.astream(
            graph_input,
            config=config,
            stream_mode="updates",
        ):
            if END in update and update[END] is not None:
                print(f"--- DEBUG ({graph_name}): END node detected with non-None value in update: {update[END]}")
                reached_end = True

            for node_id, value in update.items():
                if node_id == END:
                    continue

                # --- App Graph Specific Logic ---
                if graph_name == "app" and node_id == switch_point_node and session_data is not None:
                     print(f"Node '{switch_point_node}' executed in app_graph. Setting switch flag for session {config['configurable']['thread_id']}.")
                     session_data["app_graph_completed_switch_point"] = True # Update session data directly

                # --- Generic Message Processing (AI and Tool Messages) ---
                if node_id != END: # Redundant check, but safe
                    if isinstance(value, dict) and value.get("messages") and isinstance(value["messages"], list):
                        messages_in_update = value["messages"]
                        # Collect messages (AI or Tool) since the last Human message in this chunk
                        messages_since_last_human = []

                        for msg in reversed(messages_in_update):
                            msg_content = None
                            msg_type_name = ""
                            tool_name = None # For ToolMessages

                            if isinstance(msg, BaseMessage):
                                msg_content = msg.content
                                msg_type_name = type(msg).__name__
                                if isinstance(msg, ToolMessage):
                                    tool_name = getattr(msg, 'name', 'unknown_tool') # Get tool name if available
                            elif isinstance(msg, dict):
                                msg_content = msg.get("content")
                                role = msg.get("role")
                                msg_type = msg.get("type")
                                if role == "assistant" or msg_type == "ai":
                                     msg_type_name = "AIMessage"
                                elif role == "user":
                                     msg_type_name = "HumanMessage"
                                elif msg_type == "tool": # Check dict type for tool
                                     msg_type_name = "ToolMessage"
                                     tool_name = msg.get('name', 'unknown_tool')

                            if msg_type_name == "HumanMessage":
                                break # Stop at the last human message in this chunk

                            formatted_message = None
                            raw_content_key = None # Key for duplicate check

                            if msg_type_name == "AIMessage" and msg_content:
                                node_prefix = f"[{graph_name}:{node_id}]"
                                formatted_message = f"{node_prefix} {msg_content}"
                                raw_content_key = msg_content # Use raw AI content for deduplication
                            elif msg_type_name == "ToolMessage" and msg_content:
                                # Format tool messages with a specific prefix
                                formatted_message = f"[tool:{tool_name}] {msg_content}"
                                raw_content_key = formatted_message # Use formatted tool message for deduplication (includes name)

                            if formatted_message and raw_content_key:
                                messages_since_last_human.append((formatted_message, raw_content_key))

                        messages_since_last_human.reverse() # Restore chronological order

                        # Add unique messages to the correct node group
                        for formatted_msg, raw_key in messages_since_last_human:
                            if raw_key not in added_raw_content_this_turn:
                                messages_by_node[node_id].append(formatted_msg) # Group by node_id
                                added_raw_content_this_turn.add(raw_key)

        # Convert defaultdict to the desired list format
        grouped_messages_list = [
            {"node_id": node_id, "messages": msgs}
            for node_id, msgs in messages_by_node.items() if msgs # Only include nodes that produced messages
        ]

        print(f"--- DEBUG ({graph_name}): Stream finished. reached_end = {reached_end}")
        return grouped_messages_list, reached_end # Return grouped messages

    except Exception as e:
        # Return error within the grouped structure
        print(f"Error during server-side graph execution ({graph_name}): {e}")
        error_message = f"[{graph_name}:error] An error occurred: {e}"
        return [{"node_id": "error", "messages": [error_message]}], False


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    user_id = req.user_id
    user_input = req.user_input

    # --- Session Management ---
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "active_graph": "app",
            "app_graph_completed_switch_point": False,
            "turn_count": 0, # Track turns for first message logic
        }
    session_data = SESSIONS[session_id]
    session_data["turn_count"] += 1

    # --- Graph Execution Logic (Replicating run_graph) ---
    run_config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        }
    }

    # Decide if we need to switch permanently
    if session_data["app_graph_completed_switch_point"] and session_data["active_graph"] == "app":
        print(f"Session {session_id}: Switching permanently to movie_graph.")
        session_data["active_graph"] = "movie"

    active_graph_name = session_data["active_graph"]
    graph_input = None
    graph_to_run = None
    grouped_messages = []
    graph_ended = False # Initialize flag for graph end state

    print(f"\n--- Session {session_id}: Running Graph: {active_graph_name} (Turn {session_data['turn_count']}) ---")
    print(f"User Input: {user_input}")

    if active_graph_name == "app":
        graph_to_run = app_graph_with_memory
        # Determine input format for app_graph
        # Check turn count for this specific session
        is_first_message_ever_in_session = session_data["turn_count"] == 1

        if is_first_message_ever_in_session and user_input:
             graph_input = {"messages": [HumanMessage(content=user_input)]}
             print("App Graph Input Type: Initial Dict")
        elif user_input:
             graph_input = Command(resume=user_input)
             print("App Graph Input Type: Command(resume)")
        else:
             return {"assistant_message": "[app:error] No user input provided."}

        # Stream the app graph server-side
        grouped_messages, graph_ended = await run_and_stream_updates_server( # Capture grouped messages
            graph_to_run, graph_input, run_config, graph_name="app", session_data=session_data
        )

    elif active_graph_name == "movie":
        graph_to_run = movie_graph_with_memory

        # Determine if this is the *very first* time movie_graph is run in this session
        # We check if the switch *just* happened (active_graph was app, now movie)
        # A simple way: check if turn_count > 1 and active_graph became "movie" in *this* request
        # Or, more reliably, check if movie_graph has run before in this session (needs another flag)
        if "movie_graph_has_run" not in session_data:
             session_data["movie_graph_has_run"] = False # Initialize flag

        is_first_movie_run_ever = not session_data["movie_graph_has_run"]

        if is_first_movie_run_ever and user_input:
            graph_input = {"messages": [HumanMessage(content=user_input)]}
            print(f"Movie Graph Input Type: Initial Dict (Input: '{user_input}')")
            session_data["movie_graph_has_run"] = True # Mark as run
        elif user_input:
            graph_input = Command(resume=user_input)
            print("Movie Graph Input Type: Command(resume)")
        else:
             return {"assistant_message": "[movie:error] No user input provided."}

        # Stream the movie graph server-side
        grouped_messages, graph_ended = await run_and_stream_updates_server( # Capture grouped messages
            graph_to_run, graph_input, run_config, graph_name="movie", session_data=session_data # Pass session_data if needed by stream func
        )

    # --- Response Formatting ---
    # Handle potential goodbye message addition
    if active_graph_name == "movie" and graph_ended:
        print("--- DEBUG: Adding goodbye message ---")
        goodbye_message = "[system] Thank you for using the movie service! Goodbye."
        # Check if the last group already contains messages, otherwise add a new group
        if grouped_messages and grouped_messages[-1]["messages"]:
             grouped_messages[-1]["messages"].append(goodbye_message)
        else:
             # Add a new group for the system message if no messages were generated or last group was empty
             grouped_messages.append({"node_id": "system", "messages": [goodbye_message]})


    # Handle case where no messages were generated at all (and no goodbye message)
    if not grouped_messages:
        print("--- DEBUG: Setting 'No response generated' message ---")
        grouped_messages = [{"node_id": "system", "messages": ["[system] No response generated."]}]

    # --- DEBUG ---
    print(f"--- DEBUG Response Formatting ---")
    print(f"    active_graph_name: {active_graph_name}")
    print(f"    graph_ended: {graph_ended}")
    print(f"    grouped_messages: {grouped_messages}") # Now includes grouped messages
    # --- END DEBUG ---

    # Return the structured list of message groups
    return {"grouped_assistant_messages": grouped_messages}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
