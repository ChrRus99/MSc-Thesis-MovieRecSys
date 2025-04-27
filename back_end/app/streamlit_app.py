import streamlit as st
import asyncio
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import sys
from dotenv import load_dotenv

# --- Path and Env Setup ---
sys.path.append("D:\\Internship\\recsys\\back_end")
dotenv_path = "D:\\Internship\\recsys\\.env"
load_dotenv(dotenv_path)

# --- Graph Imports ---
try:
    from app.app_agent import create_app_agent
    # Import the builder object, NOT the already compiled graph
    from app.app_graph.movie_graph.graph import builder as movie_graph_builder
except ImportError:
    st.error("Failed to import graph components. Make sure paths are correct.")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="Movie RecSys Chatbot", page_icon="🎬")
st.title("🎬 Movie RecSys Chatbot")

# --- Session State Initialization ---
# State variables for one-way graph switching
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    #st.session_state.user_id = str(uuid.uuid4())
    st.session_state.user_id = "bdd62e39-8999-468b-be8a-c36277a93bdc"
if "messages" not in st.session_state:
    st.session_state.messages = []
# active_graph: 'app' initially, switches permanently to 'movie'
if "active_graph" not in st.session_state:
    st.session_state.active_graph = "app"
# Flag to know if the switch point has been reached
if "app_graph_completed_switch_point" not in st.session_state:
     st.session_state.app_graph_completed_switch_point = False
# movie_graph_start_input is no longer needed, the first input *after* the switch is used directly

# --- Checkpointer Setup ---
# Checkpointer for the main app_graph
if "graph_with_memory" not in st.session_state:
    st.session_state.graph_with_memory = create_app_agent(with_memory=True)

# Checkpointer for the movie_graph (NEW)
movie_checkpointer = MemorySaver()
if "movie_graph_with_memory" not in st.session_state:
    # Compile the imported movie_graph_builder with its own checkpointer
    st.session_state.movie_graph_with_memory = movie_graph_builder.compile(checkpointer=movie_checkpointer) # Use builder here

# --- Runnable Config ---
# Use the same base config, thread_id ensures checkpointers work for the session
runnable_config = {
    "configurable": {
        "thread_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
    }
}

# --- Display Chat History ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        # Display user message with prefix, but store raw content
        st.chat_message(msg["role"]).write(f"[you] {msg.get('raw_content', msg['content'])}")
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# --- Graph Execution Logic ---
from langgraph.types import Command
from langgraph.constants import END

# Add graph_name back to the function signature
async def stream_graph_updates(graph_to_stream, graph_input, config, graph_name="app"):
    """Generic function to stream updates from a graph, detecting the switch point."""
    message_placeholder = st.empty()
    # Keep track of the RAW content of AI messages added this turn to avoid duplicates
    added_raw_ai_content_this_turn = set()
    last_displayed_content = "" # Track the last thing shown in the placeholder

    try:
        async for update in graph_to_stream.astream(
            graph_input,
            config=config,
            stream_mode="updates",
        ):
            # --- Debug: Print raw update ---
            # print(f"--- Raw Update ({graph_name}) ---")
            # print(update)
            # print("--------------------------")

            for node_id, value in update.items():

                # --- App Graph Specific Logic: Detect switch point completion ---
                # Check if the switch point node has finished executing in the app graph stream
                if graph_name == "app" and node_id == "get_first_user_query":
                     print(f"Node 'get_first_user_query' executed in app_graph. Ready to switch to movie_graph on next input.")
                     st.session_state.app_graph_completed_switch_point = True

                # --- Movie Graph Specific Logic: NO switch back ---
                # Removed logic checking for END in movie_graph to switch back

                # --- Generic Message Processing (Refined Duplicate Check) ---
                if node_id != END:
                    # Check for messages directly under the node_id
                    if isinstance(value, dict) and value.get("messages") and isinstance(value["messages"], list):
                        messages_in_update = value["messages"]
                        ai_messages_content_since_last_human = []

                        # Iterate in reverse order (as in demo code)
                        for msg in reversed(messages_in_update):
                            msg_content = None
                            msg_type_name = ""
                            if isinstance(msg, BaseMessage):
                                msg_content = msg.content
                                msg_type_name = type(msg).__name__
                            elif isinstance(msg, dict):
                                msg_content = msg.get("content")
                                if msg.get("role") == "assistant" or msg.get("type") == "ai":
                                     msg_type_name = "AIMessage"
                                elif msg.get("role") == "user":
                                     msg_type_name = "HumanMessage"

                            if msg_type_name == "HumanMessage":
                                # Stop collecting if we encounter a HumanMessage (as in demo code)
                                break
                            if msg_type_name == "AIMessage" and msg_content:
                                ai_messages_content_since_last_human.append(msg_content)

                        # Reverse collected AI messages back to original order (as in demo code)
                        ai_messages_content_since_last_human.reverse()

                        # Process the extracted new AI messages for this update
                        for ai_msg_content in ai_messages_content_since_last_human:
                            # Check if the RAW AI content has already been processed *this turn*
                            if ai_msg_content not in added_raw_ai_content_this_turn:
                                node_prefix = f"[{graph_name}:{node_id}]"
                                full_display_content = f"{node_prefix} {ai_msg_content}"

                                # Add to main history
                                st.session_state.messages.append({"role": "assistant", "content": full_display_content})
                                # Mark RAW content as added for this turn
                                added_raw_ai_content_this_turn.add(ai_msg_content)
                                # Update placeholder
                                message_placeholder.chat_message("assistant").write(full_display_content)
                                last_displayed_content = full_display_content
                            # Else: If raw content already added this turn, just update placeholder if needed to show progress
                            else:
                                node_prefix = f"[{graph_name}:{node_id}]"
                                full_display_content = f"{node_prefix} {ai_msg_content}"
                                if last_displayed_content != full_display_content:
                                     message_placeholder.chat_message("assistant").write(full_display_content)
                                     last_displayed_content = full_display_content


        # After the stream finishes, ensure the placeholder is cleared if nothing was added
        if not added_raw_ai_content_this_turn and message_placeholder: # Use the raw content set here too
             message_placeholder.empty()


    except Exception as e:
        error_msg = f"An error occurred during {graph_name} graph execution: {e}"
        st.error(error_msg)
        print(f"Error details: {e}")
        if message_placeholder: message_placeholder.empty()


async def run_graph(user_input: str | None):
    """Determines which graph to run based on whether the switch point was passed."""

    # Decide if we need to switch permanently
    if st.session_state.app_graph_completed_switch_point and st.session_state.active_graph == "app":
        print("Switch point passed, permanently switching to movie_graph.")
        st.session_state.active_graph = "movie"
        # The current user_input will be the first input for movie_graph

    active_graph_name = st.session_state.active_graph
    graph_input = None
    graph_to_run = None
    run_config = runnable_config

    print(f"\n--- Running Graph: {active_graph_name} ---")
    print(f"User Input: {user_input}")
    # print(f"App Graph Switch Point Completed: {st.session_state.app_graph_completed_switch_point}") # Debug

    if active_graph_name == "app":
        graph_to_run = st.session_state.graph_with_memory
        # Determine input format for app_graph
        is_first_message_ever = not any(msg["role"] == "user" for msg in st.session_state.messages[:-1])

        if is_first_message_ever and user_input:
             graph_input = {"messages": [HumanMessage(content=user_input)]}
             print("App Graph Input Type: Initial Dict")
        elif user_input:
             graph_input = Command(resume=user_input)
             print("App Graph Input Type: Command(resume)")
        else:
             print("[Warning] App graph active but no user_input provided.")
             return

        # Stream the app graph
        await stream_graph_updates(graph_to_run, graph_input, run_config, graph_name="app")

    elif active_graph_name == "movie":
        # Use the movie_graph compiled WITH its checkpointer (NEW)
        graph_to_run = st.session_state.movie_graph_with_memory

        # Determine if this is the *very first* time movie_graph is run in this session
        # We check if the switch *just* happened (active_graph was app, now movie)
        # A simpler way: check if any message from movie_graph exists in history yet.
        is_first_movie_run_ever = not any(msg["content"].startswith("[movie:") for msg in st.session_state.messages if msg["role"] == "assistant")

        if is_first_movie_run_ever and user_input:
            # Use the current user_input as the initial message for movie_graph
            graph_input = {"messages": [HumanMessage(content=user_input)]}
            print(f"Movie Graph Input Type: Initial Dict (Input: '{user_input}')")
        elif user_input:
            # Subsequent turns within movie graph use resume
            graph_input = Command(resume=user_input)
            print("Movie Graph Input Type: Command(resume)")
        else:
             print("[Warning] Movie graph active but no user_input provided.")
             return

        # Stream the movie graph
        if graph_input:
            await stream_graph_updates(graph_to_run, graph_input, run_config, graph_name="movie")
        else:
             print("[Info] No valid input determined for movie graph, skipping stream.")

# --- Chat Input ---
if prompt := st.chat_input("What would you like to talk about?"):
    # Store raw prompt and display formatted version
    st.session_state.messages.append({"role": "user", "content": prompt, "raw_content": prompt})
    st.chat_message("user").write(f"[you] {prompt}")

    # Run the graph logic
    try:
        # Check for existing event loop
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(run_graph(user_input=prompt))
        except RuntimeError: # No running event loop
            asyncio.run(run_graph(user_input=prompt))

        # Rerun Streamlit to immediately redraw the history with all appended messages
        st.rerun() # ADD THIS LINE

    except Exception as e:
        st.error(f"Failed to run the graph: {e}")

# --- Interruption Note ---
# Interrupts are handled by app_graph's human_node. Streamlit waits for chat_input.
# If movie_graph interrupts internally, app_graph needs to handle/propagate that if needed.
# From Streamlit's perspective, it just waits for the app_graph stream to finish or interrupt.
