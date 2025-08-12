import streamlit as st
import requests
import uuid
import re # Import re for parsing tool prefix

st.set_page_config(page_title="Movie RecSys Chatbot (Client)", page_icon="🎬")
st.title("🎬 Movie RecSys Chatbot (Client)")

API_URL = "http://localhost:8000/chat" # Make sure this matches your server address

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    #st.session_state.user_id = str(uuid.uuid4())
    st.session_state.user_id = "bdd62e39-8999-468b-be8a-c36277a93bdc"
# Store messages locally for display purposes
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
# Add state for tool visibility toggle
if "show_tool_calls" not in st.session_state:
    st.session_state.show_tool_calls = True # Default to showing tool calls

# --- UI Elements ---
# Add the toggle button (e.g., in the sidebar or below the title)
st.session_state.show_tool_calls = st.toggle(
    "Show Tool Calls",
    value=st.session_state.show_tool_calls,
    help="Enable to see details about the tools being used by the assistant."
)

# --- Display Chat History ---
# Use the local display_messages list
for msg in st.session_state.display_messages:
    role = msg["role"]

    if role == "user":
        # Display user message in one bubble
        with st.chat_message(role):
            full_content = msg["content"]
            display_content = full_content
            if display_content.startswith("[you] "):
                display_content = display_content[len("[you] "):]
            st.write(f"[you] {display_content}")
    elif role == "assistant":
        # Assistant messages are now potentially grouped
        grouped_content = msg.get("grouped_content", []) # Get the list of groups
        for group in grouped_content:
            node_id = group.get("node_id", "assistant") # Fallback node_id
            messages_in_group = group.get("messages", [])

            if not messages_in_group: # Skip empty groups
                continue

            # Create ONE chat bubble for EACH group (i.e., each node's output)
            with st.chat_message(role): # Role is 'assistant'
                # Optionally display node_id as a small header (can be commented out)
                # st.caption(f"From: {node_id}")

                # Process and display all lines within this group's bubble
                for line in messages_in_group:
                    if not line: # Skip empty lines within the group
                        continue

                    # Check for tool message prefix
                    tool_match = re.match(r"\[tool:([^\]]+)\]\s*(.*)", line)
                    if tool_match:
                        if st.session_state.show_tool_calls:
                            tool_name = tool_match.group(1)
                            tool_content = tool_match.group(2)
                            st.markdown(f"🛠️ **Tool:** `{tool_name}`: {tool_content}")
                    # Check for system message prefix
                    elif line.startswith("[system]"):
                         st.markdown(f"⚙️ {line}")
                    # Check for error message prefix
                    elif line.startswith("[") and ":error]" in line:
                         st.markdown(f"🚨 {line}")
                    # Otherwise, display as regular assistant message
                    else:
                        st.markdown(line, unsafe_allow_html=True)

# --- Chat Input and Server Interaction ---
if prompt := st.chat_input("What would you like to talk about?"):
    # Add user message to local display state
    # Store the raw prompt, the display loop handles the "[you]" prefix
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    # Display immediately using the same logic as the history loop
    with st.chat_message("user"):
        st.write(f"[you] {prompt}")

    # Prepare payload for API (minimal, relying on server checkpointer)
    payload = {
        "session_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
        "user_input": prompt,
    }

    # Show loading spinner while waiting for the server
    with st.spinner("Thinking..."):
        try:
            # Send request to the server
            response = requests.post(API_URL, json=payload, timeout=120) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            # Get the grouped assistant messages from the response
            grouped_assistant_messages = data.get("grouped_assistant_messages", [])

            # Add grouped assistant messages to local display state
            if grouped_assistant_messages: # Only add if there are messages
                st.session_state.display_messages.append({
                    "role": "assistant",
                    "grouped_content": grouped_assistant_messages # Store the list of groups
                })

            # Rerun Streamlit to display the new messages
            st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to contact server: {e}")
            # Add error as a system message group
            st.session_state.display_messages.append({
                "role": "assistant",
                "grouped_content": [{"node_id": "error", "messages": [f"[system:error] Failed to get response: {e}"]}]
            })
            st.rerun()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # Add error as a system message group
            st.session_state.display_messages.append({
                "role": "assistant",
                "grouped_content": [{"node_id": "error", "messages": [f"[system:error] Unexpected error: {e}"]}]
            })
            st.rerun()
