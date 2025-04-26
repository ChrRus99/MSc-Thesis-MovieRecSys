import streamlit as st
import requests
import uuid # To generate a unique user ID for the session

# --- Configuration ---
# Replace with the actual URL where your FastAPI server is running
FASTAPI_SERVER_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{FASTAPI_SERVER_URL}/chat"

# --- Streamlit App ---

st.set_page_config(page_title="LangGraph Chat Client", layout="wide")
st.title("LangGraph Chat Client")

# Initialize session state variables
if 'user_id' not in st.session_state:
    # Generate a unique user ID for this session
    st.session_state['user_id'] = str(uuid.uuid4())
    st.write(f"Your session ID: {st.session_state['user_id']}") # Display for debugging/info

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Display chat messages from history
for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user message
user_input = st.chat_input("Enter your message:")

if user_input:
    # Add user message to history and display immediately
    st.session_state['chat_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare data to send to FastAPI server
    payload = {
        "user_id": st.session_state['user_id'],
        "message": user_input
    }

    # Send message to FastAPI server
    try:
        # Use a loading spinner while waiting for the response
        with st.spinner("Thinking..."):
            response = requests.post(CHAT_ENDPOINT, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            agent_response_data = response.json()

        # Extract the agent's response and the full message history
        agent_response_text = agent_response_data.get("response", "Error: No response from agent.")
        # Optionally, update the entire chat history from the server's response
        # This ensures the client's history is perfectly in sync with the server's state
        # However, the format needs to match Streamlit's expectation {"role": ..., "content": ...}
        server_messages = agent_response_data.get("messages", [])
        # Convert server's message format (e.g., {"type": "AIMessage", "content": "..."})
        # to Streamlit's format {"role": ..., "content": ...}
        # You might need to adjust this mapping based on the actual 'type' strings
        # your LangChain/LangGraph messages use.
        formatted_server_history = []
        for msg in server_messages:
            role = "assistant" if msg.get("type") == "AIMessage" else "user" if msg.get("type") == "HumanMessage" else "tool"
            formatted_server_history.append({"role": role, "content": msg.get("content", "...")})

        # Update the session state history with the history from the server
        # This is a robust way to keep client and server history aligned
        st.session_state['chat_history'] = formatted_server_history


        # Display the last agent message (which should now be in the updated history)
        # Find the last message that is not from the user we just sent
        last_agent_message = None
        for msg in reversed(st.session_state['chat_history']):
             if msg["role"] != "user":
                  last_agent_message = msg
                  break

        if last_agent_message:
             with st.chat_message(last_agent_message["role"]):
                  st.markdown(last_agent_message["content"])
        else:
             # Fallback if no agent message is found in the history (shouldn't happen if server works)
             st.chat_message("assistant").markdown(agent_response_text)


    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to FastAPI server or server error: {e}")
        # Optionally remove the last user message if sending failed
        if st.session_state['chat_history'] and st.session_state['chat_history'][-1]["role"] == "user":
             st.session_state['chat_history'].pop()

    # Rerun the app to clear the input box and display the new message
    st.rerun()

# To run this client:
# 1. Save the code as client.py
# 2. Install necessary libraries: pip install streamlit requests uuid
# 3. Make sure your FastAPI server is running (uvicorn main:app --reload)
# 4. Run from your terminal: streamlit run streamlit_app.py
