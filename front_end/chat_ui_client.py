import asyncio
from typing import List, Optional, Tuple

from langserve import RemoteRunnable    # Connects to a backend API for processing user input
import streamlit as st                  # Used for the UI, chat messages, and session management
from streamlit.logger import get_logger

##############################################################
#   Run Instruction (from CMD): 'streamlit run chat_ui_client.py'   #
#   Client avaliable at web page: http://localhost:8501      #
##############################################################

# Workflow:
#   1. User enters a message in the chat input box.
#   2. Message is displayed in the chat UI.
#   3. A request is sent to the backend API (http://api:8080/movie-agent/).
#   4. Streaming response starts, updating the UI token by token.
#   5. Status updates are shown while processing.
#   6. Final response is displayed in the chat.
#   7. Chat history is saved for future interactions.


# Initialize logger
logger = get_logger(__name__)

# Set Streamlit app title
st.title("Movie agent")

# Initialize chat history in session state
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "user_input" not in st.session_state:
    st.session_state["user_input"] = []

# Display chat history (showing last three exchanges only)
if st.session_state["generated"]:
    size = len(st.session_state["generated"])
    for i in range(max(size - 3, 0), size):
        with st.chat_message("user"):
            st.markdown(st.session_state["user_input"][i])
        with st.chat_message("assistant"):
            st.markdown(st.session_state["generated"][i])

class StreamHandler:
    """
        Class to handle streamed responses from the chatbot.

        Attributes:
            container (st.empty): UI container for displaying text. 
            status (st.status): Status indicator. 
            initial_text (str): Initial text for streaming response.
    """
    def __init__(self, container, status, initial_text=""):
        self.container = container  # Stores the Streamlit UI component where text is updated.
        self.status = status        # Displays the chatbot's progress.
        self.text = initial_text    # Keeps track of text being streamed by the chatbot.

    # Append new tokens to the response and update UI
    def new_token(self, token: str) -> None:
        self.text += token
        self.container.markdown(self.text)

    # Update status message while generating response
    def new_status(self, status_update: str) -> None:
        status.update(label="Generating answer🤖", state="running", expanded=True)
        with status:
            st.write(status_update)

async def get_agent_response(
    input: str, 
    stream_handler: StreamHandler, 
    chat_history: Optional[List[Tuple]] = []
):
    """
        Function to get a response from the movie agent API asynchronously.
    
        Arguments:
            input (str): User input.
            stream_handler (StreamHandler): StreamHandler object.
            chat_history (Optional[List[Tuple]]): Chat history.
    """
    # Connects to a backend API to process the user's request
    #url = "http://api:8080/movie-agent/"
    url = "http://localhost:8000/movie-agent"   # Backend API URL
    st.session_state["generated"].append("")    # Initialize empty response
    remote_runnable = RemoteRunnable(url)       # Create a remote execution object

    # Stream the response from the API (piece by piece)
    async for chunk in remote_runnable.astream_log(
        {"input": input, "chat_history": chat_history}
    ):
        log_entry = chunk.ops[0]
        value = log_entry.get("value")

        # If response contains action steps, update status
        if isinstance(value, dict) and isinstance(value.get("steps"), list):
            for step in value.get("steps"):
                stream_handler.new_status(step["action"]["log"].strip("\n"))
        # If response contains text output, update UI
        elif isinstance(value, str) and "ChatOpenAI" in log_entry["path"]:
            st.session_state["generated"][-1] += value
            stream_handler.new_token(value)

def generate_history():
    """
        Function to generate chat history for contextual responses. 
        This function extracts the last three messages to provide context for the next response.
    """
    context = []
    if st.session_state["generated"]:
        size = len(st.session_state["generated"])
        for i in range(max(size - 3, 0), size):
            context.append(
                (st.session_state["user_input"][i], st.session_state["generated"][i])
            )
    return context

# Wait for the user to type a message
if prompt := st.chat_input("How can I help you today?"):
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Initialize the StreamHandler to display real-time updates
    with st.chat_message("assistant"):
        status = st.status("Generating answer 🤖")          # Show loading status
        stream_handler = StreamHandler(st.empty(), status)  # Create stream handler

    # Retrieve chat history
    chat_history = generate_history()  
    
    # Create a new event loop for asynchronous execution
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(get_agent_response(prompt, stream_handler, chat_history))  # Fetch response
    loop.close()

    status.update(label="Finished!", state="complete", expanded=False)  # Mark completion
    
    # Store the user's input for future reference
    st.session_state.user_input.append(prompt)
