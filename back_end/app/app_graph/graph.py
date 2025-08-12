"""Main entrypoint for the conversational recommendation graph.

This module defines the core structure and functionality of the conversational recommendation graph. 
It includes the main graph definition, state management, and key functions for processing & routing
user queries, handling the sign-up, the sing-in, and the issues report of the user, and to provide
personalized movies recommendations to the user.
"""

import copy
from typing import Any, Annotated, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.prompts import ChatPromptTemplate

from app.app_graph.configuration import AgentConfiguration
from app.shared.state import InputState
from app.app_graph.state import AppAgentState
from app.app_graph.movie_graph.graph import graph as movie_graph

from app.agents.greeting_agent import create_greeting_agent
from app.agents.sign_up_agent import create_sign_up_agent
from app.agents.sign_in_agent import create_sign_in_agent
from app.agents.user_ratings_agent import create_user_ratings_agent
from app.agents.get_first_query_agent import create_get_first_query_agent
from app.agents.utils import format_agent_structured_output

from app.shared.utils import load_chat_model
from app.shared.debug_utils import (
    state_log,
    tool_log,
    log_node_state_after_return,
)


@log_node_state_after_return
async def human_node(
    state: AppAgentState, *, config: RunnableConfig
) -> Command[Literal["sign_up", "ask_user_ratings", "get_first_user_query", "human"]]:
    """A node for collecting user input."""
    user_input = interrupt(value="Ready for user input.")

    # identify the last active agent
    # (the last active node before returning to human)
    langgraph_triggers = config["metadata"]["langgraph_triggers"]
    if len(langgraph_triggers) != 1:
        raise AssertionError("Expected exactly 1 trigger in human node")

    active_agent = langgraph_triggers[0].split(":")[1]

    return Command(
        update={
            "messages": [
                {
                    "role": "human",
                    "content": user_input,
                }
            ]
        },
        goto=active_agent,
    )


@log_node_state_after_return
async def greeting_and_route_query(
    state: AppAgentState, *, config: RunnableConfig
) -> Command[Literal["sign_up", "sign_in"]]:
    """
    Greet the user, then analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow, by checking whether the user is already registered.

    Args:
        state (AppAgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        Command: The routing command indicating the next step.
    """
    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Retrieve the user_id from config
    state.user_id = config["configurable"]["user_id"]
    
    # Create a greeting agent
    greeting_agent = create_greeting_agent(state=state, llm=model, verbose=True)   

    # Extract the last user message
    user_message = state.messages[-1]
    
    # Invoke the greeting agent with the user message and chat history
    structured_response = await greeting_agent.ainvoke({
        "input": user_message.content, 
        "chat_history": [] #state.messages[:-1]  # Pass chat history (excluding the last message)
    })

    # Extract the structured response from the agent's output
    structured_response_dict = format_agent_structured_output(structured_response["output"])
    agent_messages = structured_response_dict["messages"]
    route = structured_response_dict["route"]
    state.messages.extend(agent_messages)  # Update the state

    # Return the structured routing result
    return Command(
        update={
            "messages": state.messages,
            "user_id": state.user_id,
            "is_user_registered": state.is_user_registered
        },
        goto=route
    )


# # TODO: da implementare e integrare
# async def report_issue(
#     state: AppAgentState, *, config: RunnableConfig
# ) -> dict[str, list[BaseMessage]]:
#     """Generate a response asking the user to describe the issue.

#     This node is called when the router determines that the user is experiencing some troubles in
#     using this recommendation system or is reporting an issue.

#     This node calls a tool to save the issue, in order to notify the support team

#     Args:
#         state (AppAgentState): The current state of the agent, including conversation history and 
#                             router logic.
#         config (RunnableConfig): Configuration with the model used to respond.

#     Returns:
#         dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
#     """
#     pass


@log_node_state_after_return
async def sign_up(
    state: AppAgentState, *, config: RunnableConfig
) -> Command[Literal["sign_in", "human"]]:
    """
    Handles a user sign-up process using the sign-up agent runnable.

    Args:
        state (AppAgentState): The current conversation state.
        config (RunnableConfig): The configuration containing model and tool bindings.

    Returns:
        Command: The command indicating the next step (usually 'human' until handoff).
    """
    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Create the sign-up agent
    sign_up_agent = create_sign_up_agent(state=state, llm=model)

    # Filter only HumanMessage and AIMessage for the agent input
    filtered_state = copy.deepcopy(state)
    filtered_state.messages = [
        m for m in state.messages
        if type(m).__name__ in ("HumanMessage", "AIMessage")
    ]

    # Pass the filtered state (with only HumanMessage and AIMessage) to the agent
    response = await sign_up_agent.ainvoke(filtered_state)

    # Extract the last message from the response
    response_messages = response.get('messages', [])

    if response_messages:
        last_response_message = response_messages[-1]
        state.messages.append(last_response_message)  # Update the state
    else:
        # If the response does not contain messages, handle the error
        print("[WARNING] Sign-up agent's response did not contain 'messages'.")
        pass

    return Command(
        update={
            "messages": state.messages,
            "is_user_registered": state.is_user_registered,
            "user_data": state.user_data
        },
        goto="human"
    )


@log_node_state_after_return
async def sign_in(
    state: AppAgentState, *, config: RunnableConfig
) -> Command: #) -> Command[Literal["recommendation", "ask_user_ratings"]]:
    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    
    # Create a sign-in agent
    sign_in_agent = create_sign_in_agent(state=state, llm=model, verbose=True)   
    
    # Extract the last user message
    user_message = state.messages[-1]
    
    # Invoke the sign-in agent with the user message and chat history
    structured_response = await sign_in_agent.ainvoke({
        "input": user_message.content, 
        "chat_history": [] #state.messages[:-1]  # Pass chat history (excluding the last message)
    })

    # Extract the structured response from the agent's output
    structured_response_dict = format_agent_structured_output(structured_response["output"])
    agent_messages = structured_response_dict["messages"]
    state.messages.extend(agent_messages)  # Update the state

    # Return the structured routing result
    return Command(
        update={
            "messages": state.messages,
            "seen_movies": state.seen_movies
        },
        #goto="recommendation" if state.seen_movies else "ask_user_ratings"
    )

    # Load user preferences and other data
    # Do not generate any message, just load user preferences in state


@log_node_state_after_return
async def ask_user_ratings(
    state: AppAgentState, *, config: RunnableConfig
) -> Command[Literal["get_first_user_query", "human"]]:
    """
    Collects user-movie ratings and transitions to the next node in the graph.
    """
    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Create the user-ratings agent
    user_ratings_agent = create_user_ratings_agent(state=state, llm=model)  

    # Filter only HumanMessage and AIMessage for the agent input
    filtered_state = copy.deepcopy(state)
    filtered_state.messages = [
        m for m in state.messages
        if type(m).__name__ in ("HumanMessage", "AIMessage")
    ]

    # Pass the filtered state (with only HumanMessage and AIMessage) to the agent
    response = await user_ratings_agent.ainvoke(filtered_state)

    # Extract the last message from the response
    response_messages = response.get('messages', [])

    if response_messages:
        last_response_message = response_messages[-1]
        state.messages.append(last_response_message)  # Update the state
    else:
        # If the response does not contain messages, handle the error
        print("[WARNING] User-ratings agent's response did not contain 'messages'.")
        pass

    return Command(
        update={
            "messages": state.messages,
            "seen_movies": state.seen_movies
        },
        goto="human"
    )

@log_node_state_after_return
async def get_first_user_query(
    state: AppAgentState, *, config: RunnableConfig
) -> Command[Literal["human", "movie_info_and_recommendation"]]:
    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Create the get_first_query_agent
    get_first_query_agent = create_get_first_query_agent(state=state, llm=model)

    # Filter only HumanMessage and AIMessage for the agent input
    filtered_state = copy.deepcopy(state)
    filtered_state.messages = [
        m for m in state.messages
        if type(m).__name__ in ("HumanMessage", "AIMessage")
    ]
    # Pass only the last message to the agent
    filtered_state.messages = filtered_state.messages[-1] if filtered_state.messages else []

    # Pass the filtered state (with only HumanMessage and AIMessage) to the agent
    response = await get_first_query_agent.ainvoke(filtered_state)

    # Extract the last message from the response (if any)
    response_messages = response.get('messages', [])
    
    if response_messages:
        last_response_message = response_messages[-1]
        state.messages.append(last_response_message)  # Update the state
    else:
        # If the response does not contain messages, handle the error
        print("[WARNING] Get first user query agent's response did not contain 'messages'.")
        pass

    return Command(
        update={
            "messages": state.messages,
        },
        goto="human"
    )


#@log_node_state_after_return  # Use debug inside the movie_graph
async def movie_info_and_recommendation(
    state: AppAgentState, *, config: RunnableConfig
):
    """
    Invokes the movie agent sub-graph to handle movie information requests and recommendations.

    This node acts as an entry point to the specialized movie interaction graph. It prepares the
    necessary state and configuration, calls the movie agent, and updates the main application
    state upon completion of the sub-graph's execution.

    Args:
        state (AppAgentState): The current state of the main application agent.
        config (RunnableConfig): The configuration for the agent execution.

    Returns:
        Command: A command indicating the next step, typically ending the main graph flow
                 as the movie interaction is self-contained within the sub-graph.
    """
    
    print("--- Entering movie_graph subgraph ---")

    # Extract the last user message from the chat history
    last_user_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg
            break
    if last_user_message is None:
        # Handle the case where no human message is found, though this might indicate an issue
        print("[WARNING] No HumanMessage found in state.messages for movie_info_and_recommendation node.")

    print("----------------> State messages: ", state.messages)
    print("----------------> Last user message: ", last_user_message)

    # Call the movie agent sub-graph passing only the last user message
    result = await movie_graph.ainvoke({"messages": last_user_message})

    print("--- Exiting movie_graph subgraph ---")


# Define the graph
builder = StateGraph(AppAgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node("human", human_node)
builder.add_node("greeting_and_route_query", greeting_and_route_query)
#builder.add_node(report_issue)
builder.add_node("sign_up", sign_up)
builder.add_node("sign_in", sign_in)
builder.add_node("ask_user_ratings", ask_user_ratings)
builder.add_node("get_first_user_query", get_first_user_query)
builder.add_node("movie_info_and_recommendation", movie_info_and_recommendation)

builder.add_edge(START, "greeting_and_route_query")
builder.add_edge("sign_in", "ask_user_ratings")
builder.add_edge("movie_info_and_recommendation", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "AppGraph"
