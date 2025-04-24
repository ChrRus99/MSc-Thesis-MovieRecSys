# info_and_recommendation_graph/graph.py

import ast
import json
import re
from typing import Any, Annotated, Literal, TypedDict, cast
import asyncio
from pydantic import ValidationError

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.app_graph.configuration import AgentConfiguration
from app.shared.state import InputState
from app.app_graph.movie_graph.state import RecommendationAgentState
from app.app_graph.prompts import ANALYZE_AND_ROUTE_MOVIE_SYSTEM_PROMPT

from app.agents.query_analyzer_agent import create_query_analyzer_agent
from app.agents.movie_recommender_agent import create_movie_recommender_agent
from app.agents.evaluator_agent import create_evaluator_agent
from app.agents.utils import format_agent_structured_output

from app.shared.utils import load_chat_model
from app.shared.debug_utils import (
    state_log,
    tool_log,
    log_node_state_after_return,
)


@log_node_state_after_return
async def human_node(
    state: RecommendationAgentState, *, config: RunnableConfig
) -> Command[Literal["evaluation"]]:
    """
    A node for collecting user input via interrupt and routing based on the triggering node.
    """
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
        goto="evaluation"
    )


@log_node_state_after_return
async def analyze_and_route_query(
    state: RecommendationAgentState, *, config: RunnableConfig
) -> Command[Literal["general_question", "movie_information", "movie_recommendation"]]:   
    ### Analyze the query and route it to the appropriate agent

    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Create a query analyzer agent
    query_analyzer_agent = create_query_analyzer_agent(
        state=state,
        prompt=ANALYZE_AND_ROUTE_MOVIE_SYSTEM_PROMPT,
        llm=model,
        verbose=True
    )

    # Extract the last user message
    user_message = state.messages[-1]

    # Invoke the query analyzer agent with the user message and chat history
    structured_response = await query_analyzer_agent.ainvoke({
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
        },
        goto=route
    )


@log_node_state_after_return
async def general_question(
    state: RecommendationAgentState, *, config: RunnableConfig
):
    ### Use simple ReactAgent to handle general questions and ask for clarification if needed
    # TODO

    pass


@log_node_state_after_return
async def movie_information(
    state: RecommendationAgentState, *, config: RunnableConfig
) -> Command[Literal["human"]]:
    ### Call graph info_graph
    # TODO


    # Update the last active agent for eventual response refinement
    state.last_active_agent = "movie_information"  # Update the last active agent
    
    # Route to human node to get feedback via interrupt
    return Command(
        # update={
        #     "messages": state.messages,
        #     "movies": state.movies,
        #     "explanations": state.explanations,
        #     "last_active_agent": state.last_active_agent, # Ensure last_active_agent is updated
        #     "report": [], # Clear previous report
        #     "mood": [], # Clear previous mood
        # },
        goto="human" # Route to human node for feedback interrupt
    )


@log_node_state_after_return
async def movie_recommendation(
    state: RecommendationAgentState, *, config: RunnableConfig
) -> Command[Literal["human"]]:
    ### Call graph recommendation_graph

    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Retrieve the user_id from config
    state.user_id = config["configurable"]["user_id"]

    # Create a movie recommender agent
    movie_recommender_agent = create_movie_recommender_agent(state=state, llm=model, verbose=True)

    # Extract the last user message (which triggered the recommendation)
    user_message = next((msg for msg in reversed(state.messages) if msg.type == 'human'), None)
    if not user_message:
         raise ValueError("No user message found in state for recommendation.")

    # Filter out ToolMessages from chat history
    chat_history = state.messages[:-1] # Chat history (excluding the last message)
    filtered_chat_history = [
        msg for msg in chat_history if not isinstance(msg, ToolMessage)
    ]

    # --- Retry Logic ---
    MAX_RETRIES = 3
    agent_messages = ["Sorry, I encountered an issue generating recommendations. Please try again."] # Default error message
    movies = []
    explanations = {}

    for attempt in range(MAX_RETRIES):
        try:
            # Invoke the movie recommender agent with the user message and chat history
            structured_response = await movie_recommender_agent.ainvoke({
                "input": user_message.content,
                "chat_history": [filtered_chat_history[-1]] if filtered_chat_history else [] # Pass only the last message in chat history (to keep it short, due to verbose recommendations)
            })

            # --- Start of monkey-patch ---
            # This is a monkey-patch to handle the JSON string parsing issue and to ensure the agent's
            # output is a valid dictionary.

            # Extract the raw output string and clean potential markdown fences
            raw_output = structured_response.get("output", "") # Use .get() for safety
            if not raw_output:
                 raise ValueError("Agent returned empty output.") # Handle empty output case
            cleaned_output = re.sub(r"```json\n?(.*?)\n?```", r"\1", raw_output, flags=re.DOTALL).strip()

            # Attempt to parse the cleaned output using ast.literal_eval
            structured_response_dict = None
            try:
                # Use ast.literal_eval to safely parse Python literals
                structured_response_dict = ast.literal_eval(cleaned_output)
            except (ValueError, SyntaxError, json.JSONDecodeError) as parse_error: # Catch potential errors from literal_eval and json
                # Raise a specific error to be caught by the outer try/except
                raise ValueError(f"Error parsing agent output string: {parse_error}. Cleaned output: {cleaned_output}") from parse_error

            # Ensure structured_response_dict is a dictionary before accessing keys
            if not isinstance(structured_response_dict, dict):
                 raise TypeError(f"Expected a dictionary after parsing, but got {type(structured_response_dict)}. Cleaned output: {cleaned_output}")
            # --- End of monkey-patch ---

            # Extract the structured response from the agent's output
            # Use .get() with defaults to prevent KeyError if keys are missing after successful parsing
            agent_messages = structured_response_dict.get("messages", ["Here are some recommendations:"]) # Default intro
            movies = structured_response_dict.get("movies", [])
            explanations = structured_response_dict.get("explanations", {})

            # If everything succeeded, break the loop
            break

        except (AttributeError, ValidationError, TypeError, ValueError, Exception) as e:
            print(f"Error during movie recommendation (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt + 1 >= MAX_RETRIES:
                print("Max retries reached. Using default error response.")
                # Keep the default error message set before the loop
                # Optionally, re-raise the last exception if you want the graph to fail hard
                # raise e
            else:
                await asyncio.sleep(1) # Wait a second before retrying
                continue # Go to the next attempt

    # --- End of Retry Logic ---


    # Format the final response message
    final_response_parts = []
    final_response_parts.extend(agent_messages)  # Add introductory message(s)
    final_response_parts.append("\n")  # Add a newline

    # Add movie recommendations with explanations (only if movies were successfully found)
    if movies:
        if explanations:
            for movie_title in movies:
                explanation = explanations.get(movie_title, "No explanation available.")
                final_response_parts.append(f"- **{movie_title}:** {explanation}")
        else:
             for movie_title in movies:
                final_response_parts.append(f"- **{movie_title}**")
    elif not movies and attempt + 1 >= max_retries: # If retries failed and movies is empty
        pass # The error message from agent_messages is already set
    else: # If agent genuinely found no movies (not due to error)
        final_response_parts.append("I couldn't find specific recommendations based on that.")


    final_response_content = "\n".join(final_response_parts)

    # Create a single AIMessage with the formatted content
    final_ai_message = AIMessage(content=final_response_content)
    state.messages.append(final_ai_message)  # Update the state
    state.movies = movies
    state.explanations = explanations

    # Clear the report and mood after they have already been used for this recommendation cycle
    state.report = []
    state.mood = []

    # Update the last active agent for an eventual recommendation response refinement
    state.last_active_agent = "movie_recommendation"

    # Ask for feedback *after* providing recommendations
    feedback_request_message = AIMessage(content="What do you think of these recommendations?")
    state.messages.append(feedback_request_message)

    # Route to human node to get feedback via interrupt
    return Command(
        update={
            "messages": state.messages,
            "movies": state.movies,
            "explanations": state.explanations,
            "last_active_agent": state.last_active_agent, # Ensure last_active_agent is updated
            "report": [], # Clear previous report
            "mood": [], # Clear previous mood
        },
        goto="human" # Route to human node for feedback interrupt
    )


@log_node_state_after_return
async def evaluation(
    state: RecommendationAgentState, *, config: RunnableConfig
) -> Command[Literal["movie_information", "movie_recommendation", "__end__"]]:
    """
    Evaluates the user's feedback (received via human_node interrupt) on the previous agent's response,
    saves preferences, and updates the state with a report, mood, and satisfaction status.
    Routes based on satisfaction.
    """
    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Retrieve the user_id from config
    state.user_id = config["configurable"]["user_id"]

    # Create an evaluator agent
    evaluator_agent = create_evaluator_agent(state=state, llm=model, verbose=True)

    # Extract the last user message (the feedback)
    user_feedback_message = state.messages[-1]

    # Filter out ToolMessages from chat history
    chat_history = state.messages[:-1] # Chat history (excluding the last message)
    filtered_chat_history = [
        msg for msg in chat_history if not isinstance(msg, ToolMessage)
    ]

    # Invoke the evaluator agent with the user feedback and chat history
    structured_response = await evaluator_agent.ainvoke({
        "input": user_feedback_message.content,
        "chat_history": filtered_chat_history,
    })

    # Extract the structured response
    structured_response_dict = format_agent_structured_output(structured_response["output"])

    # Handle potential missing keys gracefully
    preferences_saved_confirmation = structured_response_dict.get("preferences_saved_confirmation", "No preference update status available.")
    is_user_satisfied = structured_response_dict.get("is_user_satisfied", True)
    report = structured_response_dict.get("report", [])
    mood = structured_response_dict.get("mood", [])

    # Append confirmation message if provided and not empty/default
    confirmation_message = None
    if preferences_saved_confirmation and preferences_saved_confirmation != "No new preferences identified to save.":
         confirmation_message = AIMessage(content=preferences_saved_confirmation)
         state.messages.append(confirmation_message)

    # Update state with evaluation results
    state.is_user_satisfied = is_user_satisfied
    state.report = report
    state.mood = mood
    
    return Command(
        update={
            "messages": state.messages,
            "is_user_satisfied": state.is_user_satisfied,
            "report": state.report,
            "mood": state.mood,
        },
        # Determine the next step based on user satisfaction
        goto=state.last_active_agent if not is_user_satisfied else END
    )


# Define the graph
builder = StateGraph(RecommendationAgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node("human", human_node)
builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_node("general_question", general_question)
builder.add_node("movie_information", movie_information)
builder.add_node("movie_recommendation", movie_recommendation)
builder.add_node("evaluation", evaluation)

builder.add_edge(START, "analyze_and_route_query")
builder.add_edge("general_question", "analyze_and_route_query")
# builder.add_edge("movie_information", "evaluation")
# builder.add_edge("movie_recommendation", "evaluation")
# builder.add_edge("evaluation", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "InfoAndRecommendationGraph"