# info_and_recommendation_graph/graph.py

import copy
from typing import Any, Annotated, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.app_graph.configuration import AgentConfiguration
from app.shared.state import InputState
from app.app_graph.info_and_recommendation_graph.state import RecommendationAgentState
from app.app_graph.prompts import ANALYZE_AND_ROUTE_INFO_AND_RECOMMENDATION_SYSTEM_PROMPT

from app.agents.query_analyzer_agent import create_query_analyzer_agent
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
) -> Command[Literal["evaluator" "human"]]:
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
async def analyze_and_route_query(
    state: RecommendationAgentState, *, config: RunnableConfig
) -> Command[Literal["general_question", "movie_info_retrieval", "movie_recommendation"]]:   
    ### Analyze the query and route it to the appropriate agent

    # Load agent's configuration settings
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Create a query_analyzer agent
    query_analyzer_agent = create_query_analyzer_agent(
        state=state,
        prompt=ANALYZE_AND_ROUTE_INFO_AND_RECOMMENDATION_SYSTEM_PROMPT,
        llm=model,
        verbose=True
    )

    # Extract the last user message
    user_message = state.messages[-1]

    # Invoke the query_analyzer agent with the user message and chat history
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
            # "user_id": state.user_id,
            # "is_user_registered": state.is_user_registered
        },
        goto=route
    )


@log_node_state_after_return
async def general_question(
    state: RecommendationAgentState, *, config: RunnableConfig
):
    # Use simple ReactAgent to handle general questions and ask for clarification if needed
    # TODO

    pass


@log_node_state_after_return
async def movie_info(
    state: RecommendationAgentState, *, config: RunnableConfig
):
    # Call graph info_graph
    # TODO

    pass


@log_node_state_after_return
async def movie_recommendation(
    state: RecommendationAgentState, *, config: RunnableConfig
):
    # Call graph recommendation_graph
    # TODO

    pass


@log_node_state_after_return
async def evaluator(
    state: RecommendationAgentState, *, config: RunnableConfig
):
    # Evaluate the response from the human_node and generate a report
    # TODO: create evaluator agent + logic to generate a report (JSON format)

    pass


# Define the graph
builder = StateGraph(RecommendationAgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node("human", human_node)
builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_node("general_question", general_question)
builder.add_node("movie_info_retrieval", movie_info)
builder.add_node("movie_recommendation", movie_recommendation)
builder.add_node("evaluator", evaluator)

builder.add_edge(START, "analyze_and_route_query")
builder.add_edge(["general_question", "movie_info_retrieval", "movie_recommendation"], "evaluator")
builder.add_edge("evaluator", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "InfoAndRecommendationGraph"