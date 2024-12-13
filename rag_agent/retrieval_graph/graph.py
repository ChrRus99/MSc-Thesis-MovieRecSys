"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational retrieval graph. 
It includes the main graph definition, state management, and key functions for processing & routing
user queries, generating research plans to answer user questions, conducting research, and
formulating responses.
"""

from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState, Router
from shared.utils import format_docs, load_chat_model


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result 
                           (classification type and logic).
    """
    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Format the system prompt to route the query
    system_prompt = configuration.router_system_prompt

    # Combine the system prompt with the agent's conversation history
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    # Use the model to classify the query and ensure output matches the Router structure
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )

    # Return the classification result
    return {"router": response}


def route_query(
    state: AgentState
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: 
        The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    # Retrieve the classification type from the router in the agent's state
    _type = state.router["type"]

    # Route based on the classification type
    match _type:
        #case "movie-recommendation":
        case "langchain":
            return "create_research_plan"
        case "more-info":
            return "ask_for_more_info"
        case "general":
            return "respond_to_general_query"
        case _:
            # Raise an error if the type is not recognized
            raise ValueError(f"Unknown router type {_type}")


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and 
                            router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Format the system prompt (using the router's logic) to request more information
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    
    # Combine the system prompt with the agent's conversation history
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    # Return the generated response
    response = await model.ainvoke(messages)   
    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and 
                            router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Format the system prompt (using the router's logic) to respond to the general query
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )

    # Combine the system prompt with the agent's conversation history
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    # Return the response to the general query
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a specific query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]
        """A list of steps in the research plan."""

    # Load the agent's language model specified in the input config and
    # configure it to produce structured output matching the Plan TypedDict
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)

    # Combine the system prompt with the agent's conversation history   
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages

    # Use the model to generate the research plan and ensure output matches the Plan structure
    response = cast(Plan, await model.ainvoke(messages))

    # Return the research steps and a placeholder for 'documents'
    return {"steps": response["steps"], "documents": "delete"}


# TODO ADD TOOL HERE TO CONDUCT RESEARCH

async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    # Invoke the researcher_graph with the first research step to conduct research
    result = await researcher_graph.ainvoke({"question": state.steps[0]})

    # Return the research results ('documents') and the updated research plan ('steps' without the first step)
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node.
        - Otherwise, route to the `respond` node.

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is 
                                                complete.
    """
    # Route based on the state of the research process
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents
    retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and 
                            conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)

    # Format the retrieved documents into a suitable context for the response prompt
    context = format_docs(state.documents)

    # Format the system prompt (using the retrieved documents) to respond to the specific query
    system_prompt = configuration.response_system_prompt.format(context=context)

    # Combine the system prompt with the agent's conversation history
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    # Return the response based on the retrieved documents
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(create_research_plan)
builder.add_node(conduct_research)
builder.add_node(respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
