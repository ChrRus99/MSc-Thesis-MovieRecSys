"""Main entrypoint for the conversational recommendation graph.

This module defines the core structure and functionality of the conversational recommendation graph. 
It includes the main graph definition, state management, and key functions for processing & routing
user queries, handling the sign-up, the sing-in, and the issues report of the user, and to provide
personalized movies recommendations to the user.
"""

from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from main_graph.configuration import AgentConfiguration
from rag_agent.retrieval_graph.graph import graph as rag_graph
from main_graph.state import AgentState, InputState, Router
from main_graph.tool import save_report_tool, sign_up_tool
#from shared.utils import format_docs, load_chat_model


def route_restore_previous_conversation(
    state: AgentState
) -> Literal["greeting", "sing-in", "issue"]:
    """ Restore the user's conversation from the last agent's state used. Default 'greeting' node.

    Args:
        state (AgentState): The last used state of the agent, including the router's classification.

    Returns:
        Literal["sign-up", "sing-in", "issue"]: 
        The next step to take.
    """
    # Retrieve the last agent's state used by the user
    _type = state.router["type"]

    # Route to the last used node of the rag agent, if it exists in the state
    match _type:
        case "greeting":
            return "greeting"
        case "sing-in":
            return "sing_in"
        case "issue":
            return "report_issue"
        case _:
            # Default to the greeting node if no last node is found
            return "greeting"


async def greeting_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Greet the user, then analyze the user's query and determine the appropriate routing.

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
    system_prompt = configuration.greet_and_route_system_prompt

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
) -> Literal["sign-up", "sing-in", "issue"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["sign-up", "sing-in", "issue"]: 
        The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    # Retrieve the classification type from the router in the agent's state
    _type = state.router["type"]

    # Route based on the classification type
    match _type:
        case "sign-up":
            return "sign_up"
        case "sing-in":
            return "sign_in"
        case "issue":
            return "report_issue"
        case _:
            # Raise an error if the type is not recognized
            raise ValueError(f"Unknown router type {_type}")


# TODO vedi se creare 2 router diversi per non rischiare loop e casini vari
# secondo me non serve perchè a livello logico si può regolare con prompt per evitare che vada in sign-up
# verifica a livello fisico se risulta un edge condizionale verso sign-up!!! <<<---- !


async def report_issue(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user to describe the issue.

    This node is called when the router determines that the user is experiencing some troubles in
    using this recommendation system or is reporting an issue.

    This node calls a tool to save the issue, in order to notify the support team

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

    # Format the system prompt (using the router's logic) to request more information about the issue
    system_prompt = configuration.report_issue_system_prompt.format(
        logic=state.router["logic"]
    )
    
    # Combine the system prompt with the agent's conversation history
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    # Bind the tool to the model
    tool_func_instance = save_report_tool()
    model_with_tool = model.bind_tools([tool_func_instance])

    # Generate the response using the model with tools
    response = await model_with_tool.ainvoke(messages)   
    
    # Check if tool calls are required based on the conversation state
    if tools_condition(state) == "tools":
        # Create and invoke the ToolNode with the tool function instance
        tool_node = await ToolNode(tool_func_instance)
        tool_response = await tool_node.ainvoke(state)

        ##############################################
        # TODO: DA GESTIRE MEGLIO PERCHE' ARTIFACTS VANNO O INTEGRATI IN STATO AGGIUNGENDO FILED IN 
        # STATE O VANNO IGNORATI SENZA INTEGRARLI IN STATO

        # Process the tool responses and integrate them into the state
        for tool_message in tool_response["messages"]:
            messages.append({"role": "tool", "content": tool_message["content"]})
            if tool_message["artifact"]:
                state.update(tool_message["artifact"])

        # Optionally reprocess the state with the agent after tools are invoked
        return report_issue(state, config)
        ##############################################

    return {"messages": [response]}

    # HANDLE USER ISSUES, like the user does not know what to do, how this app works etc., or 
    # any real issues like some error, some bad behaviour of this app, some malfunctioning, etc.
    # in such cases create a JSON file to store issues.
    # Load the agent's language model specified in the input config


async def sign_up(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Format the system prompt (using the router's logic) to request user's information
    system_prompt = configuration.sign_up_system_prompt.format(
        logic=state.router["logic"]
    )
    
    # Combine the system prompt with the agent's conversation history
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    # Bind the tool to the model
    tool_func_instance = sign_up_tool()
    model_with_tool = model.bind_tools([tool_func_instance])

    # Generate the response using the model with tools
    response = await model_with_tool.ainvoke(messages)   
    
    # Check if tool calls are required based on the conversation state
    if tools_condition(state) == "tools":
        # Create and invoke the ToolNode with the tool function instance
        tool_node = await ToolNode(tool_func_instance)
        tool_response = await tool_node.ainvoke(state)

        ##############################################
        # TODO: DA GESTIRE MEGLIO PERCHE' ARTIFACTS VANNO O INTEGRATI IN STATO AGGIUNGENDO FILED IN 
        # STATE O VANNO IGNORATI SENZA INTEGRARLI IN STATO

        # Process the tool responses and integrate them into the state
        for tool_message in tool_response["messages"]:
            messages.append({"role": "tool", "content": tool_message["content"]})
            if tool_message["artifact"]:
                state.update(tool_message["artifact"])

        # Optionally reprocess the state with the agent after tools are invoked
        return report_issue(state, config)
        ##############################################

    return {"messages": [response]}

    # Please provide your name, surname and email to start the conversation.
    # Ask previous seen movies -> to solve cold issue problem

# TODO rimetti sign-in come scritto sotto (da fare alla fine) -> vedi bene come gestire utente
async def sign_in():
    # Load user preferences and other data
    # Do not generate any message, just load user preferences in state


# DA QUI <<<<<<<<------------
# TODO DA SISTEMARE MEGLIO recommendation -> vedi conduct_research in rag_agent, ma è un pò diverso!
# TODO deve ritornare movies anzichè documenti ---> vedi se conviene creare dati strutturati

async def recommendation(state: AgentState) -> dict[str, Any]:
    """Execute the rag agent's graph.

    This function executes the rag agent's graph for retrieving relevant movies to recommend to the 
    user.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    # Invoke the rag_graph with...
    result = await rag_graph.ainvoke({"question": state.steps[0]})

    # Return the research results ('documents') and ...
    return {"documents": result["documents"], "steps": state.steps[1:]}
    

    # (at this point the user is signed-in)
    # Hi Jhon, how can i help you today? Have you seen anything interesting lately?
    # call rag_agent


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(greeting)
builder.add_node(report_issue)
builder.add_node(sign_up)
builder.add_node(recommendation)

builder.add_conditional_edges(START, route_restore_previous_conversation)
builder.add_conditional_edges("greeting", route_query)
builder.add_edge("sign_up", "recommendation")
builder.add_edge("recommendation", END)

# Compile into a graph object that you can invoke and deploy.
# graph = builder.compile()
# graph.name = "RecommendationGraph"