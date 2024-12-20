"""Main entrypoint for the conversational recommendation graph.

This module defines the core structure and functionality of the conversational recommendation graph. 
It includes the main graph definition, state management, and key functions for processing & routing
user queries, handling the sign-up, the sing-in, and the issues report of the user, and to provide
personalized movies recommendations to the user.
"""

from typing import Any, Annotated, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, StructuredTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain.agents import Tool
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command, interrupt

from main_graph.configuration import AgentConfiguration
#from main_graph.recommendation_graph.graph import graph as recommendation_graph
from main_graph.state import AgentState, InputState, Router
from main_graph.tools import check_user_registration_tool, save_report_tool, sign_up_tool
from shared.utils import load_chat_model
from shared.debug_utils import (
    state_log,
    generic_log,
    log_state_after_return
)


def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Literal["sign_in"]]:
        """Redirect to the sign-in agent."""

        # Constructing the tool message
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            role="tool",
            name=tool_name,
            tool_call_id=tool_call_id,
        )

        generic_log(
            function_name="handoff_to_agent: " + tool_name, 
            fields={
                "tool_id": tool_call_id,
                "tool_name": tool_name,
                "transfer_to_agent": agent_name
            }
        )

        # Modifying state messages directly with tool message
        state.messages.append(tool_message)  # Correct way to modify the list

        # Constructing the command with just the necessary attributes
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": state.messages}  # Ensure the update key is properly referencing the list
        )
    
    return handoff_to_agent


@log_state_after_return
async def human_node(
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["sign_up", "human"]]:
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


@log_state_after_return
async def greeting_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["sign_up", "sign_in"]]:
    """
    Greet the user, then analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow, by checking whether the user is already registered.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        Command: The routing command indicating the next step.
    """
    # Load the agent's language model and system prompt specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.greet_and_route_system_prompt

    # Step 1: Generate greeting message
    greeting_response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Generate a greeting message for the user.")
    ])
    state.messages.append(AIMessage(content=greeting_response.content))

    # Step 2: Check user registration
    user_id = int(config["configurable"]["user_id"])
    
    check_registration_tool = check_user_registration_tool(user_id)
    tool_message = await check_registration_tool.ainvoke(
        {
            "name": "check_user_registration_tool",
            "args": {},
            "id": "123",
            "type": "tool_call"
        }
    )

    is_registered = tool_message.artifact
    # state.messages.append(tool_message) # SBAGLIATO

    # Step 3: Generate routing with structured output
    routing_response = await model.with_structured_output(Router).ainvoke([
        SystemMessage(content="Analyze user registration status and provide routing."),
        HumanMessage(content=f"User registered: {is_registered}")
    ])
    # state.messages.append(AIMessage(content=routing_response.content))
    state.router = routing_response

    # Step 4: Notify the user about the routing
    notification_response = await model.ainvoke([
        SystemMessage(content="Notify the user about the routing result."),
        HumanMessage(content=f"Routing decision: {routing_response}")
    ])
    state.messages.append(AIMessage(content=notification_response.content))

    # Return the structured routing result
    return Command(
        update={"messages": state.messages},
        goto=routing_response['type']
    )

# TODO: vedi se qua conviene usare Command per routing diretto senza usare conditional edges
# vedi: https://www.youtube.com/watch?v=6BJDKf90L9A&ab_channel=LangChain

# vedi anche questo https://langchain-ai.github.io/langgraph/how-tos/multi-agent-multi-turn-convo/

# vedi anche questo https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/#multi-turn-conversation

# vedi anche questo https://langchain-ai.github.io/langgraph/how-tos/agent-handoffs/#using-with-a-prebuilt-react-agent 

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
    # Load the agent's language model and the formatted system prompt specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.report_issue_system_prompt.format(
        logic=state.router["logic"]
    )
    
    # Combine the system prompt with the current conversation history
    messages = [SystemMessage(content=system_prompt)] + state.messages

    # Bind tools to the language model
    tools = [save_report_tool()]
    model_with_tools = model.bind_tools(tools)

    # Generate the response using the model with tools
    model_response = await model_with_tool.ainvoke(messages)   
    
    # My comment: secondo me questa parte non serve, perchè agente chiama tool già sopra, questo
    # serve solo per chiamarlo post response es. tool per salvare i risultati
    ################################################################################################
    # Append the [AI model] response to the conversation history
    #state.messages.append(model_response)
    # state.messages.append(SystemMessage(content=str(model_response)))

    # # Check if tool calls are required based on the conversation history
    # if tools_condition({"messages": state.messages}) == "tools":
    #     # Invoke tools and get their responses
    #     tool_node = ToolNode(tools)
    #     response = tool_node.invoke({"messages": state.messages})

    #     # Process the tool responses
    #     for tool_message in response["messages"]:
    #         # Append the [Tool] responses to the conversation history
    #         state.messages.append(tool_message)

    #         # Dynamically update state attributes with tool artifacts
    #         if hasattr(tool_message, "artifact") and tool_message.artifact:
    #             for key, value in tool_message.artifact.items():
    #                 setattr(state, key, value)

    #     # Optionally reprocess the state with the agent after tools are invoked
    #     if call_after_tool:
    #         return greeting_and_route_query(state, config)
    ################################################################################################

    print("DEBUG ---> ", "report_issue")
    print("DEBUG ---> ", type(model_response))
    print("DEBUG ---> ", model_response)
    print("")

    # Return a message containing the model response
    return {"messages": [model_response]}

    # HANDLE USER ISSUES, like the user does not know what to do, how this app works etc., or 
    # any real issues like some error, some bad behaviour of this app, some malfunctioning, etc.
    # in such cases create a JSON file to store issues.
    # Load the agent's language model specified in the input config


@log_state_after_return
async def sign_up(
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["sign_in", "human"]]:
    """
    Handles a user sign-up process with an iterative question-answer approach.

    Args:
        state (AgentState): The current conversation state.
        config (RunnableConfig): The configuration containing model and tool bindings.

    Returns:
        dict[str, list[BaseMessage]]: The updated messages after processing sign-up.
    """ 
    # Load the agent's language model and the formatted system prompt specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    # system_prompt = configuration.sign_up_system_prompt.format(
    #     logic=state.router["logic"]
    # )

    # Combine the system prompt with the current conversation history
    #messages = [SystemMessage(content=system_prompt)] + state.messages

    # Retrieve the user id and the thread id from the input config
    user_id = config["configurable"]["user_id"]

    sign_up_tools = [
        sign_up_tool(user_id=user_id),
        make_handoff_tool(agent_name="sign_in"),
        
    ]

    # sign_up_assistant = create_react_agent(
    #     model,
    #     sign_up_tools,
    #     state_modifier=(
    #         "You are a sign-up assistant. You will interact with the user to gather their first name, surname, and email address to register them for the service.\n"
    #         "Follow these steps:\n"
    #         "1. Ask the user for their first name.\n"
    #         "2. Once you have the first name, ask for their surname.\n"
    #         "3. After getting the surname, ask for their email address.\n"
    #         "4. When you have ALL three pieces of information (first name, surname, and email), use the `sign_up_tool` with the collected information.\n"
    #         "5. After the `sign_up_tool` confirms successful registration, it will automatically transfer the user to the sign-in agent. There's no need to call the `transfer_to_sign_in` tool directly."
    #     ),
    # )

    # sign_up_assistant = create_react_agent(
    #     model,
    #     sign_up_tools,
    #     state_modifier=(
    #         "Call and redirect to the `sign_in` agent."
    #     ),
    # )

    sign_up_assistant = create_react_agent(
        model,
        sign_up_tools,
        state_modifier=(
            "You are a sign-up assistant. You will interact with the user to gather their first name, surname, and email address to register them for the service.\n"
            "Follow these steps:\n"
            "1. Ask the user for their first name, surname, and email.\n"
            "2. If the user does not provide all the data, ask again to provide the missing data, until the user provide all the data."
            "3. Only when you have ALL three pieces of information (first name, surname, and email), use the `sign_up_tool` with the collected information.\n"
            "4. Only after you have the confirm of registration from the sign_up_tool', you can transfer the user to the `sign-in` agent."
        ),
    )


    response = None

    # Generate the response using the model with tools
    try:

        response = sign_up_assistant.invoke(state)
        #print("------------> ", response)
        #state.messages.append(AIMessage(content=response.content))
    except Exception as e:
        #state.messages.append(AIMessage(content=response))

        # ERROR LOG
        state_log(
            function_name="sign_up", 
            state=state,
            additional_fields={"Error": e},
            modality="error"
        )

        # FORZA TRASFERIMENTO
        ############################################################
        return Command(goto="sign_in")
        ############################################################

        #raise e
        return Command(
            update={
                "messages": [
                    {
                        "role": "system",
                        "content": "Error sign_in",
                    }
                ]
            }, 
        goto="sign_up")

    return Command(update=response, goto="human")


@log_state_after_return
async def sign_in(
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal[END]]:
    # Load the agent's language model and the formatted system prompt specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.sign_in_system_prompt.format(
        logic=state.router["logic"]
    )
    
    # Combine the system prompt with the current conversation history
    #messages = [SystemMessage(content=system_prompt)] + state.messages
    

    # Generate the response using the model
    model_response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Confirm the user that is signed in.")
    ])
    state.messages.append(AIMessage(content=model_response.content))

    # # Return a message containing the model response
    return Command(
        update={"messages": state.messages},
        goto=END
    )

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
    #result = await rag_graph.ainvoke({"question": state.steps[0]})

    # Return the research results ('documents') and ...
    #return {"documents": result["documents"], "steps": state.steps[1:]}

    pass
    

    # (at this point the user is signed-in)
    # Hi Jhon, how can i help you today? Have you seen anything interesting lately?
    # call rag_agent


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node("human", human_node)
builder.add_node("greeting_and_route_query", greeting_and_route_query)
#builder.add_node(report_issue)
builder.add_node("sign_up", sign_up)
builder.add_node("sign_in", sign_in)
#builder.add_node(recommendation)

builder.add_edge(START, "greeting_and_route_query")
#builder.add_conditional_edges("greeting_and_route_query", route_query)
#builder.add_conditional_edges("sign_up", check_registration_data)
builder.add_edge("sign_in", END)
#builder.add_edge("sign_in", "recommendation")
#builder.add_edge("recommendation", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RecommendationGraph"
