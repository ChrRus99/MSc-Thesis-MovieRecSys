"""Main entrypoint for the conversational recommendation graph.

This module defines the core structure and functionality of the conversational recommendation graph. 
It includes the main graph definition, state management, and key functions for processing & routing
user queries, handling the sign-up, the sing-in, and the issues report of the user, and to provide
personalized movies recommendations to the user.
"""

import copy
from operator import attrgetter
from typing import Any, Annotated, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, StructuredTool, ToolException
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
from main_graph.tools import (
    check_user_registration_tool,
    register_user_tool,
    load_user_data_tool,
    save_user_seen_movies_tool,
    save_report_tool,
)
from shared.utils import load_chat_model
from shared.debug_utils import (
    state_log,
    tool_log,
    log_node_state_after_return,
)


# My comment: I decided to pass state directly through make_handoff_tool function
# because if i pass state through handoff_to_agent, the state is passed indirectly by the model
# and the in the state passed by the model the messages are present correctly, while the other 
# fields of the state are resetted (which is strange). 
def make_handoff_tool(state: AgentState, *, agent_name: str):
    """Create a tool that can return handoff via a Command."""
    tool_name = f"transfer_to_{agent_name}"

    #@tool(tool_name)
    def handoff_to_agent(
        # optionally pass current graph state to the tool (will be ignored by the LLM)
        #state: Annotated[AgentState, InjectedState],   # state: The current state of the agent.
        # optionally pass the current tool call ID (will be ignored by the LLM)
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Redirect to another agent.

        Args:
            tool_call_id: The ID of the tool call.

        Returns:
            Command: A command to transfer to another agent.
        """
        try:
            # DEBUG LOG
            tool_log(
                function_name="handoff_to_agent: " + tool_name, 
                fields={
                    "tool_id": tool_call_id,
                    "tool_name": tool_name,
                    "transfer_to_agent": agent_name
                }
            )

            # Create a ToolMessage to signal the handoff
            tool_message = ToolMessage(
                content=f"Successfully transferred to {agent_name}.",
                name=tool_name,
                tool_call_id=tool_call_id
            )

            # Create a copy of the state and override the messages field
            updated_state = vars(state).copy()  # Copy all fields dynamically
            updated_state['messages'] = [tool_message]  # Override the messages field

            return Command(
                # navigate to another agent node in the PARENT graph
                goto=agent_name,
                graph=Command.PARENT,
                update=updated_state,
            )
        except ValueError as e:
            # Catch ValueError and rethrow as ToolException
            raise ToolException(f"An error occurred in the tool '{tool_name}': {str(e)}") from e

    def _handle_error(error: ToolException) -> str:
        return (
            "The following errors occurred during tool execution:"
            + error.args[0]
            + "Please try again passing the correct parameters to the tool."
        )

    # Wrap the tool using StructuredTool for better error handling
    return StructuredTool.from_function(
        func=handoff_to_agent,
        name=f"transfer_to_{agent_name}",
        description="A tool to redirect the user to another agent.",
        handle_tool_error=_handle_error,
    )


@log_node_state_after_return
async def human_node(
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["sign_up", "ask_user_preferences", "human"]]:
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


# TODO: da rifare tutti i prompt!!!

@log_node_state_after_return
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

    # Load user id from config
    state.user_id = config["configurable"]["user_id"]

    # Step 1: Generate greeting message
    greeting_response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Generate a greeting message for the user.")
    ])
    state.messages.append(AIMessage(content=greeting_response.content))  # Update the state

    # Step 2: Check user registration 
    tools = [check_user_registration_tool(state)]
    model_with_tools = model.bind_tools(tools)
    
    chain = model_with_tools | attrgetter("tool_calls") | check_user_registration_tool(state).map()

    tool_messages = await chain.ainvoke("call the tool")
    tool_message = tool_messages[0]
    state.messages.append(tool_message)  # Update the state
    
    #is_user_registered = tool_message.artifact

    # Step 3: Generate routing with structured output
    routing_response = await model.with_structured_output(Router).ainvoke([
        SystemMessage(content="Analyze user registration status and provide routing."),
        HumanMessage(content=tool_message.content)
    ])
    # state.messages.append(AIMessage(content=routing_response.content))
    # state.router = routing_response

    # Step 4: Notify the user about the routing
    notification_response = await model.ainvoke([
        SystemMessage(content="Notify the user about the routing result."),
        HumanMessage(content=f"Routing decision: {routing_response}")
    ])
    state.messages.append(AIMessage(content=notification_response.content))  # Update the state

    # Return the structured routing result
    return Command(
        update={
            "messages": state.messages,
            "user_id": state.user_id,
            "is_user_registered": state.is_user_registered
        },
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


@log_node_state_after_return
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
        register_user_tool(state),
        make_handoff_tool(state, agent_name="sign_in"),
    ]

    # sign_up_assistant = create_react_agent(
    #     model,
    #     sign_up_tools,
    #     state_modifier=(
    #         "You are a sign-up assistant. You will interact with the user to gather their first name, surname, and email address to register them for the service.\n"
    #         "Follow these steps:\n"
    #         "1. Ask the user for their first name, surname, and email.\n"
    #         "2. If the user does not provide all the data, ask again to provide the missing data, until the user provide all the data."
    #         "3. Only when you have ALL three pieces of information (first name, surname, and email), use the `register_user_tool` with the collected information.\n"
    #         "4. Only after you have the confirm of registration from the register_user_tool', you can transfer the user to the `sign_in` agent."
    #     ),
    # )

    sign_up_assistant = create_react_agent(
        model,
        sign_up_tools,
        state_modifier=("""
            You are a sign-up assistant. You will interact with the user to gather their first name, surname, and email address to register them for the service.\n
            Follow these steps:\n
            1. Ask the user for their first name, surname, and email.\n
            2. If the user does not provide all the data, ask again to provide the missing data, until the user provide all the data.
            3. Only when you have ALL three pieces of information (first name, surname, and email), use the `register_user_tool` with the collected information.\n
            4. Only after you have the confirm of registration from the register_user_tool, you can transfer the user to the `sign_in` agent.
            You MUST include human-readable response before transferring to another agent.

            Example:
                AIMessage: To get you registered for the service, could you please provide your name, surname and email address?
                HumanMessage: My name is Jhon, my surname is Black, my email address is jhon.black@gmail.com
                AIMessage: I'm registering you.
                Tool Call: `register_user_tool`
                AIMessage: Ok you have been successfully registered, now I will redirect you to the sign in process."
                Tool Call: `transfer_to_sign_in`
            """
        ),
    )

    # Generate the response using the model with tools
    try:
        # Extract the last message from the response in the history of messages
        state_messages = state.messages
        last_state_message = state_messages[-1]

        # Pass the model just the last message in the history of messages
        temp_state = copy.deepcopy(state)
        temp_state.messages = [last_state_message]

        response = await sign_up_assistant.ainvoke(temp_state)

        # Extract and store the last message from the response
        response_messages = response['messages']
        last_response_message = response_messages[-1]
        state.messages.append(last_response_message)  # Update the state
    except Exception as e:
        #state.messages.append(SystemMessage(content=f"Error: {e}"))  # Update the state
        state.messages.append(SystemMessage(content=f"Error"))  # Update the state

        # ERROR LOG
        state_log(
            function_name="sign_up", 
            state=state,
            additional_fields={"Error": e},
            modality="error"
        )

        raise e

        # FORZA TRASFERIMENTO (soluzione temporanea da sistemare)
        ############################################################
        # return Command(
        #     update={
        #         "messages": state.messages,
        #         "user_data": state.user_data
        #     }, 
        #     goto="sign_in")
        ############################################################
    
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
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["recommendation", "ask_user_preferences"]]:
    # Load the agent's language model and the formatted system prompt specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.sign_in_system_prompt.format(
        logic=state.router["logic"]
    )
    
    # Combine the system prompt with the current conversation history
    #messages = [SystemMessage(content=system_prompt)] + state.messages
    
    # Step 1: Generate the response using the model
    model_response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Confirm the user that is signed in.")
    ])
    state.messages.append(AIMessage(content=model_response.content))  # Update the state
    
    # Step 2: Load user's data (seen movies)
    tools = [load_user_data_tool(state)]
    model_with_tools = model.bind_tools(tools)

    chain = model_with_tools | attrgetter("tool_calls") | load_user_data_tool(state).map()

    tool_messages = await chain.ainvoke("call the tool")
    tool_message = tool_messages[0]
    state.messages.append(tool_message)  # Update the state

    # Step 3: 
    notification_response = await model.ainvoke([
        SystemMessage(content="Notify the user about the user data loading result."),
        HumanMessage(content=f"Data loading state: {tool_message.content}")
    ])
    state.messages.append(AIMessage(content=notification_response.content))  # Update the state

    #seen_movies = tool_message.artifact

    # Return a message containing the model response
    return Command(
        update={
            "messages": state.messages,
            "seen_movies": state.seen_movies
        },
        goto="recommendation" if state.seen_movies else "ask_user_preferences"
    )

    # Load user preferences and other data
    # Do not generate any message, just load user preferences in state


@log_node_state_after_return
async def ask_user_preferences(
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["recommendation", "human"]]:
    """
    Collects user preferences for movies and transitions to the next node in the graph.
    """
    # Load the agent's language model and configuration
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    # system_prompt = configuration.sign_up_system_prompt.format(
    #     logic=state.router["logic"]
    # )

    # Combine the system prompt with the current conversation history
    #messages = [SystemMessage(content=system_prompt)] + state.messages


    # Retrieve the user id and the thread id from the input config
    user_id = config["configurable"]["user_id"]

    user_preferences_tools = [
        save_user_seen_movies_tool(state),
        make_handoff_tool(state, agent_name="recommendation"),
    ]

    # Create the user preferences assistant
    user_preferences_assistant = create_react_agent(
        model,
        user_preferences_tools,
        state_modifier=(
            """
            You are a preferences assistant. Your job is to ask the user about the movies they have seen and their ratings.
            Follow these steps:
            1. Ask the user for the names and ratings of movies they have seen.
            2. Save the movies using the `save_user_seen_movies_tool` tool once you have the data.
            3. If the user has no more movies to provide, confirm their preferences are saved.
            4. Transition to the `recommendation` agent using the `transfer_to_recommendation` tool.
            5. Always respond in a human-readable way before transitioning.

            Example conversation:
                AIMessage: In order to provide you a movie recommendation based on your preferences in terms of movies I need some additional information.
                AIMessage: Can you tell me the names of some movies you've seen and your personal ratings for them (on a scale from 1 to 5)?
                HumanMessage: I've seen Inception, I rate it 5 stars, Interstellar, I rate it 4 stars, and Titanic, I rate it 3.5 stars.
                Tool Call: `save_user_seen_movies_tool`
                AIMessage: Thank you. I feel now I can provide you a better recommendation.
                AIMessage: Did you have seen any other movie? Otherwise we can proceed with the recommendation.
                HumanMessage: Actually I did. I've seen also Jumanji, I rate it 4 stars, and Toy Story, I rate it 2 stars. That's it.
                AIMessage: Thank you! I’ve saved your preferences. Now let's move on to recommendations.
                Tool Call: `transfer_to_recommendation`
            """
        ),
    )

    # Generate the response using the model with tools
    try:
        # Extract the last message from the response in the history of messages
        state_messages = state.messages
        last_state_message = state_messages[-1]

        # Pass the model just the last message in the history of messages
        temp_state = copy.deepcopy(state)
        temp_state.messages = [last_state_message]

        response = await user_preferences_assistant.ainvoke(temp_state)

        # Extract and store the last message from the response
        response_messages = response["messages"]
        last_response_message = response_messages[-1]
        state.messages.append(last_response_message)  # Update the state
    except Exception as e:
        state.messages.append(SystemMessage(content=f"Error"))  # Update the state

        # ERROR LOG
        state_log(
            function_name="ask_user_preferences", 
            state=state,
            additional_fields={"Error": str(e)},
            modality="error"
        )

        raise e

    return Command(
        update={
            "messages": state.messages,
            "seen_movies": state.seen_movies
        },
        goto="human"
    )


# DA QUI <<<<<<<<------------
# TODO DA SISTEMARE MEGLIO recommendation -> vedi conduct_research in rag_agent, ma è un pò diverso!
# TODO deve ritornare movies anzichè documenti ---> vedi se conviene creare dati strutturati

@log_node_state_after_return
async def recommendation(
    state: AgentState, *, config: RunnableConfig
) -> Command:
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
builder.add_node("ask_user_preferences", ask_user_preferences)
builder.add_node("recommendation", recommendation)

builder.add_edge(START, "greeting_and_route_query")
#builder.add_conditional_edges("greeting_and_route_query", route_query)
#builder.add_conditional_edges("sign_up", check_registration_data)
#builder.add_edge("sign_in", END)
#builder.add_edge("sign_in", "recommendation")
builder.add_edge("recommendation", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RecommendationGraph"
