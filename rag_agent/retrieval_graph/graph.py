"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational retrieval graph. 
It includes the main graph definition, state management, and key functions for processing & routing
user queries, generating research plans to answer user questions, conducting research, and
formulating responses.
"""

import copy
from operator import attrgetter
from typing import Any, Annotated, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, StructuredTool, ToolException
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import END, START, StateGraph
from langchain.agents import Tool
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command, interrupt

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState, Router
from shared.utils import format_docs, load_chat_model
from retrieval_graph.tools import (
    retrieve_movies_info_tool,
    retrieve_cast_and_crew_info_tool,
)
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
            #updated_state['messages'] = [tool_message]  # Override the messages field ##################### DA SISTEMARE QUESTO PROBLEMA CHE SE DECOMMENTO LANCIA ERRORE BAD REQUEST


            return Command(
                # navigate to another agent node in the PARENT graph
                goto=agent_name,
                graph=Command.PARENT,
                update=updated_state,
            )
        except ValueError as e:
            # Catch ValueError and rethrow as ToolException
            raise ToolException(f"An error occurred in the tool '{tool_name}': {str(e)}") from e
        except Error as e:
            # Catch Error and rethrow as ToolException
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
) -> Command[Literal["ask_user_for_more_info", "user_validation", "human"]]:
    """A node for collecting user input."""
    user_input = interrupt(value="Ready for user input.")

    # identify the last active agent
    # (the last active node before returning to human)
    langgraph_triggers = config["metadata"]["langgraph_triggers"]
    if len(langgraph_triggers) != 1:
        raise AssertionError("Expected exactly 1 trigger in human node")

    active_agent = langgraph_triggers[0].split(":")[1]

    print("human_node -----------------> ", active_agent)

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
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["create_recommendation_research_plan", "ask_user_for_more_info", "respond_to_general_movie_question"]]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        Command: The routing command indicating the next step.
    """

    # TODO: passare parametri: user_id, seen_movies, ecc. da recommendation node
    # TODO: modifica lo stato e rinominalo per non fare casino con lo stato di main_graph

    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Format the system prompt to route the query
    system_prompt = configuration.router_system_prompt

    # Combine the system prompt with the agent's conversation history
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    # Use the model to classify the query and ensure output matches the Router structure
    routing_response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    #state.router = routing_response  # Update the state
    print("analyze_and_route_query ---------> ", routing_response['type'])
    print("analyze_and_route_query ---------> ", routing_response)

    # Return the structured routing result
    return Command(
        update={
            "messages": state.messages,
            #"router": state.router,
            # "user_id": state.user_id, # TODO: DA PASSARE PARAMETRI DA recommendation node 
            # "is_user_registered": state.is_user_registered
        },
        goto=routing_response['type']
    )

# TODO:
# idea per integrare la parte di validation, sarebbe da:
# 1 - parte con routing none
# 2 - user question
# 3 - routing
# 4 - answer question (da uno dei 3 rami)
# 5 - validation
# 6 - Se validation user fallisce
#       - riparti da routing precedentemente + passa tutta history of messages in rag_graph
#       - Ripeti da step 3 con una user question sistemata with validation message (contente la risposta data in precedenza + cosa non andava/era sbagliato)
#     Altrimenti l'utente ha apprezzato la risposta e si può:
#       - chiudere la conversazione se l'utente non ha altre domande
#       - oppure ripetere il recommendation cycle (con routing e messages resettati) se l'utente ha altre domande

@log_node_state_after_return
async def ask_user_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["analyze_and_route_query", "human"]]:
    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Format the system prompt (using the router's logic) to request more information
    # system_prompt = configuration.more_info_system_prompt.format(
    #     logic=state.router["logic"]
    # )
    
    # Combine the system prompt with the agent's conversation history
    # messages = [{"role": "system", "content": system_prompt}] + state.messages

    ask_more_info_tools = [
        make_handoff_tool(state, agent_name="analyze_and_route_query"),
    ]

    # My comment: here an important note is that when we design the prompt we must take into account the human-in-the-loop mechanism
    # namely, in the prompt we must always consider the fact that we must handle each iteraction of the model with the user
    # in an independent way, so we must not work on the entire messages history, otherwise the prompt won't work as expected.
    # This means that each example must show the model how to handle a scenario in each single iteraction with the user, and not how
    # we expect the entire conversation history to be!
    ask_more_info_assistant = create_react_agent(
        model,
        ask_more_info_tools,
        state_modifier=("""
            You are a Movie Recommendation Assistant. Your job is to help users with their inquiries related to movies.

            Your boss has determined that the user is asking a generic question probably not related to movies, so any additional information is needed by the user before providing a movie recommendation or answering a query.

            Follow these steps:
            1. If the user question or request seems clear to you, namely it contains sufficient details to identify a specific movie or movie-related information need, directly transfer the user to the `analyze_and_route_query` agent.
            2. Otherwise, if the user question or request does not seems clear to you, ask the user to provide additional relevant information.
            3. Once the user provides any additional relevant information, reformulate the user’s request in a clear way.
            4. In this way, at the next iterations, you will be able to transfer the user to the `analyze_and_route_query` agent.

            Example 1:
                HumanMessage: I want to watch a good recent action movie. What do you recommend?
                Tool Call: `transfer_to_analyze_and_route_query`

            Example 2:
                ...Message History...
                AIMessage: The user wants a recommendation for a good recent fantasy movie.
                Tool Call: `transfer_to_analyze_and_route_query`

            Example 3:
                HumanMessage: I want to watch something fun.
                AIMessage: In order to provide you the best possible recommendation based on your preferences I will need some additional information. Were you looking for an animation movie or a comic movie?
                HumanMessage: I was thinking something like a comedy.
                AIMessage: The user wants a recommendation for a comedy movie to watch.

            Example 4:
                HumanMessage: I want to watch a good movie.
                AIMessage: Sure, what movie were you interested in?
                HumanMessage: I don't know, I just want to watch a good movie.
                AIMessage: In order to provide you the best possible recommendation based on your preferences I will need some additional information. Could you tell me what genre you are interested in?
                HumanMessage: I was thinking something like an action movie.
                AIMessage: The user wants a recommendation for an action movie to watch.

            Example 5:
                HumanMessage: I want information about a movie.
                AIMessage: Sure, what movie were you interested in?
                HumanMessage: I want to know which is the cast of Top Gun.
                AIMessage: The user wants to know which is the cast of Top Gun.
            
            **IMPORTANT**: Transition directly to the agent without asking redundant questions if enough detail is already provided.
        """
        ),
    )

    try:
        # Prepare state messages excluding ToolMessage instances
        filtered_messages: list[BaseMessage] = [
            msg for msg in state.messages if not isinstance(msg, ToolMessage)
        ]

        # Create a temporary state with filtered messages
        temp_state = copy.deepcopy(state)
        temp_state.messages = filtered_messages

        # Generate response from the assistant
        response = await ask_more_info_assistant.ainvoke(temp_state)

        # Add the response to state messages
        last_response_message = response["messages"][-1]
        state.messages.append(last_response_message)  # Update the state
    except Exception as e:
        state.messages.append(SystemMessage(content="Error"))  # Update the state

        # ERROR LOG
        state_log(
            function_name="ask_user_for_more_info", 
            state=state,
            additional_fields={"Error": e},
            modality="error"
        )

        raise e

    return Command(
        update={
            "messages": state.messages
        },
        goto="human"
    )


# DA QUI <<<<<<<<<<<<<<---------------------------------------------------- TODO
# TODO: <<<<<<<--------- qui bisogna aggiungere una logica per generare il nuovo HumanMessage
# migliorato da ripassare al nodo "ask_user_for_more_info"
# questo nodo va messo come ultimo messagio (HumanMessage) in output da questo nodo!!!
# <<<<----------- chiedi a chatgpt se ha un idea su come fare questo.
# NOTA: lo stesso meccanismo va poi implementato per validation

# TODO: dopo continua a implementare logica disegnata su foglio per creare il grafo



# TODO: aggiungere il tool implementato per rispondere alla domanda in base a dati in movies_df
# TODO: dopo come improvement, sarebbe da creare un piano per pianificare la risposta come per create research plan
# in modo da aggiungere parte di ricerca su internet per trovare più informazioni. ---> corrective_rag

@log_node_state_after_return
async def respond_to_general_movie_question(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Format the system prompt (using the router's logic) to respond to the general query
    # system_prompt = configuration.general_system_prompt.format(
    #     logic=state.router["logic"]
    # )

    # Combine the system prompt with the agent's conversation history
    #messages = [{"role": "system", "content": system_prompt}] + state.messages

    respond_general_question_tools = [
        retrieve_movies_info_tool(state),
        retrieve_cast_and_crew_info_tool(state)
    ]

    respond_general_question_assistant = create_react_agent(
        model,
        respond_general_question_tools,
        state_modifier=("""
            Answer the user question following these steps:
            
            Steps:
            1. **Analyze the user question**
            2. **Extract the target of the question and the filtering citeria from the question**
            3. **Generate a query dictionary containing the filtering criteria to pass to the tool**
            4. **Call the appropriate tool to retrieve the information**
                - **`retrieve_movies_info_tool`** if the question is related to general movies information, such as year, genre, budget, collection, etc..
                - **`retrieve_cast_and_crew_info_tool`** if the question is relative ONLY to cast, crew, characters, actors, directors, but not to general movies information. This tool is complementary to the previous one.
            5. **Format the response message with the information retrieved from the tool**

            Example 1:
                HumanMessage: In which year was Top Gun released?
                Question description: the user wants to know the 'year' in which the movie 'Top Gun' was released. This is a general movies information.
                Extracted information: target: 'year', filtering: {'title': 'Top Gun'}.
                Query dictionary: {'title': 'Top Gun'}
                Tool Call: `retrieve_movies_info_tool` with query: {'title': 'Top Gun'}
                AIMessage: The movie Top Gun was released in 1986.

            Example 2:  
                HumanMessage: Which is the genre of Top Gun?
                Question description: the user wants to know the 'genre' of the movie 'Top Gun'. This is a general movies information.
                Extracted information: target: 'genre', filtering: {'title': 'Top Gun'}.
                Query dictionary: {'title': 'Top Gun'}
                Tool Call: `retrieve_movies_info_tool` with query: {'title': 'Top Gun'}
                AIMessage: Top Gun" belongs to the following genres:
                    - Action
                    - Romance
                    - War

            Example 3:
                HumanMessage: "Which action movies were released in 1999 with a budget greather than 50M dollars?"
                Question description: the user wants to know the movies with features: genre 'action', year '1999' and budget '50000000'. This is a general movies information.
                Extracted information: target: list['titles'], filtering: {'year': 1999, 'genres': 'Action', 'budget': 50000000}.
                Query dictionary:  {'year': 1999, 'genres': 'Action', 'budget': 50000000}
                Tool Call: `retrieve_movies_info_tool` with query: {'year': 1999, 'genres': 'Action', 'budget': 50000000}
                AIMessage: Here are some action movies released in 1999 with a budget greater than 50 million dollars:
                    1. **Assassins**
                    2. **Fair Game**
                    3. **Broken Arrow**
                    4. **Jade**
                    5. **Casper**

            Example 4:  
                HumanMessage: Which is the cast of Top Gun?
                Question description: the user wants to know which 'actors' played a role in the movie 'Top Gun'. This is a question related to the cast and the actors.
                Extracted information: target: list['actors'], filtering: {'title': 'Top Gun'}.
                Query dictionary: {'title': 'Top Gun'}
                Tool Call: `retrieve_cast_and_crew_info_tool` with query: {'title': 'Top Gun'}
                AIMessage: The cast of the movie Top Gun is composed by:
                    1. **Tom Cruise** 
                    2. **Val Kilmer** 
                    3. **Kelly McGillis** 
                    4. **Anthony Edwards** 
                    5. **Mag Ryan** 

            Example 5:  
                HumanMessage: Which are the characters of Top Gun?
                Question description: the user wants to know which 'characters' are present in the movie 'Top Gun'. This is a question related to the cast and the actors.
                Extracted information: target: list['characters'], filtering: {'title': 'Top Gun'}.
                Query dictionary: {'title': 'Top Gun'}
                Tool Call: `retrieve_cast_and_crew_info_tool` with query: {'title': 'Top Gun'}
                AIMessage: The movie Top Gun features the following characters and their respective actors::
                    1. **Pete Mitchell (Maveric)** played by **Tom Cruise**  
                    2. **Tom Kazansky (Iceman)** played by **Val Kilmer**
                    3. **Charlotte Blackwood (Charlie)** played by **Kelly McGillis** 
                    4. **Nick Bradshaw (Goose)** played by **Anthony Edwards** 
                    5. **Carole Bradshaw** played by **Mag Ryan** 


            ** IMPORTANT **: Tools take in input dictionaries.
            """),
    )

                
            # Example 6:
            #     HumanMessage: Which movies were directed by Quentin Tarantino and played by Tom Hanks?
            #     Question description: the user wants to know which 'movies' were directed by 'Quentin Tarantino' and played by 'Tom Hanks'. This is a question related to the cast and the actors.
            #     Extracted information: target: list['titles'], filtering: {'director': 'Quentin Tarantino' 'actor', 'Tom Hanks'}.
            #     Query dictionary: {'director': 'Quentin Tarantino' 'actor', 'Tom Hanks'}
            #     Tool Call: `retrieve_cast_and_crew_info_tool` with query: {'director': 'Quentin Tarantino' 'actor', 'Tom Hanks'}
            #     AIMessage: Here are the movies featuring both Quentin Tarantino and Tom Hanks:
            #         1. **Pulp Fiction 2**
            #         2. **Fictional Crossover**

    try:
        # Extract the last message from the response in the history of messages
        state_messages = state.messages
        filtered_messages = [msg for msg in state.messages if not isinstance(msg, SystemMessage)] # Filter SystemMessage's
        last_state_message = filtered_messages[-1]

        # Pass the model just the last message in the history of messages
        temp_state = copy.deepcopy(state)
        temp_state.messages = [last_state_message]

        print("last_state_message --------> ", last_state_message.content)

        response = await respond_general_question_assistant.ainvoke({"messages": [{"role": "user", "content": last_state_message.content}]})

        print("response --------> ", response)

        # Extract and store the last message from the response
        response_messages = response["messages"]
        last_response_message = response_messages[-1]
        state.messages.append(AIMessage(content=last_response_message.content))  # Update the state
    except Exception as e:
        state.messages.append(SystemMessage(content=f"Error"))  # Update the state

        # ERROR LOG
        state_log(
            function_name="respond_to_general_movie_question", 
            state=state,
            additional_fields={"Error": str(e)},
            modality="error"
        )

        raise e

    state.router = {
        "type": "respond_to_general_movie_question", 
        "logic": "Store this node which preeceeds the validation node. This allows eventual backpropagation in case in which the user does not validate the provided response."
    }

    return {
        "messages": state.messages, 
        "router": state.router
    }

# e.g., tell me more about this movie --> corrective rag


async def create_recommendation_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]
        """A list of steps in the research plan."""
    pass
    # # Load the agent's language model specified in the input config and
    # # configure it to produce structured output matching the Plan TypedDict
    # configuration = AgentConfiguration.from_runnable_config(config)
    # model = load_chat_model(configuration.query_model).with_structured_output(Plan)

    # # Combine the system prompt with the agent's conversation history   
    # messages = [
    #     {"role": "system", "content": configuration.research_plan_system_prompt}
    # ] + state.messages

    # # Use the model to generate the research plan and ensure output matches the Plan structure
    # response = cast(Plan, await model.ainvoke(messages))

    # # Return the research steps and a placeholder for 'documents'
    # return {"steps": response["steps"], "documents": "delete"}


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
    pass
    # # Invoke the researcher_graph with the first research step to conduct research
    # result = await researcher_graph.ainvoke({"question": state.steps[0]})

    # # Return the research results ('documents') and the updated research plan ('steps' without the first step)
    # return {"documents": result["documents"], "steps": state.steps[1:]}


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
    pass
    # # Load the agent's language model specified in the input config
    # configuration = AgentConfiguration.from_runnable_config(config)
    # model = load_chat_model(configuration.response_model)

    # # Format the retrieved documents into a suitable context for the response prompt
    # context = format_docs(state.documents)

    # # Format the system prompt (using the retrieved documents) to respond to the specific query
    # system_prompt = configuration.response_system_prompt.format(context=context)

    # # Combine the system prompt with the agent's conversation history
    # messages = [{"role": "system", "content": system_prompt}] + state.messages

    # # Return the response based on the retrieved documents
    # response = await model.ainvoke(messages)
    # return {"messages": [response]}


################################################### <<<<<<<<<<<<<<<<<--------------------------- DA QUI, DA SISTEMARE LA LOGICA DI USER_VALIDATION
@log_node_state_after_return
async def user_validation(
    state: AgentState, *, config: RunnableConfig
) -> Command[Literal["human", "respond_to_general_movie_question", "create_recommendation_research_plan", END]]:
    # Load the agent's language model specified in the input config
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    # Format the system prompt (using the router's logic) to request more information
    # system_prompt = configuration.more_info_system_prompt.format(
    #     logic=state.router["logic"]
    # )
    
    # Combine the system prompt with the agent's conversation history
    # messages = [{"role": "system", "content": system_prompt}] + state.messages

    # Get the previous node from which this validation node was reached
    previous_node = state.router["type"] 
    print("user_validation -------------> ", previous_node)

    validation_tools = [
        make_handoff_tool(state, agent_name=previous_node),
        make_handoff_tool(state, agent_name=END)
    ]

    validation_assistant = create_react_agent(
        model,
        validation_tools,
        state_modifier=(f"""
            You are a Movie Recommendation Assistant. Your job is to help users with their inquiries related to movies.

            The agent `{previous_node}` has provided the user a response related to movies information or movie recommendation. 
            Your job is to validate if the user is satisfied with the provided response and, in the case in which the user is not satisfied about the answer, you must redact a brief report about the user complaint and redirect the user to the previous agent `{previous_node}`.

            Follow these steps:
            1. If the last message was the response of the agent `{previous_node}`, ask the user if it is satisfied by the response.
            2. If the user seems satisfied about the response provided by the agent `{previous_node}`, transfer the user to the `END` node.
            3. Otherwise, if the user does not seem satisfied about the provided response so far AND if the user complaints seems clear to you, then transfer the user back to the `{previous_node}` agent.
            4. Otherwise, if the user complaint is not clear to you, ask the user to provide any additional relevant information in order to understand its complaint, and write a report about it.
            5. In this way, at the next iterations, you will be able to redirect the user back to the `{previous_node}` agent, and your report will be used by the agent to provide a better response to the user.

            Example 1:
                HumanMessage: Thank you for the information.
                Tool Call: `transfer_to_END`

            Example 2:
                HumanMessage: I asked you about the cast of Top Gun, but you provided me information about the movie plot.
                AIMessage: The user is complaining about the fact that he asked for the cast of Top Gun, but the agent provided information about the movie plot instead.

            Example 3:
                ...Message History...
                AIMessage: The user is complaining about the fact that he asked for the year in which Titanic was released, but the agent provided information about the cast instead.
                Tool Call: `transfer_to_{previous_node}`

            Example 4:
                HumanMessage: You recommendation sucks.
                AIMessage: I'm sorry to hear that, can you provide me some additional information, in such a way to provide you a better recommendation?
                HumanMessage: These movies are all too old, while I asked you for some recent action movie.
                AIMessage: The user is complaining about the fact that he asked for a recommendation about some recent action movies, but the agent recommended old movies instead.

            Example 5:
                HumanMessage: This is not what i asked for.
                AIMessage: I'm sorry to hear that, can you tell me what don't you like about the provided answer, in such a way to provide you a better answer?
                HumanMessage: I never asked for the cast of the movie.
                AIMessage: Can you tell me what did you asked for, maybe providing some additional detail?
                HumanMessage: I asked you to briefly describe me the plot of the movie Top Gun. I don't want any other irrelevant information.
                AIMessage: The user is complaining about the fact that he asked for a brief description about the plot of the movie Top Gun, but the agent provided some other irrelevant information instead.
            
            **IMPORTANT**: Transition directly to the agent without asking redundant questions if enough detail is already provided.
        """
        ),
    )

    try:
        # Prepare state messages excluding ToolMessage instances
        filtered_messages: list[BaseMessage] = [
            msg for msg in state.messages if not isinstance(msg, ToolMessage)
        ]

        # Create a temporary state with filtered messages
        temp_state = copy.deepcopy(state)
        temp_state.messages = filtered_messages

        # Generate response from the assistant
        response = await validation_assistant.ainvoke(temp_state)

        # Add the response to state messages
        last_response_message = response["messages"][-1]
        state.messages.append(last_response_message)  # Update the state
    except Exception as e:
        state.messages.append(SystemMessage(content="Error"))  # Update the state

        # ERROR LOG
        state_log(
            function_name="user_validation", 
            state=state,
            additional_fields={"Error": e},
            modality="error"
        )

        raise e

    return Command(
        update={
            "messages": state.messages
        },
        goto="human"
    )

    pass
# TODO in routing passa la logica validation con richiesta aggiornata
# forse un idea migliore sarebbe quella di passare routing da nodo prima: respond_to_general_movie_question
# oppure create_recommendation_research_plan e se validation fallisce fare un routing direttamente 
# a nodo precedente, anzichè a analyze_and_route_query


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node("human", human_node)
builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_node("ask_user_for_more_info", ask_user_for_more_info)
builder.add_node("respond_to_general_movie_question", respond_to_general_movie_question)
builder.add_node("create_recommendation_research_plan", create_recommendation_research_plan)
builder.add_node("conduct_research", conduct_research)
builder.add_node("respond", respond)
builder.add_node("user_validation", user_validation)

builder.add_edge(START, "analyze_and_route_query")
builder.add_edge("create_recommendation_research_plan", "conduct_research")
#builder.add_conditional_edges("conduct_research", check_finished) # DA TOGLIERE ---> usa Command 
###########
builder.add_edge("conduct_research", "respond")
###########
#builder.add_edge("ask_user_for_more_info", END)
builder.add_edge("respond_to_general_movie_question", "user_validation")
builder.add_edge("respond", "user_validation")
#builder.add_edge("user_validation", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
