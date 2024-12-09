from typing import Callable, List
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# def create_tool_calling_agent(
#     llm,
#     system_prompt: str,
#     agent_name: str,
#     tools: List[Callable],
#     call_after_tool: bool=True
# ):
#     """
#     Creates an intelligent agent capable of invoking tools dynamically based on the conversation 
#     state and a defined system prompt.

#     This agent evaluates a state of messages, invokes an LLM to process the input, detects when 
#     tools need to be called, and integrates tool outputs back into the conversation. It can 
#     recursively call itself after tool execution if needed.

#     Args:
#         llm: The language model instance that powers the agent.
#         system_prompt (str): The initial system prompt to guide the LLM's behavior.
#         agent_name (str): The name of the agent, used for identification in the conversation.
#         tools (List[Callable]): A list of tools the agent can call during the conversation.
#         call_after_tool (bool, optional): Determines whether the agent should reprocess the state 
#                                           after a tool is invoked. Defaults to True.

#     Returns:
#         Callable: The created agent function.
#     """
#     # Bind tools to the language model
#     llm_with_tools = llm.bind_tools(tools)

#     def agent(state, config):
#         """
#         Processes the current state and handles language model response generation, 
#         tool invocation, and recursively manages conversation state.
#        
#         Args:
#             state (dict): The current state of the conversation, including messages and any
#                           artifacts.
#             config (dict): Additional configuration parameters (if needed).
        
#         Returns:
#             dict: Updated state of the conversation after processing.
#         """
#         # Generate a response from the LLM using the current state
#         llm_response = llm_with_tools.invoke([SystemMessage(system_prompt)] + state["messages"])

#         # Assign the agent's name to the response
#         llm_response.name = agent_name  

#         # Append the LLM's response to the conversation state
#         state["messages"].append(llm_response)

#         # Check if tool calls are required based on the conversation state
#         if tools_condition(state) == "tools":
#             # Create a tool node for tool invocation
#             tool_node = ToolNode(tools)  

#             # Invoke tools and get their responses
#             response = tool_node.invoke(state)  

#             # Process the tool responses and integrate them into the state
#             for tool_message in response["messages"]:
#                 # Add tool-generated messages to state
#                 state["messages"].append(tool_message)  

#                 # If tools return additional data/artifacts, update the state
#                 if tool_message.artifact:  
#                     state = {**state, **tool_message.artifact}

#             # Optionally reprocess the state with the agent after tools are invoked
#             if call_after_tool:
#                 agent(state, config)
#             else:
#                 return state

#         return state

#     return agent


def create_tool_calling_agent(
    llm,
    system_prompt: str,
    agent_name: str,
    tools: List[Callable],
    call_after_tool: bool = True
) -> Callable:
    """
    Creates an intelligent agent capable of invoking tools dynamically based on the conversation 
    state.

    This agent evaluates a state of messages, invokes an LLM to process the input, detects when 
    tools need to be called, and integrates tool outputs back into the conversation. It can 
    recursively call itself after tool execution if needed.

    Args:
        llm: The language model instance that powers the agent.
        system_prompt (str): The initial system prompt to guide the LLM's behavior.
        agent_name (str): The name of the agent, used for identification in the conversation.
        tools (List[Callable]): A list of tools the agent can call during the conversation.
        call_after_tool (bool, optional): Determines whether the agent should reprocess the state 
                                          after a tool is invoked. Defaults to True.
    Returns:
        Callable: The created agent function.
    """
    # Bind tools to the language model
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: dict, config: dict) -> dict:
        """
        Processes the current state and handles language model response generation, tool invocation,
        and recursively manages conversation state.

        Args:
            state (dict): The current state of the conversation, including messages and any artifacts.
            config (dict): Additional configuration parameters (if needed).

        Returns:
            dict: Updated state of the conversation after processing.
        """
        # Generate a response from the LLM using the current state
        llm_response = llm_with_tools.invoke([SystemMessage(system_prompt)] + state["messages"])

        # Assign the agent's name to the response
        llm_response.name = agent_name
        state["messages"].append(llm_response)

        # Check if tool calls are required based on the conversation state
        if tools_condition(state) == "tools":
            # Invoke tools and get their responses
            tool_node = ToolNode(tools)
            response = tool_node.invoke(state)

            # Process the tool responses and integrate them into the state
            for tool_message in response["messages"]:
                state["messages"].append(tool_message)
                if tool_message.artifact:
                    state.update(tool_message.artifact)

            # Optionally reprocess the state with the agent after tools are invoked
            if call_after_tool:
                return agent(state, config)

        return state

    return agent