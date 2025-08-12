import logging
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Type, Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI


class BaseAgent:
    """
    This class initializes and configures a ReAct agent using LangGraph principles.

    This agent supports:
        - External tools integration.
        - Tool usage based on reasoning steps (ReAct).
        - Conversation history management.
        - Optional structured input/output handling.
    """
    def __init__(
        self,
        agent_name: str,
        prompt: ChatPromptTemplate,
        llm: ChatOpenAI,
        tools: List[BaseTool],
        structured_input: Optional[Type[BaseModel]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
    ):
        """
        Initializes the agent with the provided configuration.

        Args:
            agent_name (str): The name of the agent.
            prompt (ChatPromptTemplate): The prompt template that guides the agent's behavior.
            llm (ChatOpenAI): The language model used by the agent.
            tools (List[BaseTool]): A list of external tools that the agent can use.
            structured_input (Optional[Type[BaseModel]], optional): A Pydantic model that defines
                the structured input. Defaults to None.
            structured_output (Optional[Type[BaseModel]], optional): A Pydantic model that defines
                the structured output. Defaults to None.
        """
        self.__agent_name = agent_name
        self.__prompt = prompt
        self.__llm = llm
        self.__tools = tools
        self.__structured_input = structured_input
        self.__structured_output = structured_output

        # Ensure prompt is a ChatPromptTemplate
        if not isinstance(prompt, ChatPromptTemplate):
            self.__prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),                          # System-level prompt (needs {tools} and {tool_names})
                MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
                ("user", "{input}"),                                # Use structured input if enabled
                ("system", "{agent_scratchpad}"),                   # Placeholder for agent's intermediate steps (thoughts, actions, observations)
            ])
        else:
            self.__prompt = prompt

    def create_agent_executor(self, verbose: bool=False) -> AgentExecutor:
        """
        Creates an executor to run the ReAct agent's tasks, optionally with structured I/O.

        Args:
            verbose (bool, optional): If True, display verbose output during execution. Defaults to
                False.

        Returns:
            AgentExecutor: An executor that manages the agent's task flow.
        """
        # Inject tool descriptions into the prompt before creating the agent
        formatted_prompt = self.__prompt.partial(
            tools=render_text_description(self.__tools),
            tool_names=", ".join([t.name for t in self.__tools]),
        )

        # Create the ReAct agent
        react_agent = create_react_agent(
            llm=self.__llm,
            tools=self.__tools,
            prompt=formatted_prompt,
        )

        # Wrap the ReAct agent with the AgentExecutor to manage its execution
        executor = AgentExecutor(
            name=self.__agent_name,
            agent=react_agent,
            tools=self.__tools,
            verbose=verbose,
            return_intermediate_steps=verbose,
            handle_parsing_errors=True, # Handle potential LLM output parsing errors
        )

        # Configure the executor to handle structured input/output if defined
        if self.__structured_input and self.__structured_output:
            return executor.with_types(
                input_type=self.__structured_input,
                output_type=self.__structured_output
            )
        elif self.__structured_input:
            return executor.with_types(
                input_type=self.__structured_input
            )
        elif self.__structured_output:
            return executor.with_types(
                output_type=self.__structured_output
            )
        else:
            return executor