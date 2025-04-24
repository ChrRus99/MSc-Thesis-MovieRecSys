from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from app.shared.state import InputState
from app.agents.base_langchain_agent import BaseAgent

# Prompt for ReAct format with JSON output and escaped examples
# see app/app_graph/prompts.py


class StructuredAgentOutput(BaseModel):
    messages: List[str] = Field(..., description="List of messages for the user.")
    route: str = Field(..., description="The determined route.")


def create_query_analyzer_agent(
    state: InputState,
    prompt: str,
    llm: ChatOpenAI = None,
    verbose: bool = False
) -> AgentExecutor:
    """
    Create a Query Analyzer Agent using LangChain-based BaseAgent.

    Args:
        state (InputState): The current state of the agent, including conversation history.
        prompt (str): The prompt to use for the agent.
        llm (ChatOpenAI): A custom OpenAI language model to use for generating responses. Defaults 
            to 'openai:gpt-4o-mini'.
        verbose (bool): If True, display verbose output during execution. Defaults to False.

    Returns:
        AgentExecutor: The compiled Query Analyzer Agent ready for deployment.
    """
    # Define the tools available to the agent
    tools = []

    # Create the greeting agent
    greeting_agent = BaseAgent(
        agent_name="QueryAnalyzerAgent",
        prompt=prompt,
        llm=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=tools,
        structured_output=StructuredAgentOutput,
    )

    # Create and return the executor
    return greeting_agent.create_agent_executor(verbose=verbose)