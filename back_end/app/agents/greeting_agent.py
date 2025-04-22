from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from app.shared.state import InputState
from app.agents.base_langchain_agent import BaseAgent
from app.app_graph.tools import check_user_registration_tool

# Prompt for ReAct format with JSON output and escaped examples
GREET_AND_ROUTE_SYSTEM_PROMPT = """
You are an agent in a movie recommendation system.
You are responsible for greeting the user and determining the appropriate next step based on their registration status.

------

**Tools:**
You have access to the following tools:
{tools}

**Tools Usage Instructions:**
To use a tool, please use the following format:
```
Thought: I need to check if the user is registered. To do that I need to use the most appropiate tool.
Action: The action to take, should be one of [{{tool_names}}]
Action Input: The input to the action (For check_user_registration, this can be empty as it uses internal state)
```

After the tool runs, you will receive an Observation. Based on this, decide the final step.

If you need to respond directly to the user (after getting the Observation), use the format:
```
Thought: I have the registration status and can now respond the user.
Final Answer: ```json
{{
  "messages": ["Your final greeting and routing message based on the Observation. e.g., 'Hello! Welcome back. Please sign in to continue.' or 'Hi there! Looks like you're new. Please sign up to get started.'"],
  "route": "The determined route. Must be one of 'sign_up', 'sign_in', or 'issues'."
}}
```
```

------

**Your Task:**
1. Receive the user's input (e.g., "Hi").
2. Always use the `check_user_registration` tool first to determine the user's status.
3. Based on the Observation from the tool:
   - If registered: Greet warmly and suggest signing in.
   - If not registered: Greet warmly and suggest signing up.
   - If there's an issue/error in the observation: Route to support.
4. Formulate your response using the "Final Answer:" format.
"""

class StructuredAgentOutput(BaseModel):
    messages: List[str] = Field(..., description="List of messages for the user.")
    route: str = Field(..., description="The determined route.")

def create_greeting_agent(
    state: InputState,
    llm: ChatOpenAI = None,
    verbose: bool = False
) -> AgentExecutor:
    """
    Create a Greeting Agent using LangChain-based BaseAgent.

    Args:
        state (InputState): The current state of the agent, including conversation history.
        llm (ChatOpenAI): A custom OpenAI language model to use for generating responses. Defaults 
            to 'openai:gpt-4o-mini'.
        verbose (bool): If True, display verbose output during execution. Defaults to False.

    Returns:
        AgentExecutor: The compiled Greeting Agent ready for deployment.
    """
    # Define the tools available to the agent
    tools = [check_user_registration_tool(state=state)]

    # Create the greeting agent
    greeting_agent = BaseAgent(
        agent_name="GreetingAgent",
        prompt=GREET_AND_ROUTE_SYSTEM_PROMPT,
        llm=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=tools,
        structured_output=StructuredAgentOutput,
    )

    # Create and return the executor
    return greeting_agent.create_agent_executor(verbose=verbose)