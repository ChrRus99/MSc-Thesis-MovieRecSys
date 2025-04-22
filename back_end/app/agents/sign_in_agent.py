from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from app.shared.state import InputState
from app.agents.base_langchain_agent import BaseAgent
from app.app_graph.tools import load_user_data_tool


# Prompt for ReAct format with JSON output
SIGN_IN_SYSTEM_PROMPT = """
You are an agent in a movie recommendation system.
You are responsible for handling the user sign-in process in a movie recommendation system.

------

**Tools:**
You have access to the following tools:
{tools}

**Tools Usage Instructions:**
To use a tool, please use the following format (do NOT use parentheses after the tool name):
```
Thought: I need to load the user's data. To do that I need to use the most appropriate tool.
Action: load_user_data
Action Input: (leave empty, as this tool uses internal state)
```
**Example:**
```
Thought: I need to load the user's data.
Action: load_user_data
Action Input: 
```

After the tool runs, you will receive an Observation containing the result of the data loading.

If you need to respond directly to the user (after getting the Observation), use the format:
```
Thought: I have the data loading status and can now respond to the user.
Final Answer: ```json
{{
  "messages": ["Message confirming sign-in.", "Message about data loading status."]
}}
```
```

------

**Your Task:**
1.  Start by confirming to the user that they are now signed in.
2.  Use the `load_user_data` tool to attempt loading the user's data.
3.  Based on the Observation from the tool, notify the user whether their data was loaded successfully or if no data was found.
4.  Formulate your final response using the "Final Answer:" format. Include all messages generated (sign-in confirmation, data loading notification).
"""

class StructuredSignInOutput(BaseModel):
    messages: List[str] = Field(..., description="List of messages for the user.")

def create_sign_in_agent(
    state: InputState,
    llm: ChatOpenAI = None,
    verbose: bool = False
) -> AgentExecutor:
    """
    Create a Sign-In Agent using LangChain-based BaseAgent.

    Args:
        state (InputState): The current state of the agent, used by tools.
        llm (ChatOpenAI): A custom OpenAI language model. Defaults to 'openai:gpt-4o-mini'.
        verbose (bool): If True, display verbose output. Defaults to False.

    Returns:
        AgentExecutor: The compiled Sign-In Agent.
    """
    # Define the tools available to the agent
    tools = [load_user_data_tool(state=state)]

    # Create the sign-in agent
    sign_in_agent = BaseAgent(
        agent_name="SignInAgent",
        prompt=SIGN_IN_SYSTEM_PROMPT,
        llm=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=tools,
        structured_output=StructuredSignInOutput,
    )

    # Create and return the executor
    return sign_in_agent.create_agent_executor(verbose=verbose)
