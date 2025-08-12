from langgraph.prebuilt import create_react_agent
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.shared.state import InputState
from app.shared.tools import make_handoff_tool
from app.app_graph.tools import register_user_tool


SIGN_UP_SYSTEM_PROMPT = """
You are a sign-up assistant agent in a movie recommendation system.
Your goal is to interact with the user to gather their first name, surname, and email address to register them for the service.

**Steps to Follow:**
1. Ask the user for their first name, surname, and email.
2. If the user does not provide all the data, ask again to provide the missing data, until the user provide all the data.
3. Only when you have ALL three pieces of information (first name, surname, and email), use the `register_user` with the collected information.
4. Only after you have the confirm of registration from the register_user, you can transfer the user to the `sign_in` agent using the `transfer_to_sign_in` tool.
You MUST include human-readable response before transferring to another agent.

**Available tools:**
{tools}

**Tool names:**
{tool_names}

**Example:**
    AIMessage: To get you registered for the service, could you please provide your name, surname and email address?
    HumanMessage: My name is Jhon, my surname is Black, my email address is jhon.black@gmail.com
    AIMessage: I'm registering you.
    Tool Call: `register_user`
    AIMessage: Ok you have been successfully registered, now I will redirect you to the sign in process."
    Tool Call: `transfer_to_sign_in`
"""


def create_sign_up_agent(state: InputState, llm: ChatOpenAI=None) -> Runnable:
    """
    Creates a runnable ReAct agent specifically for the sign-up process.

    Args:
        llm (ChatOpenAI): The language model to use.
        state (InputState): The current graph state, used for creating state-dependent tools.

    Returns:
        Runnable: The runnable agent instance configured for sign-up tasks.
    """
    # Define the tools required for the sign-up process, using the provided state
    sign_up_tools = [
        register_user_tool(state),
        make_handoff_tool(state, agent_name="sign_in"),
    ]

    # Define the system prompt template for the agent
    formatted_system_prompt = SIGN_UP_SYSTEM_PROMPT.format(
        tools=render_text_description(sign_up_tools),
        tool_names=", ".join([t.name for t in sign_up_tools]),
    )

    # Create the ReAct agent runnable
    react_agent_runnable = create_react_agent(
        model=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=sign_up_tools,
        prompt=formatted_system_prompt,
    )

    return react_agent_runnable

