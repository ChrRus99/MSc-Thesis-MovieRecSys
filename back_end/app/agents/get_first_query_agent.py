from langgraph.prebuilt import create_react_agent
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.shared.state import InputState
from app.shared.tools import make_handoff_tool

GET_FIRST_QUERY_SYSTEM_PROMPT = """
You are an assistant agent in a movie recommendation system.
You job is is to collect the user's first movie-related query.

**Process:**
1. Ask the user if they have any movie-related question.
2. As soon as you receive a query from the user, immediately use the `transfer_to_movie_info_and_recommendation` tool to hand off the conversation to the movie recommendation agent.

**Important:** Do not generate any response messages or ask follow-up questions.

**Available tools:**
{tools}

**Tool names:**
{tool_names}
"""

def create_get_first_query_agent(state: InputState, llm: ChatOpenAI = None) -> Runnable:
    """
    Creates a runnable ReAct agent for collecting the user's first query and handing off.

    Args:
        llm (ChatOpenAI): The language model to use.
        state (InputState): The current graph state, used for creating state-dependent tools.

    Returns:
        Runnable: The runnable agent instance configured for this task.
    """
    tools = [
        make_handoff_tool(state, agent_name="movie_info_and_recommendation"),
    ]

    formatted_system_prompt = GET_FIRST_QUERY_SYSTEM_PROMPT.format(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    react_agent_runnable = create_react_agent(
        model=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=tools,
        prompt=formatted_system_prompt,
    )

    return react_agent_runnable
