from langgraph.prebuilt import create_react_agent
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.shared.state import InputState
from app.shared.tools import make_handoff_tool
from app.app_graph.tools import save_user_seen_movies_tool

USER_PREFERENCES_SYSTEM_PROMPT = """
You are a preferences assistant agent in a movie recommendation system.
Your job is to ask the user about the movies they have seen and their ratings (e.g., on a scale of 1 to 5).

**Process:**
1. Ask the user for movie names and ratings.
2. When the user provides movie names and ratings, respond with a confirmation message and then **use** the `save_user_seen_movies` tool to save the information. Ensure the `user_ratings` argument is a list of dictionaries, each with "movie_title" and "rating" keys.
3. Ask if the user wants to add more movies. If yes, repeat step 2.
4. When the user indicates they are finished adding movies, confirm their preferences are saved and then **use** the `transfer_to_recommendation` tool.
5. Always provide a human-readable response before using any tool.

**Important:** You must actually **use** the tools when required, not just mention them in your response.

**Available tools:**
{tools}

**Tool names:**
{tool_names}
"""

def create_user_preferences_agent(state: InputState, llm: ChatOpenAI=None) -> Runnable:
    """
    Creates a runnable ReAct agent specifically for collecting user movie preferences.

    Args:
        llm (ChatOpenAI): The language model to use.
        state (InputState): The current graph state, used for creating state-dependent tools.

    Returns:
        Runnable: The runnable agent instance configured for preference collection tasks.
    """
    # Define the tools required for the preference collection process
    user_preferences_tools = [
        save_user_seen_movies_tool(state),
        make_handoff_tool(state, agent_name="recommendation"),
    ]

    # Define the system prompt template for the agent
    formatted_system_prompt = USER_PREFERENCES_SYSTEM_PROMPT.format(
        tools=render_text_description(user_preferences_tools),
        tool_names=", ".join([t.name for t in user_preferences_tools]),
    )

    # Create the ReAct agent runnable
    react_agent_runnable = create_react_agent(
        model=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=user_preferences_tools,
        prompt=formatted_system_prompt,
    )

    return react_agent_runnable
