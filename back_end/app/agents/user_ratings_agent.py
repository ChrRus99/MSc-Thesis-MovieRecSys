from langgraph.prebuilt import create_react_agent
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.shared.state import InputState
from app.shared.tools import make_handoff_tool
from app.app_graph.tools import save_user_seen_movies_tool

NEW_USER_RATINGS_SYSTEM_PROMPT = """
You are a preferences assistant agent in a movie recommendation system.
Your job is to ask the user about the movies they have seen and their ratings (e.g., on a scale of 1 to 5).

**Process:**
1. Ask the user for movie names and ratings.
2. When the user provides movie names and ratings, respond with a confirmation message and then **use** the `save_user_seen_movies` tool to save the information. Ensure the `user_ratings` argument is a list of dictionaries, each with "movie_title" and "rating" keys.
3. Ask if the user wants to add more movies. If yes, repeat step 2.
4. When the user indicates they are finished adding movies, confirm their preferences are saved and then **use** the `transfer_to_get_first_user_query` tool.
5. Always provide a human-readable response before using any tool.

**Important:** You must actually **use** the tools when required, not just mention them in your response.

**Available tools:**
{tools}

**Tool names:**
{tool_names}
"""

OLD_USER_RATINGS_SYSTEM_PROMPT = """
You are a preferences assistant agent in a movie recommendation system interacting with a returning user.
Your job is to ask if the user has seen any new movies since their last visit and update their preferences if necessary.

**Process:**
1. Welcome the user back and ask if they have seen any new movies they'd like to rate.
2. If the user says **yes** and provides movie names and ratings:
    a. Respond with a confirmation message.
    b. **Use** the `save_user_seen_movies` tool to save the new ratings. Ensure the `user_ratings` argument is a list of dictionaries, each with "movie_title" and "rating" keys.
    c. After saving, immediately **use** the `transfer_to_get_first_user_query` tool to proceed to recommendations. Directly call the tool without asking for more confirmation.
3. If the user says **no**:
    a. Acknowledge their response.
    b. **Use** the `transfer_to_get_first_user_query` tool to proceed to recommendations. Directly call the tool without asking for more confirmation.
4. Always provide a human-readable response before using any tool.

**Important:** You must actually **use** the tools when required, not just mention them in your response.

**Available tools:**
{tools}

**Tool names:**
{tool_names}
"""

def create_user_ratings_agent(state: InputState, llm: ChatOpenAI=None) -> Runnable:
    """
    Creates a runnable ReAct agent specifically for collecting user-movie ratings.

    Args:
        llm (ChatOpenAI): The language model to use.
        state (InputState): The current graph state, used for creating state-dependent tools.

    Returns:
        Runnable: The runnable agent instance configured for preference collection tasks.
    """
    # Retrieve the user status from the state
    is_user_new = state.is_user_new if state.is_user_new is not None else False

    # Define the tools required for the preference collection process
    user_ratings_tools = [
        save_user_seen_movies_tool(state),
        make_handoff_tool(state, agent_name="get_first_user_query"),
    ]

    # Define the system prompt template for the agent
    if is_user_new:
        formatted_system_prompt = NEW_USER_RATINGS_SYSTEM_PROMPT.format(
            tools=render_text_description(user_ratings_tools),
            tool_names=", ".join([t.name for t in user_ratings_tools]),
        )
    else:
        formatted_system_prompt = OLD_USER_RATINGS_SYSTEM_PROMPT.format(
            tools=render_text_description(user_ratings_tools),
            tool_names=", ".join([t.name for t in user_ratings_tools]),
        )

    # Create the ReAct agent runnable
    react_agent_runnable = create_react_agent(
        model=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=user_ratings_tools,
        prompt=formatted_system_prompt,
    )

    return react_agent_runnable
