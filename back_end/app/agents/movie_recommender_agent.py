import json
import re
from pydantic import BaseModel, Field, PydanticUserError
from typing import List, Dict
import ast # Import ast

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor


# --- Start of monkey-patch ---
# This is a monkey-patch to handle JSON strings and Pydantic v2 validation for fixing tool input 
# parsing issues.

# Monkey-patch BaseTool to handle JSON strings and Pydantic v2 validation
from langchain_core.tools.base import BaseTool
_original_parse_input = BaseTool._parse_input

def _patched_parse_input(self, tool_input, tool_call_id=None) -> dict:
    """
    Custom parse_input method to handle JSON strings and Pydantic v2 validation.

    This method is a monkey-patch for the original _parse_input method in BaseTool. 
    It first attempts to decode the input as JSON. If that fails, it falls back to the original 
    parsing logic.

    Args:
        tool_input (str or dict): The input to be parsed, which can be a JSON string or a dictionary.
        tool_call_id (str, optional): The ID of the tool call. Defaults to None.

    Returns:
        dict: The parsed input as a dictionary.

    """
    # If the tool_input is a JSON string, decode it to a dict
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            # Not JSON—leave as-is for original parsing
            pass
    # First, try the original parsing logic
    try:
        return _original_parse_input(self, tool_input, tool_call_id)
    except PydanticUserError:
        # Fallback: use Pydantic v2's model_validate with by_name=True
        return self.args_schema.model_validate(tool_input, by_name=True)

# Apply the monkey-patch
BaseTool._parse_input = _patched_parse_input
# --- End of monkey-patch ---


from app.shared.state import InputState
from app.agents.base_langchain_agent import BaseAgent
from app.app_graph.movie_graph.tools import (
    popularity_ranking_recommendation_tool,
    collaborative_filtering_recommendation_tool,
    hybrid_filtering_recommendation_tool,
)
from app.agents.utils import format_prompt


# Prompt for ReAct format with JSON output and escaped examples
MOVIE_RECOMMENDATION_PROMPT = """
You are an agent specialized in providing movie recommendations in a movie recommendation system.
Your goal is to understand the user's request, potentially considering past interactions and feedback, use the appropriate tool to generate a relevant movie recommendation, and then explain *why* each movie is recommended.

------

**Inputs:**
* `input`: The user's latest query requesting a movie recommendation.
* `chat_history`: The history of the conversation.
* `evaluation_report`: {evaluation_report}
* `user_mood`: {user_mood}

------

**Tools:**
You have access to the following tools:
{tools}

**Tool Usage Instructions:**
To use a tool, please use the following format:
```
Thought: I need to understand the user's request and select the best tool. The user is asking for [type of recommendation]. Based on this, the best tool is [tool_name]. I will use this tool now.
Action: The action to take, should be one of [{{tool_names}}]
Action Input: [Provide *only* the valid JSON object required by the tool on this line. Do NOT use markdown ```json blocks or any other text.]
```
**IMPORTANT**: The line starting with `Action Input:` must contain *only* the JSON object itself. No extra text, no markdown formatting (like ```json).

After the tool runs, you will receive an Observation containing the movie recommendations or relevant information. The Observation will be a tuple: (message_string, list_of_movie_titles).

Based on the Observation:
1.  Generate a brief explanation for *each* recommended movie in the list. Explain why it might be suitable based on the user's request, the recommendation type (popularity, collaborative, hybrid), the `user_mood`, or known movie details (e.g., "This is a highly-rated action movie from 2022", "This movie shares themes with [Movie User Liked]", "Based on your preference for [Genre/Actor], this might be a good fit", "Since you're looking for something funny today, this comedy might work...").
2.  Formulate your final response to the user, including the explanations. If you used feedback from an `evaluation_report`, you might briefly mention how you adjusted the recommendation (e.g., "Based on your feedback about not liking historical movies, here are some different suggestions...").

Use the following format for your final response:
```
Thought: I have received the recommendations from the tool. The tool returned the message: [message_string from Observation] and the movie list: [list_of_movie_titles from Observation]. Now I need to generate an explanation for each movie and format the final answer.
Final Answer: ```json
{{
  "messages": ["Your final message to the user, incorporating the message_string from the Observation and mentioning the recommendations."],
  "movies": [list_of_movie_titles from Observation],
  "explanations": {{
    "movie_title_1": "Explanation why movie_title_1 is recommended.",
    "movie_title_2": "Explanation why movie_title_2 is recommended."
    ...
  }}
}}
```

------

**Tool Selection Guide:**

1.  **`popularity_ranking_recommendation`**: Use this tool for queries seeking top-ranked or popular movies, not based on the user's specific preferences.
    * Examples: "What are the best movies of 2000?", "What are the best movies of all time?", "What are the most popular movies right now?"
    * Required Arguments: `top_n` (int), `genres` (list of str, optional), `year` (int, optional), `actors` (list of str, optional), `director` (str, optional)
    * Example Action Input: {{"top_n": 5, "genres": ["fantasy"]}}

2.  **`collaborative_filtering_recommendation`**: Use this tool for queries seeking recommendations based on the user's stated preferences, such as favorite genres, actors, directors, or general descriptions of what they want to watch.
    * Examples: "Suggest me a good action movie", "I want to watch a movie with Tom Hanks.", "Can you recommend a funny movie to see tonight?", "Recommend a sci-fi movie set in space."
    * Required Arguments: `top_n` (int), `genres` (list of str, optional), `year` (int, optional), `actors` (list of str, optional), `director` (str, optional)
    * Example Action Input: {{"top_n": 5, "genres": ["action"], "actors": ["Tom Hanks"]}}

3.  **`hybrid_filtering_recommendation`**: Use this tool for queries seeking recommendations similar to specific movies the user has liked or mentioned.
    * Examples: "I liked Titanic, what else should I watch?", "I want to watch a movie similar to Inception.", "Suggest movies like The Matrix."
    * Required Arguments: `top_n` (int), `movie_title` (str)
    * Example Action Input: {{"top_n": 5, "movie_title": "Pulp Fiction"}}

------

**Your Task:**
1. Analyze the user's `input`, considering the `chat_history`.
2. Analyze the `evaluation_report` and `user_mood`. If the report indicates dissatisfaction or a specific mood is present, consider how this feedback should influence your tool choice or the query you pass to the tool (e.g., try a different tool, refine the search criteria based on mood/feedback).
3. Determine the single best tool (`popularity_ranking_recommendation`, `collaborative_filtering_recommendation`, or `hybrid_filtering_recommendation`) based on the query's nature and any feedback/mood.
4. Use the selected tool, ensuring the `Action Input:` line contains *only* the valid JSON object required by the tool, with no extra formatting.
5. Receive the `Observation` (message, movie_list) from the tool.
6. Generate a brief explanation for *each* movie in the `movie_list`, considering the `user_mood` and `evaluation_report`.
7. Formulate your final response using the "Final Answer:" format, including the message, the movie list, and the generated explanations. Acknowledge feedback if applicable.
"""

class RecommenderAgentOutput(BaseModel):
    messages: List[str] = Field(..., description="List of messages for the user, including the recommendation.")
    movies: List[str] = Field(..., description="List of recommended movie titles.")
    explanations: Dict[str, str] = Field(..., description="Dictionary mapping each recommended movie title to its explanation.")

def create_movie_recommender_agent( # Renamed function for clarity
    state: InputState,
    llm: ChatOpenAI = None,
    verbose: bool = False
) -> AgentExecutor:
    """
    Create a Movie Recommender Agent using LangChain-based BaseAgent.

    Args:
        state (InputState): The current state of the agent, including conversation history, potential evaluation report, and user mood.
        llm (ChatOpenAI): A custom OpenAI language model to use for generating responses. Defaults
            to 'openai:gpt-4o-mini'.
        verbose (bool): If True, display verbose output during execution. Defaults to False.

    Returns:
        AgentExecutor: The compiled Movie Recommender Agent ready for deployment.
    """
    # Retrieve the report and mood generated by the evaluator agent (if any)
    report = state.report if state.report else ["No evaluation report provided."]  # Get report or provide default
    mood = state.mood if state.mood else ["No specific mood identified."]  # Get mood or provide default

    # Format the prompt *only* with the evaluation_report and user_mood fields
    # The {tools} and {tool_names} placeholders will be handled by BaseAgent.  
    formatted_prompt = format_prompt(
        MOVIE_RECOMMENDATION_PROMPT,
        { "evaluation_report": report, "user_mood": mood, }
    )

    # Define the tools available to the agent
    tools = [
        popularity_ranking_recommendation_tool(state=state),
        collaborative_filtering_recommendation_tool(state=state),
        hybrid_filtering_recommendation_tool(state=state),
    ]

    # Create the movie recommender agent
    movie_recommender_agent = BaseAgent(
        agent_name="MovieRecommenderAgent",
        prompt=formatted_prompt,
        llm=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=tools,
        structured_output=RecommenderAgentOutput,
    )

    # Create and return the executor
    return movie_recommender_agent.create_agent_executor(verbose=verbose)