import json
import re
import ast
from pydantic import BaseModel, Field, PydanticUserError
from typing import List, Dict, Any
import asyncio

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

# --- Start of monkey-patch ---
# This is a monkey-patch to handle JSON strings and Pydantic v2 validation for fixing tool input
# parsing issues.
from langchain_core.tools.base import BaseTool
_original_parse_input = BaseTool._parse_input

def _patched_parse_input(self, tool_input, tool_call_id=None) -> dict:
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            pass
    try:
        return _original_parse_input(self, tool_input, tool_call_id)
    except PydanticUserError:
        return self.args_schema.model_validate(tool_input, by_name=True)

BaseTool._parse_input = _patched_parse_input
# --- End of monkey-patch ---

from app.shared.state import InputState
from app.agents.base_langchain_agent import BaseAgent
from app.app_graph.movie_graph.tools import (
    movie_cast_and_crew_kg_rag_information_tool,
    movie_cast_and_crew_web_search_information_tool,
)
from app.agents.utils import format_prompt

# Prompt for ReAct format with JSON output for movie information retrieval
MOVIE_INFORMATION_PROMPT = """
You are an agent specialized in retrieving information about movies, actors, and directors within a movie recommendation system.
Your goal is to understand the user's request, potentially considering past interactions and feedback, use the appropriate tool to retrieve the information, and present it clearly to the user.

------

**Inputs:**
* `input`: The user's latest query requesting movie information.
* `chat_history`: The history of the conversation.
* `evaluation_report`: {evaluation_report}

------

**Tools:**
You have access to the following tools:
{tools}

**Tool Usage Instructions:**
To use a tool, please use the following format:
```
Thought: I need to understand the user's request and select the best tool. The user is asking for [type of information]. Based on this, the best tool is [tool_name]. I will use this tool now.
Action: The action to take, should be one of [{{tool_names}}]
Action Input: [Provide *only* the valid JSON object required by the tool on this line. Do NOT use markdown ```json blocks or any other text.]
```
**IMPORTANT**: The line starting with `Action Input:` must contain *only* the JSON object itself. No extra text, no markdown formatting (like ```json).

After the tool runs, you will receive an Observation containing the retrieved information. The Observation will be a tuple: (message_string, information_dict).

------

**Disambiguation Handling Instructions:**
If the response from the `movie_cast_and_crew_kg_rag_information` tool indicates that multiple candidates were found (for example, the message contains "Need additional information" and a list of candidates with their IDs), you must:
1. Extract the list of candidates and their IDs from the Observation's `information_dict`.
2. Select the most likely candidate based on the user's intent, context, or additional information (such as year, genre, etc.).
3. Call the `movie_cast_and_crew_kg_rag_information` tool again, this time providing the `entity_id` field in the Action Input (along with `entity` and `type`).
4. Use the new Observation to formulate your final answer.

**Example Disambiguation Workflow:**
```
Thought: The user asked for 'Titanic'. The tool returned multiple candidates with IDs. I will select the correct ID (from the list of candidates) and retry.
Action: movie_cast_and_crew_kg_rag_information
Action Input: {{"entity": "Titanic", "type": "movie", "entity_id": "597"}}
```

------

Based on the Observation, formulate your final response to the user. If you used feedback from an `evaluation_report`, you might briefly mention how you adjusted the search (e.g., "Based on your feedback, I looked for reviews specifically...").

Use the following format for your final response:
```
Thought: I have received the information from the tool. The tool returned the message: [message_string from Observation] and the information: [information_dict from Observation]. Now I need to format the final answer.
Final Answer: ```json
{{
  "messages": ["Your final message to the user, summarizing the key information found."],
  "information": {{information_dict from Observation}}
}}
```

------

**Tool Selection Guide:**

1.  **`movie_cast_and_crew_kg_rag_information`**: Use this tool for queries seeking simple, factual information that can likely be found in a knowledge graph. This includes details about release dates, directors, actors, cast lists, specific roles, etc. If the result is ambiguous, follow the Disambiguation Handling Instructions above.
    *   Examples: "When was the movie Titanic released?", "Who directed Inception?", "What movies did Tom Hanks act in?", "What is the cast of Top Gun?", "Who played the main character in The Matrix?"
    *   Required Arguments: `entity` (str: movie or person name), `type` (str: 'movie' or 'person')
    *   Optional Argument for Disambiguation: `entity_id` (str: unique ID from candidate list)
    *   Example Action Input (Initial): {{"entity": "Inception", "type": "movie"}}
    *   Example Action Input (Disambiguation): {{"entity": "Titanic", "type": "movie", "entity_id": "597"}}

2.  **`movie_cast_and_crew_web_search_information`**: Use this tool for queries requiring more detailed, articulated information, opinions, reviews, plot summaries, critical reception, or information not typically stored in a structured knowledge graph.
    *   Examples: "What is the plot of The Matrix?", "How was the Jumanji movie reviewed?", "Was the movie Matrix appreciated by critics?", "Is Top Gun a good movie?", "Tell me more about the making of Avatar.", "Tell me some trivia about The Matrix.", "What are some interesting facts about Titanic?", "What people think about the movie Inception?"
    *   Required Arguments:
        *   `type`: Literal["plot", "curiosity", "reviews"] - Specify the type of information needed.
        *   `movie_title`: str - The title of the movie.
        *   `query`: Optional[str] - The specific question or topic for 'curiosity' or 'reviews'. **Required** if `type` is 'curiosity' or 'reviews'. Not needed for 'plot'.
    *   Example Action Input (Plot): {{"type": "plot", "movie_title": "The Matrix"}}
    *   Example Action Input (Reviews): {{"type": "reviews", "movie_title": "Jumanji", "query": "How was the Jumanji movie reviewed?"}}
    *   Example Action Input (Curiosity): {{"type": "curiosity", "movie_title": "Avatar", "query": "Tell me more about the making of Avatar."}}

------

**Your Task:**
1. Analyze the user's `input`, considering the `chat_history`.
2. Analyze the `evaluation_report`. If it indicates dissatisfaction, consider how this feedback should influence your tool choice or the query you pass to the tool (e.g., try the other tool, refine the search query).
3. Determine the single best tool (`movie_cast_and_crew_kg_rag_information` or `movie_cast_and_crew_web_search_information`) based on the query's nature (factual vs. detailed/opinion) and any feedback.
4. Use the selected tool, ensuring the `Action Input:` line contains *only* the valid JSON object required by the tool.
5. If the result is ambiguous, follow the Disambiguation Handling Instructions above.
6. Receive the `Observation` (message, information_dict) from the tool.
7. Formulate your final response using the "Final Answer:" format, including a summary message and the retrieved information dictionary. Acknowledge feedback if applicable.
"""

class InformationAgentOutput(BaseModel):
    messages: List[str] = Field(..., description="List of messages for the user, summarizing the retrieved information.")
    information: Dict[str, Any] = Field(..., description="Dictionary containing the retrieved information.")

def create_movie_information_agent(
    state: InputState,
    llm: ChatOpenAI = None,
    verbose: bool = False
) -> AgentExecutor:
    """
    Create a Movie Information Agent using LangChain-based BaseAgent.

    Args:
        state (InputState): The current state of the agent, including conversation history and potential evaluation report.
        llm (ChatOpenAI): A custom OpenAI language model to use for generating responses. Defaults
            to 'openai:gpt-4o-mini'.
        verbose (bool): If True, display verbose output during execution. Defaults to False.

    Returns:
        AgentExecutor: The compiled Movie Information Agent ready for deployment.
    """
    # Retrieve the report generated by the evaluator agent (if any)
    report = state.report if state.report else ["No evaluation report provided."] # Get report or provide default

    # Format the prompt *only* with the evaluation_report field
    # The {tools} and {tool_names} placeholders will be handled by BaseAgent.
    formatted_prompt = format_prompt(
        MOVIE_INFORMATION_PROMPT,
        { "evaluation_report": report }
    )

    # Define the tools available to the agent
    tools = [
        movie_cast_and_crew_kg_rag_information_tool(state=state),
        movie_cast_and_crew_web_search_information_tool(state=state),
    ]

    # Create the movie information agent
    movie_information_agent = BaseAgent(
        agent_name="MovieInformationAgent",
        prompt=formatted_prompt,
        llm=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=tools,
        structured_output=InformationAgentOutput,
    )

    # Create and return the executor
    return movie_information_agent.create_agent_executor(verbose=verbose)
