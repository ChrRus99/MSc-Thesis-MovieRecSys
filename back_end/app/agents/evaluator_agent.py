import json
from pydantic import BaseModel, Field, PydanticUserError
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from app.shared.state import InputState
from app.agents.base_langchain_agent import BaseAgent
from app.app_graph.movie_graph.tools import save_user_preferences_tool


# Prompt for the Evaluator Agent
EVALUATOR_PROMPT = """
You are an agent responsible for evaluating user feedback on movie recommendations or information provided by another agent.
Your goal is to analyze the user's latest response (`input`), understand their satisfaction, extract persistent preferences (likes/dislikes) and temporary mood/state, save the preferences using the available tool, and generate a structured report including a satisfaction flag.

------

**Inputs:**
*   `input`: The user's latest feedback message regarding the previous agent's response.
*   `chat_history`: The history of the conversation, particularly the last AI response the user is reacting to.

------

**Tools:**
You have access to the following tools:
{tools}

**Tool Usage Instructions:**
1.  Analyze the user's `input` to identify persistent preferences (e.g., "I don't like historical movies", "I love Tom Hanks", "Suggest movies like The Hangover").
2.  If preferences are identified, use the `save_user_preferences` tool to store them. Format the input as a JSON object with a key exactly named `user_preferences` containing a list of strings as the value.
    ```
    Thought: I have identified the following user preferences: [list of preferences]. I need to use the save_user_preferences tool to store them.
    Action: save_user_preferences
    Action Input: {{"user_preferences": ["preference 1", "preference 2", ...]}}
    ```
    **IMPORTANT**: The `Action Input:` line must contain *only* the valid JSON object, and the key must be `user_preferences`.

3.  If no specific preferences suitable for saving are identified, do not call the tool.

After the tool runs (if called), you will receive an Observation confirming the action (e.g., "User preferences have been successfully saved.").

------

**Your Task:**
1.  Carefully analyze the user's `input` message in the context of the `chat_history`.
2.  Determine if the user is generally satisfied (True) or dissatisfied (False) with the previous AI response. Consider explicit statements ("great suggestions", "not what I wanted") and implicit tones.
3.  If the user expresses a complaint, negative feedback, or dissatisfaction (explicitly or implicitly) or is partially satisfied (ok but), set `is_user_satisfied` to `false`. For example, if the user says they do not like something, or that the recommendations are not what they wanted, or expresses a negative sentiment, set `is_user_satisfied` to `false`. Only set it to `true` if the user is clearly completely satisfied or positive.
4.  Extract persistent preferences (likes/dislikes of genres, actors, directors, specific movies, themes, etc.).
5.  Extract temporary mood or state (e.g., "wants something funny today", "feeling adventurous", "needs cheering up").
6.  If persistent preferences were extracted, use the `save_user_preferences` tool with those preferences formatted correctly in the `Action Input` as ` {{"user_preferences": ["preference 1", ...]}}`.
7.  Generate a concise summary `report` capturing the essence of the user's feedback (satisfaction, preferences, mood).
8.  Isolate the identified `mood` statements into a separate list.
9.  Based on whether the tool was called and its Observation, formulate a `preferences_saved_confirmation` message (e.g., "User preferences noted and saved." or "No new preferences identified to save.").
10.  Formulate your final response using the "Final Answer:" format below, including the `is_user_satisfied` flag.

**Final Answer Format:**
```
Thought: I have analyzed the user's feedback, determined satisfaction, identified preferences and mood, used the tool if necessary with the correct JSON input format, and received its observation. I can now construct the final structured output.
Final Answer: ```json
{{
  "preferences_saved_confirmation": "Confirmation message about saving preferences.",
  "is_user_satisfied": boolean_value_indicating_satisfaction,
  "report": ["Summary of feedback point 1", "Summary of feedback point 2", ...],
  "mood": ["Identified mood statement 1", "Identified mood statement 2", ...]
}}
```
```
"""

class EvaluatorAgentOutput(BaseModel):
    report: List[str] = Field(..., description="Concise summary of the user's feedback, including preferences and mood.")
    mood: List[str] = Field(..., description="User's current mood or temporary state extracted from feedback (e.g., 'wants something funny'). Empty list if none identified.")
    preferences_saved_confirmation: str = Field(..., description="Confirmation message indicating whether preferences were saved, based on the tool's output.")
    is_user_satisfied: bool = Field(..., description="Boolean indicating if the user was generally satisfied with the previous response (True) or not (False).")


def create_evaluator_agent(
    state: InputState,
    llm: ChatOpenAI = None,
    verbose: bool = False
) -> AgentExecutor:
    """
    Create an Evaluator Agent using LangChain-based BaseAgent.

    Args:
        state (InputState): The current state of the agent, used by the tool.
        llm (ChatOpenAI): A custom OpenAI language model. Defaults to 'openai:gpt-4o-mini'.
        verbose (bool): If True, display verbose output. Defaults to False.

    Returns:
        AgentExecutor: The compiled Evaluator Agent.
    """
    # Define the tools available to the agent
    tools = [save_user_preferences_tool(state=state)]

    # Create the evaluator agent
    evaluator_agent = BaseAgent(
        agent_name="EvaluatorAgent",
        prompt=EVALUATOR_PROMPT,
        llm=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=tools,
        structured_output=EvaluatorAgentOutput,
    )

    # Create and return the executor
    return evaluator_agent.create_agent_executor(verbose=verbose)

