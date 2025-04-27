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
Your goal is to analyze the user's latest response (`input`), understand their satisfaction, determine if they have further questions, extract persistent preferences (likes/dislikes) and temporary mood/state, save the preferences using the available tool, and generate a structured report including satisfaction and question flags.

------

**Inputs:**
*   `input`: The user's latest feedback message regarding the previous agent's response (e.g., "Great suggestions! Do you know who directed the first one?", "No, I don't like horror. Can you suggest comedies instead?", "That's helpful, thanks!").
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
2.  Determine user satisfaction (`is_user_satisfied`):
    *   If the user expresses a complaint, negative feedback, or dissatisfaction (explicitly or implicitly) or is only partially satisfied (e.g., "ok but...", "not really what I wanted"), set `is_user_satisfied` to `false`.
    *   If the user expresses clear satisfaction (e.g., "ok thanks, bye", "great suggestions", "perfect", "yes, that's helpful") OR asks a follow-up question related to the previous response or a new topic, set `is_user_satisfied` to `true`.
3.  Determine if the user has other questions (`has_user_other_questions`):
    *   If the user explicitly asks a new question (e.g., "Do you know...", "Can you also tell me...", "What about...") or implies a desire for more interaction/information beyond simple confirmation, set `has_user_other_questions` to `true`.
    *   If the user simply expresses satisfaction/dissatisfaction without asking anything further (e.g., "Thanks!", "No, that's wrong.", "Good recommendations."), set `has_user_other_questions` to `false`.
    *   **Crucially**: If `is_user_satisfied` is `false` (due to a complaint/dissatisfaction), `has_user_other_questions` MUST be `false`, even if the user phrases their complaint as a question (e.g., "Why did you suggest horror? I wanted comedy." -> `is_user_satisfied`: false, `has_user_other_questions`: false). The dissatisfaction takes precedence for routing.
4.  Extract persistent preferences (likes/dislikes of genres, actors, directors, specific movies, themes, etc.).
5.  Extract temporary mood or state (e.g., "wants something funny today", "feeling adventurous", "needs cheering up").
6.  If persistent preferences were extracted, use the `save_user_preferences` tool with those preferences formatted correctly in the `Action Input` as ` {{"user_preferences": ["preference 1", ...]}}`.
7.  Generate a concise summary `report` capturing the essence of the user's feedback (satisfaction, preferences, mood, other questions).
8.  Isolate the identified `mood` statements into a separate list.
9.  Based on whether the tool was called and its Observation, formulate a `preferences_saved_confirmation` message (e.g., "User preferences noted and saved." or "No new preferences identified to save.").
10. Formulate your final response using the "Final Answer:" format below, including the `is_user_satisfied` and `has_user_other_questions` flags.

**Final Answer Format:**
```
Thought: I have analyzed the user's feedback, determined satisfaction, identified preferences and mood, used the tool if necessary with the correct JSON input format, and received its observation. I can now construct the final structured output.
Final Answer: ```json
{{
  "preferences_saved_confirmation": "Confirmation message about saving preferences.",
  "is_user_satisfied": boolean_value_indicating_satisfaction,
  "has_user_other_questions": boolean_value_indicating_other_questions,
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
    has_user_other_questions: bool = Field(..., description="Boolean indicating if the user has other questions or desires further interaction (True) or not (False).")


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

