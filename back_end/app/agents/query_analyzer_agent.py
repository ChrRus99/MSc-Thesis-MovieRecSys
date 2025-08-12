from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from app.shared.state import InputState
from app.agents.base_langchain_agent import BaseAgent

# Prompt for ReAct format with JSON output and escaped examples
ANALYZE_AND_ROUTE_SYSTEM_PROMPT = """
You are an agent in a movie recommendation system.
You are responsible for analyzing the user's latest query and determining the most appropriate next step or agent to handle it. Your role is **only** to route the request, not to answer it directly.

------

**Possible Routes:**
Based on the user's query, you must decide which of the following routes is the most suitable:

1.  **`movie_information`**: Use this route if the user is asking for specific information about movies, actors, directors, plots, release dates, awards, or any factual details related to the film industry.
    *   Examples: "Who directed Inception?", "What is the plot of The Matrix?", "Tell me about Tom Hanks' filmography.", "When was the movie Titanic released?"

2.  **`movie_recommendation`**: Use this route if the user is asking for movie suggestions, recommendations based on preferences (genre, actors, mood), or wants help finding something to watch.
    *   Examples: "Recommend a good sci-fi movie.", "I liked Parasite, what else should I watch?", "Suggest a comedy movie from the 90s.", "What are the best movies of 2000?"

3.  **`general_question`**: Use this route for any queries that don't fall into the above categories, namely any general questions or casual conversation not specifically related to movie information or recommendations. This includes general conversation, greetings, questions about the system itself, clarifications, or off-topic discussions.
    *   Examples: "Hi there!", "How does this recommendation system work?", "Can you tell me a joke?", "What's the weather like?"

------

**Your Task:**
1. Analyze the user's latest input (`input`) considering the `chat_history` (if any).
2. Determine the single best route (`movie_information`, `movie_recommendation`, or `general_question`) based on the user's intent.
3. Formulate your response using the "Final Answer:" format described below. The message should confirm understanding and indicate the routing decision, **without providing the actual movie information or recommendation**.

**Final Answer Format:**
```
Thought: I have analyzed the user's query and can now respond.
Final Answer: ```json
{{
  "messages": ["Your final message to the user based on the analysis."],
  "route": "The determined route. Must be one of `movie_information`, `movie_recommendation`, or `general_question`."
}}
```
"""


class StructuredAgentOutput(BaseModel):
    messages: List[str] = Field(..., description="List of messages for the user.")
    route: str = Field(..., description="The determined route.")


def create_query_analyzer_agent(
    state: InputState,
    llm: ChatOpenAI = None,
    verbose: bool = False
) -> AgentExecutor:
    """
    Create a Query Analyzer Agent using LangChain-based BaseAgent.

    Args:
        state (InputState): The current state of the agent, including conversation history.
        llm (ChatOpenAI): A custom OpenAI language model to use for generating responses. Defaults 
            to 'openai:gpt-4o-mini'.
        verbose (bool): If True, display verbose output during execution. Defaults to False.

    Returns:
        AgentExecutor: The compiled Query Analyzer Agent ready for deployment.
    """
    # Define the tools available to the agent
    tools = []

    # Create the greeting agent
    greeting_agent = BaseAgent(
        agent_name="QueryAnalyzerAgent",
        prompt=ANALYZE_AND_ROUTE_SYSTEM_PROMPT,
        llm=llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        tools=tools,
        structured_output=StructuredAgentOutput,
    )

    # Create and return the executor
    return greeting_agent.create_agent_executor(verbose=verbose)