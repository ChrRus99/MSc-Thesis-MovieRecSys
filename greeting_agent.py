from typing import Dict
from langchain_core.tools import tool


def greeting_agent_prompt(routes: Dict[str, str]) -> str: 
    return f"""
    You are an agent in an online store. 
    You are part of a team of other agent that can perform more specialized tasks.
    You are the first in the chain of agents. 
    You goal is to greet the customer and identify their needs.
    Once you understand the user question use tools to redirect the user the specialized agent.
    
    There are the following assistants that you can redirect to:
    {''.join([f"- {key}: {value} " for key, value in routes.items()])}
        
    Example:
        User: Hello
        Agent: Hello. I'm automated assistant. How can I help you? 
        User: I'd like to open an account
        tool_call: redirect_tool
    """

@tool(parse_docstring=True, response_format="content_and_artifact")
def redirect_tool(next_agent: str) -> dict:
    """ A tool that redirects to a specific agent.

    Args:
        next_agent: The name of the agent to redirect to.

    Returns:
        dict: A message indicating the redirection and an artifact with the current route.
    """
    # Generate the artifact with routing information
    route_metadata = {"target_route": target_agent}
    
    # Serialize the reults
    serialized = f"You will be redirected to {next_agent}"
    
    return serialized, route_metadata
