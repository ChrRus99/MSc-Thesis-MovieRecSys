from langchain_core.tools import tool


report_agent_prompt = f"""
You are the assistant who receives a report about an issue. 
Ask the user to describe an issue. Once done use the tools available to save it.

Example:
    User: My internet is very slow.
    Assistant: I\'m sorry to hear that. Could you please provide me with more details about the issue?
    User: I can not tell you much. Youtube is just not working.
    Assistant: Thank you for the information. Have notified the support team.
"""

def save_report_tool(return_route: str):
    """
    Creates a tool function to handle saving user-reported issues.

    This factory function generates a tool that accepts a user-reported issue, saves it, and returns
    a response with updated routing and the report data.

    Args:
        return_route (str): The route to navigate to after the report is submitted.

    Returns:
        Callable: A tool function that accepts an issue description and returns a success message 
                  along with updated routing and the saved report as a dictionary.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func(issue: str) -> dict:
        """ A tool that saves an issue reported by the user.
        
        Args:
            issue: A brief summary of the issue reported by the user.

        Returns:
            dict: A success message and an artifact with the current route and report data.
        """
        # Simulate saving the report
        print("Saving user [save_report_tool]: ", issue)
        
        # Generate the artifact with routing information and report details
        issue_metadata = {
            "current_route": return_route,
            "reports": [issue]
        }

        # Serialize the results
        serialized = f"Report is submitted."
        
        # Return content and artifact with updated state
        return serialized, issue_metadata

    return tool_func
