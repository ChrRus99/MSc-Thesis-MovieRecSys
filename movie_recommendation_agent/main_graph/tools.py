from typing import Annotated, Dict, Tuple

from langchain_core.tools import tool, ToolException

from main_graph.state import UserData
from shared.utils import (
    is_user_registered,
    save_user_info,
)
from shared.debug_utils import (
    state_log,
    generic_log
)


def check_user_registration_tool(user_id: int):
    """
    Creates a tool function to check if a user is already registered.

    This factory function generates a tool that checks whether the user is already registered in 
    the system, helping the greeting agent to route correctly.

    Args:
        user_id (int): The unique identifier of the user.

    Returns:
        Callable: A coroutine tool function that checks the user's registration status.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    async def tool_func() -> Tuple[str, bool]:
        """ A tool that checks user registration status.

        Returns:
            Tuple:
                - str: A message indicating whether the user is registered or not.
                - bool: True if the user is registered, False otherwise.
        """
        # DEBUG LOG
        generic_log(
            function_name="check_user_registration_tool", 
            messages= [
                "Called Tool: [check_user_registration_tool]",
                f"Checking registration status for user:  {user_id}"
            ]
        )

        # Check if the user is already registered
        is_registered = is_user_registered(user_id)

        # Serialize the results
        serialized = f"User {'is' if is_registered else 'is not'} registered."
        
        # Return content and artifacts
        return serialized, is_registered

    return tool_func


# TODO: DA SISTEMARE LA GESTIONE DEGLI ERRORI SE CI SONO PARAMETRI MANCANTI (ToolException non funziona bene)
# vedi: https://python.langchain.com/docs/how_to/custom_tools/
# nota: Annotated not present error è dovuto a @tool(parse_docstring=True, response_format="content_and_artifact") (commenta per risolvere)

def sign_up_tool(user_id: int):
    """
    Creates a tool function to handle user sign-up by saving user details.

    This factory function generates a tool that saves the user's first name, last name, and email,
    helping the sign_up agent to register the user.

    Args:
        user_id (int): Unique identifier for the user.

    Returns:
        Callable: A tool function that accepts user details and returns a success message along 
                  with user data as a dictionary.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func(first_name: str, last_name: str, email: str) -> Tuple[str, Dict]:
        """ A tool that saves user first and last name.
        
        Args:
            first_name: The user's first name.
            last_name: The user's last name.
            email: The user's email address.

        Returns:
            Tuple:
                - str: A message indicating the registration details.
                - dict: A dictionary containing user data.
        """
        # DEBUG LOG
        generic_log(
            function_name="sign_up_tool", 
            messages= [
                "Called Tool: [sign_up_tool]",
                f"Creating User: {user_id}, {first_name}, {last_name}, {email}"
            ]
        )

        # Check for missing data
        #if not first_name or not surname or not email:
            #raise ValueException("Missing required information: first name, surname, and email are required.")
            #raise ToolException("Missing required information: first name, surname, and email are required.")

        # Generate the artifacts with user details 
        user_metadata = {
            "user_id": user_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": email
        }

        # Save the user data to the CSV file
        save_user_info(user_metadata)        

        # Serialize the results
        serialized = f"User {user_id} has been successfully registered."
        
        # Return content and artifact
        return serialized, user_metadata

    return tool_func


def save_report_tool():
    """
    Creates a tool function to handle saving user-reported issues.

    This factory function generates a tool that accepts a user-reported issue, saves it, and returns
    a response with the report data.

    Returns:
        Callable: A tool function that accepts an issue description and returns a success message 
                  along with the saved report as a dictionary.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    async def tool_func(issue: str) -> Tuple[str, Dict]:
        """ A tool that saves an issue reported by the user.

        Args:
            issue: A brief summary of the issue reported by the user.

        Returns:
            tuple: A success message and an artifact with the report data.
        """
        # DEBUG LOG
        generic_log(
            function_name="save_report_tool", 
            messages= [
                "Called Tool: [save_report_tool]",
                f"Saving user issue:  {issue}"
            ]
        )
        
        
        # TODO crea una data_class per prendere il report strutturato e salvalo come file JSON 
        # config in un folder dedicato.

        # Generate the artifacts with report details
        issue_metadata = {
            "reports": [issue]
        }

        # Serialize the results
        serialized = "Report is submitted."

        # Return content and artifact with updated state
        return serialized, issue_metadata

    return tool_func