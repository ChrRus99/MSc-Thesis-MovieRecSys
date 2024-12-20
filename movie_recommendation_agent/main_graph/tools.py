from typing import Annotated, Dict, Tuple, List

from langchain_core.tools import tool, ToolException

from main_graph.state import AgentState, UserData, Movie
from shared.utils import (
    is_user_registered,
    save_user_info,
    load_user_info,
    load_user_seen_movies
)
from shared.debug_utils import (
    state_log,
    generic_log
)


def check_user_registration_tool(state: AgentState):
    """
    Creates a tool function to check if a user is already registered.

    This factory function generates a tool that checks whether the user is already registered in 
    the system, helping the greeting agent to route correctly.

    Args:
        state (AgentState): The current conversation state.

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
        # Extract user id from state
        user_id = state.user_id

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
        state.is_user_registered = is_registered  # Update the state

        # Serialize the results
        serialized = f"User {'is' if is_registered else 'is not'} registered."
        
        # Return content and artifacts
        return serialized, is_registered

    return tool_func


# TODO: DA SISTEMARE LA GESTIONE DEGLI ERRORI SE CI SONO PARAMETRI MANCANTI (ToolException non funziona bene)
# vedi: https://python.langchain.com/docs/how_to/custom_tools/
# nota: Annotated not present error è dovuto a @tool(parse_docstring=True, response_format="content_and_artifact") (commenta per risolvere)

def register_user_tool(state: AgentState):
    """
    Creates a tool function to handle user sign-up by saving user details.

    This factory function generates a tool that saves the user's first name, last name, and email,
    helping the sign_up agent to register the user.

    Args:
        state (AgentState): The current conversation state.

    Returns:
        Callable: A tool function that accepts user details and returns a success message along 
                  with user data as a dictionary.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func(first_name: str, last_name: str, email: str) -> Tuple[str, Dict]:
        """ A tool that saves user first, last name and email.
        
        Args:
            first_name: The user's first name.
            last_name: The user's last name.
            email: The user's email address.

        Returns:
            Tuple:
                - str: A message indicating the registration details.
                - dict: A dictionary containing user data.
        """
        # Extract user id from state
        user_id = state.user_id

        # DEBUG LOG
        generic_log(
            function_name="register_user_tool", 
            messages= [
                "Called Tool: [register_user_tool]",
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
        state.user_data.update(user_metadata)  # Update the state
        state.is_user_registered = True  # Update the state

        # Serialize the results
        serialized = f"User {user_id} has been successfully registered."
        
        # Return content and artifact
        return serialized, user_metadata

    return tool_func


def load_user_data_tool(state: AgentState):
    """
    Creates a tool function to load user's personal details and seen movies data.

    This factory function generates a tool that loads the personal details and seen movies of a user,
    allowing the sign_in agent to load user's preferences.
    Args:
        state (AgentState): The current conversation state.

    Returns:
        Callable: A tool function that accepts user details and returns a success message along 
                  with user data as a list of dictionaries.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func() -> Tuple[str, List[Dict[str, str]]]:
        """ A tool that loads user seen movies.
        
        Args:
            None

        Returns:
            Tuple:
                - str: A message indicating the user data has been loaded.
                - list: A list of dictionaries containing the user's personal data and seen movies
                        data.
        """
        # Extract user id from state
        user_id = state.user_id

        # DEBUG LOG
        generic_log(
            function_name="load_user_data_tool",
            messages=[
                "Called Tool: [load_user_data_tool]",
                f"Loading user's personal details and seen movies for user ID: {user_id}"
            ]
        )

        # Load user's personal details (if not passed through sign_up process before)
        if not state.user_data:
            user_data = load_user_info(user_id)
            state.user_data.update(user_data)  # Update the state

        # Load user seen movies from the CSV file
        seen_movies = load_user_seen_movies(user_id)
        state.seen_movies.extend(seen_movies)  # Update the state

        # Serialize the results
        serialized = f"Loaded user's personal details and seen movies for user ID: {user_id}."

        # Return content and artifact
        return serialized, seen_movies

    return tool_func



# TODO: da sistemare e adattare al resto, vedi sopra altri tool.
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