from typing import Annotated, Dict, Tuple, List

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, ToolException

from main_graph.state import AgentState, UserData, Movie
from shared.utils import (
    is_user_registered,
    save_user_info,
    load_user_info,
    load_user_seen_movies,
    save_user_seen_movies
)
from shared.debug_utils import tool_log


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
        tool_log(
            function_name="check_user_registration_tool", 
            messages= [
                "Called Tool: [check_user_registration_tool]",
                f"Checking registration status for user ID: {user_id}"
            ]
        )

        # Check if the user is already registered
        is_registered = is_user_registered(user_id)
        state.is_user_registered = is_registered  # Update the state

        # Serialize the results
        serialized = f"User {user_id} {'is' if is_registered else 'is not'} registered."
        
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
        tool_log(
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
        
        # Create the ToolMessage for confirmation
        tool_message = ToolMessage(
            content=serialized,
            name="register_user_tool",
            tool_call_id="call_user_registration"
        )
        state.messages.append(tool_message)  # Update the state

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
    def tool_func() -> Tuple[str, List[Dict]]:
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
        tool_log(
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
        serialized = f"User {user_id} personal details and seen movies have been successfully loaded."

        # Return content and artifact
        return serialized, seen_movies

    return tool_func


def save_user_seen_movies_tool(state: AgentState):
    """
    Creates a tool function to saves the movies seen by the user.

    This factory function generates a tool that saves the movies seen by the user, thus allowing for 
    a user-based recommendation from the recommendation agent.

    Args:
        state (AgentState): The current conversation state.

    Returns:
        Callable: A tool function that accepts user seen movies and returns a success message along 
                  with user data as a list of dictionaries.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func(movies: List[Dict]) -> Tuple[str, List[Dict]]:
        """ A tool that saves the movies seen by the user.
        
        Args:
            movies: List of movies seen by the user, each represented as a dict with keys 
                    "movie_name" and "rating".

        Returns:
            Tuple:
                - str: A message indicating the movies seen by the user have been saved.
                - list: A list of dictionaries containing the user's seen movies data.
        """
        # Extract user id from state
        user_id = state.user_id

        # DEBUG LOG
        tool_log(
            function_name="save_user_seen_movies_tool",
            messages=[
                "Called Tool: [save_user_seen_movies_tool]",
                f"Saving user's seen movies for user ID: {user_id}"
            ]
        )

        # Resolve movie_name to movie_id
        def resolve_movie_name_to_id(movie_name: str) -> str:
            """Simulate resolving movie_name to movie_id (placeholder for actual logic)."""
            return f"movie_id_{hash(movie_name) % 1000}"

        # Create a list of movies with keys "movie_id" and "rating"
        seen_movies = [
            {"movie_id": resolve_movie_name_to_id(movie["movie_name"]), "rating": movie["rating"]}
            for movie in movies
        ]

        # Save user seen movies in the CSV file
        save_user_seen_movies(user_id, seen_movies)
        state.seen_movies.extend(seen_movies)  # Update the state

        # Serialize the results
        serialized = f"User {user_id} seen movies have been successfully saved."

        # Create the ToolMessage for confirmation
        tool_message = ToolMessage(
            content=serialized,
            name="save_user_seen_movies_tool",
            tool_call_id="call_save_user_seen_movies_tool"
        )
        state.messages.append(tool_message)  # Update the state

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
        tool_log(
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