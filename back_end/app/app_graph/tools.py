import os
import sys
import requests
from dotenv import load_dotenv
from pathlib import Path
from typing import Annotated, Dict, Tuple, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, ToolException

from app.app_graph.state import AppAgentState, UserData, Movie
from app.shared.debug_utils import tool_log


def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

environment = "docker" if is_docker() else "local"

# Add the project directories to the system path
if is_docker():
    #print("[INFO] Running inside a Docker container")

    # Set the project root
    project_root = "/app"
else:
    #print("[INFO] Running locally")

    # Dynamically find the project root
    project_root = Path(__file__).resolve().parents[3]

# Load environment variables from .env file
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)


def check_user_registration_tool(state: AppAgentState):
    """
    Creates a tool function to check if a user is already registered.

    This factory function generates a tool that checks whether the user is already registered in 
    the system, helping the greeting_and_route_query agent to route correctly.

    Args:
        state (AppAgentState): The current conversation state.

    Returns:
        Callable: A coroutine tool function that checks the user's registration status.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    async def check_user_registration() -> Tuple[str, bool]:
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
        BASE_URL = "http://host.docker.internal:" if is_docker() else "http://localhost:"
        PORT = os.getenv("USER_MICROSERVICE_PORT")

        response = requests.get(f"{BASE_URL}{PORT}/users/{user_id}/exists")
        assert response.status_code == 200, "Failed to check user registration"

        is_registered = response.json()["exists"]
        state.is_user_registered = is_registered  # Update the state

        # Serialize the results
        message = f"User {user_id} {'is' if is_registered else 'is not'} registered."
        
        # Create ToolMessage for confirmation
        tool_message = ToolMessage(
            content=message,
            name="check_user_registration_tool",
            tool_call_id="call_user_registration_tool"
        )
        state.messages.append(tool_message)  # Update the state

        # Return content and artifacts
        return message, is_registered

    return check_user_registration


def register_user_tool(state: AppAgentState):
    """
    Creates a tool function to handle user sign-up by saving user details.

    This factory function generates a tool that saves the user's first name, last name, and email,
    helping the sign_up agent to register the user.

    Args:
        state (AppAgentState): The current conversation state.

    Returns:
        Callable: A tool function that accepts user details and returns a success message along 
                  with user data as a dictionary.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def register_user(first_name: str, last_name: str, email: str) -> Tuple[str, Dict]:
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
                f"Creating User: {first_name}, {last_name}, {email}"
            ]
        )

        # Generate the artifacts with user details 
        user_metadata = {
            "user_id": user_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": email
        }
        
        state.user_data.update(user_metadata)  # Update the state

        # Register the user
        BASE_URL = "http://host.docker.internal:" if is_docker() else "http://localhost:"
        PORT = os.getenv("USER_MICROSERVICE_PORT")

        response = requests.post(f"{BASE_URL}{PORT}/users", json=state.user_data)

        print(f"Response: {response.json()}")
        print(f"Status Code: {response.status_code}")
        print(f"user id: {response.json()["user_id"]}")

        assert response.status_code == 200, "Failed to create user"
        assert user_id == response.json()["user_id"], "User ID mismatch"

        state.is_user_registered = True  # Update the state

        # Serialize the results
        message = f"User {state.user_id} has been successfully registered."
        
        # Create ToolMessage for confirmation
        tool_message = ToolMessage(
            content=message,
            name="register_user_tool",
            tool_call_id="call_register_user_tool"
        )
        state.messages.append(tool_message)  # Update the state

        # Return content and artifacts
        return message, user_metadata

    return register_user


# NOTE: questo tool al momento non carica nessun dato visto che è tutto lato microservice (ma potrebbe ancora servire in futuro).
def load_user_data_tool(state: AppAgentState):
    """
    Creates a tool function to load user's personal details and seen movies data.

    This factory function generates a tool that loads the personal details and seen movies of a user,
    allowing the sign_in agent to load user's preferences.

    Args:
        state (AppAgentState): The current conversation state.

    Returns:
        Callable: A tool function that accepts user details and returns a success message along 
                  with user data as a list of dictionaries.
    """
    # @tool(parse_docstring=True, response_format="content_and_artifact")
    # def load_user_data() -> Tuple[str, List[Dict]]:
    #     """ A tool that loads user's data.

    #     Returns:
    #         Tuple:
    #             - str: A message indicating the user data has been loaded.
    #             - list: A list of dictionaries containing the user's personal data and seen movies
    #                     data.
    #     """
    @tool(parse_docstring=True)
    def load_user_data() -> str:
        """ A tool that loads user's data.

        Returns:
            str: A message indicating the user data has been loaded.
                
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

        # # Load user's personal details (if not passed through sign_up process before)
        # if not state.user_data:
        #     user_data = load_user_info(user_id)
        #     state.user_data.update(user_data)  # Update the state

        # # Load user seen movies from the CSV file
        # seen_movies = load_user_seen_movies(user_id)
        # state.seen_movies.extend(seen_movies)  # Update the state

        # Serialize the results
        message = f"User {user_id} personal details and seen movies have been successfully loaded."

        # Create ToolMessage for confirmation
        tool_message = ToolMessage(
            content=message,
            name="load_user_data_tool",
            tool_call_id="call_load_user_data_tool"
        )
        state.messages.append(tool_message)  # Update the state

        # Return content and artifact
        return message#, seen_movies

    return load_user_data


def save_user_seen_movies_tool(state: AppAgentState):
    """
    Creates a tool function to save the movies seen by the user.

    This factory function generates a tool that saves the movies seen by the user, thus allowing for 
    a user-based recommendation from the recommendation agent.

    Args:
        state (AppAgentState): The current conversation state.

    Returns:
        Callable: A tool function that accepts user seen movies and returns a success message along 
                  with user data as a list of dictionaries.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def save_user_seen_movies(user_ratings: List[Dict]) -> Tuple[str, List[Dict]]:
        """ A tool that saves the movies seen by the user.
        
        Args:
            user_ratings: A list of dictionaries containing user's ratings for movies, each 
                represented as a dict with keys "movie_title" and "rating".

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

        # Generate the artifacts with user-movie ratings
        formatted_user_ratings = [
            {"user_id": user_id, "movie_title": movie.get("movie_name", movie.get("movie_title")), "rating": movie["rating"]}
            for movie in user_ratings
        ]

        state.seen_movies.extend(formatted_user_ratings)  # Update the state

        # Store user-movie ratings
        BASE_URL = "http://host.docker.internal:" if is_docker() else "http://localhost:"
        PORT = os.getenv("USER_MICROSERVICE_PORT")

        for rating in formatted_user_ratings:
            response = requests.post(f"{BASE_URL}{PORT}/users/{user_id}/ratings", json=rating)
            assert response.status_code == 200, "Failed to save user seen movies"
        
        # Serialize the results
        message = f"User {user_id} seen movies have been successfully saved."

        # Create the ToolMessage for confirmation
        tool_message = ToolMessage(
            content=message,
            name="save_user_seen_movies_tool",
            tool_call_id="call_save_user_seen_movies_tool"
        )
        state.messages.append(tool_message)  # Update the state

        # Return content and artifact
        return message, formatted_user_ratings

    return save_user_seen_movies


# TODO: da sistemare e adattare al resto, vedi sopra altri tool.
# def save_report_tool():
#     """
#     Creates a tool function to handle saving user-reported issues.

#     This factory function generates a tool that accepts a user-reported issue, saves it, and returns
#     a response with the report data.

#     Returns:
#         Callable: A tool function that accepts an issue description and returns a success message 
#                   along with the saved report as a dictionary.
#     """
#     @tool(parse_docstring=True, response_format="content_and_artifact")
#     async def save_report(issue: str) -> Tuple[str, Dict]:
#         """ A tool that saves an issue reported by the user.

#         Args:
#             issue: A brief summary of the issue reported by the user.

#         Returns:
#             tuple: A success message and an artifact with the report data.
#         """
#         # DEBUG LOG
#         tool_log(
#             function_name="save_report_tool", 
#             messages= [
#                 "Called Tool: [save_report_tool]",
#                 f"Saving user issue:  {issue}"
#             ]
#         )
        
        
#         # TODO crea una data_class per prendere il report strutturato e salvalo come file JSON 
#         # config in un folder dedicato.

#         # Generate the artifacts with report details
#         issue_metadata = {
#             "reports": [issue]
#         }

#         # Serialize the results
#         serialized = "Report is submitted."

#         # Return content and artifact with updated state
#         return serialized, issue_metadata

#     return save_report