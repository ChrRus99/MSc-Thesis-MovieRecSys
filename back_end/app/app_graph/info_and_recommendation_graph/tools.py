import os
import pandas as pd
import sys
import requests
from dotenv import load_dotenv
from pathlib import Path
from typing import Annotated, Dict, Tuple, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, ToolException

from app.app_graph.info_and_recommendation_graph.state import RecommendationAgentState
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


def popularity_ranking_recommendation_tool(state: RecommendationAgentState):
    """
    Creates a tool function to generate recommendations based on popularity ranking.

    This factory function generates a tool that provides movie recommendations NOT based on user 
    tastes, helping the popularity_ranking_recommendation agent to generate general recommendations.

    Args:
        state (RecommendationAgentState): The current conversation state.

    Returns:
        Callable: A coroutine tool function that provides movie recommendations based on
            popularity ranking.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    async def popularity_ranking_recommendation(filtering_params: dict) -> Tuple[str, List[str]]:
        """ A tool that provides movie recommendations based on top ranking.

        Args:
            filtering_params (dict): A dictionary containing filtering parameters for the 
                recommendation. Valid keys are:
                - 'genres': (Optional) A list of genres to filter the recommendations by.
                - 'year': (Optional) The year to filter the recommendations by.
                - 'actors': (Optional) A list of actors to filter the recommendations by.
                - 'directors': (Optional) The director to filter the recommendations by.
                - 'top_n': The number of top recommendations to return.
        Returns:
            Tuple:
                - str: A message containing the list of recommended movies based on popularity ranking.
                - list: The list of recommended movies based on popularity ranking.
        """
        # Extract user id from state
        user_id = state.user_id

        # DEBUG LOG
        tool_log(
            function_name="popularity_ranking_recommendation_tool", 
            messages= [
                "Called Tool: [popularity_ranking_recommendation_tool]",
                f"Top ranking recommendation"
            ]
        )

        # Check if the user is already registered
        BASE_URL = "http://host.docker.internal:" if is_docker() else "http://localhost:"
        PORT = os.getenv("MOVIE_RECOMMENDATION_MICROSERVICE_PORT")

        response = requests.get(
            url=f"{BASE_URL}{PORT}/users/{user_id}/recommendations/popularity_ranking",
            params=filtering_params
        )
        assert response.status_code == 200, "Failed to check user registration"

        response_data = response.json()
        recommended_movies_df = pd.DataFrame(response_data)
        state.recommended_movies_df = recommended_movies_df  # Update the state (store all information of the recommended movies) 

        # Extract the titles of the recommended movies
        recommended_movie_titles = recommended_movies_df["title"].tolist()

        # Serialize the results
        message = f"List of recommended movies based on popularity ranking: {recommended_movie_titles}"
        
        # Create ToolMessage for confirmation
        tool_message = ToolMessage(
            content=message,
            name="popularity_ranking_recommendation_tool",
            tool_call_id="call_popularity_ranking_recommendation_tool"
        )
        state.messages.append(tool_message)  # Update the state

        # Return content and artifacts
        return message, recommended_movie_titles

    return popularity_ranking_recommendation


def collaborative_filtering_recommendation_tool(state: RecommendationAgentState):
    """
    Creates a tool function to generate user recommendations based on collaborative filtering.

    This factory function generates a tool that provides movie recommendations to the user based on
    their preferences, helping the collaborative_filtering_recommendation agent to generate 
    personalized recommendations.

    Args:
        state (RecommendationAgentState): The current conversation state.

    Returns:
        Callable: A coroutine tool function that provides movie recommendations based on 
            collaborative filtering.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    async def collaborative_filtering_recommendation(filtering_params: dict) -> Tuple[str, List[str]]:
        """ A tool that provides movie recommendations to the user based on collaborative filtering.

        Args:
            filtering_params (dict): A dictionary containing filtering parameters for the 
                recommendation. Valid keys are:
                - 'genres': (Optional) A list of genres to filter the recommendations by.
                - 'year': (Optional) The year to filter the recommendations by.
                - 'actors': (Optional) A list of actors to filter the recommendations by.
                - 'directors': (Optional) The director to filter the recommendations by.
                - 'top_n': The number of top recommendations to return.
        Returns:
            Tuple:
                - str: A message containing the list of recommended movies based on collaborative filtering.
                - list: The list of recommended movies based on collaborative filtering.
        """
        # Extract user id from state
        user_id = state.user_id

        # DEBUG LOG
        tool_log(
            function_name="collaborative_filtering_recommendation_tool", 
            messages= [
                "Called Tool: [collaborative_filtering_recommendation_tool]",
                f"Collaborative filtering recommendation"
            ]
        )

        # Check if the user is already registered
        BASE_URL = "http://host.docker.internal:" if is_docker() else "http://localhost:"
        PORT = os.getenv("MOVIE_RECOMMENDATION_MICROSERVICE_PORT")

        response = requests.get(
            url=f"{BASE_URL}{PORT}/users/{user_id}/recommendations/collaborative_filtering",
            params=filtering_params
        )
        assert response.status_code == 200, "Failed to check user registration"

        response_data = response.json()
        recommended_movies_df = pd.DataFrame(response_data)
        state.recommended_movies_df = recommended_movies_df  # Update the state (store all information of the recommended movies) 

        # Extract the titles of the recommended movies
        recommended_movie_titles = recommended_movies_df["title"].tolist()

        # Serialize the results
        message = f"List of recommended movies based on collaborative filtering: {recommended_movie_titles}"
        
        # Create ToolMessage for confirmation
        tool_message = ToolMessage(
            content=message,
            name="collaborative_filtering_recommendation_tool",
            tool_call_id="call_collaborative_filtering_recommendation_tool"
        )
        state.messages.append(tool_message)  # Update the state

        # Return content and artifacts
        return message, recommended_movie_titles

    return collaborative_filtering_recommendation


def hybrid_filtering_recommendation_tool(state: RecommendationAgentState):
    """
    Creates a tool function to generate user movie suggestions based on hybrid filtering.

    This factory function generates a tool that provides movie suggestions to the user based on
    their preferences and on movie similarities, helping the hybrid_filtering_recommendation agent
    to generate personalized suggestions.

    Args:
        state (RecommendationAgentState): The current conversation state.

    Returns:
        Callable: A coroutine tool function that provides movie suggestions based on hybrid 
            filtering.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    async def hybrid_filtering_recommendation(filtering_params: dict) -> Tuple[str, List[str]]:
        """ A tool that provides movie suggestions to the user based on hybrid filtering.

        Args:
            filtering_params (dict): A dictionary containing filtering parameters for the 
                recommendation. Valid keys are:
                - 'movie_title': The title of the movie to find similar movies for.
                - 'top_n': The number of top recommendations to return.
        Returns:
            Tuple:
                - str: A message containing the list of recommended movies based on hybrid filtering.
                - list: The list of recommended movies based on hybrid filtering.
        """
        # Extract user id from state
        user_id = state.user_id

        # DEBUG LOG
        tool_log(
            function_name="hybrid_filtering_recommendation_tool", 
            messages= [
                "Called Tool: [hybrid_filtering_recommendation_tool]",
                f"Hybrid filtering recommendation"
            ]
        )

        # Check if the user is already registered
        BASE_URL = "http://host.docker.internal:" if is_docker() else "http://localhost:"
        PORT = os.getenv("MOVIE_RECOMMENDATION_MICROSERVICE_PORT")

        response = requests.get(
            url=f"{BASE_URL}{PORT}/users/{user_id}/recommendations/hybrid_filtering",
            params=filtering_params
        )
        assert response.status_code == 200, "Failed to check user registration"

        response_data = response.json()
        recommended_movies_df = pd.DataFrame(response_data)
        state.recommended_movies_df = recommended_movies_df  # Update the state (store all information of the recommended movies) 

        # Extract the titles of the suggested movies
        suggested_movie_titles = recommended_movies_df["title"].tolist()

        # Serialize the results
        message = f"List of suggested movies based on hybrid filtering: {suggested_movie_titles}"
        
        # Create ToolMessage for confirmation
        tool_message = ToolMessage(
            content=message,
            name="hybrid_filtering_recommendation_tool",
            tool_call_id="call_hybrid_filtering_recommendation_tool"
        )
        state.messages.append(tool_message)  # Update the state

        # Return content and artifacts
        return message, suggested_movie_titles

    return hybrid_filtering_recommendation


def save_user_preferences_tool(state: RecommendationAgentState):
    """
    Creates a tool function to save user's preferences.

    This factory function generates a tool that saves users's preferences, to be used for re-ranking
    purposes in the recommendation process.

    Args:
        state (RecommendationAgentState): The current conversation state.

    Returns:
        Callable: A coroutine tool function that saves user's preferences.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def save_user_preferences(user_preferences: List[str]) -> Tuple[str, List[str]]:
        """ A tool that saves user's preferences.
        
        Args:
            user_preferences: A list of user's preferences, each represented as a string.

        Returns:
            Tuple:
                - str: A message indicating the user's preferences have been saved.
                - list: A list of user's preferences that have been saved.
        """
        # Extract user id from state
        user_id = state.user_id

        # DEBUG LOG
        tool_log(
            function_name="save_user_preferences_tool",
            messages=[
                "Called Tool: [save_user_preferences_tool]",
                f"Saving user's preferences for user ID: {user_id}"
            ]
        )

        # Generate the artifacts with user preferences
        formatted_user_preferences = [
            {"user_id": user_id, "preference": preference}
            for preference in user_preferences
        ]

        state.seen_movies.extend(formatted_user_preferences)  # Update the state

        # Store user-movie ratings
        BASE_URL = "http://host.docker.internal:" if is_docker() else "http://localhost:"
        PORT = os.getenv("USER_MICROSERVICE_PORT")

        for preference in formatted_user_preferences:
            response = requests.post(f"{BASE_URL}{PORT}/users/{user_id}/preferences", json=preference)
            assert response.status_code == 200, "Failed to save user preferences"
        
        # Serialize the results
        message = f"User {user_id} preferences have been successfully saved."

        # Create the ToolMessage for confirmation
        tool_message = ToolMessage(
            content=message,
            name="save_user_preferences_tool",
            tool_call_id="call_save_user_preferences_tool"
        )
        state.messages.append(tool_message)  # Update the state

        # Return content and artifact
        return message, formatted_user_preferences

    return save_user_preferences