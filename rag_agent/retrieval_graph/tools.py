import pandas as pd
from typing import Annotated, Dict, Tuple, List

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, ToolException

from retrieval_graph.state import AgentState
from shared.utils import (
    retrieve_filtered_movies,
    retrieve_filtered_cast_and_crew,
)
from shared.debug_utils import tool_log


# TODO: add ToolMessage's in state
def retrieve_movies_info_tool(state: AgentState):
    """
    Creates a tool function to retrieve movies info.

    This factory function generates a tool that retrieves movies information based on a given query
    containing filtering criteria, helping the respond_to_general_movie_question agent to provide
    the correct movies information to the user.

    Args:
        state (AgentState): The current conversation state.

    Returns:
        Callable: A coroutine tool function that retrieve movies information.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func(query: dict) -> Tuple[str, list]:
        """ A tool that retrieves rows from the movies_df DataFrame based on a given query.
        
        Args:
            query (dict): A dictionary where keys are column names and values are the filter criteria.
        
        Returns:
            Tuple:
                - str: A message indicating the extracted movies details.
                - list: the retrieved rows as a list of dictionaries.
        """
        # DEBUG LOG
        tool_log(
            function_name="retrieve_movies_info_tool", 
            messages= [
                "Called Tool: [retrieve_movies_info_tool]",
                f"Filtering and retrieving movies based on the query filtering criteria: {query}"
            ]
        )

        # Retrieve filtered movies based on the query filtering criteria
        filtered_df = retrieve_filtered_movies(query)
        retrieved_rows = filtered_df.to_dict(orient="records")

        # Serialize the results
        serialized = f"Found {len(retrieved_rows)} matching rows."

        # Create the ToolMessage for confirmation
        tool_message = ToolMessage(
            content=serialized,
            name="retrieve_movies_info_tool",
            tool_call_id="call_retrieve_movies_info_tool"
        )
        #state.messages.append(tool_message)  # Update the state

        # Return content and artifact
        return serialized, retrieved_rows
    return tool_func


# TODO: add ToolMessage's in state
def retrieve_cast_and_crew_info_tool(state: AgentState):
    """
    Creates a tool function to retrieve casts and crews info.

    This factory function generates a tool that retrieves cast and crew information based on a given
    query containing filtering criteria, helping the respond_to_general_movie_question agent to 
    provide the correct cast and crew information to the user.

    Args:
        state (AgentState): The current conversation state.

    Returns:
        Callable: A coroutine tool function that retrieves casts and crews information.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func(query: dict) -> Tuple[str, list]:
        """ A tool that retrieves rows from the movies_df and the credits_df DataFrames based on a
        given query.
        
        Args:
            query (dict): A dictionary where keys are column names and values are the filter criteria.
        
        Returns:
            Tuple:
                - str: A message indicating the extracted casts and crews details.
                - list: the retrieved rows as a list of dictionaries.
        """
        # DEBUG LOG
        tool_log(
            function_name="retrieve_cast_and_crew_info_tool", 
            messages= [
                "Called Tool: [retrieve_cast_and_crew_info_tool]",
                f"Filtering and retrieving casts and crews based on the query filtering criteria: {query}"
            ]
        )

        # Retrieve filtered casts and crews based on the query filtering criteria
        filtered_df = retrieve_filtered_cast_and_crew(query)
        print("TOOL --------> filtered_df", filtered_df) 
        retrieved_rows = filtered_df.to_dict(orient="records")

        # Serialize the results
        serialized = f"Found {len(retrieved_rows)} matching rows."

        # Create the ToolMessage for confirmation
        tool_message = ToolMessage(
            content=serialized,
            name="retrieve_cast_and_crew_info_tool",
            tool_call_id="call_retrieve_cast_and_crew_info_tool"
        )
        #state.messages.append(tool_message)  # Update the state
        
        # Return content and artifact
        return serialized, retrieved_rows
    return tool_func