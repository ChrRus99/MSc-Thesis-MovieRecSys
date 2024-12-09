from langchain_core.tools import tool


def retrieve_agent(routes: Dict[str, str]) -> str: 
    return f"""
    You are an agent that queries a movie database. 
    The movie database has the following columns:
    {column_info}
    
    The user's query will ask about specific movies or details like release year, genre, or rating.
    Your task is to:
        1. Extract the movie title.
        2. Identify what information the user is asking for (e.g., year, genres).
        3. Return the appropriate filter query to retrieve that data.

    Example queries:
    1. "In which year was Titanic released?" -> Query: {{'title': 'Titanic', 'request': 'release_date'}}
    2. "What are the genres of Titanic?" -> Query: {{'title': 'Titanic', 'request': 'genres'}}
    """

def retrieve_movies_tool(return_route: str, movies_df):
    """
    Creates a tool function to retrieve rows from a DataFrame based on a query.

    This factory function generates a tool that accepts a query, filters the DataFrame, and returns
    a response with a success message and updated routing along with the retrieved rows.

    Args:
        return_route (str): The route to navigate to after the retrieval is completed.
        movies_df (DataFrame): The DataFrame to retrieve rows from.

    Returns:
        Callable: A tool function that accepts a query dictionary and returns matching rows 
                  along with updated routing and a summary message.
    """
    @tool(parse_docstring=True, response_format="content_and_artifact")
    def tool_func(query: dict) -> dict:
        """ A tool that retrieves movies information from the DataFrame.
        
        Args:
            query: A dictionary where keys are column names and values are filter criteria.

        Returns:
            dict: A success message and an artifact containing the current route and retrieved rows.
        """
        # Retrieve relevant movie records from the from movies_df DataFrame
        filtered_df = filter_movies(movies_df, query)
        retrieved_rows = filtered_df.to_dict(orient="records")
        
        # Generate the artifact with routing information and movies details 
        retrieved_rows_metadata = {
            "current_route": return_route,
            "retrieved_rows": retrieved_rows
        }
        
        # Serialize the results
        serialized = f"Found {len(retrieved_rows)} matching rows."
        
        # Return the serialized message and the artifact with routing and data
        return serialized, retrieved_rows_metadata

    return tool_func
