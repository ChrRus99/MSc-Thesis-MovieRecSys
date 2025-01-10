import ast
import os
import re
import sys
import pandas as pd
import numpy as np
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from conventional_recommendation_models.movie_recommendation_system.Src.scripts.data.tabular_dataset_handler import TabularDatasetHandler


# Tabular dataset datapath
processed_data_path = "D:\\Internship\\resources\\movielens_processed"
tabular_dataset_filepath = os.path.join(processed_data_path, "tabular_dataset_handler_instance.pkl")

# Tabular dataset column types
MOVIE_COLUMNS_TYPES = {
    "adult": bool,
    "belongs_to_collection": "object",
    "budget": "Int64",
    "genres": "object",  # Keeping as object to handle list-like structures
    "homepage": "object",
    "id": "int32",
    #"imdb_id": "object",
    "original_language": "object",
    "original_title": "object",
    "overview": "object",
    "popularity": "float64",
    #"poster_path": "object",
    "production_companies": "object", # Keeping as object to handle list-like structures
    "production_countries": "object", # Keeping as object to handle list-like structures
    "release_date": "object",  # Convert to datetime later if needed
    "revenue": "float64",
    "runtime": "float64",
    "spoken_languages": "object",  # Keeping as object to handle list-like structures
    "status": "object",
    "tagline": "object",
    "title": "object",
    #"video": bool,
    "vote_average": "float64",
    "vote_count": "Int64",
    "year": "Int16",
}

# TODO: da caricare l'istanza tdh una sola volta non ad ogni chiamata
# per farlo bisogna mantenere un'instanza nello stato e estrarla dallo stato per passarla al tool
# oppure creare un database sql in modo da fare solo chiamate sql per filtrare dati
# Stessa cosa per funzione sotto 
def retrieve_filtered_movies(query: dict) -> pd.DataFrame:
    """ 
    Filters and retrieve movies from the 'movies_df' based on a given query.

    This function handles different column types:
        - List-like columns: Checks if the query values exist within the list.
        - Numeric columns: Performs a direct equality comparison.
        - String/object columns: Compares for exact matches.

    Args:
        query (dict): A dictionary where keys are column names and values are the filter criteria.
    
    Returns:
        pd.DataFrame: A DataFrame containing the retrieved filtered movies.
    
    Note:
        - If a column specified in the query does not exist in the DataFrame, it is skipped.
    """
    # Instantiate a tabular dataset handler
    tdh = TabularDatasetHandler.load_class_instance(tabular_dataset_filepath)

    # Get a copy of the 'movies_df'
    movies_df = tdh.get_movies_df_deepcopy()

    # Process the 'movies_df'
    movies_df = __process_movies_df(movies_df)
    
    # Retrieve filtered movies from the 'movies_df'
    filtered_movies_df = movies_df.copy()

    for column, value in query.items():
        # Skip non-existent columns
        if column not in filtered_movies_df.columns:
            continue
        
        # Special handling for list-like columns
        if isinstance(movies_df[column].iloc[0], list):
            # Normalize the query value to a list format
            if isinstance(value, str):
                try:
                    # Attempt to evaluate the string as a literal list
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If evaluation fails, try to parse a comma-separated string
                    value = re.findall(r'\b\w+\b', value)
            elif not isinstance(value, list):
                value = [value]
            
            # Filter movies based on the normalized list
            filtered_movies_df = filtered_movies_df[filtered_movies_df[column].apply(lambda x: all(v in x for v in value) if isinstance(x, list) else False)]
        # Handling numeric comparisons
        elif filtered_movies_df[column].dtype in [np.float64, np.int32, np.int16]:
            filtered_movies_df = filtered_movies_df[filtered_movies_df[column] == float(value)]
        # General case: string or object comparison
        else:
            filtered_movies_df = filtered_movies_df[filtered_movies_df[column] == value]
    
    return filtered_movies_df


def retrieve_filtered_cast_and_crew(query: dict) -> pd.DataFrame:
    """
    Filters and retrieves movies from the 'credits_df' based on a given query.

    Args:
        query (dict): A dictionary containing filtering criteria. 
                      Supported keys: 'title', 'actor', 'director'.

    Returns:
        pd.DataFrame: A filtered DataFrame where all query conditions are met.
    """
    # Instantiate a tabular dataset handler
    tdh = TabularDatasetHandler.load_class_instance(tabular_dataset_filepath)

    # Get a copy of the 'movies_df'
    movies_df = tdh.get_movies_df_deepcopy()
    credits_df = tdh.get_credits_df_deepcopy()

    # Get only relevant columns from 'movies_df'
    movies_df = movies_df[['id', 'title']]

    # Add the columns 'cast' and 'crew' to the 'movies_df'
    movies_cast_and_crew_df = movies_df.merge(credits_df, on='id')

    # Add the columns 'cast_size' and 'crew_size' to the 'movies_df'
    movies_cast_and_crew_df['cast_size'] = movies_cast_and_crew_df['cast'].apply(lambda x: len(x))
    movies_cast_and_crew_df['crew_size'] = movies_cast_and_crew_df['crew'].apply(lambda x: len(x))

    # Start with the full DataFrame and filter it progressively
    filtered_df = movies_cast_and_crew_df.copy()

    # Check if keys in the query are valid
    for key in query.keys():
        if key not in ["title", "actor", "director"]:
            raise KeyError(f"Column '{key}' not found in the DataFrame. Valid keys are: 'title', 'actor', 'director'")

    for key, value in query.items():
        # Filter by movie ID
        if key == "title":
            filtered_df = filtered_df[filtered_df['title'] == value]
        
        # Filter by actor in the 'cast' column
        elif key == "actor":
            filtered_df = filtered_df[
                filtered_df['cast'].apply(lambda cast: any(actor['name'] == value for actor in cast))
            ]

        # Filter by director in the 'crew' column
        elif key == "director":
            filtered_df = filtered_df[
                filtered_df['crew'].apply(lambda crew: any(member['job'] == 'Director' and member['name'] == value for member in crew))
            ]

    return filtered_df


def __process_movies_df(movies_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Processes the movies_df by handling missing values and converting data types.

    Args:
        movies_df (pd.DataFrame): A DataFrame containing movie data.

    Returns:
        pd.DataFrame: The processed DataFrame with cleaned data and standardized types.

    Processing steps:
        - Converts the 'year' column to numeric, replacing NaN values with -1.
        - Casts all columns to predefined data types for consistency.
    """
    # Handle invalid or missing values in the 'year' column: replace NaN with -1
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    movies_df['year'] = movies_df['year'].fillna(-1).astype('Int16')

    # Convert dataframe columns to the correct type
    movies_df = movies_df.astype(MOVIE_COLUMNS_TYPES)

    return movies_df


################################# DA CANCELLARE ####################################################
def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""
################################# DA CANCELLARE ####################################################


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)