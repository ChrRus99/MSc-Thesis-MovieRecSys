import math
import os
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display
from langchain_community.graphs import Neo4jGraph
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from yfiles_jupyter_graphs import GraphWidget

from db_handlers.utils import (
    is_docker,
    environment,
    generate_full_text_query,
)


try:
    # INSTALL: pip install apache-airflow-providers-neo4j==3.7.0

    # If running in Airflow use the Neo4jHook
    from airflow.providers.neo4j.hooks.neo4j import Neo4jHook
    
    AIRFLOW_AVAILABLE = "AIRFLOW_HOME" in os.environ
except ImportError:
    # Fallback to using neo4j
    AIRFLOW_AVAILABLE = False

AIRFLOW_AVAILABLE = False  # TEMP: forces to avoid using Airflow Hooks

if AIRFLOW_AVAILABLE:
    print(f"[LOG] Detected Airflow environment, with Docker: [{is_docker()}]")
else:
    print(f"[LOG] Detected local environment, with Docker: [{is_docker()}]")
    
    from neo4j import GraphDatabase
    from dotenv import load_dotenv

    # Dynamically find the project root (assumes .env is always in recsys)
    project_root = Path(__file__).resolve().parents[2]  # Move up two levels
    dotenv_path = project_root / ".env"  # Path to .env

    # Load environment variables from .env file
    load_dotenv(dotenv_path)


def get_db_connection():
    """Establishes and returns a MongoDB database connection, using Airflow's MongoHook if available."""
    # Instantiate connection to Neo4j
    return Neo4jGraph()


def create_knowledge_graph(movies_cast_and_crew_df: pd.DataFrame, batch_size: int = 1000) -> None:
    """
    Creates a knowledge graph in Neo4j from the given DataFrame in batches.
    
    The DataFrame is expected to contain the following columns:
    ["id", "title", "budget", "genres", "overview", "production_companies",
     "production_countries", "release_date", "revenue", "runtime", 
     "spoken_languages", "tagline", "cast", "crew"]
     
    The columns such as 'genres', 'production_companies', 'production_countries'
    are assumed to be lists of strings, while 'spoken_languages', 'cast', and 'crew'
    are assumed to be lists of dictionaries (with keys such as 'name', 'character', or 'job').
    
    Ingestion is performed in batches (default: 1000 rows per batch) to support large datasets.
    """
    graph = get_db_connection()

    # Define unique constraints and indices
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE;")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (pc:ProductionCompany) REQUIRE pc.name IS UNIQUE;")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE;")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Language) REQUIRE l.name IS UNIQUE;")

    # Convert the DataFrame to a list of dictionaries.
    movies = movies_cast_and_crew_df.to_dict("records")

    # Define the Cypher query with CASE expressions...
    query = """
    UNWIND $movies as row
    MERGE (m:Movie {id: row.id})
    SET m.title = row.title,
        m.budget = toInteger(row.budget),
        m.overview = row.overview,
        m.release_date = CASE WHEN row.release_date IS NULL OR row.release_date = "" THEN null ELSE date(row.release_date) END,
        m.revenue = toFloat(row.revenue),
        m.runtime = toInteger(row.runtime),
        m.tagline = row.tagline

    // Process genres (assumed to be a list of strings)
    FOREACH (genre IN row.genres |
        MERGE (g:Genre {name: genre})
        MERGE (m)-[:IN_GENRE]->(g)
    )

    // Process production companies (assumed to be a list of strings)
    FOREACH (comp IN row.production_companies |
        MERGE (pc:ProductionCompany {name: comp})
        MERGE (m)-[:PRODUCED_BY]->(pc)
    )

    // Process production countries (assumed to be a list of strings)
    FOREACH (country IN row.production_countries |
        MERGE (c:Country {name: country})
        MERGE (m)-[:PRODUCED_IN]->(c)
    )

    // Process spoken languages (assumed to be a list of strings)
    FOREACH (lang IN row.spoken_languages |
        MERGE (l:Language {name: lang})
        MERGE (m)-[:SPOKEN_LANGUAGE]->(l)
    )

    // Process cast (assumed to be a list of dicts with keys 'name' and 'character')
    FOREACH (actor IN row.cast |
        MERGE (p:Person {name: actor.name})
        MERGE (p)-[r:ACTED_IN]->(m)
        SET r.character = actor.character
    )

    // Process crew (assumed to be a list of dicts with keys 'name', 'job', and 'department')
    FOREACH (crewMember IN row.crew |
        MERGE (p:Person {name: crewMember.name})
        FOREACH (_ IN CASE WHEN crewMember.job = 'Director' THEN [1] ELSE [] END |
            MERGE (p)-[:DIRECTED]->(m)
        )
        FOREACH (_ IN CASE WHEN crewMember.job <> 'Director' THEN [1] ELSE [] END |
            MERGE (p)-[r:WORKED_ON]->(m)
            SET r.job = crewMember.job, r.department = crewMember.department
        )
    )
    """

    # Ingest data in batches
    batch_size = 8000 # Define the batch size for ingestion (consider tuning this)
    total = len(movies)
    for i in tqdm(range(0, total, batch_size), desc="Ingesting batches"):
        batch = movies[i:i + batch_size]
        graph.query(query, params={"movies": batch})

    # Define fulltext indices WITH EXPLICIT NAMES
    print("[LOG] Creating/Ensuring full-text indices...")
    graph.query("CREATE FULLTEXT INDEX movie IF NOT EXISTS FOR (m:Movie) ON EACH [m.title]")
    graph.query("CREATE FULLTEXT INDEX person IF NOT EXISTS FOR (p:Person) ON EACH [p.name]")
    print("[LOG] Full-text indices creation commands sent.")

    # # Wait for the indices to become online and queryable
    # print("[LOG] Waiting for indices to become online (up to 300s each)...")
    # try:
    #     # Wait up to 300 seconds (5 minutes) for each index
    #     graph.query("CALL db.awaitIndex('movie', 300)")
    #     print("[LOG] Index 'movie' is online.")
    #     graph.query("CALL db.awaitIndex('person', 300)")
    #     print("[LOG] Index 'person' is online.")
    # except Exception as e:
    #     # Handle cases where the index doesn't exist or timeout occurs
    #     print(f"[ERROR] Failed waiting for indices to become online: {e}")
    #     print("[WARN] Proceeding without confirmation indices are online. Queries might fail.")
    #     # Depending on your workflow, you might want to raise the error here
    #     # raise e

    graph_url = graph.url if hasattr(graph, 'url') else "Unknown URL"
    print(f"[LOG] Knowledge graph created successfully in [{environment}] at URL: [{graph_url}].")
    print(f"[LOG] Ingested {len(movies)} movies into the knowledge graph.")


# def reset_knowledge_graph() -> None:
#     """ Reset the Neo4j graph instance by deleting all nodes and relationships. """
#     graph = get_db_connection()
    
#     # Drop all nodes and relationships in batches using APOC
#     graph.query("CALL apoc.periodic.iterate('MATCH (n) RETURN n','DETACH DELETE n',{batchSize:1000})")

#     print("[LOG] Knowledge graph reset successfully.")

def reset_knowledge_graph() -> None:
    """
    Reset the Neo4j graph instance by dropping known full-text indices
    and deleting all nodes and relationships.
    """
    graph = get_db_connection()
    print("[LOG] Starting knowledge graph reset...")

    # Drop Full-Text Indices (Use IF EXISTS to avoid errors)
    print("[LOG] Dropping known full-text indices (if they exist)...")
    indices_to_drop = [
        "movie",           # Index intended by the application
        "person",          # Index intended by the application
        "index_87a50586",  # Problematic auto-named index seen in logs
        "index_45eed07a"   # Problematic auto-named index seen in logs
    ]
    for index_name in indices_to_drop:
        try:
            query = f"DROP INDEX {index_name} IF EXISTS;"
            graph.query(query)
            print(f"[LOG]   - Attempted to drop index '{index_name}'.")
        except Exception as e:
            # Catch errors during index drop (e.g., syntax error if version mismatch, permissions)
            # Using IF EXISTS should prevent IndexNotFound errors.
            print(f"[ERROR] Failed trying to drop index '{index_name}': {e}")
            print("[WARN] Continuing reset process despite potential index drop failure.")
    print("[LOG] Finished attempting to drop indices.")

    # Drop all nodes and relationships using APOC
    print("[LOG] Deleting all nodes and relationships...")
    try:
        # Using apoc.periodic.iterate for batching deletion
        # Ensure the APOC plugin is installed in your Neo4j instance
        graph.query("CALL apoc.periodic.iterate('MATCH (n) RETURN n','DETACH DELETE n',{batchSize:1000, parallel:true})")
        print("[LOG] Node and relationship deletion completed.")
    except Exception as e:
        print(f"[ERROR] Failed during node/relationship deletion (APOC might be missing or query failed): {e}")
        # This is a critical failure for a reset
        print("[ERROR] Knowledge graph reset failed during data deletion.")
        raise # Re-raise the exception as the reset failed significantly

    print("[LOG] Knowledge graph reset finished successfully.")


def store_movie_rating(user_id: str, movie_id: str, rating: int):
    """Stores/Updates the knowledge graph with new movies, cast and crew nodes or information"""
    graph = get_db_connection()

    # Cypher query to store user's movie rating in the Neo4j database
    store_rating_query = """
    MERGE (u:User {userId:$user_id})
    WITH u
    UNWIND $candidates as row
    // Use the unique ID for matching the movie
    MATCH (m:Movie {id: toInteger(row.id)})
    MERGE (u)-[r:RATED]->(m)
    SET r.rating = toFloat($rating)
    RETURN distinct 'Noted' AS response
    """

    # Retrieve possible matching candidates including their IDs
    candidates = _get_candidates(movie_id, "movie")

    if not candidates:
        return "This movie is not in our database"
    elif len(candidates) > 1:
        newline = "\n"
        # Include the ID in the message
        return (
            "Need additional information, which of these "
            f"did you mean: {newline + newline.join(str(d) for d in candidates)}"
            "\nPlease provide the 'id' for the movie."
        )

    # If only one candidate, proceed with storing the rating using its ID
    single_candidate_list = [candidates[0]]

    response = graph.query(
        store_rating_query,
        params={"user_id": user_id, "candidates": single_candidate_list, "rating": rating},
    )

    # Return a confirmation message
    try:
        return response[0]["response"]
    except Exception as e:
        print(e)
        return "Something went wrong"


def get_movie_cast_and_crew_information(entity: str, type: str, entity_id: Optional[str] = None) -> str:
    """
    Fetch detailed information about a given movie or person from the database.

    Args:
        entity (str): The name/title of the movie or person to search for (used if entity_id is None).
        type (str): The type of entity ('movie' or 'person').
        entity_id (Optional[str]): The unique ID of the movie (numeric string) or person (name string)
                                   to fetch directly. If provided, 'entity' and 'type' are ignored for lookup.

    Returns:
        str: A formatted string containing the information or an error/ambiguity message.
    """
    graph = get_db_connection()
    candidate_info = None

    # If entity_id is provided, try to fetch directly
    if entity_id:
        id_property = "id" if entity_id.isdigit() else "name"
        label = "Movie" if id_property == "id" else "Person"
        match_value = int(entity_id) if id_property == "id" else entity_id

        direct_fetch_query = f"""
        MATCH (m:{label} {{{id_property}: $match_value}})
        RETURN coalesce(m.title, m.name) AS candidate
        LIMIT 1
        """
        result = graph.query(direct_fetch_query, params={"match_value": match_value})
        if result:
            candidate_info = {"candidate": result[0]["candidate"]}
        else:
            return f"No information found for the provided ID: {entity_id}"

    # If no entity_id or direct fetch failed, perform search
    if not candidate_info:
        candidates = _get_candidates(entity, type)

        if not candidates:
            return "No information was found about the movie or person in the database"
        elif len(candidates) > 1:
            newline = "\n"
            return (
                "Need additional information, which of these "
                f"did you mean: {newline + newline.join(str(d) for d in candidates)}"
                f"\nPlease call the function again providing the 'id' for the desired entity."
            )
        else:
            candidate_info = candidates[0]

    # Fetch description using the determined candidate name/title
    description_query = """
    MATCH (m:Movie|Person)
    WHERE m.title = $candidate OR m.name = $candidate
    MATCH (m)-[r:ACTED_IN|DIRECTED|IN_GENRE|WORKED_ON]-(t)
    WITH m, type(r) as type, collect(coalesce(t.name, t.title)) as names
    WITH m, type + ": " + reduce(s="", n IN names | s + n + ", ") as types
    WITH m, collect(types) as contexts
    WITH m, "type:" + labels(m)[0] +
           "\ntitle: " + coalesce(m.title, m.name) +
           "\nyear: " + coalesce(toString(m.release_date.year), "") + "\n" +
           reduce(s="", c in contexts | s + substring(c, 0, size(c)-2) +"\n") as context
    RETURN context
    LIMIT 1
    """
    data = graph.query(
        description_query, params={"candidate": candidate_info["candidate"]}
    )

    if data:
        return data[0]["context"]
    else:
        return f"Found {candidate_info['candidate']} but no further details available."


def show(cypher_query: Optional[str] = None) -> GraphWidget:
    """
    Displays the graph resulting from the given Cypher query.
    
    Parameters:
        cypher_query: Optional Cypher query to execute and visualize the graph.
                        If None, a default query is used to display a sample of the graph.
    Returns:
        A GraphWidget object displaying the queried graph.
    """
    # Set a default Cypher query if none is provided
    if cypher_query is None:
        # Directly show the graph resulting from the given Cypher query
        cypher_query = "MATCH (s)-[r]->(t) RETURN s,r,t LIMIT 500"
    
    # Create a graph widget to display the graph
    try:
        with GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
        ) as neo4j_driver:
            with neo4j_driver.session() as session:
                result_graph = session.run(cypher_query).graph()
                
                widget = GraphWidget(graph=result_graph)
                widget.node_label_mapping = 'id'
                
                # Uncomment if using in Jupyter or IPython environment
                #display(widget)
                
                return widget
    except Exception as e:
        print(f"Failed to display graph visualization: {e}")
        raise


def _get_candidates(input: str, type: str, limit: int = 3) -> List[Dict[str, str]]:
    """
    Retrieves a list of candidate entities from database based on the input string.

    Returns a list of dictionaries, each containing 'candidate' (name/title),
    'label' ('Person' or 'Movie'), and 'id' (unique identifier: movie ID or person name).
    """
    graph = get_db_connection()

    candidate_query = """
    CALL db.index.fulltext.queryNodes($index, $fulltextQuery, {limit: $limit})
    YIELD node
    WITH node, [el in labels(node) WHERE el IN ['Person', 'Movie'] | el][0] AS label
    RETURN coalesce(node.name, node.title) AS candidate,
           label,
           CASE label WHEN 'Movie' THEN toString(node.id) ELSE node.name END AS id
    """

    ft_query = generate_full_text_query(input)
    candidates = graph.query(
        candidate_query, {"fulltextQuery": ft_query, "index": type, "limit": limit}
    )

    return [dict(c) for c in candidates] if candidates else []