"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import json
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState
from shared import retrieval
from shared.state import reduce_docs


async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID, adds them to the 
    retriever's index, and then signals for the documents to be deleted from the state.

    If docs are not provided in the state, they will be loaded from the 
    configuration.docs_file JSON file.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    # Ensure that a configuration is provided.
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    # Load the index configuration from the provided RunnableConfig object. 
    configuration = IndexConfiguration.from_runnable_config(config)

    # Retrieve documents from the state.
    docs = state.docs

    # If no documents are present in the state, load them from the specified JSON file.
    if not docs:
        with open(configuration.docs_file) as f:
            serialized_docs = json.load(f)
            docs = reduce_docs([], serialized_docs)

    # Create and initialize a retriever using the provided configuration.
    with retrieval.make_retriever(config) as retriever:
        await retriever.aadd_documents(docs)

     # Return a placeholder for 'documents'
    return {"docs": "delete"}


# Define the graph
builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge(START, "index_docs")
builder.add_edge("index_docs", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"