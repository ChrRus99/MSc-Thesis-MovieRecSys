from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from retrieval_graph.graph import builder

def create_rag_agent(with_memory: bool = False) -> CompiledStateGraph:
    """
    Create and compile a RAG (Retrieval-Augmented Generation) Agent with optional memory 
    capabilities.

    Args:
        with_memory (bool): If True, integrate a memory saver for storing and managing user memories
                            during the agent's session. Defaults to False.

    Returns:
        CompiledStateGraph: The compiled RAG agent ready for deployment with optional memory
                            management.
    """

    # Compile the graph based on whether memory management is needed
    if with_memory:
        # Compile with memory management for managing user memories
        memory = MemorySaver()  
        graph = builder.compile(checkpointer=memory)  
    else:
        # Compile without memory management
        graph = builder.compile()  

    # Set the name of the graph for identification purposes
    graph.name = "RetrievalGraph"
    return graph