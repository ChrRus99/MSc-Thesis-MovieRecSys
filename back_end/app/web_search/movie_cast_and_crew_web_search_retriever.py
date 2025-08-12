import re
import os
import asyncio # Added asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import scraping functions
from app.web_search.crawl4ai_web_scraping import (
    search_movie_plot_info,
    search_movie_curiosities_info,
    search_movie_reviews,
)


if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set. OpenAIEmbeddings will fail.")
    # raise RuntimeError("OPENAI_API_KEY environment variable not set.")


CHUNK_SIZE_CHARS = 1000    # Target chunk size in characters
CHUNK_OVERLAP_CHARS = 100  # Overlap in characters
MAX_PLOT_CHARS = 25000     # Max characters for plot
MAX_REVIEW_CHARS = 25000   # Max characters for combined reviews


class MovieWebSearchRetriever:
    """
    Handles loading, chunking, embedding, indexing, and retrieving movie data provided as a list of
    search results.
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_CHARS,
            chunk_overlap=CHUNK_OVERLAP_CHARS,
            length_function=len, # Use character count
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            add_start_index=False,
        )

    def build_index(self, search_results: List[Dict[str, str]]):
        """
        Builds the FAISS index from a list of search result dictionaries.

        Args:
            search_results: A list of dictionaries, where each dictionary should have 'content' and
                'url' keys representing scraped web page data.
        """
        if not search_results:
            print("Warning: No search results provided.")
            self.vectorstore = None
            return

        # Convert raw search result dictionaries into LangChain Document objects.
        source_docs = [
            Document(
                page_content=result.get('content', ''),
                metadata={'source': result.get('url', 'Unknown')} # Store URL as metadata.
            )
            for result in search_results if result.get('content') # Ensure content exists
        ]

        if not source_docs:
            print("Warning: No valid documents created from search results.")
            self.vectorstore = None
            return

        # Split the documents into smaller chunks based on the configured splitter.
        all_chunks = self.text_splitter.split_documents(source_docs)
        if not all_chunks:
            print("Warning: No text chunks generated.")
            self.vectorstore = None
            return

        try:
            # Create the FAISS vector store from the chunks and embeddings.
            self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            print(f"FAISS index built successfully with {self.vectorstore.index.ntotal} vectors.")
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            self.vectorstore = None

    def build_index_from_docs(self, documents: List[Document]):
        """
        Builds the FAISS index directly from a list of LangChain Document objects.
        Used internally by retrieve_movie_reviews for custom splitting.
        """
        if not documents:
            print("Warning: No documents provided to build_index_from_docs.")
            self.vectorstore = None
            return

        # Split the documents into smaller chunks based on the configured splitter.
        all_chunks = self.text_splitter.split_documents(documents)
        if not all_chunks:
            print("Warning: No text chunks generated.")
            self.vectorstore = None
            return

        try:
            # Create the FAISS vector store from the chunks and embeddings.
            self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            print(f"FAISS index built successfully with {self.vectorstore.index.ntotal} vectors from pre-split docs.")
        except Exception as e:
            print(f"Error creating FAISS index from pre-split docs: {e}")
            self.vectorstore = None


    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant chunks for a query."""
        if self.vectorstore is None:
            raise RuntimeError("Index is not built. Call build_index() first.")
        if k <= 0:
            return []

        # FAISS returns Documents and L2 distance scores
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        # Format the results into a list of dictionaries.
        formatted_results = []
        for doc, score in results_with_scores:
            formatted_results.append({
                'text': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'score': float(score)
            })

        # Sort by score ascending (lower distance is better)
        formatted_results.sort(key=lambda x: x['score'])
        return formatted_results


# --- New Retrieval Functions ---

async def retrieve_movie_plot(movie_title: str) -> Optional[str]:
    """
    Retrieves the plot for a movie, prioritizing IMDb.
    Truncates the plot if it exceeds MAX_PLOT_CHARS.

    Args:
        movie_title: The title of the movie.

    Returns:
        The plot text (potentially truncated) or None if not found.
    """
    print(f"Retrieving plot for: {movie_title}")
    # Prioritize IMDb, only need the first result
    plot_results = await search_movie_plot_info(movie_title, num_results=1)

    if not plot_results:
        print(f"No plot found for {movie_title}.")
        return None

    content = plot_results[0].get('content', '')
    if not content:
        print(f"Plot result for {movie_title} had no content.")
        return None

    if len(content) > MAX_PLOT_CHARS:
        print(f"Plot for {movie_title} truncated to {MAX_PLOT_CHARS} characters.")
        return content[:MAX_PLOT_CHARS] + "..."
    else:
        print(f"Retrieved full plot for {movie_title} ({len(content)} characters).")
        return content


async def retrieve_movie_curiosities(movie_title: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k relevant curiosity/trivia chunks for a movie based on a query.

    Args:
        movie_title: The title of the movie.
        query: The query to find relevant curiosities.
        k: The number of chunks to retrieve.

    Returns:
        A list of top-k relevant chunk dictionaries, sorted by relevance.
    """
    print(f"Retrieving curiosities for: {movie_title} with query: '{query}'")
    # Fetch potential sources (e.g., 3 sources)
    curiosity_sources = await search_movie_curiosities_info(movie_title, num_results=3)

    if not curiosity_sources:
        print(f"No curiosity sources found for {movie_title}.")
        return []

    retriever = MovieWebSearchRetriever()
    retriever.build_index(curiosity_sources)

    if retriever.vectorstore is None:
        print(f"Failed to build curiosity index for {movie_title}.")
        return []

    retrieved_chunks = retriever.retrieve(query, k=k)
    print(f"Retrieved {len(retrieved_chunks)} curiosity chunks for {movie_title}.")
    return retrieved_chunks


async def retrieve_movie_reviews(movie_title: str, query: str, initial_k: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves relevant review chunks for a movie based on a query, ensuring each chunk
    represents a single review and the total character count is limited.

    Args:
        movie_title: The title of the movie.
        query: The query to find relevant reviews.
        initial_k: The initial number of chunks to retrieve before filtering by character limit.

    Returns:
        A list of relevant review chunk dictionaries, sorted by relevance, up to MAX_REVIEW_CHARS.
    """
    print(f"Retrieving reviews for: {movie_title} with query: '{query}'")
    # Fetch potential sources (e.g., 3 sources)
    review_sources = await search_movie_reviews(movie_title, num_results=3)

    if not review_sources:
        print(f"No review sources found for {movie_title}.")
        return []

    # --- Custom Splitting Logic ---
    individual_review_docs = []
    imdb_separator = "\n\n---\n\n" # Separator used in _extract_imdb_reviews
    general_separator = "\n\n"     # Separator used in _extract_reviews fallback

    for source_result in review_sources:
        content = source_result.get('content', '')
        url = source_result.get('url', 'Unknown')
        if not content:
            continue

        # Determine the separator based on the source or content structure
        separator = imdb_separator if imdb_separator in content else general_separator

        split_reviews = content.split(separator)
        for review_text in split_reviews:
            cleaned_review = review_text.strip()
            if len(cleaned_review) > 50: # Basic filter for meaningful reviews
                individual_review_docs.append(
                    Document(page_content=cleaned_review, metadata={'source': url})
                )
    # --- End Custom Splitting ---

    if not individual_review_docs:
        print(f"Could not extract individual reviews from sources for {movie_title}.")
        return []

    print(f"Extracted {len(individual_review_docs)} individual reviews for indexing.")

    retriever = MovieWebSearchRetriever()
    # Build index using the manually split documents
    retriever.build_index_from_docs(individual_review_docs)

    if retriever.vectorstore is None:
        print(f"Failed to build review index for {movie_title}.")
        return []

    # Retrieve an initial larger set of chunks
    retrieved_chunks = retriever.retrieve(query, k=initial_k)

    # Filter chunks by total character count
    final_chunks = []
    total_chars = 0
    for chunk in retrieved_chunks:
        chunk_len = len(chunk.get('text', ''))
        if total_chars + chunk_len <= MAX_REVIEW_CHARS:
            final_chunks.append(chunk)
            total_chars += chunk_len
        else:
            # If adding the next chunk exceeds the limit, stop.
            print(f"Reached character limit ({total_chars} chars) for reviews. Stopping.")
            break

    print(f"Retrieved {len(final_chunks)} review chunks for {movie_title} within character limit.")
    return final_chunks
