"""State management for the app graph.

This module defines the state structures used in the app graph. It includes the definition for
AppAgentState, which is an extended version of the InputState.
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict, Optional, List, Dict, Tuple

from app.shared.state import InputState


class UserData(TypedDict):
    """User data structure."""
    first_name: str
    """The user's first name."""
    last_name: str
    """The user's last name."""
    email: str
    """The user's email address."""


class Movie(TypedDict):
    """Movie data structure."""
    movie_title: str
    """The title of the movie."""
    rating: float
    """The rating given to the movie (from 1 to 5)."""
    timestamp: int
    """The timestamp when the movie was rated."""


# This is the primary state of your agent, where you can store any information.
@dataclass(kw_only=True)
class AppAgentState(InputState):
    """State of the retrieval graph / agent."""
    # User's personal details
    user_id: Optional[str] = None
    """The user's unique identifier."""
    is_user_registered: Optional[bool] = None
    """A flag indicating whether the user is registered in the system."""
    is_user_new: Optional[bool] = None
    """A flag indicating whether the user is new to the system."""
    user_data: UserData = field(default_factory=dict)
    """The user's personal data."""

    # User's seen movies
    seen_movies: List[Movie] = field(default_factory=list)
    """The list of movies the user has seen."""
