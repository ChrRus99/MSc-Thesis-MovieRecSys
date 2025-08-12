# movie_recommender/recommenders/__init__.py

from .popularity_ranking import PopularityRanking
from .content_based_filtering import ContentBasedFiltering
from .collaborative_filtering import CollaborativeFiltering
from .hybrid_filtering import HybridFiltering

__all__ = [
    "PopularityRanking", "ContentBasedFiltering", "CollaborativeFiltering", "HybridFiltering"
]
