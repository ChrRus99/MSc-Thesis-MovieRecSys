"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and retrieval processes, 
    including embedding model selection, retriever provider choice, and search parameters.
    """

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        # Retrieve the "configurable" section from the provided configuration
        config = ensure_config(config)
        configurable = config.get("configurable") or {}

        # Get the set of field names from the class that are initialized during creation
        _fields = {f.name for f in fields(cls) if f.init}

        # Filter out fields from the "configurable" dictionary that match the fields
        # in the class and return a new instance of the class with those configurations.
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)