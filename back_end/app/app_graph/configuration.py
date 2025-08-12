"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from app.app_graph import prompts
from app.shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    ## llm models

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    ## agents
    # TODO

    ## prompts

    # greet_and_route_system_prompt: str = field(
    #     default=prompts.GREET_AND_ROUTE_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for greeting and for classifying user questions to route them to the correct node."
    #     },
    # )

    # report_issue_system_prompt: str = field(
    #     default=prompts.REPORT_ISSUE_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for asking for more information from the user."
    #     },
    # )

    # sign_up_system_prompt: str = field(
    #     default=prompts.SIGN_UP_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for signing up a new user."
    #     },
    # )

    # sign_in_system_prompt: str = field(
    #     default=prompts.SIGN_IN_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for signing in an already registered user."
    #     },
    # )
