import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage


def format_agent_structured_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from a given string, parses it into a dictionary, and wraps messages in
    LangChain's AIMessage objects.

    Args:
        text (str): The input string that may contain a JSON object.

    Returns:
        Optional[Dict[str, Any]]: A dictionary with messages wrapped in AIMessage objects.
            Returns None if no valid JSON is found.
    """
    # Regex to extract JSON inside triple backticks or raw JSON
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        json_str = match.group(1) if "```json" in text else match.group(0)
        try:
            parsed_data = json.loads(json_str)

            # Convert "messages" list to AIMessage objects if present
            if "messages" in parsed_data and isinstance(parsed_data["messages"], list):
                parsed_data["messages"] = [AIMessage(content=msg) for msg in parsed_data["messages"]]

            return parsed_data  # Return updated dictionary
        except json.JSONDecodeError:
            print("Error: Failed to parse JSON.")
            return None
    return None  # No JSON found


def format_prompt(prompt_template: str, params_dict: Dict[str, Any]) -> str:
    """
    Replaces the specified placeholders in a prompt using key-value pairs from a dictionary.

    Args:
        prompt_template (str): The prompt containing placeholders like {key}.
        params_dict (Dict[str, Any]): A dictionary where keys are placeholder names and values are
            the replacements.

    Returns:
        str: The prompt with specified placeholders replaced.
    """
    result_string = prompt_template

    for key, value in params_dict.items():
        placeholder = f"{{{key}}}"
        # Convert value to string for replacement
        result_string = result_string.replace(placeholder, str(value))
    
    return result_string