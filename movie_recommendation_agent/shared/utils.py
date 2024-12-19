import csv
import os
from datetime import datetime
from typing import Any, Annotated, Optional, Dict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

CSV_FILE_PATH = os.path.abspath("user_registry.csv")


def is_user_registered(user_id):
    """Check if a user is already registered.

    Args:
        user_id (str): Unique ID of the user.

    Returns:
        bool: True if the user is registered, False otherwise.
    """
    if not os.path.isfile(CSV_FILE_PATH):
        return False

    with open(CSV_FILE_PATH, mode='r') as file:
        # Attempt to read headers
        try:
            reader = csv.DictReader(file)
        except csv.Error:
            # Fallback in case the file does not have headers
            reader = csv.reader(file)
            fieldnames = ["user_id", "name", "surname", "email", "registered_at"]
            # Skip the first line if it's just data without headers
            next(reader, None)  # skip the header

        for row in reader:
            if str(row.get("user_id") or row[0]) == str(user_id):
                return True
    return False


def save_user_info(user_data: dict):
    """Save user information to a CSV file.

    Args:
        user_data (dict): Dictionary containing user information with keys:
                          "user_id", "name", "surname", "email".
    """
    # Check if the file exists, if not, create it with the header
    if not os.path.isfile(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["user_id", "first_name", "last_name", "email", "registered_at"])

    # Append the user information to the CSV file
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            user_data.get("user_id"),
            user_data.get("first_name"),
            user_data.get("last_name"),
            user_data.get("email"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])


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