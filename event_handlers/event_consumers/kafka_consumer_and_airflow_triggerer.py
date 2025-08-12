import json
import os
import requests
from dotenv import load_dotenv
from kafka import KafkaConsumer
from pathlib import Path
from typing import List, Dict, Any

def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

# Dynamically find the project root (assumes .env is always in recsys)
project_root = Path(__file__).resolve().parents[2]  # Move up two levels
dotenv_path = project_root / ".env"  # Path to .env

# Load environment variables from .env file
load_dotenv(dotenv_path)


# Kafka connection parameters
KAFKA_BROKER = os.environ.get("DOCKER_KAFKA_BROKER") if is_docker() else os.environ.get("KAFKA_BROKER")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC")

# Airflow API endpoint for triggering DAG runs
AIRFLOW_BASE_URL = os.environ.get("DOCKER_AIRFLOW_BASE_URL") if is_docker() else os.environ.get("AIRFLOW_BASE_URL")
AUTH = (os.environ.get("AIRFLOW_USER"), os.environ.get("AIRFLOW_PASSWORD"))
DAG_ID = "online_pipeline"  # The ID of the DAG to trigger
AIRFLOW_API_ENDPOINT = f"{AIRFLOW_BASE_URL}/api/v1/dags/{DAG_ID}/dagRuns"


def trigger_batch_dag(batch_users: List[Dict[str, Any]]):
    """
    Sends a batch request to the Airflow REST API to trigger the DAG for multiple users.

    Args:
        batch_users (List[Dict]): A list of dictionaries, where each dict contains user information
            expected by the DAG's config (e.g., 'user_id', 'num_new_ratings').

    Workflow:
        1. Constructs the JSON payload with the correct configuration key ("users_to_retrain")
            expected by the DAG.
        2. Sends an HTTP POST request to the correct Airflow API endpoint.
        3. Logs success or failure based on Airflow's response.
    """
    if not batch_users:
        print("[WARN] No users in the batch to trigger DAG run.")
        return

    # Construct the payload for the Airflow API
    payload = {"conf": {"users_to_retrain": batch_users}}

    user_ids = [user.get("user_id", "UNKNOWN") for user in batch_users]
    print(f"[LOG] Attempting to trigger DAG '{DAG_ID}' for users: {user_ids} with payload: {json.dumps(payload)}")

    try:
        # Send the request to trigger the DAG run
        response = requests.post(AIRFLOW_API_ENDPOINT, json=payload, auth=AUTH)

        # Check the response status code
        if response.status_code == 200:
            print(f"[LOG] Successfully triggered DAG run for users {user_ids}. Response: {response.text}")
        elif response.status_code == 401:
            print(f"[LOG] Authentication failed. Check Airflow credentials. Status: {response.status_code}, Response: {response.text}")
        elif response.status_code == 403:
            print(f"[LOG] Permission denied. Ensure the user '{AUTH[0]}' has permissions to trigger DAG '{DAG_ID}'. Status: {response.status_code}, Response: {response.text}")
        elif response.status_code == 404:
            print(f"[LOG] DAG '{DAG_ID}' not found or API endpoint incorrect. Check URL '{AIRFLOW_API_ENDPOINT}'. Status: {response.status_code}, Response: {response.text}")
        else:
            # Other potential errors (e.g., 400 Bad Request if payload is wrong format, 500 Server Error)
            print(f"[ERROR] Failed to trigger DAG run for users {user_ids}. Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connection Error: Could not connect to Airflow at {AIRFLOW_API_ENDPOINT}. Is Airflow running? Error: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred when trying to trigger the DAG: {e}")


def main():
    """
    Listens to Kafka messages in batches and triggers the Airflow DAG efficiently.

    Workflow:
        1. Subscribes to the `user_retrain` Kafka topic.
        2. Enters a loop to continuously consume messages from Kafka.
        3. Uses `poll(timeout_ms=1000)` to fetch multiple messages at once.
        4. Collects `user_id`s into a batch list.
        5. If there are users in the batch, sends a single request to trigger the Airflow DAG.
        6. Gracefully handles shutdowns and ensures proper resource cleanup.
    """
    print(f"[INFO] Starting Kafka consumer for topic '{KAFKA_TOPIC}' on broker '{KAFKA_BROKER}'...")
    print(f"[INFO] Will trigger Airflow DAG '{DAG_ID}' via API at '{AIRFLOW_API_ENDPOINT}'\n")

    try:
        # Create a Kafka consumer instance which subscribes to the topic 'user_retrain'
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='earliest',                               # Start reading from the beginning if no offset is stored
            enable_auto_commit=True,                                    # Commit offsets automatically
            group_id='airflow_trigger_group',                           # Assign a group ID
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=1000                                    # How long poll() waits if no messages
        )
    except Exception as e:
        print(f"[ERROR] Failed to create Kafka consumer: {e}")
        return # Exit if consumer cannot be created

    print("[INFO] Kafka consumer created successfully. Waiting for messages...\n")

    try:
        while True:
            # Poll Kafka for new messages.
            messages_dict = consumer.poll(timeout_ms=5000) # Poll every 5 seconds

            if not messages_dict:
                # No messages received in this poll interval
                continue

            batch_users = []  # Reset batch for each poll cycle

            for topic_partition, messages_list in messages_dict.items():
                print(f"[LOG] Received {len(messages_list)} messages from {topic_partition}")

                for message in messages_list:
                    try:
                        message_value = message.value # Already deserialized
                        # Expecting message value like: {"user_id": "some_id", "num_new_ratings": 5}

                        # Extract user_id and num_new_ratings from the message value
                        user_id = message_value.get("user_id")
                        num_new_ratings = message_value.get("num_new_ratings")

                        if user_id is not None and num_new_ratings is not None:
                            print(f"[LOG]  - Adding user: {user_id}, ratings: {num_new_ratings} to batch.")

                            # Append to batch_users list
                            batch_users.append({"user_id": user_id, "num_new_ratings": num_new_ratings})
                        else:
                            print(f"[WARN] Skipping message with missing 'user_id' or 'num_new_ratings': {message_value}")

                    except Exception as e:
                         print(f"[ERROR] Failed to process message: {message}. Error: {e}")

            # If the batch collected any valid users, trigger the DAG run
            if batch_users:
                print(f"Collected batch of {len(batch_users)} users. Triggering DAG run...")
                trigger_batch_dag(batch_users)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Shutting down consumer...")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the main loop: {e}")
    finally:
        if 'consumer' in locals() and consumer:
            print("[INFO] Closing Kafka consumer.")
            consumer.close()
        print("[INFO] Shutdown complete.")

if __name__ == "__main__":
    main()