import json
import os
import time
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError, NotLeaderForPartitionError
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import PyMongoError

def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

# Dynamically find the project root (assumes .env is always in recsys)
project_root = Path(__file__).resolve().parents[2]  # Move up two levels
dotenv_path = project_root / ".env"  # Path to .env

# Load environment variables from .env file
load_dotenv(dotenv_path)


# MongoDB connection parameters
MONGO_URI = os.environ.get("DOCKER_MONGODB_URI") if is_docker() else os.environ.get("MONGODB_URI")
DB_NAME = "movie_app"
COLLECTION_NAME = "user_movie_ratings"

# Kafka connection parameters
KAFKA_BROKER = os.environ.get("DOCKER_KAFKA_BROKER") if is_docker() else os.environ.get("KAFKA_BROKER")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC")
KAFKA_RETRY_ATTEMPTS = int(os.environ.get("KAFKA_RETRY_ATTEMPTS", 3))
KAFKA_RETRY_DELAY_SECONDS = int(os.environ.get("KAFKA_RETRY_DELAY_SECONDS", 2))

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
ratings_collection = db[COLLECTION_NAME]


# In-memory counter for tracking user rating updates
user_ratings_count = {}

def initialize_kafka_producer():
    """Initializes the Kafka producer with error handling."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks=1
        )
        print("[INFO] Kafka producer created successfully.")
        return producer
    except KafkaError as e:
        print(f"[ERROR] Failed to initialize Kafka producer: {e}")
        return None

def send_kafka_message(producer, topic, message, retry_attempts=KAFKA_RETRY_ATTEMPTS, retry_delay=KAFKA_RETRY_DELAY_SECONDS):
    """Sends a message to Kafka with retries on failure."""
    if producer is None:
        print("[ERROR] Kafka producer is not initialized. Cannot send message.")
        return False

    for attempt in range(KAFKA_RETRY_ATTEMPTS):
        try:
            producer.send(topic, value=message).get(timeout=10)  # Block until sent
            producer.flush()  # Ensure delivery
            print(f"[LOG] Message sent to Kafka topic '{topic}': {message}")
            return True
        except NotLeaderForPartitionError:
            print("[WARN] Kafka leader changed, retrying...")
            time.sleep(KAFKA_RETRY_DELAY_SECONDS)
        except KafkaError as e:
            print(f"[ERROR] Failed to send message to Kafka (attempt {attempt + 1}/{retry_attempts}): {e}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] Max retry attempts reached. Message not sent: {message}")
                return False
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while sending to Kafka: {e}")
            return False

    print(f"[ERROR] Max retry attempts reached. Message not sent: {message}")
    return False

def process_change(change: dict, producer, trigger_threshold: int=5):
    """
    Process a new MongoDB change event and track new user ratings.

    Args:
        change (dict): A MongoDB change event.
        producer (KafkaProducer): The Kafka producer instance.
        trigger_threshold (int): Number of new ratings to trigger retraining.

    Workflow:
        1. Detects an "insert" operation in the ratings collection.
        2. Extracts the `user_id` from the inserted document.
        3. Increments the rating count for that user in memory.
        4. If the count reaches the threshold, sends a message to Kafka.
        5. Resets the user's rating count after publishing to Kafka.
    """

    # Only process insert events
    if change["operationType"] == "insert":
        # Extract user ID from the new rating
        user_id = change["fullDocument"]["user_id"]

        # Increment user rating count
        user_ratings_count[user_id] = user_ratings_count.get(user_id, 0) + 1
        print(f"[LOG] User {user_id} new rating count: {user_ratings_count[user_id]}")

        # If threshold reached, publish message to Kafka and reset count
        if user_ratings_count[user_id] >= trigger_threshold:
            message = {"user_id": user_id, "num_new_ratings": trigger_threshold, "action": "retrain"}
            if send_kafka_message(producer, KAFKA_TOPIC, message):
                # Reset counter for this user only if the message was sent successfully
                user_ratings_count[user_id] = 0
                print(f"[LOG] Published retraining event for user {user_id} to Kafka")
            else:
                print(f"[ERROR] Failed to publish retraining event for user {user_id} to Kafka.")


def main():
    """
    Poll MongoDB for new ratings and process them.

    Workflow:
        1. Initializes a Kafka producer.
        2. Polls MongoDB for new ratings based on timestamps.
        3. Calls `process_change()` for each new rating found.
        4. Updates the last checked timestamp to track new entries.
        5. Continues polling every 5 seconds.
    """
    TRIGGER_THRESHOLD = 5  # Number of new ratings to trigger retraining
    last_checked = time.time()

    print(f"[INFO] Starting Kafka producer for topic '{KAFKA_TOPIC}' on broker '{KAFKA_BROKER}'...")

    producer = initialize_kafka_producer()
    if producer is None:
        return

    try:
        while True:
            try:
                # Fetch documents with ratings that have timestamps greater than the last checked timestamp
                new_ratings = ratings_collection.find({
                    "ratings.timestamp": {"$gt": last_checked}  # Filter by timestamp in the ratings array
                })

                for user in new_ratings:
                    # Iterate over the ratings array to find individual ratings
                    for rating in user.get('ratings', []):
                        if rating['timestamp'] > last_checked:  # Ensure it's a new rating based on timestamp
                            process_change({"operationType": "insert", "fullDocument": user}, producer, trigger_threshold=TRIGGER_THRESHOLD)

                # Update last checked timestamp to the current time
                last_checked = time.time()

            except PyMongoError as e:
                print(f"[ERROR] Error polling MongoDB: {e}")

            time.sleep(5)  # Poll every 5 seconds
    except KeyboardInterrupt:
        print("[INFO] Shutting down producer...")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the main loop: {e}")
    finally:
        if producer:
            producer.close()
            print("[INFO] Kafka producer closed.")

if __name__ == "__main__":
    main()
