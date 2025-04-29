# Use an official Python base image  
FROM python:3.12-slim

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Working directory inside the container
WORKDIR /app

# Copy project files
COPY requirements.txt . 
COPY .env /app/.env
COPY event_handlers/ /app/event_handlers/
COPY wait-for-kafka.sh /app/wait-for-kafka.sh

# Install system dependencies needed for Kafka libraries
RUN apt-get update && apt-get install -y \
    libpq-dev gcc librdkafka-dev curl gnupg2 software-properties-common \
    netcat-openbsd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make the wait script executable
RUN chmod +x /app/wait-for-kafka.sh

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Optional entrypoint command for testing
CMD ["python", "-c", "print('Event handlers dependencies are ready!')"]