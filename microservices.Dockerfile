# Use an official Python base image
FROM python:3.12-slim

# Working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY requirements.txt .
COPY .env /app/.env
COPY microservices/ /app/microservices/
COPY movie_recommendation_system/src /app/movie_recommendation_system/
COPY db_handlers/ /app/db_handlers/
COPY data/ /app/data/

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
  libpq-dev gcc

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Optional entrypoint for testing the container functionality
CMD ["python", "-c", "print('Microservices dependencies are ready!')"]