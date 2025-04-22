# Use an official Python base image
FROM python:3.12-slim

# Working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY requirements.txt .
COPY .env /app/.env
COPY db_handlers/ /app/db_handlers/

# Install system dependencies (for building Python packages that depend on C libraries)
# and install PostgreSQL development libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so other services can import this library
ENV PYTHONPATH="/app:/app/db_handlers"

# Optional entrypoint for testing the container functionality
CMD ["python", "-c", "import db_handlers; print('db_handlers is ready!')"]