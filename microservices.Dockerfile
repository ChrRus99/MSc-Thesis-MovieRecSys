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

# Install necessary system libraries and CUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
  libpq-dev gcc wget \
  && rm -rf /var/lib/apt/lists/*

# Install CUDA 12.4 (matching airflow.Dockerfile)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-4

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch to match the version in Airflow
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional entrypoint for testing the container functionality
CMD ["python", "-c", "print('Microservices dependencies are ready!')"]