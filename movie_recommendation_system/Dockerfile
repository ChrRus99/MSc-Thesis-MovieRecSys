# Use an official Python base image
FROM python:3.12-slim

# Working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY requirements.txt .
COPY src/ /app/src/

# Install system dependencies (for building Python packages that depend on C libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA 12.1
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

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Set the PYTHONPATH so other services can import this library
ENV PYTHONPATH="/app/src:/app/src/movie_recommender"

# Optional entrypoint for testing the container functionality
CMD ["python", "-c", "import movie_recommender; print('movie_recommender is ready!')"]