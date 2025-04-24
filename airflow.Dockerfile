# Custom container image to extend the official Apache Airflow image with custom dependencies!
# Edit docker-compose: lines 49-54, to mount this image.
# References:
#   - Building a custom Airflow image (to install new dependencies): 
#       - https://airflow.apache.org/docs/docker-stack/build.html
#   - Using MongoDB with Apache Airflow: 
#       - https://www.mongodb.com/developer/products/mongodb/mongodb-apache-airflow/
#   - Connection type missing in Airflow (for Airflow UI connections configuration):
#       - https://www.youtube.com/watch?v=UzGQ8R4F6z4
#       - https://www.youtube.com/watch?v=sVNvAtIZWdQ  
#   - Airflow Providers installation (example link below - click on "Use module/provider"):
#       - https://registry.astronomer.io/providers/apache-airflow-providers-mongo/versions/4.2.1/modules/MongoHook
#   - Adding packages from requirements.txt:
#       - https://airflow.apache.org/docs/docker-stack/build.html#adding-packages-from-requirements-txt

FROM apache/airflow:2.10.5

USER root

# Install system dependencies (for building Python packages that depend on C libraries)
# and install PostgreSQL development libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA 12.4
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-4

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

USER airflow

# Copy the project files into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch
#RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#RUN pip3 install torch torchvision torchaudio

# Install additional Python dependencies
# RUN pip install --no-cache-dir psycopg2==2.9.10 apache-airflow-providers-postgres==5.13.1
# RUN pip install --no-cache-dir pymongo==4.11.2 apache-airflow-providers-mongo==4.2.1
# RUN pip install --no-cache-dir neo4j apache-airflow-providers-neo4j==3.7.0

# Optional entrypoint for testing the container functionality
CMD ["python", "-c", "print('Airflow provider dependencies are ready!')"]