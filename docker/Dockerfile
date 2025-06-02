# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY environment.yml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the project in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/checkpoints models/final logs

# Set permissions
RUN chmod +x scripts/*.py

# Expose ports for MLflow and other services
EXPOSE 5000 8000

# Default command
CMD ["python", "scripts/train.py", "--config", "configs/training/default.yaml"] 