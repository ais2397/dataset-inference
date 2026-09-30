# Use an official Python image with CUDA support if needed, or base CPU
FROM python:3.10-slim

# Add pipeline metadata labels
LABEL org.opencontainers.image.dataset="CIFAR-10"
LABEL org.opencontainers.image.defense_stage="deploy_defense"
LABEL org.opencontainers.image.defense_type="fingerprinting"
LABEL org.opencontainers.image.framework="pytorch"

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your script files
COPY dataset_inference.py /app/
COPY main.py /app/

# Optional: copy requirements inline or via file
# Here, we assume torch and sklearn are required
RUN pip install --no-cache-dir \
    torch torchvision \
    numpy \
    scikit-learn \
    joblib

# Set default command (can override with `docker run`)
# Remove ENTRYPOINT to allow pipeline to control execution
