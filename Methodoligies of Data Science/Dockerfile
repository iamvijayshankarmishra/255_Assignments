# Data Mining Methodologies Portfolio - Docker Image
FROM python:3.9-slim

LABEL maintainer="your-email@example.com"
LABEL description="Production environment for CRISP-DM, SEMMA, and KDD methodologies"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p crisp_dm/data/raw crisp_dm/data/processed \
    semma/data/raw semma/data/processed \
    kdd/data/raw kdd/data/processed

# Expose ports
EXPOSE 8888 8000

# Default command: Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Alternative commands:
# Run FastAPI (CRISP-DM): CMD ["uvicorn", "crisp_dm.deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
# Run FastAPI (KDD): CMD ["uvicorn", "kdd.deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
