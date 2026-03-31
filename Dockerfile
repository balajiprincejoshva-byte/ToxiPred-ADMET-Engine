# ==============================================================================
# ToxiPred — Dockerfile
# ==============================================================================
# Multi-stage build for reproducible deployment.
# Uses miniconda for RDKit + pip for remaining dependencies.
#
# Build:   docker build -t toxipred .
# Run:     docker run -p 8501:8501 toxipred
# ==============================================================================

FROM continuumio/miniconda3:latest AS base

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with RDKit
RUN conda install -y -c conda-forge rdkit=2023.09 python=3.11 \
    && conda clean -afy

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/sample_inputs \
    models artifacts/plots artifacts/explanations artifacts/metrics

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run the Streamlit app
CMD ["streamlit", "run", "src/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
