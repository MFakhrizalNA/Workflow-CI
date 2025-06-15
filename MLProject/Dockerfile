FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Set working directory
WORKDIR /app

# Copy environment file
COPY conda.yaml /app/conda.yaml

# Install conda (via Miniconda)
RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

ENV PATH="/opt/conda/bin:$PATH"

# Create conda env
RUN conda env create -f conda.yaml

# Set environment
SHELL ["conda", "run", "-n", "titanic_env", "/bin/bash", "-c"]

# Copy project files
COPY . /app

# Entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "titanic_env", "python", "modelling.py"]