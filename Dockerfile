# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# install native libraries rasterio expects
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    gdal-bin \
    libgdal-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy your actual source code
COPY src /app

# Add /app to Python path (so it can find biodata/)
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "biodata.enrich"]
