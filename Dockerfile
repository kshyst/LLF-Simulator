# Use the official slim Python base image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workspace directory
WORKDIR /app

# Create a non-root system user and group for security
RUN groupadd -r streamlit && useradd -r -g streamlit -d /app -s /sbin/nologin streamlit

# Install minimal system dependencies required for building Python modules
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies (no-cache to keep build size small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Set ownership of files to the non-root user
RUN chown -R streamlit:streamlit /app

# Switch execution context to the non-root user
USER streamlit

# Expose the application port
EXPOSE 8501

# Run Streamlit and bind to 0.0.0.0, disabling usage statistics prompt
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
