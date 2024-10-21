# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 9005

# Command to run the FastAPI application with increased timeouts
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "9005", "--timeout-keep-alive", "120", "--timeout-graceful-shutdown", "120"]
