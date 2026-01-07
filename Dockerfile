# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .

# Install system dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

ENV FLASK_RUN_HOST=0.0.0.0

# Run the application
CMD ["python", "endpoint_stuff/endpoint.py"]
