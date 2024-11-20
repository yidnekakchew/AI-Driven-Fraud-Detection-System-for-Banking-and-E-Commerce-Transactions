# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file from the host's app/ directory to the container's /app
COPY requirements.txt .

# Install setuptools globally using pip to ensure it's available for all packages
RUN pip install --no-cache-dir setuptools

# Install the dependencies globally
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code from the host's app/ folder to the container's /app
COPY  . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]