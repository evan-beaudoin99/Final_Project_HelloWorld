# # Python image to use.
# FROM python:3.12-alpine

# # Set the working directory to /app
# WORKDIR /app

# # copy the requirements file used for dependencies
# COPY requirements.txt .

# # Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt


# # Copy the rest of the working directory contents into the container at /app
# COPY . .

# # Run app.py when the container launches
# ENTRYPOINT ["python", "app.py"]

# Use the official Python image from the Docker Hub.
FROM python:3.12-alpine

# Set the working directory in the container to /app.
WORKDIR /app

# Install system dependencies.
RUN apk update && apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    build-base \
    libsndfile-dev

# Copy the requirements file into the container at /app.
COPY requirements.txt .

RUN cat requirements.txt .


# Install the Python packages specified in requirements.txt.
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the working directory contents into the container at /app.
COPY . .

# Specify the command to run on container start.
ENTRYPOINT ["python", "app.py"]
