# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY Requirements.txt .

# Install the Python dependencies
RUN pip install -r Requirements.txt

# Copy the application code and HTML templates
COPY app.py .
COPY index.html .
COPY recommendations.html .

# Copy any static assets
COPY static/ ./static/

# Copy the data and model files
COPY data/ ./data/
COPY models/ ./models/

# Expose the port that Flask will use
EXPOSE 5000

# Set the environment variable to tell Flask which app to run
ENV FLASK_APP=app.py

# Command to run the Flask application
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]