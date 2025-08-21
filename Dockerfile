# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port that Gradio uses
EXPOSE 7860

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
