FROM python:3.9-bullseye

WORKDIR /app

# Copy requirements first to leverage Docker caching. This avoids
# the pip saving the packages in its cache, making the container lighter
# and relying only in its own cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories, config is not in there as it
RUN mkdir -p ./data/raw ./data/features ./data/predictions ./models/trained ./models/hyperparameter ./logs

# Set environment variables
ENV PYTHONPATH=/app

# Command to run when the container starts
ENTRYPOINT [ "python", "main.py"]

# Default arguments
CMD ["--download", "--preprocess", "--train", "--predict"]