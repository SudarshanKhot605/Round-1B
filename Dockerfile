# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for non-interactive commands
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by some Python packages (e.g., PyMuPDF, pyenchant)
# '--no-install-recommends' helps keep the image size smaller
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libenchant-2-dev \
    # Clean up apt caches to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
# This step is done early to leverage Docker's build cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
# Install 'torch' specifically with CPU support to keep the image size smaller
# '--no-cache-dir' prevents pip from storing download caches in the image layer
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Create the 'models' directory where the pre-trained models will be stored
RUN mkdir -p models

# Download and save the 'all-mpnet-base-v2' SentenceTransformer model
# This ensures the model is available locally in 'models/all-mpnet-base-v2' as required by your script
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-mpnet-base-v2'); model.save_pretrained('models/all-mpnet-base-v2')"

# Download and save the 'distilbart-cnn-6-6' Hugging Face Transformers model and its tokenizer
# This ensures both are available locally in 'models/distilbart-cnn-6-6' as required by your script
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    tokenizer = AutoTokenizer.from_pretrained('distilbart-cnn-6-6'); \
    model = AutoModelForSeq2SeqLM.from_pretrained('distilbart-cnn-6-6'); \
    tokenizer.save_pretrained('models/distilbart-cnn-6-6'); \
    model.save_pretrained('models/distilbart-cnn-6-6')"

# Copy the rest of your application code into the container
# This includes your 'main.py' script, 'text_extraction.py', 'structure_analysis.py', and the 'Challenge_1b' directory
COPY . /app

# Specify the command to run your Python script when the container starts
# Replace 'main.py' with the actual name of your primary Python script if it's different
CMD ["python", "main.py"]