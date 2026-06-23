FROM python:3.13.5-slim

WORKDIR /app

# 1. Install system dependencies from your packages.txt intent
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the configuration files first to leverage Docker caching
COPY requirements.txt ./
# packages.txt is optional for local docker, but good to have if HF relies on it
COPY packages.txt ./ 

# 3. Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# 4. Copy your specific modular layout into the container
COPY app.py ./
COPY config/ ./config/
COPY src/ ./src/

# Hugging Face Spaces defaults to port 7860 internally, but if you're deploying as a 
# standard Streamlit Space, HF handles port redirection. 8501 is perfect for local testing too.
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 5. Point directly to your root app.py as the entry point
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]