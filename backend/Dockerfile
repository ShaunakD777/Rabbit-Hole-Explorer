FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU-only torch build first. Plain `pip install torch==2.3.1` on Linux
# resolves to the CUDA build by default, pulling ~2GB of unused nvidia-* wheels
# (cuBLAS, cuDNN, etc.) into an image that has no GPU. Once this exact version is
# satisfied, the requirements.txt install below is a no-op for torch.
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000
