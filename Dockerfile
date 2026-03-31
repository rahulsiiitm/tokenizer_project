# Use a Python base image with build tools
FROM python:3.11-slim

# Install C++ compiler
RUN apt-get update && apt-get install -y g++ build-essential

WORKDIR /app

# Copy and compile the C++ tokenizer engine
COPY ./engine/fast_vocab.cpp .
RUN g++ -O3 -shared -fPIC -o tokenizer_engine.so fast_vocab.cpp

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app
COPY . .

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]