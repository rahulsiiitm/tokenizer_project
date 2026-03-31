FROM python:3.11-slim

# 1. Install build tools
RUN apt-get update && apt-get install -y g++ build-essential

WORKDIR /app

# 2. Copy all files (This includes your engine/ and vocabs/ folders)
COPY . .

# 3. FIX: Point g++ to the file inside the engine folder
# We keep the output (-o) in the root so api.py can load it
RUN g++ -O2 -shared -fPIC -o fast_vocab.so engine/fast_vocab.cpp

# 4. Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Start command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]