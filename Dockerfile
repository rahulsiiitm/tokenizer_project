FROM python:3.11-slim

# 1. Install build tools for your O(1) C++ Engine
RUN apt-get update && apt-get install -y g++ build-essential

WORKDIR /app

# 2. Copy all files
COPY . .

# 3. CRITICAL: Compile the engine to the EXACT name used in api.py
# Your code uses f'fast_vocab.so', so we name it exactly that.
RUN g++ -O2 -shared -fPIC -o fast_vocab.so fast_vocab.cpp

# 4. Install dependencies (Ensure scipy==1.12.0 is in requirements.txt)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Force bind to Render's default port
# We use 'api:app' assuming your file is named api.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]