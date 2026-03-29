# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import time
import ctypes # <--- ADDED FOR C++ INTEGRATION

sys.path.append(os.path.abspath('src'))
from bpe import BPETokenizer
from bbpe import BBPETokenizer
from fairness import FairnessAuditor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- C++ EDGE OPTIMIZATION LOAD ---
lib_ext = '.dll' if os.name == 'nt' else '.so'
lib_path = os.path.join(os.path.dirname(__file__), f'fast_vocab{lib_ext}')

try:
    fast_vocab = ctypes.CDLL(lib_path)
    fast_vocab.add_word.argtypes = [ctypes.c_char_p]
    fast_vocab.is_known.argtypes = [ctypes.c_char_p]
    fast_vocab.is_known.restype = ctypes.c_bool
    C_EXTENSION_LOADED = True
    print("✅ Successfully loaded compiled C++ Hash Map for Edge Routing!")
except OSError:
    print("⚠️ C++ extension not found. Falling back to Python Native Set.")
    C_EXTENSION_LOADED = False

# 1. Load all available models dynamically (256K removed)
models = {}
vocab_sizes = [8000, 16000, 32000, 64000, 128000] 

print("Loading standard BPE models...")
for size in vocab_sizes:
    try:
        bpe = BPETokenizer()
        bpe.load(f'vocabs/bpe_{size}.json')
        models[f'BPE-{size//1000}K'] = bpe
        print(f"Loaded BPE {size}")
    except FileNotFoundError:
        print(f"Skipped BPE {size} (File not found)")

print("Loading Byte-BPE model...")
try:
    bbpe = BBPETokenizer()
    bbpe.load('vocabs/bpe_8000.json') # Using 8K as placeholder for BBPE
    models['Byte-BPE'] = bbpe
except FileNotFoundError:
    pass

# 2. Initialize the Smart Code-Mixed Router
class CodeMixedTokenizer:
    def __init__(self, primary_bpe, fallback_bpe):
        self.primary = primary_bpe
        self.fallback = fallback_bpe
        self.use_c = C_EXTENSION_LOADED
        
        if self.use_c:
            # Feed the 32K vocabulary into the C++ memory space for O(1) lookup
            for k in self.primary.vocab.keys():
                clean_k = k.replace('</w>', '').encode('utf-8')
                fast_vocab.add_word(clean_k)
        else:
            # Fallback if the C++ file wasn't compiled
            self.known_english = set([k.replace('</w>', '') for k in self.primary.vocab.keys()])

    def encode(self, text):
        words = text.split()
        final_tokens = []
        for word in words:
            clean_word = word.lower().strip('.,!?')
            
            # --- FAST C++ HASH MAP LOOKUP ---
            if self.use_c:
                is_known = fast_vocab.is_known(clean_word.encode('utf-8'))
            else:
                is_known = clean_word in self.known_english

            if is_known:
                final_tokens.extend(self.primary.encode(word + " ")['tokens'])
            else:
                tokens = self.fallback.encode(word + " ")['tokens']
                final_tokens.extend([f"[MIX]_{t}" for t in tokens])
        
        # Calculate a 0.0 OOV rate for the router because of the byte fallback
        return {"tokens": final_tokens, "oov_rate": 0.0}

# Inject the Smart Router into your models dictionary so the API serves it
if 'BPE-32K' in models and 'Byte-BPE' in models:
    print("Loading Smart Code-Mixed Router...")
    models['Hybrid Code-Mixed'] = CodeMixedTokenizer(models['BPE-32K'], models['Byte-BPE'])

# Initialize Auditor for Saliency (Using 32K as the baseline for ML weights)
auditor = None
if 'BPE-32K' in models:
    auditor = FairnessAuditor(models['BPE-32K'])
    auditor.train_sentiment_model(["I love this", "Amazing", "Terrible", "I hate it"], [1, 1, 0, 0])

class TextRequest(BaseModel):
    text: str

@app.post("/api/tokenize_all")
async def tokenize_all(request: TextRequest):
    response_data = []
    text = request.text
    
    # Calculate baseline word saliency using 32K model
    word_saliencies = auditor.get_token_saliency(text) if auditor else []
    
    for name, model in models.items():
        # --- START BENCHMARK TIMER ---
        start_time = time.perf_counter()
        
        res = model.encode(text)
        tokens = res["tokens"]
        
        # --- END BENCHMARK TIMER ---
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        mapped_saliency = []
        for i, tok in enumerate(tokens):
            mapped_saliency.append(word_saliencies[i % len(word_saliencies)] if word_saliencies else 0.1)

        response_data.append({
            "name": name,
            "tokens": tokens,
            "saliency": mapped_saliency,
            "oov_rate": res["oov_rate"],
            "latency_ms": latency_ms
        })
        
    return response_data