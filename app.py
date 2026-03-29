# app.py
import streamlit as st
import sys
import os
import pandas as pd
import time
import psutil

# Add src/ to path so we can import our custom modules
sys.path.append(os.path.abspath('src'))

from bpe import BPETokenizer
from bbpe import BBPETokenizer
from optimized_bpe import EdgeBPETokenizer
from fairness import FairnessAuditor

st.set_page_config(page_title="Tokenizer AI Dashboard", layout="wide")
st.title("🔬 Tokenizer Explainability & Edge Performance")

@st.cache_resource
def initialize_system():
    # 1. Load Standard 32K BPE
    bpe = BPETokenizer()
    bpe.load('vocabs/bpe_32000.json')
    
    # 2. Initialize Edge Optimizer (Part 5)
    edge_bpe = EdgeBPETokenizer()
    edge_bpe.load('vocabs/bpe_32000.json')
    edge_bpe.prepare_for_edge()
    
    # 3. Load Byte-Level BPE (Part 2)
    bbpe = BBPETokenizer()
    # Loading a small slice for the demo
    with open('data/raw/wikitext_sample.txt', 'r', encoding='utf-8') as f:
        bbpe.train(f.read()[:50000], 8000)
    
    # 4. Train Sentiment Model for Saliency (Part 6)
    auditor = FairnessAuditor(bpe)
    train_texts = ["I love this", "Amazing!", "This is bad", "Terrible quality", "Great work", "I hate it"]
    train_labels = [1, 1, 0, 0, 1, 0]
    auditor.train_sentiment_model(train_texts, train_labels)
    
    # FIX: Changed edge_model to edge_bpe to match the initialization above
    return bpe, bbpe, edge_bpe, auditor

# Unpack the returned models
bpe_model, bbpe_model, edge_model, auditor = initialize_system()

# --- SIDEBAR: MOBILE BENCHMARK (PART 5) ---
st.sidebar.header("📱 Simulated Mobile Environment")
if st.sidebar.button("Run Latency Benchmark"):
    # Simulate mobile by restricting process to a single CPU core
    p = psutil.Process(os.getpid())
    try:
        p.cpu_affinity([0])
        st.sidebar.caption("CPU Affinity restricted to Core 0 (Mobile Simulation)")
    except Exception:
        st.sidebar.warning("CPU Affinity modification not supported on this OS.")

    test_text = "The artificial intelligence algorithm processed the data." * 50
    
    # Benchmark Standard
    t0 = time.time()
    bpe_model.encode(test_text)
    t_std = time.time() - t0
    
    # Benchmark Optimized
    t0 = time.time()
    edge_model.encode(test_text)
    t_edge = time.time() - t0
    
    st.sidebar.metric("Latency Speedup", f"{t_std/t_edge:.2f}x")
    st.sidebar.write(f"Standard BPE: {t_std:.4f}s")
    # This refers to the hash-map optimized version required by the assignment [cite: 31, 32]
    st.sidebar.write(f"Edge-Optimized: {t_edge:.4f}s")

# --- MAIN UI: EXPLAINABILITY (PART 6) ---
st.subheader("Interactive Analysis")
user_input = st.text_area("Input Text for Analysis:", "The training was amazing but the result was terrible.")

if st.button("Tokenize & Explain", type="primary"):
    res = bpe_model.encode(user_input)
    tokens = res['tokens']
    # Saliency analysis maps token contributions back to words [cite: 38]
    saliency = auditor.get_token_saliency(user_input)

    st.subheader("Token Saliency Visualization")
    st.caption("Green = Positive | Red = Negative | Grey = Neutral")
    
    html_out = "<div style='line-height: 2.5;'>"
    for i, token in enumerate(tokens):
        score = saliency[i] if i < len(saliency) else 0
        alpha = min(abs(score) * 2, 1.0)
        color = f"rgba(0, 150, 0, {alpha})" if score > 0 else f"rgba(200, 0, 0, {alpha})"
        if score == 0: color = "#eeeeee"
        
        display_token = token.replace('</w>', ' ↵')
        html_out += f"<span style='background-color: {color}; color: black; padding: 5px 8px; border-radius: 5px; margin: 3px; border: 1px solid #999; font-family: monospace;'>{display_token}</span>"
    html_out += "</div>"
    st.markdown(html_out, unsafe_allow_html=True)

    st.divider()
    # Visualization that aligns input characters with tokens [cite: 37]
    st.subheader("Character Alignment Table")
    st.table(pd.DataFrame({
        "Token Sequence #": range(1, len(tokens) + 1),
        "Sub-word Token": tokens, 
        "Saliency Score": [round(s, 4) for s in saliency]
    }))