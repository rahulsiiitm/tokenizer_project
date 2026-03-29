# Explainable Tokenizer & Edge-Optimized NLP Pipeline

**Submission by:** Rahul Sharma (B.Tech CSE, IIIT Manipur)\
**Deadline:** March 28, 2026

------------------------------------------------------------------------

## Project Overview

This project implements a comprehensive sub-word tokenization suite,
featuring:

-   **Custom BPE & Byte-Level BPE** trained on English corpora (8K -
    128K vocab sizes).
-   **Hybrid Code-Mixed Router**: An adaptive mitigation strategy for
    Indian multilingual scenarios.
-   **C++ Optimized Inference**: A low-latency lookup engine using
    compiled hash maps for edge devices.
-   **Explainability Dashboard**: A Next.js visualization tool for token
    saliency and boundary analysis.

------------------------------------------------------------------------

## Installation & Setup

### 1. Backend (FastAPI + C++ Extension)

The backend handles the tokenization logic and saliency calculations.

``` bash
# Install Python dependencies
pip install -r requirements.txt

# Compile the C++ Hash Map Extension (Windows)
g++ -O3 -shared -o fast_vocab.dll fast_vocab.cpp -fPIC

# Start the API server
python api.py
```

### 2. Frontend (Next.js Dashboard)

The dashboard provides a visual interface for the explainable tokenizer.

``` bash
cd dashboard
npm install
npm run dev
```

Access the dashboard at: http://localhost:3000

------------------------------------------------------------------------

## Deliverables & Folder Structure

-   **/notebooks**: Contains 01--05 covering training, robustness under
    noise, bias audits, and edge benchmarking.
-   **/reports**: Performance plots, bias audit results, and the
    edge-device comparative table.
-   **/vocabs**: Trained vocabulary JSON files (8K, 16K, 32K, 64K,
    128K).
-   **/src**: Core logic for BPE and BBPE implementations.
-   **api.py**: FastAPI wrapper for the inference pipeline.
-   **fast_vocab.cpp**: C++ source for optimized dictionary lookups.

------------------------------------------------------------------------

## Fairness & Robustness Analysis

As required in Part 2 and Part 3, this project includes:

-   **Noise Robustness**: Evaluation of token drift under OCR
    distortions and transliteration errors.
-   **Bias Mitigation**: The hybrid code-mixed model reduces
    fragmentation in Hinglish text by routing unknown scripts to a
    byte-level fallback.
-   **Edge Simulation**: Benchmarking performed using CPU-affinity
    pinning to simulate single-core mobile environments.

------------------------------------------------------------------------

## Author

Rahul Sharma\
B.Tech Computer Science & Engineering\
IIIT Manipur
