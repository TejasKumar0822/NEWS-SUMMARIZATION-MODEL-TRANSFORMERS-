# NEWS-SUMMARIZATION-MODEL-TRANSFORMERS-
# 📰 Hybrid Multilingual News Summarizer

A hybrid extractive + abstractive news summarization system supporting **6 Indian and global languages**, built with Hugging Face Transformers, Sentence-BERT, and Pegasus.

---

## 🌐 Supported Languages

| Language | Code | Script |
|----------|------|--------|
| English  | `en` | Latin  |
| Hindi    | `hi` | Devanagari |
| Telugu   | `te` | Telugu |
| Tamil    | `ta` | Tamil  |
| Kannada  | `kn` | Kannada |
| Bengali  | `bn` | Bengali |

---

## 🧠 How It Works

This project uses a **3-stage hybrid pipeline**:

### Stage 1 — Extractive Summarization
Sentences are ranked using a weighted combination of three algorithms:
- **TF-IDF scoring** (20%) — term frequency relevance
- **TextRank / PageRank** (30%) — graph-based sentence importance
- **Sentence-BERT cosine similarity** (50%) — semantic similarity to the article centroid

Final sentence selection uses **MMR (Maximal Marginal Relevance)** to balance salience and diversity.

### Stage 2 — Abstractive Summarization (Pegasus)
The extractive output is used to build a structured prompt fed into **Google Pegasus (CNN/DailyMail)**, which generates 6 candidate summaries via beam search. The best candidate is selected by a reranking step that scores:
- Semantic similarity to the original article
- Entity coverage (factual completeness)
- Hallucination penalty (entities not in source)
- Redundancy penalty

### Stage 3 — Translation (Indic ↔ English)
For non-English articles, the system:
1. Detects the source language (script-based regex + `langdetect`)
2. Translates to English using **IndicTrans2 (Indic → En)**
3. Runs the full pipeline in English
4. Translates the final summary back using **IndicTrans2 (En → Indic)**

---

## 🔧 Architecture

```
Input (URL or raw text)
        │
        ▼
  Language Detection
        │
        ├── Non-English? → Translate to English (IndicTrans2)
        │
        ▼
  Article Cleaning & Scraping (newspaper3k + BeautifulSoup)
        │
        ▼
  Extractive Core (TF-IDF + TextRank + BERT + MMR)
        │
        ▼
  Pegasus Input Builder (structured notes + context)
        │
        ▼
  Abstractive Generation (Pegasus, 6-beam candidates)
        │
        ▼
  Reranking (semantic sim + entity coverage + hallucination penalty)
        │
        ├── Non-English? → Back-translate to source language (IndicTrans2)
        │
        ▼
  Final Summary Output (Streamlit UI)
```

---

## ⚠️ Large Model Files (Not Included in Repo)

> **The transformer model weights are too large to upload to GitHub and are excluded from this repository.**

The following models must be downloaded separately via Hugging Face:

| Model | Purpose | Size (approx.) |
|-------|---------|----------------|
| `all-MiniLM-L6-v2` | Sentence embeddings (BERT) | ~90 MB |
| `google/pegasus-cnn_dailymail` | Abstractive summarization | ~2.3 GB |
| `ai4bharat/indictrans2-indic-en-dist-200M` | Indic → English translation | ~800 MB |
| `ai4bharat/indictrans2-en-indic-dist-200M` | English → Indic translation | ~800 MB |

### Downloading Models

Models are automatically downloaded the first time you run the app, provided you have an internet connection and sufficient disk space (~4–5 GB total).

You can also pre-download them manually:

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer, PegasusForConditionalGeneration

SentenceTransformer("all-MiniLM-L6-v2")
PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)
AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)
AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
```

Models are cached by Hugging Face in `~/.cache/huggingface/` by default.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multilingual-news-summarizer.git
cd multilingual-news-summarizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the App

```bash
streamlit run app.py
```

> **Recommended:** Use a machine with a GPU (CUDA) for faster inference. CPU mode works but will be significantly slower, especially for Pegasus and IndicTrans2.

---

## 📦 Requirements

```
streamlit
torch
transformers
sentence-transformers
scikit-learn
networkx
numpy
spacy
nltk
langdetect
newspaper3k
beautifulsoup4
requests
```

---

## 📁 Project Structure

```
├── app.py              # Streamlit UI
├── summarizer.py       # Core pipeline (extraction, generation, translation)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 💻 Hardware Requirements

| Mode | Minimum RAM | Recommended |
|------|------------|-------------|
| CPU only | 16 GB RAM | 32 GB RAM |
| GPU (CUDA) | 6 GB VRAM | 8+ GB VRAM |

---

## 📌 Notes

- The scraper supports both **direct article text** and **news URLs** (via `newspaper3k` + `BeautifulSoup` fallback).
- Pegasus `<n>` tokens (newline artifacts) are automatically cleaned from outputs.
- Language detection uses both Unicode script matching (higher priority) and `langdetect` as fallback, making it robust for Indian scripts.

---

