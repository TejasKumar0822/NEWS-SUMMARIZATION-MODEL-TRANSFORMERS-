# ================= MULTILINGUAL HYBRID NEWS SUMMARIZER =================

import re
import requests
import numpy as np
import networkx as nx
import torch
import spacy
import nltk

from bs4 import BeautifulSoup
from newspaper import Article
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langdetect import detect

# ================= SETUP =================

torch.set_grad_enabled(False)
nltk.download("punkt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# ================= LOAD MODELS =================

print("Loading models...")

bert = SentenceTransformer("all-MiniLM-L6-v2", device=device)

pegasus_tok = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

pegasus = PegasusForConditionalGeneration.from_pretrained(
    "google/pegasus-cnn_dailymail"
).to(device)

# Indic → English
indic_en_tok = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-indic-en-dist-200M",
    trust_remote_code=True
)

indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-indic-en-dist-200M",
    trust_remote_code=True
).to(device)

# English → Indic
en_indic_tok = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True
)

en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True
).to(device)

nlp = spacy.load("en_core_web_sm")

print("Models Loaded")

# ================= LANGUAGE MAP =================

LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "bn": "ben_Beng"
}

# ================= LANGUAGE DETECTION =================

def detect_language(text):

    sample = text[:500]

    try:
        lang = detect(sample)
    except:
        lang = "en"

    if re.search(r"[ऀ-ॿ]", sample):
        return "hi"

    if re.search(r"[ఀ-౿]", sample):
        return "te"

    if re.search(r"[஀-௿]", sample):
        return "ta"

    if re.search(r"[ಀ-೿]", sample):
        return "kn"

    if re.search(r"[ঀ-৿]", sample):
        return "bn"

    return lang if lang in LANG_MAP else "en"

# ================= TRANSLATION =================

def translate(text, src_lang, tgt_lang):

    if src_lang == tgt_lang:
        return text

    src = LANG_MAP.get(src_lang, "eng_Latn")
    tgt = LANG_MAP.get(tgt_lang, "eng_Latn")

    sentences = re.split(r'[.!?।]+', text)[:40]

    outputs = []

    for s in sentences:

        s = s.strip()

        if len(s.split()) < 3:
            continue

        try:

            formatted = f"{src} {tgt} {s}"

            if tgt_lang == "en":

                inputs = indic_en_tok(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(device)

                with torch.no_grad():

                    generated = indic_en_model.generate(
                        **inputs,
                        num_beams=4,
                        max_length=128,
                        repetition_penalty=1.2
                    )

                translated = indic_en_tok.decode(
                    generated[0],
                    skip_special_tokens=True
                )

            else:

                inputs = en_indic_tok(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(device)

                with torch.no_grad():

                    generated = en_indic_model.generate(
                        **inputs,
                        num_beams=4,
                        max_length=128,
                        repetition_penalty=1.2
                    )

                translated = en_indic_tok.decode(
                    generated[0],
                    skip_special_tokens=True
                )

            translated = translated.replace("▁", " ").strip()

            outputs.append(translated)

        except:
            outputs.append(s)

    result = " ".join(outputs)

    result = re.sub(r"\s+", " ", result).strip()

    return result

# ================= CLEANING =================

BAD_PATTERNS = [
    "newsletter","sign up","advertisement",
    "sponsored","all rights reserved",
    "cookie policy","follow us",
    "instagram","youtube","twitter","facebook",
    "image source","image caption",
    "getty images","bbc news","read more",
    "पढ़ना जारी रखें","और पढ़ें","यह भी पढ़ें"
]

def clean_text(text):

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r'[.!?।]', text)

    cleaned = []

    for s in sentences:

        s = s.strip()

        if len(s.split()) < 8 or len(s.split()) > 45:
            continue

        if any(p in s.lower() for p in BAD_PATTERNS):
            continue

        cleaned.append(s)

    return ". ".join(cleaned)

# ================= SCRAPER =================

HEADERS = {"User-Agent": "Mozilla/5.0"}

def scrape(url):

    try:

        article = Article(url)

        article.download()

        article.parse()

        txt = clean_text(article.text)

        if len(txt.split()) > 200:

            return txt

    except:

        pass

    try:

        r = requests.get(url, headers=HEADERS, timeout=10)

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script","style","nav","footer"]):

            tag.decompose()

        txt = " ".join(p.get_text() for p in soup.find_all("p"))

        return clean_text(txt)

    except:

        return ""

# ================= EXTRACTIVE =================

def tfidf_scores(sents):

    vec = TfidfVectorizer(stop_words="english")

    tfidf = vec.fit_transform(sents)

    scores = tfidf.sum(axis=1).A1

    return scores / scores.max()

def textrank_scores(sents):

    tf = TfidfVectorizer(stop_words="english").fit_transform(sents)

    sim = cosine_similarity(tf)

    np.fill_diagonal(sim, 0)

    graph = nx.from_numpy_array(sim)

    scores = np.array(list(nx.pagerank(graph).values()))

    return scores / scores.max()

def bert_scores(sents):

    emb = bert.encode(sents, convert_to_numpy=True)

    centroid = emb.mean(axis=0)

    sims = cosine_similarity(emb, [centroid]).flatten()

    return sims / sims.max()

def extractive_core(article):

    sentences = sent_tokenize(article)[:80]

    sentences = [s for s in sentences if 8 <= len(s.split()) <= 45]

    salience = (
        0.20 * tfidf_scores(sentences)
        + 0.30 * textrank_scores(sentences)
        + 0.50 * bert_scores(sentences)
    )

    emb = bert.encode(sentences, convert_to_numpy=True)

    target = int(len(article.split()) * 0.25)

    selected = []
    words = 0

    first = int(np.argmax(salience))
    selected.append(first)

    words += len(sentences[first].split())

    while words < target:

        remaining = list(set(range(len(sentences))) - set(selected))

        best_score = -999
        best_idx = None

        for i in remaining:

            sim = max(
                cosine_similarity([emb[i]], emb[selected]).flatten()
            )

            mmr = 0.6 * salience[i] - 0.4 * sim

            if mmr > best_score:

                best_score = mmr
                best_idx = i

        selected.append(best_idx)

        words += len(sentences[best_idx].split())

    selected = sorted(selected)

    return " ".join([sentences[i] for i in selected])
# ================= NOTES BUILDER =================

def build_notes(text):

    sentences = sent_tokenize(text)

    notes = []

    for s in sentences:

        s = s.strip()

        if len(s.split()) < 6:
            continue

        notes.append("• " + s)

    return "\n".join(notes[:12])


# ================= PEGASUS =================

def generate_candidates(context, article):

    article_words = len(article.split())

    min_len = max(35, int(article_words * 0.10))
    max_len = min(150, int(article_words * 0.15))

    inputs = pegasus_tok(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = pegasus.generate(
        inputs["input_ids"],
        num_beams=8,
        num_return_sequences=6,
        min_length=min_len,
        max_length=max_len,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=1.1,
        early_stopping=True
    )

    summaries = []
    for o in outputs:
        decoded = pegasus_tok.decode(o, skip_special_tokens=True)
        # ================= NEW CLEANING BLOCK =================
        # Pegasus often outputs <n> for newlines. We replace it with a space.
        cleaned = decoded.replace("<n>", " ").strip()
        # Clean up extra whitespace that might result from the replacement
        cleaned = re.sub(r"\s+", " ", cleaned)
        # ======================================================
        summaries.append(cleaned)

    return list(set(summaries))
# ================= FACTUALITY HELPERS =================

def get_entities(text):

    return set(e.text.lower() for e in nlp(text).ents)


def entity_coverage(summary, article):

    art = get_entities(article)
    summ = get_entities(summary)

    if not art:
        return 0

    return len(summ & art) / len(art)


def hallucination_penalty(summary, article):

    art = get_entities(article)
    summ = get_entities(summary)

    extra = summ - art

    return len(extra) / (len(summ) + 1e-5)

# ================= RERANK =================

def rerank(candidates, article):

    if not candidates:
        return ""

    art_emb = bert.encode([article], convert_to_numpy=True)[0]

    best_score = -999
    best = None

    for c in candidates:

        sents = sent_tokenize(c)

        if not sents:
            continue

        emb = bert.encode(sents, convert_to_numpy=True)

        sim = np.max(
            cosine_similarity(emb, [art_emb]).flatten()
        )

        redundancy = 0

        if len(emb) > 1:

            sim_matrix = cosine_similarity(emb)

            redundancy = np.mean(
                sim_matrix[np.triu_indices(len(emb), k=1)]
            )

        ent_cov = entity_coverage(c, article)

        hall = hallucination_penalty(c, article)

        score = (
            0.50 * sim +
            0.25 * ent_cov -
            0.30 * redundancy -
            0.40 * hall
        )

        if score > best_score:

            best_score = score
            best = c

    return best

# ================= MAIN PIPELINE =================

def summarize(input_text):

    if input_text.startswith("http"):

        article = scrape(input_text)

    else:

        article = clean_text(input_text)

    if len(article.split()) < 120:

        raise ValueError("Extraction failed")

    lang = detect_language(article)

    article_en = article

    if lang != "en":

        article_en = translate(article, lang, "en")

    # ---------------- EXTRACTIVE ----------------

    extractive_en = extractive_core(article_en)

    # ---------------- PEGASUS INPUT ----------------

    # first part of article (context)
    article_part = " ".join(article_en.split()[:120])

    # structured notes from extractive summary
    notes = build_notes(extractive_en)

    context = "Key points:\n" + notes + "\n\nContext:\n" + article_part

    # safety fallback
    if len(context.split()) < 40:
        context = article_en[:900]

    # ---------------- GENERATE ----------------

    candidates = generate_candidates(context, article_en)

    abstractive_en = rerank(candidates, article_en)

    # ---------------- BACK TRANSLATION ----------------

    if lang != "en":

        extractive_final = translate(extractive_en, "en", lang)

        abstractive_final = translate(abstractive_en, "en", lang)

    else:

        extractive_final = extractive_en

        abstractive_final = abstractive_en

    return {

        "article": article,
        "extractive": extractive_final,
        "summary": abstractive_final,
        "detected_language": lang,

        # DEBUG
        "extractive_original": extractive_en,
        "extractive_english": extractive_en,
        "pegasus_input": context,
        "pegasus_output_en": abstractive_en,
        "article_en_preview": article_en[:500]
    }