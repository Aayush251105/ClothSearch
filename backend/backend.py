# Imports
import math
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# BASIC SETUP

BASE_DIR = Path(__file__).resolve().parent
CORPUS_FILE = BASE_DIR / "corpus_100.txt"

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# REQUEST MODELS

class SearchRequest(BaseModel):
    query: str


class ProximityRequest(BaseModel):
    query: str
    k: int



# PART A - CORPUS PARSING
def parse_corpus(filename):
    """
    Read the 100-document corpus and extract:
    DOCID, CATEGORY, TITLE and TEXT.
    """

    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()

    # The corpus contains multiple DOC elements,
    # so wrap everything inside a root element.
    root = ET.fromstring("<ROOT>" + content + "</ROOT>")

    documents = {}

    for doc in root.findall("DOC"):
        doc_id = doc.findtext("DOCID", "").strip()
        category = doc.findtext("CATEGORY", "").strip()
        title = doc.findtext("TITLE", "").strip()
        text = doc.findtext("TEXT", "").strip()

        documents[doc_id] = {
            "doc_id": doc_id,
            "category": category,
            "title": title,
            "text": text
        }

    return documents

# PART A - PREPROCESSING

def preprocess(text):
    """
    Preprocessing:
    1. Convert to lowercase
    2. Replace hyphens with spaces
    3. Tokenize
    4. Remove punctuation/non-alphabetic tokens
    5. Remove stop words
    6. Apply Porter stemming
    """

    text = text.lower()
    text = text.replace("-", " ")

    tokens = nltk.word_tokenize(text)

    processed_tokens = []

    for token in tokens:

        # Remove punctuation and non-alphabetic tokens
        if not token.isalpha():
            continue

        # Stop-word removal
        if token in STOP_WORDS:
            continue

        # Stemming
        stemmed_token = STEMMER.stem(token)

        processed_tokens.append(stemmed_token)

    return processed_tokens

# PART A - INVERTED INDEX

def build_inverted_index(documents):
    """
    Build dictionary/inverted index.

    Structure:

    term -> {
        "df": document frequency,
        "postings": {
            docID: term frequency
        }
    }
    """

    index = defaultdict(lambda: {
        "df": 0,
        "postings": {}
    })

    document_lengths = {}
    processed_documents = {}

    for doc_id, doc in documents.items():

        # Search across category, title and description
        full_text = (
            doc["category"] + " " +
            doc["title"] + " " +
            doc["text"]
        )

        tokens = preprocess(full_text)

        processed_documents[doc_id] = tokens
        document_lengths[doc_id] = len(tokens)

        term_frequency = Counter(tokens)

        for term, tf in term_frequency.items():

            index[term]["postings"][doc_id] = tf

    # Calculate document frequency
    for term in index:
        index[term]["df"] = len(index[term]["postings"])

    return dict(index), document_lengths, processed_documents

# PART B - LNC WEIGHT

def lnc_weight(tf):
    """
    Document weight:

    lnc:
    1 + log10(tf), if tf > 0
    """

    if tf <= 0:
        return 0

    return 1 + math.log10(tf)


# PART B - LTC WEIGHT

def ltc_weight(tf, df, N):
    """
    Query weight:

    ltc =
    (1 + log10(tf)) * log10(N / df)
    """

    if tf <= 0 or df <= 0:
        return 0

    return (1 + math.log10(tf)) * math.log10(N / df)


# PART B - DOCUMENT VECTOR NORMS

def calculate_document_norms(index, documents):
    """
    Calculate the complete lnc vector norm for every document.

    This is important because cosine similarity requires
    normalization using the complete document vector,
    not only the query terms.
    """

    document_norms = {}

    for doc_id, doc in documents.items():

        full_text = (
            doc["category"] + " " +
            doc["title"] + " " +
            doc["text"]
        )

        tokens = preprocess(full_text)
        term_frequency = Counter(tokens)

        squared_sum = 0

        for term, tf in term_frequency.items():
            weight = lnc_weight(tf)
            squared_sum += weight ** 2

        document_norms[doc_id] = math.sqrt(squared_sum)

    return document_norms


# PART B - RANKED RETRIEVAL

def ranked_retrieval(
    query,
    index,
    document_norms,
    documents,
    top_k=10
):
    """
    Ranked free-text retrieval using lnc.ltc and cosine similarity.
    """

    query_tokens = preprocess(query)

    if not query_tokens:
        return []

    query_tf = Counter(query_tokens)
    N = len(documents)

    # Calculate query weights

    query_weights = {}

    for term, tf in query_tf.items():

        if term not in index:
            continue

        df = index[term]["df"]

        weight = ltc_weight(tf, df, N)

        query_weights[term] = weight

    if not query_weights:
        return []

    # Normalize query vector

    query_norm = math.sqrt(
        sum(weight ** 2 for weight in query_weights.values())
    )

    if query_norm == 0:
        return []

    normalized_query = {
        term: weight / query_norm
        for term, weight in query_weights.items()
    }

    # Calculate cosine scores

    scores = defaultdict(float)

    for term in normalized_query:

        if term not in index:
            continue

        query_weight = normalized_query[term]

        postings = index[term]["postings"]

        for doc_id, tf in postings.items():

            doc_weight = lnc_weight(tf)

            if document_norms[doc_id] == 0:
                continue

            normalized_doc_weight = (
                doc_weight / document_norms[doc_id]
            )

            scores[doc_id] += (
                query_weight * normalized_doc_weight
            )

    # Sort:
    # 1. Higher score first
    # 2. Smaller DOCID first for ties

    ranked = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    results = []

    for doc_id, score in ranked[:top_k]:

        results.append({
            "doc_id": doc_id,
            "title": documents[doc_id]["title"],
            "category": documents[doc_id]["category"],
            "score": round(score, 6)
        })

    return results


