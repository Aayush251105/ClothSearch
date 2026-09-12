# Imports
import math
import json
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
DELIVERABLES_DIR = BASE_DIR / "deliverables"

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# FASTAPI SETUP

app = FastAPI(title="Clothing Search Engine")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

# PART C - POSITIONAL INDEX
def build_positional_index(documents):
    """
    Build a positional index.

    Structure:

    term -> {
        "df": number of documents,
        "postings": {
            docID: {
                "tf": term frequency,
                "positions": [positions]
            }
        }
    }
    """

    positional_index = defaultdict(lambda: {
        "df": 0,
        "postings": {}
    })

    for doc_id, doc in documents.items():

        full_text = (
            doc["category"] + " " +
            doc["title"] + " " +
            doc["text"]
        )

        tokens = preprocess(full_text)

        positions = defaultdict(list)

        for position, term in enumerate(tokens):
            positions[term].append(position)

        for term, term_positions in positions.items():

            positional_index[term]["postings"][doc_id] = {
                "tf": len(term_positions),
                "positions": term_positions
            }

    # Calculate document frequency
    for term in positional_index:
        positional_index[term]["df"] = len(
            positional_index[term]["postings"]
        )

    return dict(positional_index)


# PART C - EXACT PHRASE SEARCH

def phrase_search(phrase, positional_index):
    """
    Exact phrase search.

    Example:

    Query:
        cotton shirt

    A document matches only if:

        cotton position = p
        shirt position = p + 1
    """

    query_terms = preprocess(phrase)

    if not query_terms:
        return {}

    # If only one term is supplied
    if len(query_terms) == 1:

        term = query_terms[0]

        if term not in positional_index:
            return {}

        return {
            doc_id: posting["positions"]
            for doc_id, posting
            in positional_index[term]["postings"].items()
        }

    # Find documents containing every query term
    candidate_docs = None

    for term in query_terms:

        if term not in positional_index:
            return {}

        docs = set(
            positional_index[term]["postings"].keys()
        )

        if candidate_docs is None:
            candidate_docs = docs
        else:
            candidate_docs &= docs

    results = {}

    for doc_id in candidate_docs:

        first_term_positions = (
            positional_index[query_terms[0]]
            ["postings"][doc_id]["positions"]
        )

        matching_positions = []

        for start_position in first_term_positions:

            match = True

            for offset, term in enumerate(query_terms):

                positions = (
                    positional_index[term]
                    ["postings"][doc_id]["positions"]
                )

                if start_position + offset not in positions:
                    match = False
                    break

            if match:
                matching_positions.append(start_position)

        if matching_positions:
            results[doc_id] = matching_positions

    return results


# PART C - PROXIMITY SEARCH

def proximity_search(query, k, positional_index):
    """
    Ordered proximity search for two terms.

    Example:

        cotton WITHIN/3 shirt

    means cotton must occur before shirt and
    their positional distance must be <= 3.
    """

    query_terms = preprocess(query)

    if len(query_terms) != 2:
        return {}

    first_term = query_terms[0]
    second_term = query_terms[1]

    if first_term not in positional_index:
        return {}

    if second_term not in positional_index:
        return {}

    first_docs = set(
        positional_index[first_term]["postings"].keys()
    )

    second_docs = set(
        positional_index[second_term]["postings"].keys()
    )

    candidate_docs = first_docs & second_docs

    results = {}

    for doc_id in candidate_docs:

        first_positions = (
            positional_index[first_term]
            ["postings"][doc_id]["positions"]
        )

        second_positions = (
            positional_index[second_term]
            ["postings"][doc_id]["positions"]
        )

        matches = []

        for p1 in first_positions:

            for p2 in second_positions:

                # Ordered proximity:
                # first term occurs before second term
                # and distance is at most k.
                if 0 < p2 - p1 <= k:
                    matches.append((p1, p2))

        if matches:
            results[doc_id] = matches

    return results


# DELIVERABLE EXPORTS

def write_index_deliverables(inverted_index, positional_index):
    """Write the assignment's two index structures as readable JSON files."""

    DELIVERABLES_DIR.mkdir(exist_ok=True)

    outputs = {
        "dictionary_inverted_index.json": inverted_index,
        "positional_index.json": positional_index,
    }

    for filename, index_data in outputs.items():
        output_file = DELIVERABLES_DIR / filename

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(index_data, file, indent=2, sort_keys=True)
            file.write("\n")

# LOAD CORPUS AND BUILD INDEXES

documents = parse_corpus(CORPUS_FILE)

inverted_index, document_lengths, processed_documents = (
    build_inverted_index(documents)
)

document_norms = calculate_document_norms(
    inverted_index,
    documents
)

positional_index = build_positional_index(documents)

write_index_deliverables(inverted_index, positional_index)

# API - HOME

@app.get("/")
def home():
    return {
        "message": "Clothing Search Engine API is running",
        "documents": len(documents)
    }


# API - RANKED SEARCH
@app.post("/search/ranked")
def ranked_search(request: SearchRequest):

    results = ranked_retrieval(
        request.query,
        inverted_index,
        document_norms,
        documents,
        top_k=10
    )

    return {
        "query": request.query,
        "mode": "ranked",
        "results": results
    }


# API - PHRASE SEARCH
@app.post("/search/phrase")
def phrase_search_api(request: SearchRequest):

    matches = phrase_search(
        request.query,
        positional_index
    )

    results = []

    for doc_id, positions in sorted(matches.items()):

        results.append({
            "doc_id": doc_id,
            "title": documents[doc_id]["title"],
            "category": documents[doc_id]["category"],
            "positions": positions
        })

    return {
        "query": request.query,
        "mode": "phrase",
        "results": results
    }


# API - PROXIMITY SEARCH
@app.post("/search/proximity")
def proximity_search_api(request: ProximityRequest):

    matches = proximity_search(
        request.query,
        request.k,
        positional_index
    )

    results = []

    for doc_id, positions in sorted(matches.items()):

        results.append({
            "doc_id": doc_id,
            "title": documents[doc_id]["title"],
            "category": documents[doc_id]["category"],
            "positions": positions
        })

    return {
        "query": request.query,
        "mode": "proximity",
        "k": request.k,
        "results": results
    }
