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

