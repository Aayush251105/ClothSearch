# Clothing Search Engine

A full-stack information retrieval project for searching a 100-document clothing corpus.

## What is included

- **FastAPI backend** with corpus parsing and CORS support.
- Text preprocessing: lowercase conversion, punctuation removal, stop-word removal, and Porter stemming.
- **Dictionary / inverted index** with document frequency and term-frequency postings.
- **Positional index** for term locations within processed documents.
- Three search methods:
  - Ranked retrieval using lnc.ltc weighting and cosine similarity
  - Exact phrase search
  - Ordered proximity search
- A minimal React interface for submitting queries, choosing a search mode, and viewing results.
- Automatically generated index deliverables:
  - `backend/deliverables/dictionary_inverted_index.json`
  - `backend/deliverables/positional_index.json`

## Project structure

```text
backend/                 FastAPI API, corpus, and index generation
frontend/                React + Vite user interface
backend/deliverables/    Exported inverted and positional indexes
instructions.md          Complete local setup and run instructions
```

## Run the project

See [instructions.md](instructions.md) for the exact backend and frontend setup commands.

Once running:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
