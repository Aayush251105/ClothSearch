# Clothing Search Engine — Run Instructions

These instructions run the FastAPI backend and React frontend locally on Windows using PowerShell.

## Prerequisites

Install the following before starting:

- Python 3.10 or newer
- Node.js 18 or newer (includes npm)

Check that both are available:

```powershell
py --version
node --version
npm --version
```

> If `py --version` reports no installed Python version, install Python from [python.org](https://www.python.org/downloads/). During installation, select **Add Python to PATH**.

## 1. Open the project folder

Open PowerShell and navigate to the project root:

```powershell
cd "C:\Users\aayus\OneDrive\Desktop\ClothSearch"
```

## 2. Set up the backend

Open a first PowerShell terminal in the project root and run:

```powershell
cd backend
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the NLTK resources required by the backend:

```powershell
python -m nltk.downloader punkt punkt_tab stopwords
```

Start the API server:

```powershell
uvicorn backend:app --reload --port 8000
```

The backend is ready when the terminal shows that Uvicorn is running at:

`http://127.0.0.1:8000`

You can confirm it in a browser at `http://127.0.0.1:8000`. It should return a JSON message saying the Clothing Search Engine API is running.

### Backend deliverables

When the backend starts, it rebuilds the indexes and writes these files automatically:

- `backend/deliverables/dictionary_inverted_index.json`
- `backend/deliverables/positional_index.json`

They contain the inverted dictionary and positional index generated from `backend/corpus_100.txt`.

## 3. Set up the frontend

Keep the backend terminal running. Open a **second** PowerShell terminal and run:

```powershell
cd "C:\Users\aayus\OneDrive\Desktop\ClothSearch\frontend"
npm install
npm run dev
```

Vite will display a local URL, normally:

`http://localhost:5173`

Open that URL in your browser.

## 4. Use the application

1. Enter a clothing-related query, such as `linen shirt`.
2. Select one of the search modes:
   - **Ranked search** — returns the most relevant results.
   - **Exact phrase search** — finds query terms in the same processed-token order.
   - **Proximity search** — enter exactly two terms and select the maximum distance.
3. Select **Search** or press Enter.

Example queries:

```text
cotton shirt
linen
regular fit
cotton kurta
```

## Stop the servers

In each terminal, press `Ctrl + C`.

## Troubleshooting

### `py` or `python` is not recognized

Install Python, close and reopen PowerShell, then repeat the backend setup. If an older virtual environment points to a removed Python installation, delete only `backend/venv` and recreate it with `py -m venv venv`.

### Backend says an NLTK resource is missing

Activate the backend virtual environment and run:

```powershell
python -m nltk.downloader punkt punkt_tab stopwords
```

### Frontend cannot connect to the backend

Confirm that the first terminal is still running `uvicorn backend:app --reload --port 8000` and that the frontend is running at `http://localhost:5173`. The backend's CORS configuration permits that local frontend address.

### Port 8000 or 5173 is already in use

Stop the process using that port, or start the server on a different port. If you change the backend port, update the API URLs in `frontend/src/App.jsx` to match it.
