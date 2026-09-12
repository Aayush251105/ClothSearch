/* eslint-disable no-unused-vars */
import { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("ranked");
  const [k, setK] = useState(3);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const search = async () => {
    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);

    try {
      let url = "http://localhost:8000/search/ranked";

      let body = {
        query: query
      };

      if (mode === "phrase") {
        url = "http://localhost:8000/search/phrase";
      }

      if (mode === "proximity") {
        url = "http://localhost:8000/search/proximity";

        body = {
          query: query,
          k: Number(k)
        };
      }

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        throw new Error("Search request failed");
      }

      const data = await response.json();

      setResults(data.results || []);
    } catch (err) {
      setError(
        "Could not connect to the backend. Make sure FastAPI is running."
      );
    }

    setLoading(false);
  };

  return (
    <div className="app">

      <div className="container">

        <h1>Clothing Search Engine</h1>

        <p className="subtitle">
          Information Retrieval Assignment
        </p>

        <div className="search-box">

          <input
            type="text"
            placeholder="Enter clothing query..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                search();
              }
            }}
          />

          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="ranked">
              Ranked Search
            </option>

            <option value="phrase">
              Exact Phrase
            </option>

            <option value="proximity">
              Proximity Search
            </option>
          </select>

          {mode === "proximity" && (
            <input
              className="k-input"
              type="number"
              min="1"
              value={k}
              onChange={(e) => setK(e.target.value)}
              placeholder="k"
            />
          )}

          <button onClick={search}>
            Search
          </button>

        </div>

        {loading && (
          <p className="status">
            Searching...
          </p>
        )}

        {error && (
          <p className="error">
            {error}
          </p>
        )}

        {!loading && !error && results.length === 0 && (
          <p className="status">
            Enter a query to search the corpus.
          </p>
        )}

        {results.length > 0 && (
          <div className="results">

            <h2>
              Search Results
            </h2>

            {results.map((result, index) => (

              <div
                className="result-card"
                key={result.doc_id}
              >

                <div className="result-header">

                  <span className="rank">
                    #{index + 1}
                  </span>

                  <strong>
                    {result.doc_id}
                  </strong>

                  {mode === "ranked" && (
                    <span className="score">
                      Score: {result.score}
                    </span>
                  )}

                </div>

                <h3>
                  {result.title}
                </h3>

                <p>
                  <strong>Category:</strong>{" "}
                  {result.category}
                </p>

                {mode !== "ranked" && (
                  <p>
                    <strong>Matching Positions:</strong>{" "}
                    {JSON.stringify(result.positions)}
                  </p>
                )}

              </div>

            ))}

          </div>
        )}

      </div>

    </div>
  );
}

export default App;