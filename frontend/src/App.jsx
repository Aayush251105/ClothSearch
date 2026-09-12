import { useState } from "react";
import "./App.css";

const searchModes = {
  ranked: { label: "Ranked", description: "Find the most relevant matches" },
  phrase: { label: "Exact phrase", description: "Match words in the same order" },
  proximity: { label: "Proximity", description: "Match two terms within a distance" },
};

function SearchIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4 4" /></svg>;
}

function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("ranked");
  const [k, setK] = useState(3);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);

  const search = async () => {
    if (!query.trim() || loading) return;
    setLoading(true); setError(""); setResults([]); setHasSearched(true);
    try {
      let url = "http://localhost:8000/search/ranked";
      let body = { query };
      if (mode === "phrase") url = "http://localhost:8000/search/phrase";
      if (mode === "proximity") { url = "http://localhost:8000/search/proximity"; body = { query, k: Number(k) }; }
      const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!response.ok) throw new Error("Search request failed");
      const data = await response.json();
      setResults(data.results || []);
    } catch {
      setError("Could not connect to the backend. Make sure FastAPI is running.");
    } finally { setLoading(false); }
  };

  const formatPositions = (positions) => positions.map((position) => (Array.isArray(position) ? position.join(" – ") : position)).join(", ");

  return (
    <main className="app">
      <section className="container" aria-labelledby="page-title">
        <header className="hero">
          <p className="eyebrow">Search the collection</p>
          <h1 id="page-title">Find what fits.</h1>
          <p className="subtitle">Explore the clothing corpus using ranked, phrase, and proximity search.</p>
        </header>

        <form className="search-panel" onSubmit={(event) => { event.preventDefault(); search(); }}>
          <div className="search-field">
            <SearchIcon />
            <label className="sr-only" htmlFor="search-query">Search clothing</label>
            <input id="search-query" type="search" placeholder="Try “linen shirt” or “winter jacket”" value={query} onChange={(event) => setQuery(event.target.value)} />
          </div>
          <div className="search-controls">
            <label className="sr-only" htmlFor="search-mode">Search mode</label>
            <select id="search-mode" value={mode} onChange={(event) => setMode(event.target.value)}>
              {Object.entries(searchModes).map(([value, item]) => <option key={value} value={value}>{item.label} search</option>)}
            </select>
            {mode === "proximity" && <label className="distance-control" htmlFor="distance"><span>Within</span><input id="distance" type="number" min="1" value={k} onChange={(event) => setK(event.target.value)} /></label>}
            <button type="submit" disabled={!query.trim() || loading}>{loading ? "Searching…" : "Search"}</button>
          </div>
          <p className="mode-hint">{searchModes[mode].description}{mode === "proximity" ? "; enter exactly two terms." : "."}</p>
        </form>

        <section className="results-area" aria-live="polite">
          {loading && <p className="status">Looking through the collection…</p>}
          {error && <p className="error" role="alert">{error}</p>}
          {!loading && !error && !hasSearched && <p className="status">Start with a clothing item, material, or style.</p>}
          {!loading && !error && hasSearched && results.length === 0 && <p className="status">No matches found. Try a broader search term.</p>}
          {results.length > 0 && !loading && <div className="results">
            <div className="results-heading"><div><p className="eyebrow">Results</p><h2>{results.length} {results.length === 1 ? "match" : "matches"} for “{query}”</h2></div><span className="result-mode">{searchModes[mode].label}</span></div>
            <div className="result-list">
              {results.map((result, index) => <article className="result-card" key={result.doc_id}>
                <span className="rank">{String(index + 1).padStart(2, "0")}</span>
                <div className="result-content"><div className="result-meta"><span>{result.category}</span><span className="dot" aria-hidden="true" /><span>{result.doc_id}</span></div><h3>{result.title}</h3>{mode !== "ranked" && <p className="positions">Matches at positions {formatPositions(result.positions)}</p>}</div>
                {mode === "ranked" && <span className="score">{Number(result.score).toFixed(3)}</span>}
              </article>)}
            </div>
          </div>}
        </section>
      </section>
    </main>
  );
}

export default App;
