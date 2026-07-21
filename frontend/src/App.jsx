import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function App() {
  const [form, setForm] = useState({
    synopsis:
      "A young ninja seeks recognition from his peers and dreams of becoming the leader of his village.",
    genres: "Action, Adventure, Shounen",
    score: 8.4,
    episodes: 220,
    members: 1500000,
    popularity: 5,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const update = (k) => (e) =>
    setForm((f) => ({
      ...f,
      [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value,
    }));

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          synopsis: form.synopsis,
          genres: form.genres,
          numeric_features: {
            score: form.score,
            episodes: form.episodes,
            members: form.members,
            popularity: form.popularity,
          },
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.details || data.error || "Request failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="hero">
        <span className="badge">ML • KMeans + TF-IDF + PCA</span>
        <h1>Anime Cluster Predictor</h1>
        <p>Enter an anime's details and get its predicted cluster.</p>
      </header>

      <div className="card">
        <form onSubmit={submit}>
          <label>
            Synopsis
            <textarea
              value={form.synopsis}
              onChange={update("synopsis")}
              placeholder="Short description of the anime..."
            />
          </label>

          <label>
            Genres (comma-separated)
            <input value={form.genres} onChange={update("genres")} />
          </label>

          <div className="grid-2">
            <label>
              Score
              <input type="number" step="0.1" value={form.score} onChange={update("score")} />
            </label>
            <label>
              Episodes
              <input type="number" value={form.episodes} onChange={update("episodes")} />
            </label>
            <label>
              Members
              <input type="number" value={form.members} onChange={update("members")} />
            </label>
            <label>
              Popularity
              <input type="number" value={form.popularity} onChange={update("popularity")} />
            </label>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting…" : "Predict Cluster"}
          </button>
          <p className="muted">API: {API_URL}</p>
        </form>

        {error && (
          <div className="result">
            <p className="error">Error: {error}</p>
          </div>
        )}

        {result && (
          <div className="result">
            <span className="badge">Predicted cluster</span>
            <div className="cluster-num">#{result.cluster}</div>
            <p className="muted">Raw response:</p>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>

      <p className="footer">Frontend: Vercel • Backend: Render</p>
    </div>
  );
}
