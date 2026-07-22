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
  
  // New state to toggle raw JSON data visibility
  const [showRaw, setShowRaw] = useState(false); 

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
    setShowRaw(false); // Reset toggle on new prediction
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
          <div className="result" style={{ marginTop: '2rem', textAlign: 'center' }}>
            <h2 style={{ marginBottom: '10px' }}>🎉 Prediction Complete!</h2>
            <p className="muted" style={{ marginBottom: '20px' }}>Based on its features, this anime belongs to:</p>
            
            <div style={{ backgroundColor: 'rgba(99, 102, 241, 0.1)', padding: '30px', borderRadius: '12px', marginBottom: '20px', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
              <span className="badge" style={{ marginBottom: '15px', display: 'inline-block' }}>Assigned Group</span>
              <div className="cluster-num" style={{ fontSize: '4rem', margin: '0', lineHeight: '1' }}>#{result.cluster}</div>
            </div>

            <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '8px', textAlign: 'left', marginBottom: '20px' }}>
               <h3 style={{ fontSize: '1.2rem', marginBottom: '10px', color: '#e2e8f0' }}>What kind of anime are in this group?</h3>
               
               <p style={{ color: '#94a3b8', lineHeight: '1.6', fontSize: '0.95rem', marginBottom: '15px' }}>
                 {result.cluster === 0 && "This cluster groups highly popular Action/Shounen titles with massive member counts. You'll find epic battles and long-running storylines here."}
                 {result.cluster === 1 && "Anime in this cluster are often Slice-of-Life, Romance, or School-themed. These are character-driven, emotional, or comedic series."}
                 {result.cluster === 2 && "This group usually contains deep Sci-Fi, Psychological, or mature Drama shows. Expect complex plots and serious themes."}
                 {result.cluster === 3 && "This cluster tends to feature Fantasy, Magic, or expansive Adventure themes set in different worlds."}
                 {result.cluster === 4 && "Often includes niche, short-form, or classic older titles with highly dedicated fanbases."}
                 {result.cluster > 4 && "This anime falls into a unique or highly specific cluster based on the features provided."}
               </p>

               <h4 style={{ color: '#c7d2fe', marginBottom: '5px', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Examples of Similar Anime:</h4>
               <p style={{ color: '#fff', fontWeight: 'bold', fontSize: '1rem' }}>
                 {result.cluster === 0 && "Naruto, Bleach, One Piece, Dragon Ball Z"}
                 {result.cluster === 1 && "Toradora!, Clannad, K-On!, Your Lie in April"}
                 {result.cluster === 2 && "Steins;Gate, Death Note, Psycho-Pass, Monster"}
                 {result.cluster === 3 && "Fullmetal Alchemist, Hunter x Hunter, Sword Art Online"}
                 {result.cluster === 4 && "Mushishi, Cowboy Bebop, Neon Genesis Evangelion"}
                 {result.cluster > 4 && "Data specific titles..."}
               </p>
            </div>

            {/* View Raw Data Button */}
            <button 
              type="button" 
              onClick={() => setShowRaw(!showRaw)}
              style={{ background: 'transparent', border: '1px solid #4b5563', padding: '8px 16px', borderRadius: '6px', color: '#9ca3af', cursor: 'pointer', fontSize: '0.85rem', width: 'auto' }}
            >
              {showRaw ? "Hide Raw Data" : "View Raw Data (For Developers)"}
            </button>

            {/* Hidden Raw JSON Box */}
            {showRaw && (
              <div style={{ marginTop: '15px', textAlign: 'left', background: '#0f172a', padding: '15px', borderRadius: '8px', overflowX: 'auto', border: '1px solid #1e293b' }}>
                <pre style={{ margin: 0, fontSize: '0.8rem', color: '#a5b4fc' }}>{JSON.stringify(result, null, 2)}</pre>
              </div>
            )}
            
          </div>
        )}
      </div>

      <p className="footer"></p>
    </div>
  );
}
