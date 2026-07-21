"""
Anime Clustering & Recommendation - Flask Backend
Deploy target: Render
"""
import os
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the Vercel frontend to call this API

# ---------------------------------------------------------------------------
# Model loading (lazy / safe)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

kmeans_model = None
tfidf_vectorizer = None
standard_scaler = None
pca_transformer = None
model_load_error = None


def _load_pickle(filename):
    import joblib  # joblib handles sklearn pickles most reliably
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {filename}")
    return joblib.load(path)


def load_models():
    """Attempt to load every model artifact. Store errors instead of crashing."""
    global kmeans_model, tfidf_vectorizer, standard_scaler, pca_transformer, model_load_error
    try:
        kmeans_model = _load_pickle("kmeans_model.pkl")
        tfidf_vectorizer = _load_pickle("tfidf_vectorizer.pkl")
        standard_scaler = _load_pickle("standard_scaler.pkl")
        pca_transformer = _load_pickle("pca_transformer.pkl")
        model_load_error = None
        print("[OK] All models loaded.")
    except Exception as e:  # noqa: BLE001
        model_load_error = f"{type(e).__name__}: {e}"
        print(f"[WARN] Could not load models: {model_load_error}")


load_models()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "Anime Clustering API",
            "status": "running",
            "models_loaded": model_load_error is None,
            "model_error": model_load_error,
            "endpoints": ["/health", "/predict"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "models_loaded": model_load_error is None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON payload (adjust to match your training pipeline):
    {
      "synopsis": "A ninja on a quest ...",
      "genres": "Action, Adventure, Shounen",
      "numeric_features": {
        "score": 8.4,
        "episodes": 220,
        "members": 1500000,
        "popularity": 5
      }
    }
    """
    if model_load_error is not None:
        return (
            jsonify(
                {
                    "error": "Models not loaded on server.",
                    "details": model_load_error,
                    "hint": "Drop the four .pkl files into /backend/saved_models and redeploy.",
                }
            ),
            503,
        )

    try:
        payload = request.get_json(force=True) or {}
        synopsis = payload.get("synopsis", "") or ""
        genres = payload.get("genres", "") or ""
        numeric = payload.get("numeric_features", {}) or {}

        # ---- Text features (TF-IDF) ----
        text_blob = f"{synopsis} {genres}".strip()
        tfidf_vec = tfidf_vectorizer.transform([text_blob])

        # ---- Numeric features (scaled) ----
        # Order should match whatever you trained on; adjust as needed.
        numeric_df = pd.DataFrame([numeric])
        numeric_arr = standard_scaler.transform(numeric_df.values)

        # ---- Combine + PCA ----
        combined = np.hstack([tfidf_vec.toarray(), numeric_arr])
        reduced = pca_transformer.transform(combined)

        # ---- KMeans cluster ----
        cluster = int(kmeans_model.predict(reduced)[0])

        # Distance to each centroid (useful for "similar anime" logic)
        try:
            distances = kmeans_model.transform(reduced)[0].tolist()
        except Exception:  # noqa: BLE001
            distances = None

        return jsonify(
            {
                "cluster": cluster,
                "distances_to_centroids": distances,
                "input_echo": {
                    "synopsis_chars": len(synopsis),
                    "genres": genres,
                    "numeric_features": numeric,
                },
            }
        )
    except Exception as e:  # noqa: BLE001
        return (
            jsonify(
                {
                    "error": "Prediction failed.",
                    "details": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc(limit=3),
                }
            ),
            400,
        )


@app.route("/reload-models", methods=["POST"])
def reload_models():
    load_models()
    return jsonify({"models_loaded": model_load_error is None, "error": model_load_error})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
