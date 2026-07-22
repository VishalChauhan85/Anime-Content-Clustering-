"""
Anime Clustering & Recommendation - Flask Backend
Deploy target: Render

Pipeline: TF-IDF(genre) + one-hot(type) + [rating, members_log, episodes_log]
          -> StandardScaler -> PCA -> KMeans(4)

This mirrors the exact feature engineering from the training notebook
(Section 6). The single most important idea: instead of hand-building a
fixed-size array and zero-padding it to whatever shape happens to fit,
we build a named DataFrame and reindex it to standard_scaler.feature_names_in_
- the exact column list + order the scaler was actually fit on. That
list is baked into the pickle itself, so this can never silently drift
out of sync with the training pipeline again.
"""
import os
import re
import string
import traceback
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allow Vercel frontend to call this API

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_PATH = os.path.join(BASE_DIR, "anime.csv")  # ship this file with the backend

# ---------------------------------------------------------------------------
# Globals (populated by load_models() / build_anime_database())
# ---------------------------------------------------------------------------
kmeans_model = None
tfidf_vectorizer = None
standard_scaler = None
pca_transformer = None
model_load_error = None

anime_db = None  # full anime.csv, cleaned, with a precomputed 'cluster' column
anime_db_reduced = None  # PCA-space coordinates for every row in anime_db, same order
db_build_error = None

# ---------------------------------------------------------------------------
# Genre text cleaning - mirrors the notebook's preprocessing EXACTLY.
# If this drifts from the notebook, TF-IDF vectors won't match training
# and clusters become meaningless again, so keep the two in sync.
# ---------------------------------------------------------------------------
from nltk.stem import WordNetLemmatizer
import nltk

for pkg in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_punct_table = str.maketrans("", "", string.punctuation.replace(",", ""))


def clean_genre_text(raw_genre: str) -> str:
    text = (raw_genre or "").lower()
    text = text.translate(_punct_table)          # strip punctuation, keep commas
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.replace(" ,", ",").replace(", ", ",").strip()
    tokens = [t for t in text.split(",") if t]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Feature building - one function, used for BOTH the whole-dataset batch
# pass at startup and every live /predict call, so the two paths can never
# disagree with each other.
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """df needs columns: genre, type, rating, episodes, members."""
    cleaned_genre = df["genre"].apply(clean_genre_text)
    tfidf_cols = tfidf_vectorizer.get_feature_names_out()
    tfidf_matrix = tfidf_vectorizer.transform(cleaned_genre).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=df.index)

    type_dummies = pd.get_dummies(df["type"], prefix="type")

    numeric_df = pd.DataFrame(
        {
            "rating": df["rating"].astype(float),
            "members_log": np.log1p(df["members"].astype(float)),
            "episodes_log": np.log1p(df["episodes"].astype(float)),
        },
        index=df.index,
    )

    combined = pd.concat([type_dummies, numeric_df, tfidf_df], axis=1)

    # THE fix, replacing the old blind zero-pad-to-55 hack: reindex to the
    # exact columns (and order) standard_scaler was fit on. Any column the
    # scaler expects but we didn't build (e.g. a type that isn't in this
    # batch) becomes 0. Any column we built that the scaler doesn't know
    # about is dropped. This can never go out of shape-sync again.
    combined = combined.reindex(columns=standard_scaler.feature_names_in_, fill_value=0)
    return combined


def predict_clusters(df: pd.DataFrame):
    features = build_feature_matrix(df)
    scaled = standard_scaler.transform(features)
    # Re-wrap with column names before PCA so sklearn doesn't warn about
    # missing feature names (and so any future column-order mismatch
    # surfaces as a loud error instead of a silent misprediction).
    scaled_df = pd.DataFrame(scaled, columns=features.columns, index=features.index)
    reduced = pca_transformer.transform(scaled_df)
    clusters = kmeans_model.predict(reduced)
    distances = kmeans_model.transform(reduced)
    return clusters, distances, reduced


# ---------------------------------------------------------------------------
# Model + dataset loading
# ---------------------------------------------------------------------------
def load_models():
    global kmeans_model, tfidf_vectorizer, standard_scaler, pca_transformer, model_load_error
    try:
        kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
        tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        standard_scaler = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.pkl"))
        pca_transformer = joblib.load(os.path.join(MODEL_DIR, "pca_transformer.pkl"))
        model_load_error = None
        print("[OK] All models loaded successfully.")
    except Exception as e:
        model_load_error = f"{type(e).__name__}: {e}"
        print(f"[WARN] Could not load models: {model_load_error}")


def build_anime_database():
    """Cleans anime.csv and assigns every title a cluster ONCE at startup.
    /predict then just filters this cached table -> real recommendations
    instead of nothing but a bare cluster id."""
    global anime_db, anime_db_reduced, db_build_error
    try:
        df = pd.read_csv(DATA_PATH)
        df["episodes"] = pd.to_numeric(df["episodes"].replace("Unknown", np.nan))
        df["rating"] = df["rating"].fillna(df["rating"].median())
        df["episodes"] = df["episodes"].fillna(df["episodes"].median())
        df["type"] = df["type"].fillna(df["type"].mode()[0])
        df["genre"] = df["genre"].fillna(df["genre"].mode()[0])
        df = df.drop_duplicates().reset_index(drop=True)

        clusters, _, reduced = predict_clusters(df)
        df["cluster"] = clusters

        anime_db = df
        anime_db_reduced = reduced  # row i corresponds to anime_db.iloc[i]
        db_build_error = None
        print(f"[OK] Indexed {len(df)} anime into {df['cluster'].nunique()} clusters.")
    except Exception as e:
        db_build_error = f"{type(e).__name__}: {e}"
        anime_db = None
        anime_db_reduced = None
        print(f"[WARN] Could not build anime database: {db_build_error}")


load_models()
if model_load_error is None:
    build_anime_database()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "status": "running",
            "models_loaded": model_load_error is None,
            "anime_indexed": int(len(anime_db)) if anime_db is not None else 0,
            "db_build_error": db_build_error,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    ok = model_load_error is None and anime_db is not None
    return jsonify({"status": "ok" if ok else "degraded"}), (200 if ok else 503)


@app.route("/reload-models", methods=["POST"])
def reload_models():
    load_models()
    if model_load_error is None:
        build_anime_database()
    return jsonify(
        {
            "models_loaded": model_load_error is None,
            "model_load_error": model_load_error,
            "db_build_error": db_build_error,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model_load_error is not None:
        return jsonify({"error": "Models not loaded on server.", "details": model_load_error}), 503
    if anime_db is None:
        return jsonify({"error": "Anime database not indexed yet.", "details": db_build_error}), 503

    try:
        payload = request.get_json(force=True) or {}

        genres = payload.get("genres", "")
        anime_type = payload.get("type", "TV")  # frontend should send this - see note below
        numeric_data = payload.get("numeric_features", {})
        rating = numeric_data.get("score", numeric_data.get("rating", 0))
        episodes = numeric_data.get("episodes", 0)
        members = numeric_data.get("members", 0)

        query_df = pd.DataFrame(
            [{"genre": genres, "type": anime_type, "rating": rating, "episodes": episodes, "members": members}]
        )
        clusters, distances, reduced = predict_clusters(query_df)
        cluster = int(clusters[0])
        query_point = reduced[0]

        # Real recommendations: other anime in the same cluster, ranked by
        # nearest-neighbor distance in PCA space (not just raw popularity -
        # a broad cluster can hold thousands of titles, so this keeps
        # results genuinely close to the query instead of always surfacing
        # the same "most popular in cluster" handful), excluding an exact
        # name match if given.
        name = (payload.get("name") or "").strip().lower()
        mask = anime_db["cluster"] == cluster
        if name:
            mask &= anime_db["name"].str.lower() != name

        pool = anime_db[mask].copy()
        pool_points = anime_db_reduced[mask.to_numpy()]
        pool["distance"] = np.linalg.norm(pool_points - query_point, axis=1)

        top_n = int(payload.get("top_n", 10))
        recommendations = (
            pool.sort_values(["distance", "rating"], ascending=[True, False])
            .head(top_n)[["name", "genre", "type", "rating", "members"]]
            .to_dict(orient="records")
        )

        return jsonify(
            {
                "cluster": cluster,
                "distances_to_centroids": distances[0].tolist(),
                "recommendations": recommendations,
                "cluster_size": int(mask.sum()),
            }
        )

    except Exception as e:
        return jsonify(
            {
                "error": "Prediction failed.",
                "details": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3),
            }
        ), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
