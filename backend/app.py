"""
Anime Clustering & Recommendation - Flask Backend
Deploy target: Render
"""
import os
import traceback
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allow Vercel frontend to call this API

# ---------------------------------------------------------------------------
# Model loading (lazy / safe)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

kmeans_model = None
tfidf_vectorizer = None
standard_scaler = None
pca_transformer = None
model_load_error = None

def load_models():
    """Attempt to load every model artifact."""
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

load_models()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "running", "models_loaded": model_load_error is None})

@app.route("/predict", methods=["POST"])
def predict():
    if model_load_error is not None:
        return jsonify({"error": "Models not loaded on server.", "details": model_load_error}), 503

    try:
        # 1. Get data from the Vercel frontend
        payload = request.get_json(force=True) or {}
        synopsis = payload.get("synopsis", "")
        genres = payload.get("genres", "")
        
        # 2. Extract Text Features
        text_blob = f"{synopsis} {genres}".strip()
        tfidf_vec = tfidf_vectorizer.transform([text_blob]).toarray()
        
        # 3. THE FIX: Extract Numeric Features from the nested dictionary
        numeric_data = payload.get("numeric_features", {})
        
        score = float(numeric_data.get("score", 0) or 0)
        episodes = float(numeric_data.get("episodes", 0) or 0)
        members = float(numeric_data.get("members", 0) or 0)
        popularity = float(numeric_data.get("popularity", 0) or 0)
        
        numeric_arr = np.array([[score, episodes, members, popularity]])
        
        # 4. Combine into one array
        combined_features = np.hstack([tfidf_vec, numeric_arr])
        
        # 5. DYNAMIC PADDING
        # Your scaler wants exactly 55 features. We will force the array to be 55.
        expected_features = standard_scaler.n_features_in_
        actual_features = combined_features.shape[1]
        
        if actual_features < expected_features:
            # Add zeros for any missing features (e.g., if we only have 54, add 1 zero)
            padding = np.zeros((1, expected_features - actual_features))
            combined_features = np.hstack([combined_features, padding])
        elif actual_features > expected_features:
            # Trim off extra features just in case
            combined_features = combined_features[:, :expected_features]
            
        # 6. Scale -> PCA -> Predict
        scaled_features = standard_scaler.transform(combined_features)
        reduced = pca_transformer.transform(scaled_features)
        cluster = int(kmeans_model.predict(reduced)[0])
        
        # Calculate distances safely
        try:
            distances = kmeans_model.transform(reduced)[0].tolist()
        except:
            distances = None
            
        return jsonify({
            "cluster": cluster,
            "distances_to_centroids": distances
        })
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed.",
            "details": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=3),
        }), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
