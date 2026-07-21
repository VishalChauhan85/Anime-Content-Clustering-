# Saved Models

Drop your trained pickle files into this directory:

- `kmeans_model.pkl`
- `tfidf_vectorizer.pkl`
- `standard_scaler.pkl`
- `pca_transformer.pkl`

They are loaded on server startup by `app.py`. If any file is missing,
the `/predict` endpoint will return HTTP 503 with a helpful message
until you add them and redeploy (or hit `POST /reload-models`).

> These files are intentionally NOT committed to the repo when large.
> On Render, either commit them here or upload via a persistent disk.
