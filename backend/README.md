# Backend — Anime Clustering API (Flask)

## Local run
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py   # http://localhost:5000
```

## Deploy to Render
1. Push this repo to GitHub.
2. On Render → **New Web Service** → connect the repo.
3. Root directory: `backend`
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
6. Add your `.pkl` files under `backend/saved_models/` (commit them or use a Render disk).

## Endpoints
- `GET /` – service info
- `GET /health` – health probe
- `POST /predict` – body:
  ```json
  {
    "synopsis": "A ninja on a quest...",
    "genres": "Action, Adventure",
    "numeric_features": {"score": 8.4, "episodes": 220, "members": 1500000, "popularity": 5}
  }
  ```
- `POST /reload-models` – re-read the pickles from disk.
