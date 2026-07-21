# Anime Clustering & Recommendation — Full Stack

Dual-folder monorepo:

```
.
├── backend/    # Flask API → deploy to Render
└── frontend/   # React + Vite UI → deploy to Vercel
```

## Quick start
```bash
# 1) Backend
cd backend
pip install -r requirements.txt
python app.py                  # http://localhost:5000

# 2) Frontend (new terminal)
cd frontend
cp .env.example .env           # VITE_API_URL=http://localhost:5000
npm install
npm run dev                    # http://localhost:5173
```

## Deploy
- **Backend → Render**: New Web Service, root dir `backend`, start command
  `gunicorn app:app --bind 0.0.0.0:$PORT`. Drop your 4 `.pkl` files into
  `backend/saved_models/`.
- **Frontend → Vercel**: Import repo, root dir `frontend`, add env var
  `VITE_API_URL` pointing at your Render URL.

See each folder's README for details.
