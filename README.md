# J.A.R.V.I.S — Enterprise RAG + Smart Access + AI Tools

Black/red themed enterprise assistant with role/level-aware RAG, Smart Access behavioral security, Performance Meter, Auto Team Assembler, and Advanced AI Interviewer.

## Architecture
- FastAPI backend `app/main.py`
- React 19 + Vite frontend `J.A.R.V.I.S/` (auto-started by backend)
- Qdrant vector DB (Docker) for documents and Smart Access behavioral embeddings
- JWT auth (roles: staff/manager/admin; numeric levels)
- Gemini for generation and summaries (optional OpenRouter for Auto Team)

## Major features
- RAG Chat and Uploads
  - Upload `.txt` files → embedded to Qdrant with audience controls (roles/levels)
  - Query endpoint uses filtered vector search + local reranker + Gemini answer
- Screening & AI Interviewer (classic)
  - Pages under React: list jobs, candidates, create job, applicant apply page
  - Gemini evaluation parsing for score/fit/summary
- Smart Access
  - Endpoints in `app/smart_access/routes.py` for collect/check/admin dashboards
  - Auto-collection from Main page: starts after login, posts behavior every 5–15s
  - Admin dashboard, user details, and settings (themed)
- Performance Meter (Windows edge-device)
  - Mounted Flask app at `/perf` with JR-themed UI `/perf/ui`
  - Start/Flush/Stop, stores sessions to `perf.db` by employee_id, view Trends
  - Local JSONL remains for compatibility (`app/Performance_Meter/data`)
- Auto Team Assembler
  - JR-themed UI at `/autoteam` to chat + include `employees.json` context
  - Parses Markdown table of EID | Name | Email; Download or Email actions
  - Backend: `/autoteam/chat` uses OpenRouter (if configured) or Gemini fallback
- Advanced AI Interviewer (Whisper + Gemini)
  - JR-themed UI at `/interviewer-advanced`
  - Paste/upload transcript → Gemini summary
  - Windows live recording: Start/Stop capture into 5s chunks, continuous transcription with Faster‑Whisper, then Gemini summary

## URLs (local)
- Root redirect: http://localhost:8000/ → React app at 5173
- RAG demo (embedded in backend page used by iframe): http://localhost:8000/demo
- Smart Access admin: http://localhost:8000/smart-access/admin
- Performance Meter UI: http://localhost:8000/perf/ui
- Auto Team Assembler: http://localhost:8000/autoteam
- Advanced AI Interviewer: http://localhost:8000/interviewer-advanced
- Screening Admin: http://localhost:8000/screening/jobs (JWT via token query)

## Data stores
- Qdrant (Docker) at http://localhost:6333
  - RAG collection for documents
  - Smart Access collection for behavior embeddings
- SQLite:
  - `screening.db` for screening/jobs/candidates
  - `perf.db` for performance sessions (by employee_id)
  - `app/accounts` models (optional) for users/projects/memberships

## Environment variables
Create `.env` in repo root (and optional ones in feature folders):
- Required
  - `JWT_SECRET` — any strong secret
- Optional (recommended)
  - `GEMINI_API_KEY` — for generations and summaries
  - `MODEL_NAME` — defaults to `gemini-2.5-flash`
  - `OPENROUTER_API_KEY` — to enable OpenRouter in Auto Team
  - `OPENROUTER_MODEL` — e.g. `qwen/qwen3-coder:free`
  - `PROJECTS` — comma list used when accounts DB is not active
  - `QDRANT_URL` — defaults: http://localhost:6333

Performance Meter specific (optional, in its folder):
- `GEMINI_API_KEY`, `GEMINI_MODEL` for stress analysis

Whisper project specific (optional, in its folder):
- `GOOGLE_API_KEY` for Gemini summaries if running standalone scripts

## Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (for Qdrant)
- Windows only (for live audio capture in Advanced Interviewer & Perf Meter hooks)
  - FFmpeg on PATH
  - Device names: list via
    - DirectShow: `ffmpeg -hide_banner -list_devices true -f dshow -i dummy`
    - WASAPI: `ffmpeg -hide_banner -list_devices true -f wasapi -i dummy`

## Setup (Windows and Linux)
1) Clone and env
```
git clone <repo>
cd RAG
cp .env.example .env   # or create .env and fill JWT_SECRET, GEMINI_API_KEY, etc.
```

2) Start Qdrant
```
docker compose up -d
```

3) Python env
```
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/mac
source .venv/bin/activate
pip install -r requirements.txt
```

4) Start backend (auto-starts React dev server)
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5) Open app
- http://localhost:8000 → React app (5173) with Login
- Demo users: `alice/alice123` (staff), `bob/bob123` (manager), `carol/carol123` (admin)

## How features are wired
- React SPA (J.A.R.V.I.S)
  - Routes in `J.A.R.V.I.S/src/App.jsx`
  - Main page `pages/Main.jsx` embeds `/demo` and triggers Smart Access auto-collection post-login
  - Links added for Smart Access, Performance Meter, Auto Team, Advanced AI Interviewer
- RAG
  - `/documents` — upload `.txt`, tagged with uploader role/level/dept/project
  - `/query` — vector search (Qdrant) → rerank → Gemini answer
- Screening
  - `app/screening/routes.py` HTML admin + JSON APIs for React Interviewer pages
  - Improved Gemini eval parsing for score/fit/summary
- Smart Access
  - `app/smart_access/routes.py` — collect/check/admin; Qdrant embeddings + anomaly
  - Auto-collector in backend `/demo` and React `Main.jsx`
- Performance Meter
  - Mounted Flask app via WSGI at `/perf`
  - UI `/perf/ui` themed to JR; Start/Flush/Stop → persists to `perf.db` by employee
  - Trends endpoint `/perf/trends?employee_id=<id>` shows per-day metrics
- Auto Team Assembler
  - UI `/autoteam` (themed). Backend `/autoteam/chat` uses OpenRouter (if configured) or Gemini
  - Parses Markdown table, provides Download + Gmail compose for each employee
- Advanced AI Interviewer
  - UI `/interviewer-advanced` (themed)
  - Paste/upload transcript → `/interviewer-advanced/summary` calls Whisper helper `summarize_with_gemini` or Gemini fallback
  - Live record (Windows):
    - Start: `/interviewer-advanced/record/start` builds FFmpeg cmd (wasapi/dshow), spawns Faster‑Whisper loop
    - Stop: `/interviewer-advanced/record/stop` terminates, summarizes to `output/summary.txt` and returns text
    - Status: `/interviewer-advanced/record/status`

## Typical workflows
- RAG
  - Login → Upload `.txt` under Upload Documents → Ask questions in Chat
- Smart Access
  - Login → use app → events collected automatically
  - Admin: open `/smart-access/admin` for dashboards and flagged activity
- Performance Meter
  - Open `/perf/ui` → enter Employee → Start → work → Stop → see Output + Trends
- Auto Team
  - Open `/autoteam` → write a prompt (toggle include employees.json) → Send → Download table or Email candidates
- Advanced Interviewer
  - For files: paste/upload transcript → Summarize
  - For live: on Windows, set mic/speaker names → Start Recording → Stop & Summarize

## Troubleshooting
- Qdrant not reachable: ensure Docker is running and port 6333 is free
- Gemini 401/404: verify GEMINI_API_KEY and region/model access
- Live record disabled: only supported on Windows with FFmpeg; device names must be exact
- React not opening: backend auto-starts Vite dev server; if blocked, run `npm install && npm run dev` in `J.A.R.V.I.S`

## Security notes
- Do not commit real API keys
- JWT secret should be strong in production
- Consider gating `/perf` and `/autoteam` with JWT/roles if exposing outside localhost
