# Enterprise RAG (Roles + Levels)

## Stack
- FastAPI backend
- Qdrant (Docker) for vector DB
- Local CPU embeddings: sentence-transformers/all-MiniLM-L6-v2
- Local CPU reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Gemini 1.5 Flash for generation
- JWT auth with role + level claims

## Setup
1. Create and fill .env
```
cp .env.example .env
# Add GEMINI_API_KEY (optional) and JWT_SECRET
```
2. Start Qdrant
```
docker compose up -d
```
3. Create venv and install deps
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
4. Run API
```
uvicorn app.main:app --reload
```
5. Open http://localhost:8000/ and try:
- Login with alice/alice123 (staff, level 2), bob/bob123 (manager, level 4), carol/carol123 (admin, level 5)
- Upload .txt file (optionally set allowed roles)
- Query; results filtered by uploader_level <= your level and role allowlist if provided

## Notes
- Demo supports .txt upload only to avoid heavy dependencies.
- Qdrant payload fields: text, filename, uploader, uploader_role, uploader_level, allow_roles.
- Access rule: only same or higher level can view (user.level >= uploader_level). If `allow_roles` is set, user.role must be in it.
