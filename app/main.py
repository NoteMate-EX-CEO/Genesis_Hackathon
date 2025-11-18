import os
import io
from typing import List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app import auth
from app.models import LoginRequest, TokenResponse, User, UploadResponse, QueryRequest, QueryResponse
from app.embedding import embed_texts
from app.reranker import rerank
from app.qdrant_client import ensure_collection, upsert_text_chunks, filtered_search
from app.utils import simple_chunk
from app.screening.db import init_db as screening_init_db
from app.screening import routes as screening
from app.smart_access import routes as smart_access

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
MODEL_NAME = "gemini-2.5-flash"

app = FastAPI(title="Enterprise RAG with Roles/Levels")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(screening.router, prefix="/screening")
app.include_router(smart_access.router)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

class MeResponse(BaseModel):
    username: str
    role: str
    level: int
    dept: str
    project: str

@app.on_event("startup")
async def startup():
    ensure_collection()
    screening_init_db()
    # prepare smart access collection
    try:
        smart_access.ensure_collection()
    except Exception:
        pass

@app.post("/auth/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth.create_access_token(
        sub=user["username"], role=user["role"], level=user["level"], dept=user.get("dept", ""), project=user.get("project", "")
    )
    return TokenResponse(access_token=token)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return User(username=payload["sub"], role=payload["role"], level=payload["level"], dept=payload.get("dept", ""), project=payload.get("project", ""))

@app.get("/me", response_model=MeResponse)
async def me(user: User = Depends(get_current_user)):
    return MeResponse(username=user.username, role=user.role, level=user.level, dept=user.dept, project=user.project)

@app.post("/documents", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    audience: str = Form(default="all"),  # one of: all, managers, employees, custom
    allow_roles_custom: str = Form(default=""),  # used when audience=custom, comma-separated roles
    user: User = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported in demo")
    raw = (await file.read()).decode("utf-8", errors="ignore")
    chunks = simple_chunk(raw)
    vecs = embed_texts(chunks)
    audience = (audience or "all").strip().lower()
    if audience not in {"all", "managers", "employees", "custom"}:
        raise HTTPException(status_code=400, detail="Invalid audience")
    roles_final = []
    if audience == "all":
        roles_final = ["staff", "manager", "admin"]
    elif audience == "managers":
        roles_final = ["manager", "admin"]
    elif audience == "employees":
        roles_final = ["staff"]
    else:
        roles_final = [r.strip() for r in allow_roles_custom.split(",") if r.strip()]
        if not roles_final:
            raise HTTPException(status_code=400, detail="Custom audience requires allow_roles_custom")
    payloads = [
        {
            "uploader": user.username,
            "uploader_role": user.role,
            "uploader_level": user.level,
            "filename": file.filename,
            "dept": user.dept,
            "project": user.project,
            "audience": audience,
            "allow_roles": roles_final,
        }
        for _ in chunks
    ]
    upsert_text_chunks(chunks, vecs, payloads)
    return UploadResponse(ids=[file.filename])

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, user: User = Depends(get_current_user)):
    # Embed query
    qvec = embed_texts([req.query])[0]
    # Filtered vector search
    passages = filtered_search(
        qvec,
        user_level=user.level,
        user_role=user.role,
        user_dept=user.dept,
        user_project=user.project,
        top_k=max(10, req.top_k)
    )
    # Rerank locally; reranked items are ((text, payload), score)
    reranked = rerank(req.query, passages, top_k=req.top_k)
    contexts = [pair[0] for pair, score in reranked]

    # Call Gemini for answer
    if not os.getenv("GEMINI_API_KEY"):
        answer = "[Gemini API key not configured]\n\nTop contexts:\n" + "\n---\n".join(contexts)
        sources = [pair[1] for pair, score in reranked]
        return QueryResponse(answer=answer, sources=sources)

    model = genai.GenerativeModel(MODEL_NAME)
    prompt = (
        "You are a corporate assistant. Use the provided context chunks to answer.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Question: {req.query}\n\n"
        "Context:\n" + "\n\n".join(f"- {c}" for c in contexts) + "\n\nAnswer:"
    )
    resp = model.generate_content(prompt)
    answer = resp.text if hasattr(resp, "text") else str(resp)
    sources = [pair[1] for pair, score in reranked]
    return QueryResponse(answer=answer, sources=sources)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Enterprise RAG Demo</title>
  <style>
    body{font-family:system-ui, sans-serif; max-width:900px; margin:40px auto;}
    input,button,textarea{padding:8px; margin:6px 0; width:100%;}
    .row{display:flex; gap:12px}
    .col{flex:1}
    pre{background:#f6f8fa; padding:12px;}
  </style>
</head>
<body>
  <h1>Enterprise RAG Demo</h1>
  <section id="login">
    <h2>Login</h2>
    <div class="row">
      <div class="col">
        <label>Username</label>
        <input id="username" value="carol"/>
      </div>
      <div class="col">
        <label>Password</label>
        <input id="password" type="password" value="carol123"/>
      </div>
    </div>
    <button onclick="login()">Login</button>
    <div id="me"></div>
    <div style="margin-top:8px">
      <button onclick="openScreening()">Open Screening (admin)</button>
      <button onclick="openSmartAccess()">Open Smart Access (admin)</button>
    </div>
  </section>
  <hr/>
  <section id="upload">
    <h2>Upload .txt Document</h2>
    <input type="file" id="file"/>
    <div>
      <label>Audience</label>
      <select id="audience">
        <option value="all">All</option>
        <option value="managers">Managers only</option>
        <option value="employees">Employees only</option>
        <option value="custom">Custom roles</option>
      </select>
    </div>
    <label>Custom roles (comma separated, used only if audience=custom)</label>
    <input id="roles_custom" placeholder="e.g. manager,admin"/>
    <button onclick="uploadDoc()">Upload</button>
    <div id="uploadResult"></div>
  </section>
  <hr/>
  <section id="query">
    <h2>Query</h2>
    <textarea id="q" rows="3" placeholder="Ask a question..."></textarea>
    <button onclick="doQuery()">Search</button>
    <h3>Answer</h3>
    <pre id="answer"></pre>
    <h3>Sources</h3>
    <pre id="sources"></pre>
  </section>
<script>
let token = localStorage.getItem('token')||'';
let deviceId = localStorage.getItem('device_id')||'';
if(!deviceId){
  deviceId = 'dev-' + Math.random().toString(36).slice(2) + '-' + Date.now();
  localStorage.setItem('device_id', deviceId);
}
let employeeId = '';
// Smart Access collectors
let saMouseMoves = 0;
let saKeyTimes = [];
let saTimer = null;

async function login(){
  const form = new URLSearchParams();
  form.append('username', document.getElementById('username').value);
  form.append('password', document.getElementById('password').value);
  form.append('grant_type', 'password');
  const r = await fetch('/auth/login', {method:'POST', body: form});
  if(!r.ok){ alert('login failed'); return; }
  const data = await r.json();
  token = data.access_token; localStorage.setItem('token', token);
  const me = await fetch('/me', {headers:{'Authorization':'Bearer '+token}});
  const meTxt = await me.text();
  document.getElementById('me').innerText = meTxt;
  try { const m = JSON.parse(meTxt); employeeId = m.username || ''; } catch(e) { employeeId = document.getElementById('username').value; }
  startSmartAccessCollector();
}

function openScreening(){
  if(!token){ alert('Please login first'); return; }
  const url = '/screening/jobs?token='+encodeURIComponent(token);
  window.open(url, '_blank');
}

function openSmartAccess(){
  if(!token){ alert('Please login first'); return; }
  const url = '/smart-access/admin?token='+encodeURIComponent(token);
  window.open(url, '_blank');
}

// ---------- Smart Access Collector ----------
function startSmartAccessCollector(){
  // mouse
  window.addEventListener('mousemove', () => { saMouseMoves++; });
  // typing
  window.addEventListener('keydown', () => { saKeyTimes.push(Date.now()); });
  if(saTimer) clearInterval(saTimer);
  saTimer = setInterval(sendSmartAccessEvent, 15000); // every 15s
  document.addEventListener('visibilitychange', () => { if(document.visibilityState==='hidden'){ sendSmartAccessEvent(true); } });
  window.addEventListener('beforeunload', () => { sendSmartAccessEvent(true); });
}

async function sendSmartAccessEvent(sync=false){
  if(!employeeId) return;
  // compute typing CPM and burstiness
  const now = Date.now();
  const horizonMs = 15000; // last 15s
  saKeyTimes = saKeyTimes.filter(t => now - t <= horizonMs);
  const strokes = saKeyTimes.length;
  const typingCpm = Math.round((strokes / (horizonMs/60000)));
  let typingBurst = 0;
  if(saKeyTimes.length > 2){
    const intervals = [];
    for(let i=1;i<saKeyTimes.length;i++){ intervals.push(saKeyTimes[i]-saKeyTimes[i-1]); }
    const avg = intervals.reduce((a,b)=>a+b,0)/intervals.length;
    const variance = intervals.reduce((a,b)=>a+Math.pow(b-avg,2),0)/intervals.length;
    typingBurst = Math.min(1, Math.max(0, Math.sqrt(variance)/1000));
  }
  const payload = {
    employee_id: employeeId,
    page: location.pathname,
    mouse_moves: saMouseMoves,
    typing_cpm: typingCpm,
    typing_burstiness: typingBurst,
    device_id: deviceId,
    seen_device_before: null,
    user_agent: navigator.userAgent,
    timestamp: new Date().toISOString(),
  };
  // reset mouse counter after sampling
  saMouseMoves = 0;
  try{
    await fetch('/smart-access/collect', {
      method:'POST',
      headers:{'Content-Type':'application/json', 'Authorization': token ? ('Bearer '+token) : ''},
      body: JSON.stringify(payload)
    });
  }catch(e){ /* ignore */ }
}

async function uploadDoc(){
  const f = document.getElementById('file').files[0];
  if(!f){ alert('select a .txt file'); return; }
  const fd = new FormData();
  fd.append('file', f);
  fd.append('audience', document.getElementById('audience').value);
  fd.append('allow_roles_custom', document.getElementById('roles_custom').value);
  const r = await fetch('/documents', {method:'POST', headers:{'Authorization':'Bearer '+token}, body: fd});
  document.getElementById('uploadResult').innerText = await r.text();
}

async function doQuery(){
  const r = await fetch('/query', {method:'POST', headers:{'Authorization':'Bearer '+token, 'Content-Type':'application/json'}, body: JSON.stringify({query: document.getElementById('q').value, top_k: 5})});
  const data = await r.json();
  document.getElementById('answer').innerText = data.answer || '';
  document.getElementById('sources').innerText = JSON.stringify(data.sources, null, 2);
}
</script>
</body>
</html>
"""
