import os
import io
import threading
import subprocess
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
try:
    from app.accounts.db import init_db as accounts_init_db, get_session as accounts_session
    from app.accounts.models import Project, ProjectMembership
    from sqlmodel import select
    _HAS_ACCOUNTS = True
except Exception:
    _HAS_ACCOUNTS = False

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

class ProjectsResponse(BaseModel):
    projects: List[str]

@app.on_event("startup")
async def startup():
    ensure_collection()
    screening_init_db()
    # init accounts DB if available
    try:
        if _HAS_ACCOUNTS:
            accounts_init_db()
    except Exception:
        pass
    # prepare smart access collection
    try:
        smart_access.ensure_collection()
    except Exception:
        pass
    # Try to start J.A.R.V.I.S vite dev server in background
    def _start_jarvis():
        try:
            subprocess.Popen(["npm", "run", "dev"], cwd=os.path.join(os.getcwd(), "J.A.R.V.I.S"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    threading.Thread(target=_start_jarvis, daemon=True).start()

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

@app.get("/projects", response_model=ProjectsResponse)
async def get_projects(user: User = Depends(get_current_user)):
    allowed = _allowed_projects_for_user(user)
    return ProjectsResponse(projects=allowed)

def _allowed_projects_for_user(user: User) -> List[str]:
    """Derive list of projects user may access.
    Prefer accounts DB memberships if available; otherwise fallback to env + user's project.
    """
    if _HAS_ACCOUNTS:
        try:
            with accounts_session() as s:
                # Admin/manager: all projects
                if user.role in {"manager", "admin"}:
                    names = [p.name for p in s.exec(select(Project)).all()]
                    if user.project and user.project not in names:
                        names.append(user.project)
                    return sorted(list(dict.fromkeys(names)))
                # Staff: memberships only
                mems = s.exec(select(ProjectMembership).where(ProjectMembership.username == user.username)).all()
                names = [m.project_name for m in mems]
                if not names and user.project:
                    names = [user.project]
                return sorted(list(dict.fromkeys(names)))
        except Exception:
            pass
    # Fallback to env var
    env_projects = os.getenv("PROJECTS", "")
    available = [p.strip() for p in env_projects.split(",") if p.strip()]
    if user.project and user.project not in available:
        available.append(user.project)
    if user.role in {"manager", "admin"}:
        return sorted(list(dict.fromkeys(available)))
    allowed = [p for p in available if p == user.project]
    if not allowed and user.project:
        allowed = [user.project]
    return allowed

@app.post("/documents", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    audience: str = Form(default="all"),  # one of: all, managers, employees, custom
    allow_roles_custom: str = Form(default=""),  # used when audience=custom, comma-separated roles
    project_override: str | None = Form(default=None),  # optional: override user's project for tagging
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
    # Enforce project permission: only allow override if within allowed projects
    allowed_projects = _allowed_projects_for_user(user)
    desired_project = (project_override or user.project or "").strip() or user.project
    if desired_project and desired_project not in allowed_projects:
        raise HTTPException(status_code=403, detail="Not allowed to upload to this project")
    project_final = desired_project
    payloads = [
        {
            "uploader": user.username,
            "uploader_role": user.role,
            "uploader_level": user.level,
            "filename": file.filename,
            "dept": user.dept,
            "project": project_final,
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

from fastapi.responses import RedirectResponse

@app.get("/", response_class=RedirectResponse)
async def root():
    # Redirect to J.A.R.V.I.S dev server home
    return RedirectResponse(url="http://localhost:5173/")


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>J.A.R.V.I.S — Enterprise RAG</title>
  <style>
    :root{--bg:#000; --card:#0b0b0b; --muted:#9ca3af; --fg:#fff; --accent:#7A0000; --accent2:#520000; --border:#1f2937}
    *{box-sizing:border-box}
    body{font-family:system-ui, sans-serif; margin:0; background:var(--bg); color:var(--fg);}
    a{color:var(--accent)}
    .container{max-width:1100px; margin:0 auto; padding:24px}
    .row{display:flex; gap:12px}
    .col{flex:1}
    .card{background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px}
    .muted{color:var(--muted)}
    input,button,textarea,select{padding:10px 12px; margin:8px 0; width:100%; background:#0f0f10; color:var(--fg); border:1px solid var(--border); border-radius:10px}
    button{background:var(--accent); border:none; cursor:pointer; transition:transform .08s ease, background .2s}
    button:hover{background:var(--accent2)}
    button:active{transform:translateY(1px)}
    .btn-secondary{background:#111827}
    .header{position:sticky; top:0; z-index:5; background:linear-gradient(180deg, rgba(0,0,0,.9), rgba(0,0,0,.6)); border-bottom:1px solid var(--border)}
    .brand{display:flex; align-items:center; gap:12px}
    .logo{width:40px; height:40px; border:2px solid var(--accent); border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:700}
    .title{font-weight:900; letter-spacing:6px}
    .grid{display:grid; grid-template-columns: 1fr; gap:16px}
    .section-title{margin:0 0 8px 0}
    /* Chat styles */
    .chat{display:flex; flex-direction:column; gap:10px; height:420px; overflow:auto; padding:8px; border:1px solid var(--border); border-radius:12px; background:#0a0a0a}
    .msg{max-width:80%; padding:10px 12px; border-radius:12px; line-height:1.4; white-space:pre-wrap}
    .msg.user{background:#101010; border:1px solid var(--border); align-self:flex-end}
    .msg.ai{background:#120606; border:1px solid #2a0a0a}
    .msg .tag{display:block; font-size:12px; color:var(--muted); margin-bottom:4px}
    .typing{display:inline-flex; gap:6px; align-items:center}
    .dot{width:6px; height:6px; border-radius:50%; background:var(--muted); opacity:.4; animation:blink 1s infinite}
    .dot:nth-child(2){animation-delay:.15s}
    .dot:nth-child(3){animation-delay:.3s}
    @keyframes blink{0%{opacity:.2} 50%{opacity:1} 100%{opacity:.2}}
    pre{background:#0f0f10; padding:12px; border:1px solid var(--border); border-radius:10px;}
    .pill{display:inline-block; padding:6px 10px; background:#120606; border:1px solid #2a0a0a; color:#fda4a4; border-radius:999px; font-size:12px}
    .actions{display:flex; gap:8px}
    .divider{height:1px; background:var(--border); margin:20px 0}
  </style>
</head>
<body>
  <div class="header">
    <div class="container" style="display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 24px;">
      <div class="brand">
        <div class="logo">JR</div>
        <div class="title">J.A.R.V.I.S</div>
      </div>
      <div class="actions">
        <button class="btn-secondary" onclick="openScreening()">Screening Admin</button>
        <button class="btn-secondary" onclick="openSmartAccess()">Smart Access</button>
      </div>
    </div>
  </div>
  <div class="container" style="padding-top:24px">
  <div class="grid">
    <section id="query" class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:8px">
        <h2 class="section-title">Chat</h2>
        <div style="display:flex; gap:8px; align-items:center">
          <select id="projectSelect" title="Project" style="width:220px">
            <option value="">Default Project</option>
          </select>
          <a id="uploadBtn" class="pill" href="#" onclick="openUpload(event)">Upload Documents</a>
        </div>
      </div>
      <div id="chat" class="chat">
        <div class="msg ai"><span class="tag">J.A.R.V.I.S</span>Ask me anything about your documents. I follow your role/level/dept/project.</div>
      </div>
      <div class="row">
        <div class="col">
          <textarea id="q" rows="2" placeholder="Type your question..."></textarea>
        </div>
        <div style="width:140px; display:flex; align-items:flex-end">
          <button style="width:100%" onclick="doQuery()">Send</button>
        </div>
      </div>
      <div class="divider"></div>
      <h3 class="section-title">Sources</h3>
      <pre id="sources"></pre>
    </section>
  </div>
  </div>
<script>
// Accept token from URL and persist
try{
  const urlToken = new URLSearchParams(location.search).get('token');
  if(urlToken){ localStorage.setItem('token', urlToken); }
}catch(e){}
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

async function bootstrapUser(){
  if(!token){ return; }
  const me = await fetch('/me', {headers:{'Authorization':'Bearer '+token}});
  const meTxt = await me.text();
  try { const m = JSON.parse(meTxt); employeeId = m.username || ''; await setupProjects(); } catch(e) { /* ignore */ }
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

function openUpload(e){
  e.preventDefault();
  const proj = document.getElementById('projectSelect').value||'';
  const url = '/demo/upload?token='+encodeURIComponent(token)+'&project='+encodeURIComponent(proj);
  window.open(url, '_blank');
}

async function doQuery(){
  const chat = document.getElementById('chat');
  const q = document.getElementById('q').value;
  if(!q.trim()) return;
  chat.insertAdjacentHTML('beforeend', `<div class="msg user"><span class="tag">You</span>${q.replace(/</g,'&lt;')}</div>`);
  document.getElementById('q').value='';
  const typing = document.createElement('div');
  typing.className='msg ai';
  typing.innerHTML = '<span class="tag">J.A.R.V.I.S</span><span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
  chat.appendChild(typing); chat.scrollTop = chat.scrollHeight;

  const r = await fetch('/query', {method:'POST', headers:{'Authorization':'Bearer '+token, 'Content-Type':'application/json'}, body: JSON.stringify({query: q, top_k: 5})});
  const data = await r.json();
  typing.remove();
  const ans = (data.answer||'').replace(/</g,'&lt;');
  chat.insertAdjacentHTML('beforeend', `<div class="msg ai"><span class="tag">J.A.R.V.I.S</span>${ans}</div>`);
  document.getElementById('sources').innerText = JSON.stringify(data.sources, null, 2);
  chat.scrollTop = chat.scrollHeight;
}
bootstrapUser();
</script>
</body>
</html>
"""


@app.get("/demo/upload", response_class=HTMLResponse)
async def demo_upload():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Upload Documents — J.A.R.V.I.S</title>
  <style>
    :root{--bg:#000; --card:#0b0b0b; --muted:#9ca3af; --fg:#fff; --accent:#7A0000; --accent2:#520000; --border:#1f2937}
    body{font-family:system-ui, sans-serif; margin:0; background:var(--bg); color:var(--fg)}
    .container{max-width:800px; margin:40px auto; padding:24px}
    .card{background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px}
    input,button,textarea,select{padding:10px 12px; margin:8px 0; width:100%; background:#0f0f10; color:var(--fg); border:1px solid var(--border); border-radius:10px}
    button{background:var(--accent); border:none; cursor:pointer}
    .muted{color:var(--muted)}
  </style>
  <script>
  let token='';
  function getParam(n){ try{ return new URLSearchParams(location.search).get(n)||'' }catch(e){ return '' } }
  async function init(){
    const t = getParam('token'); if(t){ localStorage.setItem('token', t); }
    token = localStorage.getItem('token')||'';
    await populateProjects();
    const proj = getParam('project'); if(proj){ const sel=document.getElementById('project'); for(const o of sel.options){ if(o.value===proj){ sel.value=proj; break; } } }
  }
  async function populateProjects(){
    try{
      const r = await fetch('/projects', {headers:{'Authorization':'Bearer '+token}});
      if(!r.ok) return;
      const data = await r.json();
      const sel = document.getElementById('project');
      sel.innerHTML='';
      (data.projects||[]).forEach(p=>{
        const opt = document.createElement('option'); opt.value=p; opt.textContent=p; sel.appendChild(opt);
      });
    }catch(e){}
  }
  async function upload(){
    const f = document.getElementById('file').files[0]; if(!f){ alert('Choose a .txt file'); return; }
    const fd = new FormData();
    fd.append('file', f);
    fd.append('audience', document.getElementById('audience').value);
    fd.append('allow_roles_custom', document.getElementById('roles_custom').value);
    fd.append('project_override', document.getElementById('project').value);
    const r = await fetch('/documents', {method:'POST', headers:{'Authorization':'Bearer '+token}, body: fd});
    const txt = await r.text();
    document.getElementById('result').innerText = txt;
  }
  </script>
</head>
<body onload="init()">
  <div class="container">
    <h1>Upload Documents</h1>
    <div class="card">
      <label>Project</label>
      <select id="project"></select>
      <label>Audience</label>
      <select id="audience">
        <option value="all">All</option>
        <option value="managers">Managers only</option>
        <option value="employees">Employees only</option>
        <option value="custom">Custom roles</option>
      </select>
      <label>Custom roles (comma separated)</label>
      <input id="roles_custom" placeholder="e.g. manager,admin"/>
      <label>File (.txt)</label>
      <input type="file" id="file"/>
      <button onclick="upload()">Upload</button>
      <pre class="muted" id="result"></pre>
    </div>
  </div>
</body>
</html>
"""
