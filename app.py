import streamlit as st
import os
import re
import time
import json
import csv
import requests as http_requests
from datetime import datetime
from pathlib import Path
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

from indexer import PDFIndexer, PAPER_TYPES, INDEX_TYPES, GEN_MODELS, EMBED_MODEL

st.set_page_config(
    page_title="Vertex RAG — Powered by Claude",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    box-sizing: border-box;
}

.stApp { background: #212121; color: #ececec; }
.main  { background: #212121; }

#MainMenu, footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }
header { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

section[data-testid="stSidebar"] {
    min-width: 240px !important;
    max-width: 240px !important;
    transform: none !important;
    background: #171717 !important;
    border-right: 1px solid #2f2f2f !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 12px !important;
    overflow-x: hidden;
    box-sizing: border-box !important;
}
[data-testid="stSidebarCollapseButton"] button {
    color: #9b9b9b !important; background: transparent !important; border: none !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    color: #ececec !important; background: #2f2f2f !important; border-radius: 6px !important;
}

.sb-header {
    padding: 28px 16px 16px 16px; border-bottom: 1px solid #2f2f2f;
    font-size: 1.45rem; font-weight: 700; color: #ececec; letter-spacing: -0.02em;
    display: flex; align-items: center; justify-content: center; text-align: center;
}
.sb-section-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: #666; padding: 14px 16px 6px 16px;
}
.sb-nav-btn {
    display: block; width: 100%; text-align: left; padding: 8px 16px;
    font-size: 0.83rem; font-weight: 400; color: #9b9b9b; background: transparent;
    border: none; border-radius: 0; cursor: pointer; transition: background 0.12s, color 0.12s;
}
.sb-nav-btn:hover  { background: #2f2f2f; color: #ececec; }
.sb-nav-btn.active { background: #2f2f2f; color: #ececec; font-weight: 500; }

.sb-metrics { display: flex; gap: 6px; padding: 8px 16px 12px 16px; }
.sb-metric {
    flex: 1; background: #2a2a2a; border: 1px solid #333; border-radius: 8px;
    padding: 10px 6px; text-align: center;
}
.sb-metric .val { font-size: 1rem; font-weight: 600; color: #ececec; line-height: 1; }
.sb-metric .lbl { font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.06em; color: #666; margin-top: 3px; }

.file-pill {
    margin: 2px 16px; background: #2a2a2a; border: 1px solid #333; border-radius: 6px;
    padding: 6px 10px; font-size: 0.71rem; color: #bbb;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ftype {
    display: inline-block; font-size: 0.58rem; font-weight: 600; padding: 1px 5px;
    border-radius: 3px; background: #3a2f55; color: #a78bfa; margin-left: 4px;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #9b9b9b !important; font-size: 0.78rem !important;
    font-weight: 500 !important; padding-left: 16px !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #2a2a2a !important; border: 1px solid #333 !important;
    color: #ececec !important; border-radius: 6px !important;
    margin: 0 !important; width: 100% !important; box-sizing: border-box !important;
}
section[data-testid="stSidebar"] .stSlider { padding: 0 8px !important; width: 100% !important; }
section[data-testid="stSidebar"] div[role="slider"] { background: #e53935 !important; }
section[data-testid="stSidebar"] .stButton > button {
    margin: 2px 4px !important; width: 100% !important; background: #2a2a2a !important;
    border: 1px solid #333 !important; color: #9b9b9b !important; border-radius: 6px !important;
    font-size: 0.8rem !important; font-weight: 500 !important; padding: 7px 8px !important;
    white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
    box-sizing: border-box !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #333 !important; color: #ececec !important; border-color: #444 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #e53935 !important; color: #fff !important; border-color: #e53935 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #c62828 !important; border-color: #c62828 !important;
}
section[data-testid="stSidebar"] hr { border-color: #2f2f2f !important; margin: 8px 16px !important; }

.block-container {
    max-width: 1100px !important; width: 100% !important;
    margin: 0 auto !important; padding: 0 48px 120px 48px !important;
}
.page-heading { text-align: center; padding: 48px 0 32px 0; }
.page-heading h1 { font-size: 2rem; font-weight: 600; color: #ececec; letter-spacing: -0.03em; margin: 0 0 6px 0; }
.page-heading p  { font-size: 0.78rem; color: #666; letter-spacing: 0.06em; text-transform: uppercase; margin: 0; }

div[data-testid="stTabs"] { border-bottom: 1px solid #2f2f2f !important; margin-bottom: 24px !important; text-align: center !important; }
div[data-testid="stTabs"] [role="tablist"] { justify-content: center !important; display: flex !important; }
div[data-testid="stTabs"] [role="tab"] {
    font-size: 5rem !important; font-weight: 500 !important; color: #666 !important;
    border: none !important; background: transparent !important; padding: 10px 40px !important;
}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #ececec !important; border-bottom: 2px solid #e53935 !important;
}
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] [role="tab"] span,
div[data-testid="stTabs"] [role="tab"] div { font-size: 1.1rem !important; font-weight: 500 !important; }

div[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important; padding: 16px 0 !important;
    border-bottom: 1px solid #2f2f2f !important; border-radius: 0 !important; max-width: 100% !important;
}
div[data-testid="stChatMessage"] p { color: #ececec !important; font-size: 0.93rem !important; line-height: 1.65 !important; }
div[data-testid="stChatMessage"][data-testid*="user"] {
    background: #2a2a2a !important; border-radius: 12px !important;
    border: 1px solid #333 !important; padding: 14px 18px !important;
}

div[data-testid="stChatInput"] {
    position: fixed !important; bottom: 0 !important;
    left: calc(50% + 120px) !important; transform: translateX(-50%) !important;
    width: min(1000px, calc(100vw - 260px)) !important; background: #212121 !important;
    padding: 16px 0 28px 0 !important; z-index: 999 !important;
}
div[data-testid="stChatInput"] > div {
    background: transparent !important; border: 1px solid #3f3f3f !important;
    border-radius: 16px !important; padding: 16px 20px !important;
    box-shadow: 0 0 0 1px #3f3f3f !important; min-height: 72px !important;
}
div[data-testid="stChatInput"] > div:focus-within {
    border-color: #555 !important; box-shadow: 0 0 0 1px #555, 0 4px 24px rgba(0,0,0,0.4) !important;
}
div[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #ececec !important; font-size: 1.25rem !important;
    border: none !important; outline: none !important; resize: none !important;
    min-height: 44px !important; line-height: 1.6 !important;
}
div[data-testid="stChatInput"] textarea::placeholder { color: #555 !important; font-size: 1.25rem !important; }
div[data-testid="stChatInput"] button {
    background: #e53935 !important; border-radius: 10px !important; color: #fff !important;
    width: 38px !important; height: 38px !important;
}
div[data-testid="stChatInput"] button:hover { background: #c62828 !important; }

.chunk-preview {
    background: #2a2a2a; border-left: 2px solid #444; padding: 0.65rem 0.9rem;
    border-radius: 0 6px 6px 0; margin: 0.4rem 0; font-size: 0.78rem; color: #bbb;
    white-space: pre-wrap; font-family: 'SF Mono','Fira Code','Menlo',monospace !important; line-height: 1.5;
}

.badge {
    display: inline-block; background: #2a2a2a; color: #9b9b9b; padding: 1px 7px;
    border-radius: 3px; font-size: 0.66rem; font-weight: 500; margin-right: 3px;
    font-family: 'SF Mono','Fira Code',monospace !important; border: 1px solid #333;
}
.badge-type  { background: #1e1533; color: #a78bfa; border-color: #3a2f55; }
.badge-url   { background: #0f1e33; color: #60a5fa; border-color: #1e3a55; }
.badge-score { background: #0f2218; color: #34d399; border-color: #1a3d2b; }
.badge-claude { background: #1a1200; color: #f59e0b; border-color: #3d2e00; }

.reflect-pass  { background: #0f2218; border-left: 2px solid #34d399; padding: 0.4rem 0.7rem; border-radius: 0 5px 5px 0; font-size: 0.74rem; color: #34d399; margin: 0.25rem 0; }
.reflect-fail  { background: #220f0f; border-left: 2px solid #f87171; padding: 0.4rem 0.7rem; border-radius: 0 5px 5px 0; font-size: 0.74rem; color: #f87171; margin: 0.25rem 0; }
.reflect-fixed { background: #1f1a0f; border-left: 2px solid #fbbf24; padding: 0.4rem 0.7rem; border-radius: 0 5px 5px 0; font-size: 0.74rem; color: #fbbf24; margin: 0.25rem 0; }

.paper-card { background: #2a2a2a; border: 1px solid #333; border-radius: 8px; padding: 12px 14px; margin: 5px 0; }
.paper-card .ptitle { font-weight: 600; font-size: 0.84rem; color: #ececec; margin-bottom: 4px; }
.paper-card .purl   { font-size: 0.71rem; color: #60a5fa; }
.paper-card .pscore { font-size: 0.67rem; color: #555; margin-top: 3px; font-family: 'SF Mono',monospace !important; }

/* ── Claude Reasoning Trace ── */
.trace-container {
    background: #1a1a1a; border: 1px solid #2f2f2f; border-radius: 10px;
    padding: 12px 16px; margin: 12px 0;
}
.trace-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: #f59e0b; padding-bottom: 10px; border-bottom: 1px solid #2f2f2f; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.trace-step {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #222;
}
.trace-step:last-child { border-bottom: none; }
.trace-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.trace-body { flex: 1; min-width: 0; }
.trace-label { font-size: 0.8rem; font-weight: 600; color: #ececec; margin-bottom: 3px; }
.trace-content {
    font-size: 0.71rem; color: #888; line-height: 1.55; white-space: pre-wrap;
    font-family: 'SF Mono','Fira Code',monospace !important;
    background: #242424; border-radius: 5px; padding: 6px 10px; margin-top: 4px;
}
.trace-pass    { border-left: 3px solid #34d399 !important; }
.trace-fail    { border-left: 3px solid #f87171 !important; }
.trace-draft   { border-left: 3px solid #60a5fa !important; }
.trace-revised { border-left: 3px solid #fbbf24 !important; }
.trace-ok      { border-left: 3px solid #6b7280 !important; }
.verdict-pass { display: inline-block; background: #0f2218; color: #34d399; border: 1px solid #1a3d2b; border-radius: 4px; padding: 1px 8px; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.05em; margin-left: 6px; }
.verdict-fail { display: inline-block; background: #220f0f; color: #f87171; border: 1px solid #3d1515; border-radius: 4px; padding: 1px 8px; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.05em; margin-left: 6px; }

div[data-testid="stStatus"] {
    background: #2a2a2a !important; border: 1px solid #333 !important;
    border-radius: 8px !important; color: #ececec !important;
}
div[data-testid="stProgressBar"] > div { background: #e53935 !important; }

.stSelectbox > div > div {
    background: #2a2a2a !important; border: 1px solid #333 !important;
    color: #ececec !important; border-radius: 6px !important;
}
.stSelectbox label { color: #9b9b9b !important; font-size: 0.8rem !important; }

.block-container .stButton > button {
    background: #2a2a2a !important; border: 1px solid #333 !important;
    color: #9b9b9b !important; border-radius: 6px !important;
}
.block-container .stButton > button:hover {
    background: #333 !important; color: #ececec !important; border-color: #444 !important;
}
.block-container .stButton > button[kind="primary"] {
    background: #e53935 !important; color: #fff !important; border-color: #e53935 !important;
}
.block-container .stButton > button[kind="primary"]:hover {
    background: #c62828 !important; border-color: #c62828 !important;
}

.stTextInput input, .stTextArea textarea {
    background: #2a2a2a !important; border: 1px solid #333 !important;
    color: #ececec !important; border-radius: 6px !important;
}
.stTextInput label, .stTextArea label { color: #9b9b9b !important; font-size: 0.8rem !important; }
.stCheckbox label { color: #9b9b9b !important; }
.stSlider label   { color: #9b9b9b !important; }

.stAlert { border-radius: 8px !important; }
div[data-testid="stMetric"] {
    background: #2a2a2a !important; border: 1px solid #333 !important;
    border-radius: 8px !important; padding: 12px 16px !important;
}
div[data-testid="stMetric"] label { color: #666 !important; font-size: 0.72rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ececec !important; }
div[data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden; border: 1px solid #333 !important; }

.stExpander { background: #2a2a2a !important; border: 1px solid #333 !important; border-radius: 8px !important; }
.stExpander summary { color: #9b9b9b !important; font-size: 0.82rem !important; }
hr { border-color: #2f2f2f !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
_CHAT_HISTORY_PATH   = Path("./chat_history.json")
_INDEXED_FILES_PATH  = Path("./indexed_files.json")

def _save_chat_history():
    try:
        _CHAT_HISTORY_PATH.write_text(json.dumps(st.session_state.chat_history), encoding="utf-8")
    except Exception:
        pass

def _load_chat_history():
    try:
        if _CHAT_HISTORY_PATH.exists():
            return json.loads(_CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def _save_indexed_files():
    try:
        _INDEXED_FILES_PATH.write_text(json.dumps({
            "indexed_files": st.session_state.indexed_files,
            "file_url_map":  st.session_state.file_url_map,
        }), encoding="utf-8")
    except Exception:
        pass

def _load_indexed_files():
    try:
        if _INDEXED_FILES_PATH.exists():
            data = json.loads(_INDEXED_FILES_PATH.read_text(encoding="utf-8"))
            return data.get("indexed_files", []), data.get("file_url_map", {})
    except Exception:
        pass
    return [], {}


# ── Session state defaults ─────────────────────────────────────────────────────
_defaults = [
    ("indexer",           None),
    ("indexed_files",     _load_indexed_files()[0]),
    ("file_url_map",      _load_indexed_files()[1]),
    ("index_stats",       {}),
    ("chat_history",      _load_chat_history()),
    ("prepared_chunks",   None),
    ("prepared_files",    None),
    ("benchmark_results", None),
    ("benchmark_csv_path",None),
    ("benchmark_run_dir", None),
    ("interrupt_requested",False),
    ("is_indexing",       False),
    ("is_querying",       False),
    ("is_benchmarking",   False),
    ("last_index_error",  None),
    ("sidebar_tab",       "config"),
    ("cfg_index_type",    "HNSW"),
    ("cfg_chunk_size",    400),
    ("cfg_chunk_overlap", 50),
    ("cfg_language",      "English"),
    ("cfg_model",         list(GEN_MODELS.keys())[0]),
    ("cfg_top_k",         4),
    ("cfg_max_iterations",2),
    ("cfg_min_words",     80),
    ("cfg_max_words",     400),
    ("file_assignments",  {}),
    ("custom_paper_types",[]),
]
for k, v in _defaults:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auto-reconnect to existing Milvus collection ───────────────────────────────
if st.session_state.indexer is None:
    try:
        _candidate = PDFIndexer(
            chunk_size=st.session_state.cfg_chunk_size,
            chunk_overlap=st.session_state.cfg_chunk_overlap,
            model=st.session_state.cfg_model,
            index_type=st.session_state.cfg_index_type,
            collection_name="papers_rag_interactive",
            drop_old_collection=False,
        )
        if _candidate._collection and _candidate._collection.num_entities > 0:
            _candidate._collection.load()
            _candidate._build_graph()
            st.session_state.indexer = _candidate
            if not st.session_state["index_stats"].get("total_chunks"):
                st.session_state["index_stats"] = {
                    "status": "ok",
                    "total_chunks": _candidate._collection.num_entities,
                    "total_files":  len(st.session_state.indexed_files),
                    "index_type":   st.session_state.cfg_index_type,
                    "embed_model":  EMBED_MODEL,
                    "chunk_size":   st.session_state.cfg_chunk_size,
                    "embed_dim":    1024,
                    "gen_model":    st.session_state.cfg_model,
                    "paper_types":  [],
                }
            if hasattr(_candidate, "_indexed_files") and _candidate._indexed_files:
                if not st.session_state.indexed_files:
                    st.session_state.indexed_files = _candidate._indexed_files
                if not st.session_state.file_url_map:
                    st.session_state.file_url_map  = _candidate._file_url_map
    except Exception:
        pass


# ── Helpers ────────────────────────────────────────────────────────────────────
def _check_interrupt():
    if st.session_state.get("interrupt_requested", False):
        raise RuntimeError("Interrupted by user.")

def _now_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_tmp_upload_dir() -> Path:
    p = Path("/tmp/rag_uploads"); p.mkdir(parents=True, exist_ok=True); return p

def _estimate_storage_mb(n, dim):
    return (n * dim * 4) / (1024 * 1024)

def _unique_papers_from_sources(sources, file_url_map, k=2):
    seen, recs = set(), []
    for s in sorted(sources, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        name = s.get("source", "")
        if not name or name in seen: continue
        seen.add(name)
        recs.append({
            "paper": name, "url": file_url_map.get(name, ""),
            "paper_type": s.get("paper_type", ""), "best_score": float(s.get("score", 0.0)),
        })
        if len(recs) >= k: break
    return recs

def _safe_set_failed_stats(index_type, model_choice):
    st.session_state["index_stats"] = {
        "status": "failed", "index_type": index_type,
        "gen_model": model_choice, "embed_model": EMBED_MODEL,
    }

def _save_benchmark_csv(rows, run_dir: Path) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "benchmark_results.csv"
    if not rows:
        csv_path.write_text("no_results\n"); return str(csv_path)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)
    return str(csv_path)


# ── URL Reflect ────────────────────────────────────────────────────────────────
def _is_url_reachable(url: str, timeout: int = 5) -> bool:
    if not url or not url.startswith("http"): return False
    try:
        return http_requests.head(url, allow_redirects=True, timeout=timeout).status_code < 400
    except Exception:
        return False

def _llm_find_url(paper_title: str) -> str:
    """Use Claude Haiku to find a paper URL."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=128,
            system="You are a research paper URL resolver. Return ONLY the single most likely canonical URL (arXiv, DOI, or venue). Raw URL only, no explanation. If not confident, return empty string.",
            messages=[{"role": "user", "content": f"Paper title: {paper_title}"}],
        )
        raw = resp.content[0].text.strip().split()[0] if resp.content[0].text.strip() else ""
        return raw if re.match(r"^https?://", raw) else ""
    except Exception:
        return ""

def _reflect_and_fix_urls(recs: list) -> tuple:
    trace, updated = [], []
    for rec in recs:
        title = rec.get("paper", "")
        url   = rec.get("url",   "").strip()
        entry = {"paper": title, "original_url": url}
        if url and _is_url_reachable(url):
            entry.update({"status": "PASS", "final_url": url, "note": "URL present and reachable."})
        else:
            entry["status"] = "FETCH"; entry["note"] = "URL missing — querying Claude."
            found = _llm_find_url(title)
            if found:
                ok = _is_url_reachable(found)
                entry.update({"status": "FIXED", "final_url": found,
                               "note": f"URL found ({'HEAD OK' if ok else 'HEAD inconclusive'}): {found}"})
                url = found
            else:
                entry.update({"status": "UNRESOLVED", "final_url": "", "note": "Could not find URL."})
        trace.append(entry)
        updated.append({**rec, "url": entry.get("final_url", url)})
    return updated, trace


def _render_recommendations(recs: list, url_trace: list = None):
    if not recs: return
    st.markdown('<div style="font-size:0.82rem;font-weight:600;color:#9b9b9b;margin:12px 0 6px 0;">Recommended Papers</div>', unsafe_allow_html=True)
    for i, r in enumerate(recs, 1):
        url      = r.get("url", "")
        url_html = f'<div class="purl">{url}</div>' if url else '<div class="pscore">No URL available</div>'
        st.markdown(
            f'<div class="paper-card">'
            f'<div class="ptitle">{i}. {r.get("paper","")}'
            f'  <span class="badge badge-type">{r.get("paper_type","")}</span></div>'
            f'{url_html}<div class="pscore">score: {r.get("best_score",0.0):.4f}</div></div>',
            unsafe_allow_html=True,
        )
    if url_trace:
        st.markdown('<div style="font-size:0.72rem;font-weight:600;color:#555;margin:10px 0 4px 0;">URL reflection trace</div>', unsafe_allow_html=True)
        for t in url_trace:
            status = t.get("status", "")
            css = "reflect-pass" if status == "PASS" else "reflect-fixed" if status == "FIXED" else "reflect-fail"
            st.markdown(
                f'<div class="{css}"><strong>{t.get("paper","")}</strong> — {status}<br>'
                f'Original: {t.get("original_url","") or "(none)"}<br>'
                f'Final: {t.get("final_url","") or "(none)"}<br>{t.get("note","")}</div>',
                unsafe_allow_html=True,
            )

def _render_sources(sources):
    if not sources: return
    st.markdown(
        f'<div style="font-size:0.78rem;font-weight:600;color:#9b9b9b;margin:12px 0 6px 0;">'
        f'{len(sources)} retrieved chunks</div>', unsafe_allow_html=True,
    )
    for s in sources:
        url      = st.session_state.file_url_map.get(s["source"], "")
        url_line = f"<br><span class='badge badge-url'>URL</span> {url}" if url else ""
        st.markdown(
            f'<div class="chunk-preview">'
            f'<span class="badge">{s["source"]}</span>'
            f'<span class="badge">p{s["page"]}</span>'
            f'<span class="badge badge-type">{s.get("paper_type","?")}</span>'
            f'<span class="badge badge-score">score {s["score"]:.3f}</span>'
            f'{url_line}<br><br>'
            f'{s["text"][:500]}{"..." if len(s["text"])>500 else ""}</div>',
            unsafe_allow_html=True,
        )


# ── Claude Reasoning Trace renderer ───────────────────────────────────────────
def _render_reasoning_trace(trace: list):
    """Renders the Claude LangGraph reasoning trace as a beautiful visual panel."""
    if not trace:
        return

    STATUS_CSS = {
        "pass":    "trace-pass",
        "fail":    "trace-fail",
        "draft":   "trace-draft",
        "revised": "trace-revised",
        "ok":      "trace-ok",
    }

    steps_html = ""
    for event in trace:
        label   = event.get("label",   "")
        content = event.get("content", "")
        status  = event.get("status",  "ok")
        verdict = event.get("verdict", "")
        css     = STATUS_CSS.get(status, "trace-ok")

        verdict_html = ""
        if verdict == "PASS":
            verdict_html = '<span class="verdict-pass">PASS</span>'
        elif verdict == "FAIL":
            verdict_html = '<span class="verdict-fail">FAIL</span>'

        content_escaped = content.replace("<", "&lt;").replace(">", "&gt;")

        steps_html += f"""
        <div class="trace-step {css}">
            <div class="trace-body">
                <div class="trace-label">{label}{verdict_html}</div>
                <div class="trace-content">{content_escaped}</div>
            </div>
        </div>
        """

    st.markdown(
        f"""
        <div class="trace-container">
            <div class="trace-header">
                <span>⚡</span> Claude Reasoning Trace
                <span class="badge badge-claude">{len(trace)} steps</span>
            </div>
            {steps_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Benchmark ──────────────────────────────────────────────────────────────────
def _run_index_benchmark_all_methods(embedded_chunks, model_choice, top_k, input_language,
                                      sample_query, paper_filter, max_iterations=2):
    _check_interrupt()
    st.session_state["is_benchmarking"] = True
    for k in ["benchmark_results", "benchmark_csv_path", "benchmark_run_dir"]:
        st.session_state[k] = None

    run_id  = _now_run_id()
    run_dir = Path("./benchmarks") / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    shared_summary = {
        "run_id": run_id, "n_chunks": len(embedded_chunks), "embed_model": EMBED_MODEL,
        "embed_dim": 1024,
        "estimated_vector_storage_mb": round(_estimate_storage_mb(len(embedded_chunks), 1024), 3),
        "note": "Storage estimate of vector bytes only.",
    }

    results = []
    raw_mb  = round(_estimate_storage_mb(len(embedded_chunks), 1024), 3)
    _OH     = {"HNSW": 1.4, "IVF_PQ": 0.3, "DiskANN": 1.3}

    for method in [m for m in INDEX_TYPES if m in ("HNSW", "IVF_PQ", "DiskANN")]:
        _check_interrupt()
        t0     = time.perf_counter()
        est_mb = round(raw_mb * _OH.get(method, 1.0), 3)
        try:
            idx = PDFIndexer(
                chunk_size=st.session_state.index_stats.get("chunk_size", 400) or 400,
                chunk_overlap=st.session_state.index_stats.get("chunk_overlap", 50) or 50,
                model=model_choice, index_type=method,
                collection_name=f"papers_rag_{run_id}_{method.lower()}", drop_old_collection=True,
            )
            _check_interrupt()
            tb0 = time.perf_counter(); idx.build_index(embedded_chunks); tb1 = time.perf_counter()
            _check_interrupt()
            tq0 = time.perf_counter()
            answer, sources, _trace = idx.query(
                sample_query, top_k=top_k, model=model_choice,
                paper_filter=paper_filter, output_language=input_language,
                max_iterations=max_iterations,
            )
            tq1 = time.perf_counter(); t1 = time.perf_counter()
            recs      = _unique_papers_from_sources(sources, st.session_state.get("file_url_map", {}), k=2)
            recs, _   = _reflect_and_fix_urls(recs)
            results.append({
                "method": method, "n_chunks": len(embedded_chunks),
                "index_build_sec": round(tb1-tb0, 4), "end_to_end_sec": round(t1-t0, 4),
                "one_query_sec": round(tq1-tq0, 4), "estimated_storage_mb": est_mb,
                "top_k": top_k, "sample_query": sample_query, "paper_filter": paper_filter or "None",
                "recommended_1": recs[0]["paper"] if recs else "",
                "recommended_1_url": recs[0]["url"] if recs else "",
                "recommended_2": recs[1]["paper"] if len(recs)>1 else "",
                "recommended_2_url": recs[1]["url"] if len(recs)>1 else "",
                "notes": "DiskANN support depends on Milvus build." if method == "DiskANN" else "",
            })
            (run_dir / f"{method}_answer.txt").write_text(answer or "", encoding="utf-8")
            (run_dir / f"{method}_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
        except Exception as e:
            results.append({
                "method": method, "n_chunks": len(embedded_chunks),
                "index_build_sec": "", "end_to_end_sec": "", "one_query_sec": "",
                "estimated_storage_mb": est_mb, "top_k": top_k, "sample_query": sample_query,
                "paper_filter": paper_filter or "None", "recommended_1": "", "recommended_1_url": "",
                "recommended_2": "", "recommended_2_url": "", "notes": f"FAILED: {e}",
            })

    csv_path = _save_benchmark_csv(results, run_dir)
    st.session_state["is_benchmarking"] = False
    return results, csv_path, str(run_dir), shared_summary


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0px 16px 12px 16px;text-align:center;">
        <div style="
            display:inline-block; background:#050505; border:1px solid #00eaff;
            border-radius:10px; padding:12px 20px;
            box-shadow:0 0 12px rgba(0,234,255,0.4), 0 0 28px rgba(0,234,255,0.15), inset 0 0 12px rgba(0,0,0,0.8);
            margin-bottom:12px;
        ">
            <div style="
                font-size:1.1rem; font-weight:600; color:#00eaff; letter-spacing:0.12em;
                text-transform:uppercase; font-family:'Georgia',serif;
                text-shadow:0 0 8px rgba(0,234,255,0.8), 0 0 20px rgba(0,234,255,0.4);
                line-height:1.5;
            ">Devavrath<br>Sandeep's</div>
        </div>
    </div>
    <div class="sb-header" style="font-size:1.2rem;font-weight:700;letter-spacing:-0.02em;padding-top:0;">
        Vertex RAG
    </div>
    <div style="text-align:center;padding:4px 0 8px 0;">
        <span class="badge badge-claude">⚡ Powered by Claude</span>
    </div>
    """, unsafe_allow_html=True)

    for tid, tlabel in [
        ("config",      "Configuration"),
        ("generate",    "Generation"),
        ("files",       "Indexed Files"),
        ("vectorstore", "Vector Store"),
    ]:
        is_active = st.session_state.sidebar_tab == tid
        if st.button(tlabel, key=f"vtab_{tid}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.sidebar_tab = tid
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    status_txt = (
        "Indexing..."    if st.session_state.get("is_indexing")    else
        "Benchmarking..." if st.session_state.get("is_benchmarking") else
        "Querying..."    if st.session_state.get("is_querying")    else ""
    )
    if status_txt:
        st.caption(status_txt)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Interrupt", use_container_width=True, key="interrupt_global"):
            st.session_state["interrupt_requested"] = True
    with c2:
        if st.button("Reset", use_container_width=True, key="reset_global"):
            for k, v in [
                ("indexer", None), ("indexed_files", []), ("file_url_map", {}),
                ("index_stats", {}), ("chat_history", []), ("prepared_chunks", None),
                ("prepared_files", None), ("benchmark_results", None),
                ("benchmark_csv_path", None), ("benchmark_run_dir", None),
                ("last_index_error", None), ("interrupt_requested", False),
                ("file_assignments", {}), ("custom_paper_types", []),
            ]:
                st.session_state[k] = v
            _save_chat_history(); _save_indexed_files()
            st.success("App state cleared."); st.rerun()

    if st.session_state.get("interrupt_requested"):
        st.warning("Interrupt requested.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── CONFIG ─────────────────────────────────────────────────────────────────
    if st.session_state.sidebar_tab == "config":
        st.markdown('<div class="sb-section-label">Index algorithm</div>', unsafe_allow_html=True)
        st.session_state.cfg_index_type = st.selectbox(
            "index_alg", ("HNSW", "IVF_PQ", "DiskANN"),
            index=("HNSW", "IVF_PQ", "DiskANN").index(st.session_state.cfg_index_type),
            label_visibility="collapsed",
        )
        st.caption({
            "HNSW":    "Fast ANN, low latency, best recall.",
            "IVF_PQ":  "Compressed vectors, good at scale.",
            "DiskANN": "Disk-based ANN, large corpora.",
        }.get(st.session_state.cfg_index_type, ""))

        st.markdown('<div class="sb-section-label">Claude model</div>', unsafe_allow_html=True)
        st.session_state.cfg_model = st.selectbox(
            "model", list(GEN_MODELS.keys()),
            index=list(GEN_MODELS.keys()).index(st.session_state.cfg_model)
                  if st.session_state.cfg_model in GEN_MODELS else 0,
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-section-label">Chunking</div>', unsafe_allow_html=True)
        st.session_state.cfg_chunk_size    = st.slider("Chunk size (words)",   100, 800, st.session_state.cfg_chunk_size,    50)
        st.session_state.cfg_chunk_overlap = st.slider("Chunk overlap (words)", 0,  200, st.session_state.cfg_chunk_overlap, 10)

        st.markdown('<div class="sb-section-label">Retrieval</div>', unsafe_allow_html=True)
        st.session_state.cfg_top_k          = st.slider("Top-K results",         1, 12, st.session_state.cfg_top_k,          1)
        st.session_state.cfg_max_iterations = st.slider("Max reflection iters",  1,  5, st.session_state.cfg_max_iterations, 1)

        st.markdown('<div class="sb-section-label">Output language</div>', unsafe_allow_html=True)
        _LANGS = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Arabic", "Portuguese", "Hindi"]
        _cur   = st.session_state.cfg_language if st.session_state.cfg_language in _LANGS else "English"
        st.session_state.cfg_language = st.selectbox(
            "lang", _LANGS, index=_LANGS.index(_cur), label_visibility="collapsed",
        )

    # ── GENERATION ─────────────────────────────────────────────────────────────
    elif st.session_state.sidebar_tab == "generate":
        st.markdown('<div class="sb-section-label">Claude settings</div>', unsafe_allow_html=True)
        max_tokens_claude = st.slider("Max tokens", 256, 2048, 1024, 64)
        st.session_state["_claude_max_tokens"] = max_tokens_claude

    # ── INDEXED FILES ──────────────────────────────────────────────────────────
    elif st.session_state.sidebar_tab == "files":
        st.markdown('<div class="sb-section-label">Indexed files</div>', unsafe_allow_html=True)
        files = st.session_state.indexed_files
        if not files:
            st.caption("No files indexed yet.")
        else:
            st.markdown(
                f'<div class="sb-metrics">'
                f'<div class="sb-metric"><div class="val">{len(files)}</div><div class="lbl">Files</div></div>'
                f'<div class="sb-metric"><div class="val">{st.session_state["index_stats"].get("total_chunks",0)}</div><div class="lbl">Chunks</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for name, ptype in files:
                st.markdown(
                    f'<div class="file-pill">{name}<span class="ftype">{ptype}</span></div>',
                    unsafe_allow_html=True,
                )

    # ── VECTOR STORE ───────────────────────────────────────────────────────────
    elif st.session_state.sidebar_tab == "vectorstore":
        st.markdown('<div class="sb-section-label">Vector store</div>', unsafe_allow_html=True)
        s = st.session_state.get("index_stats", {})
        if s:
            for label, key in [("Index",  "index_type"), ("Chunks", "total_chunks"),
                                ("Dim",   "embed_dim"),   ("Model",  "gen_model")]:
                val = s.get(key, "--")
                st.markdown(
                    f'<div class="sb-metric" style="margin:3px 16px;">'
                    f'<div class="val">{val}</div><div class="lbl">{label}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Not indexed yet.")

        st.markdown('<div class="sb-section-label">NPZ snapshot</div>', unsafe_allow_html=True)
        npz_path = st.text_input("NPZ path", value="./rag_snapshot.npz", label_visibility="collapsed")

        c1e, c2i = st.columns(2)
        with c1e:
            if st.button("Export", use_container_width=True):
                try:
                    n = st.session_state.indexer.export_to_npz(npz_path)
                    st.success(f"Exported {n} chunks → {npz_path}")
                except Exception as e:
                    st.error(str(e))
        with c2i:
            append_npz = st.checkbox("Append", value=False)
            if st.button("Import", use_container_width=True):
                try:
                    meta = st.session_state.indexer.import_from_npz(npz_path, append=append_npz)
                    st.session_state["index_stats"] = {
                        "status": "ok", "total_chunks": meta["n_chunks"],
                        "total_files": len(st.session_state.indexed_files),
                        "index_type": meta["index_type"], "embed_model": meta["embed_model"],
                        "embed_dim": 1024, "gen_model": st.session_state.cfg_model, "paper_types": [],
                    }
                    if meta.get("indexed_files"):
                        st.session_state.indexed_files = meta["indexed_files"]
                    if meta.get("file_url_map"):
                        st.session_state.file_url_map  = meta["file_url_map"]
                    _save_indexed_files()
                    st.success(f"Imported {meta['n_chunks']} chunks.")
                except Exception as e:
                    st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
# Read sidebar config
model_choice    = st.session_state.cfg_model
index_type      = st.session_state.cfg_index_type
chunk_size      = st.session_state.cfg_chunk_size
chunk_overlap   = st.session_state.cfg_chunk_overlap
input_language  = st.session_state.cfg_language
top_k           = st.session_state.cfg_top_k
max_iterations  = st.session_state.cfg_max_iterations
max_tokens_claude = st.session_state.get("_claude_max_tokens", 1024)

tab_index, tab_query, tab_stats_tab = st.tabs(["Upload & Index", "Query", "Stats & Benchmark"])


# ── TAB 1: Upload & Index ──────────────────────────────────────────────────────
with tab_index:
    st.markdown(
        '<div class="page-heading"><h1>Upload & Index</h1>'
        '<p>BGE Embeddings · Milvus · Claude-ready pipeline</p></div>',
        unsafe_allow_html=True,
    )

    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed",
        )
        append_mode       = st.checkbox("Append to existing index", value=False)
        cache_for_benchmark = st.checkbox("Cache chunks for benchmarking", value=True)
        show_chunks       = st.checkbox("Show chunk preview after indexing", value=False)
        submitted         = st.form_submit_button("Index PDFs", type="primary", use_container_width=True)

    if submitted and uploaded_files:
        # Paper type + URL assignment per file
        all_paper_types = PAPER_TYPES + st.session_state.get("custom_paper_types", [])
        st.markdown("**Assign paper types and optional URLs:**")
        file_paper_types, file_urls = {}, {}

        for uf in uploaded_files:
            col_name, col_type, col_url = st.columns([2, 1, 2])
            with col_name:
                st.markdown(f'<div style="padding-top:8px;font-size:0.85rem;color:#bbb;">{uf.name}</div>', unsafe_allow_html=True)
            with col_type:
                file_paper_types[uf.name] = st.selectbox(
                    f"type_{uf.name}", all_paper_types, index=0, label_visibility="collapsed",
                )
            with col_url:
                file_urls[uf.name] = st.text_input(
                    f"url_{uf.name}", placeholder="Optional URL", label_visibility="collapsed",
                )

        if st.button("Confirm & Index", type="primary", use_container_width=True):
            st.session_state["is_indexing"] = True
            st.session_state["interrupt_requested"] = False

            try:
                tmp_dir = _ensure_tmp_upload_dir()
                saved   = []
                for uf in uploaded_files:
                    dest = tmp_dir / uf.name
                    dest.write_bytes(uf.read())
                    saved.append((dest, file_paper_types.get(uf.name, "Other"), file_urls.get(uf.name, "")))

                status_box = st.empty()
                bar        = st.progress(0, text="Starting...")
                log_box    = st.empty()
                logs       = []

                def log(msg):
                    logs.append(msg)
                    log_box.markdown(
                        '<div style="background:#2a2a2a;border:1px solid #333;border-radius:8px;'
                        'padding:12px 16px;font-size:0.8rem;color:#bbb;line-height:1.7;">'
                        + "".join(f"· {l}<br>" for l in logs[-10:])
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                def tick(pct, label):
                    bar.progress(min(float(pct), 1.0), text=f"{label} — {int(min(pct,1.0)*100)}%")
                    status_box.markdown(
                        f'<div style="font-size:0.82rem;color:#9b9b9b;padding:4px 0 8px 0;">{label}</div>',
                        unsafe_allow_html=True,
                    )

                tick(0.02, "Creating indexer")
                indexer_obj = PDFIndexer(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model_choice,
                    index_type=index_type, collection_name="papers_rag_interactive",
                    drop_old_collection=(not append_mode),
                )
                log(f"Indexer ready [{index_type}] — Claude model: {model_choice}")

                all_chunks = []
                n_files    = len(saved)

                for fi, (pdf_path, ptype, url) in enumerate(saved):
                    _check_interrupt()
                    p0 = 0.05 + (fi / max(n_files, 1)) * 0.25
                    p1 = 0.05 + ((fi+1) / max(n_files, 1)) * 0.25
                    tick(p0, f"Extracting {pdf_path.name}")
                    chunks = indexer_obj.extract_and_chunk(str(pdf_path), paper_type=ptype)
                    for c in chunks: c["url"] = url
                    all_chunks.extend(chunks)
                    tick(p1, f"{pdf_path.name} — {len(chunks)} chunks")
                    log(f"{pdf_path.name} [{ptype}] — {len(chunks)} chunks")

                total = len(all_chunks)
                tick(0.33, f"Extraction done — {total} chunks")

                log("Embedding chunks locally (BGE)...")
                _check_interrupt()
                embedder      = indexer_obj._get_embedder()
                BATCH_SIZE    = 64
                total_batches = math.ceil(total / BATCH_SIZE)

                for bi in range(0, total, BATCH_SIZE):
                    _check_interrupt()
                    batch_num   = bi // BATCH_SIZE + 1
                    batch_texts = [
                        "Represent this passage for retrieval: " + all_chunks[bi+j]["text"]
                        for j in range(min(BATCH_SIZE, total-bi))
                    ]
                    embeddings = embedder.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
                    for j, emb in enumerate(embeddings):
                        all_chunks[bi+j]["embedding"] = emb.tolist()
                    pct = 0.33 + (min(bi+BATCH_SIZE, total) / max(total, 1)) * 0.42
                    tick(pct, f"Embedding batch {batch_num}/{total_batches}")
                    log(f"Batch {batch_num}/{total_batches} done")

                tick(0.75, f"{total} embeddings done")
                log("All embeddings complete")

                if cache_for_benchmark:
                    st.session_state.prepared_chunks = all_chunks
                    st.session_state.prepared_files  = saved

                _check_interrupt()
                tick(0.80, f"Building {index_type} index...")
                log(f"Building {index_type} index...")

                if append_mode and st.session_state.indexer:
                    st.session_state.indexer.add_to_index(all_chunks)
                    st.session_state.indexer.set_output_language(input_language)
                else:
                    indexer_obj.build_index(all_chunks)
                    indexer_obj.set_output_language(input_language)
                    st.session_state.indexer = indexer_obj

                tick(1.0, "Indexing complete ✓")

                new_entries = [(p.name, pt) for p, pt, _ in saved]
                url_map     = st.session_state.file_url_map or {}
                for p, _, u in saved:
                    if u: url_map[p.name] = u
                st.session_state.file_url_map = url_map

                if append_mode: st.session_state.indexed_files.extend(new_entries)
                else:           st.session_state.indexed_files = new_entries

                st.session_state.indexer._indexed_files = new_entries
                st.session_state.indexer._file_url_map  = url_map
                _save_indexed_files()

                st.session_state.index_stats = {
                    "status": "ok", "total_chunks": total,
                    "total_files": len(st.session_state.indexed_files),
                    "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                    "embed_dim": 1024, "embed_model": EMBED_MODEL,
                    "index_type": index_type, "gen_model": model_choice,
                    "paper_types": list({pt for _, pt in st.session_state.indexed_files}),
                }
                st.success(f"**{n_files} file(s)** — **{total} chunks** — **{index_type}** index built.")

                if show_chunks and all_chunks:
                    st.markdown("**Chunk preview (first 5)**")
                    for c in all_chunks[:5]:
                        url       = st.session_state.file_url_map.get(c["source"], "")
                        url_badge = " <span class='badge badge-url'>URL</span>" if url else ""
                        st.markdown(
                            f'<div class="chunk-preview">'
                            f'<span class="badge">{c["source"]}</span>'
                            f'<span class="badge">p{c["page"]}</span>'
                            f'<span class="badge badge-type">{c["paper_type"]}</span>'
                            f'<span class="badge">#{c["chunk_id"]}</span>{url_badge}<br><br>'
                            f'{c["text"][:480]}{"..." if len(c["text"])>480 else ""}</div>',
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.session_state["last_index_error"] = str(e)
                _safe_set_failed_stats(index_type=index_type, model_choice=model_choice)
                if "Interrupted by user" in str(e): st.warning("Indexing interrupted by user.")
                else: st.error(str(e)); st.exception(e)
            finally:
                st.session_state["is_indexing"] = False


# ── TAB 2: Query ───────────────────────────────────────────────────────────────
with tab_query:
    if not st.session_state.indexer:
        st.markdown("""
        <div class="page-heading">
            <h1>PDF RAG Studio</h1>
            <p>Claude · BGE Embeddings · Milvus · LangGraph</p>
        </div>
        <div style="text-align:center;color:#555;font-size:0.88rem;margin-top:8px;">
            Index PDFs first using the Upload &amp; Index tab.
        </div>""", unsafe_allow_html=True)
    else:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="page-heading">
                <h1>PDF RAG Studio</h1>
                <p>Claude · BGE Embeddings · Milvus · LangGraph</p>
            </div>""", unsafe_allow_html=True)

        # Filter bar
        available_types = sorted({pt for _, pt in st.session_state.indexed_files})
        filter_options  = ["All paper types"] + list(available_types)
        col_f, col_info = st.columns([3, 1])
        with col_f:
            selected_filter = st.selectbox(
                "Filter by paper type", filter_options, index=0, label_visibility="collapsed",
            )
        with col_info:
            _s2 = st.session_state.get("index_stats", {})
            st.markdown(
                f'<div style="padding-top:6px;font-size:0.72rem;color:#555;">'
                f'{_s2.get("total_files",0)} files · {_s2.get("total_chunks",0)} chunks · {index_type}</div>',
                unsafe_allow_html=True,
            )
        paper_filter = None if selected_filter == "All paper types" else selected_filter

        # Render full chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;margin:16px 0;">'
                    f'<div style="width:32px;height:32px;border-radius:50%;background:#e53935;'
                    f'display:flex;align-items:center;justify-content:center;font-size:0.75rem;'
                    f'font-weight:700;color:#fff;flex-shrink:0;">You</div>'
                    f'<div style="color:#ececec;font-size:0.95rem;line-height:1.65;padding-top:4px;">'
                    f'{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;margin:16px 0;">'
                    f'<div style="width:32px;height:32px;border-radius:50%;background:#2a2a2a;border:1px solid #333;'
                    f'display:flex;align-items:center;justify-content:center;font-size:0.7rem;'
                    f'font-weight:700;color:#f59e0b;flex-shrink:0;">CL</div>'
                    f'<div style="color:#ececec;font-size:0.95rem;line-height:1.65;padding-top:4px;width:100%;">',
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])
                if msg.get("trace"):
                    with st.expander("🔍 View Claude Reasoning Trace", expanded=False):
                        _render_reasoning_trace(msg["trace"])
                if msg.get("recs"):
                    _render_recommendations(msg["recs"])
                if msg.get("sources"):
                    _render_sources(msg["sources"])
                st.markdown('</div></div>', unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask anything about your PDFs..."):
            st.session_state["interrupt_requested"] = False
            st.session_state["is_querying"]         = True
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            _save_chat_history()

            with st.container():
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;margin:16px 0;">'
                    f'<div style="width:32px;height:32px;border-radius:50%;background:#e53935;'
                    f'display:flex;align-items:center;justify-content:center;font-size:0.75rem;'
                    f'font-weight:700;color:#fff;flex-shrink:0;">You</div>'
                    f'<div style="color:#ececec;font-size:0.95rem;padding-top:4px;">{prompt}</div></div>',
                    unsafe_allow_html=True,
                )

            answer, sources, recs, trace_events = "", [], [], []

            with st.container():
                try:
                    if st.session_state.get("interrupt_requested", False):
                        raise RuntimeError("Interrupted by user.")

                    with st.spinner("Claude is reasoning over your documents…"):
                        answer, sources, trace_events = st.session_state.indexer.query(
                            prompt, top_k=top_k, model=model_choice,
                            paper_filter=paper_filter, output_language=input_language,
                            max_iterations=max_iterations,
                        )

                    if st.session_state.get("interrupt_requested", False):
                        raise RuntimeError("Interrupted by user.")

                    # Render Claude avatar + answer
                    st.markdown(
                        f'<div style="display:flex;align-items:flex-start;gap:10px;margin:16px 0;">'
                        f'<div style="width:32px;height:32px;border-radius:50%;background:#2a2a2a;border:1px solid #333;'
                        f'display:flex;align-items:center;justify-content:center;font-size:0.7rem;'
                        f'font-weight:700;color:#f59e0b;flex-shrink:0;">CL</div>'
                        f'<div style="width:100%;">',
                        unsafe_allow_html=True,
                    )
                    st.markdown(answer)

                    # ── CLAUDE REASONING TRACE (the wow moment) ────────────────
                    if trace_events:
                        with st.expander("🔍 View Claude Reasoning Trace", expanded=True):
                            _render_reasoning_trace(trace_events)

                    recs = _unique_papers_from_sources(
                        sources=sources,
                        file_url_map=st.session_state.get("file_url_map", {}),
                        k=2,
                    )
                    with st.spinner("Verifying paper URLs with Claude…"):
                        recs, url_trace = _reflect_and_fix_urls(recs)

                    _render_recommendations(recs, url_trace=url_trace)
                    _render_sources(sources)
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    if "Interrupted by user" in str(e):
                        st.warning("Query interrupted by user.")
                        answer, sources = "Query interrupted by user.", []
                    else:
                        st.error(f"Query failed: {e}")
                        answer, sources = f"Query failed: {e}", []
                finally:
                    st.session_state["is_querying"] = False

            if isinstance(answer, str) and "interrupted by user" not in answer.lower():
                st.session_state.chat_history.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "recs": recs, "trace": trace_events,
                })
                _save_chat_history()

        if st.session_state.chat_history:
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                _save_chat_history()
                st.rerun()


# ── TAB 3: Stats & Benchmark ───────────────────────────────────────────────────
with tab_stats_tab:
    st.markdown(
        '<div class="page-heading"><h1>Stats & Benchmark</h1>'
        '<p>Index performance comparison — HNSW vs IVF_PQ vs DiskANN</p></div>',
        unsafe_allow_html=True,
    )

    s = st.session_state.get("index_stats", {}) or {}

    if not s or "total_files" not in s:
        st.info("No successful indexing stats available yet.")
        if st.session_state.get("last_index_error"):
            st.error(f"Last error: {st.session_state['last_index_error']}")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Files",      s.get("total_files", 0))
        c2.metric("Chunks",     s.get("total_chunks", 0))
        c3.metric("Index",      s.get("index_type", "--"))
        c4.metric("Embed dim",  s.get("embed_dim", "--"))
        c5.metric("Chunk size", f'{s.get("chunk_size","--")}w')
        c6.metric("LLM",        s.get("gen_model", "--"))

        st.divider()
        st.markdown("**Embed model**"); st.code(s.get("embed_model", "--"))

        st.markdown("**Paper types**")
        for pt in sorted(s.get("paper_types", [])):
            st.markdown(f'<span class="badge badge-type">{pt}</span>', unsafe_allow_html=True)

        st.markdown("**Indexed files**")
        for name, ptype in st.session_state.get("indexed_files", []):
            url = st.session_state.file_url_map.get(name, "")
            url_badge = f'  <span class="badge badge-url">URL</span> {url}' if url else ""
            st.markdown(
                f'`{name}`  <span class="badge badge-type">{ptype}</span>{url_badge}',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("**Benchmark — HNSW vs IVF_PQ vs DiskANN**")

    if st.session_state.prepared_chunks is None:
        st.info("Index once with caching enabled to unlock benchmarking.")
    else:
        st.caption("Benchmarks reuse cached chunks to keep comparisons fair.")
        sample_query = st.text_input(
            "Sample benchmark query",
            value="Summarize the main contribution of the papers and cite sources.",
        )
        bench_filter       = st.selectbox("Benchmark filter (optional)", ["None"] + PAPER_TYPES, index=0)
        bench_paper_filter = None if bench_filter == "None" else bench_filter

        colx, coly = st.columns(2)
        with colx:
            run_bench = st.button("Run benchmark for all methods", type="primary", use_container_width=True)
        with coly:
            if st.button("Clear cached benchmark prep", use_container_width=True):
                for k in ["prepared_chunks", "prepared_files", "benchmark_results",
                          "benchmark_csv_path", "benchmark_run_dir"]:
                    st.session_state[k] = None
                st.success("Cleared."); st.rerun()

        if run_bench:
            st.session_state["interrupt_requested"] = False
            st.session_state["is_benchmarking"]     = True
            try:
                _check_interrupt()
                with st.spinner("Running benchmark across all index methods…"):
                    results, csv_path, run_dir, shared_summary = _run_index_benchmark_all_methods(
                        embedded_chunks=st.session_state.prepared_chunks,
                        model_choice=model_choice, top_k=top_k,
                        input_language=input_language, sample_query=sample_query,
                        paper_filter=bench_paper_filter, max_iterations=max_iterations,
                    )
                st.session_state.benchmark_results  = results
                st.session_state.benchmark_csv_path = csv_path
                st.session_state.benchmark_run_dir  = run_dir
                st.success("Benchmark complete"); st.json(shared_summary)
            except Exception as e:
                if "Interrupted by user" in str(e): st.warning("Benchmark interrupted.")
                else: st.error(f"Benchmark failed: {e}"); st.exception(e)
            finally:
                st.session_state["is_benchmarking"] = False

        if st.session_state.get("benchmark_results"):
            st.markdown("**Results**")
            st.dataframe(st.session_state["benchmark_results"], use_container_width=True)
            csv_path = st.session_state.get("benchmark_csv_path")
            if csv_path and Path(csv_path).exists():
                st.caption(f"Saved: `{csv_path}`")
                st.download_button(
                    "Download benchmark CSV", data=Path(csv_path).read_bytes(),
                    file_name=Path(csv_path).name, mime="text/csv", use_container_width=True,
                )
            if st.session_state.get("benchmark_run_dir"):
                st.caption(f"Run folder: `{st.session_state['benchmark_run_dir']}`")
