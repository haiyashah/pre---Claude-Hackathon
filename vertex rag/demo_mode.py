"""
demo_mode.py — Vertex RAG DEMO MODE (no Milvus/Docker required)

A fully self-contained Streamlit app that demonstrates the full
Claude-powered reasoning pipeline using in-memory vector search.

Run:
    streamlit run demo_mode.py

This is your HACKATHON FALLBACK — works anywhere, instantly,
with just: pip install streamlit anthropic sentence-transformers torch numpy
"""

import streamlit as st
import os
import json
import time
import math
import numpy as np
from typing import List, Dict, Tuple, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

st.set_page_config(
    page_title="Vertex RAG — Research Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.stApp { background: #0f0f0f; color: #e8e8e8; }
.main  { background: #0f0f0f; }
#MainMenu, footer, header { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

section[data-testid="stSidebar"] {
    min-width: 260px !important; max-width: 260px !important;
    background: #141414 !important; border-right: 1px solid #222 !important;
}
section[data-testid="stSidebar"] > div { padding: 0 12px !important; }

.block-container { max-width: 1000px !important; margin: 0 auto !important; padding: 0 40px 120px 40px !important; }

/* Hero */
.hero { text-align:center; padding: 48px 0 36px; }
.hero h1 { font-size: 2.4rem; font-weight: 700; color: #fff; letter-spacing:-0.04em; margin:0; }
.hero h1 span { color: #ff4444; }
.hero p { color: #666; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 8px; }

/* Cards */
.card { background: #1a1a1a; border: 1px solid #252525; border-radius: 12px; padding: 20px 24px; margin: 12px 0; }
.card-title { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #555; margin-bottom: 8px; }
.card-value { font-size: 0.95rem; color: #e8e8e8; line-height: 1.6; }

/* Trace step */
.trace-step { border-left: 2px solid #333; padding: 10px 16px; margin: 6px 0; border-radius: 0 8px 8px 0; }
.trace-step.pass { border-color: #22c55e; background: rgba(34,197,94,0.05); }
.trace-step.revised { border-color: #f59e0b; background: rgba(245,158,11,0.05); }
.trace-step.ok { border-color: #3b82f6; background: rgba(59,130,246,0.05); }
.trace-step.generated { border-color: #8b5cf6; background: rgba(139,92,246,0.05); }
.trace-step-label { font-size: 0.72rem; font-weight: 600; color: #888; margin-bottom: 4px; }
.trace-step-body { font-size: 0.82rem; color: #bbb; line-height: 1.5; }

/* Source pill */
.source-pill { display: inline-block; background: #1e2130; border: 1px solid #2a3060; 
    border-radius: 6px; padding: 4px 10px; font-size: 0.72rem; color: #8899ff; margin: 3px; }

/* Answer block */
.answer-block { background: #151515; border: 1px solid #222; border-radius: 12px; 
    padding: 20px 24px; margin: 16px 0; }
.answer-meta { font-size: 0.68rem; color: #555; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.08em; }

/* Status badge */
.badge-pass { background: rgba(34,197,94,0.15); color: #22c55e; border-radius: 4px; 
    padding: 2px 8px; font-size: 0.65rem; font-weight: 700; }
.badge-fail { background: rgba(239,68,68,0.15); color: #ef4444; border-radius: 4px; 
    padding: 2px 8px; font-size: 0.65rem; font-weight: 700; }

/* Sidebar */
.sb-logo { padding: 24px 16px 16px; border-bottom: 1px solid #222; text-align: center; }
.sb-logo h2 { font-size: 1.1rem; font-weight: 700; color: #fff; margin: 0; }
.sb-logo p  { font-size: 0.65rem; color: #555; margin: 2px 0 0; }
.sb-section { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; 
    letter-spacing: 0.1em; color: #444; padding: 14px 16px 6px; }
.sb-doc { background: #1a1a1a; border: 1px solid #222; border-radius: 6px; 
    padding: 6px 10px; margin: 3px 8px; font-size: 0.72rem; color: #aaa; 
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sb-type { display: inline-block; background: #2a1a40; color: #a78bfa; 
    font-size: 0.58rem; padding: 1px 5px; border-radius: 3px; margin-left: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Sample knowledge base (built-in, no PDF upload needed for demo) ──────────
DEMO_DOCUMENTS = [
    {
        "source": "attention_is_all_you_need.pdf",
        "paper_type": "AI / ML",
        "url": "https://arxiv.org/abs/1706.03762",
        "chunks": [
            "The Transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), relies entirely on attention mechanisms, dispensing with recurrence and convolutions. It achieves state-of-the-art results on translation tasks.",
            "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Each head learns different aspects of the input relationships.",
            "The encoder consists of 6 identical layers, each with two sub-layers: multi-head self-attention and position-wise feed-forward network. Residual connections and layer normalization are applied around each sub-layer.",
            "On WMT 2014 English-to-German translation, the Transformer achieves 28.4 BLEU, outperforming all previously published models including ensembles. Training takes 3.5 days on 8 P100 GPUs.",
            "Positional encoding is added to input embeddings to inject sequence order information, since the architecture contains no recurrence or convolution. Sine and cosine functions of different frequencies are used.",
        ]
    },
    {
        "source": "bert_pretraining_2018.pdf",
        "paper_type": "AI / ML",
        "url": "https://arxiv.org/abs/1810.04805",
        "chunks": [
            "BERT (Bidirectional Encoder Representations from Transformers) pre-trains deep bidirectional representations by jointly conditioning on both left and right context in all layers, unlike previous models which were unidirectional.",
            "BERT uses two pre-training tasks: Masked Language Model (MLM), which randomly masks tokens and predicts them, and Next Sentence Prediction (NSP), which predicts whether two sentences are consecutive.",
            "Fine-tuning BERT is straightforward: for each downstream task, task-specific inputs and outputs are plugged in and all parameters are fine-tuned end-to-end, achieving new SOTA on 11 NLP benchmarks.",
            "BERT-large achieves 93.2 F1 on SQuAD v1.1, 86.7% on MultiNLI, and 80.5 on GLUE benchmark — representing significant absolute improvements over previous state-of-the-art systems.",
            "The key innovation is bidirectionality: previous models like GPT trained left-to-right, limiting context. BERT's masked training objective enables truly bidirectional representations.",
        ]
    },
    {
        "source": "llm_security_survey_2024.pdf",
        "paper_type": "Security",
        "url": "https://arxiv.org/abs/2402.00888",
        "chunks": [
            "Prompt injection attacks manipulate LLM behavior by embedding adversarial instructions in inputs. Direct attacks override system prompts while indirect attacks embed malicious instructions in retrieved external content.",
            "Jailbreaking techniques bypass LLM safety filters through role-playing scenarios, token manipulation, and multi-step reasoning that gradually guides models toward producing harmful content.",
            "Training data poisoning attacks inject malicious examples to introduce backdoors or biases before deployment, making them particularly hard to detect through standard evaluation methods.",
            "Defense mechanisms for LLM security include: input filtering and sanitization, adversarial training, constitutional AI approaches, output monitoring, and differential privacy during training.",
            "As LLMs integrate into critical applications, standardized security evaluation benchmarks and defense frameworks are urgently needed. Current defenses often reduce utility while not fully eliminating attack surfaces.",
        ]
    },
    {
        "source": "rag_survey_2024.pdf",
        "paper_type": "AI / ML",
        "url": "https://arxiv.org/abs/2312.10997",
        "chunks": [
            "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant documents at inference time, grounding responses in external knowledge and reducing hallucination without expensive retraining.",
            "RAG systems consist of three components: a retriever (dense or sparse), a knowledge base (vector store), and a generator (LLM). Dense retrieval using transformer embeddings typically outperforms BM25 on complex queries.",
            "Advanced RAG techniques include query rewriting, iterative retrieval, and self-reflection loops where the LLM evaluates and improves its own answers — significantly boosting factual accuracy.",
            "Milvus, Pinecone, and Weaviate are leading vector databases for production RAG. Index types like HNSW offer high recall with fast approximate nearest-neighbor search.",
            "Evaluation of RAG systems requires measuring both retrieval quality (recall@k, MRR) and generation quality (faithfulness, answer relevance). RAGAS is a popular framework for automated evaluation.",
        ]
    },
]

# ─── In-memory vector search (no Milvus required) ─────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast

@st.cache_resource(show_spinner="Building knowledge base…")
def build_index():
    embedder = load_embedder()
    all_chunks = []
    for doc in DEMO_DOCUMENTS:
        for chunk in doc["chunks"]:
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "paper_type": doc["paper_type"],
                "url": doc["url"],
            })
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    return all_chunks, embeddings

def retrieve(query: str, top_k: int = 4, paper_filter: Optional[str] = None) -> List[Dict]:
    embedder = load_embedder()
    chunks, embeddings = build_index()
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(embeddings, q_emb)
    idxs = np.argsort(scores)[::-1]
    results = []
    for i in idxs:
        c = chunks[i]
        if paper_filter and c["paper_type"] != paper_filter:
            continue
        results.append({**c, "score": float(scores[i])})
        if len(results) >= top_k:
            break
    return results

# ─── Claude pipeline ──────────────────────────────────────────────────────────
import anthropic

def _claude(prompt: str, system: str = "You are a helpful AI assistant.", max_tokens: int = 1200) -> str:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()

def run_rag_pipeline(question: str, top_k: int = 4, paper_filter: Optional[str] = None, max_iter: int = 2):
    """Full generate → reflect → revise pipeline with trace."""
    trace = []

    # Step 1: Retrieve
    docs = retrieve(question, top_k=top_k, paper_filter=paper_filter)
    context = "\n\n".join([f"[{d['source']}] {d['text']}" for d in docs])
    papers_list = list({d["source"]: d["url"] for d in docs}.items())
    papers_block = "\n".join([f"• {src} — {url}" for src, url in papers_list])
    trace.append({"label": "🔍 Retrieved Chunks", "status": "ok",
                  "content": f"Found {len(docs)} relevant chunks from {len(papers_list)} papers."})

    # Step 2: Generate
    gen_prompt = f"""Answer the following question using ONLY the context below.
Be concise (3-5 sentences), grounded, and cite paper filenames.

Question: {question}

Context:
{context}

Available papers:
{papers_block}

Format your answer as:
<your answer here>

Recommended Research Papers:
• <paper title/filename>. URL: <url>
• <paper title/filename>. URL: <url>
"""
    answer = _claude(gen_prompt, system="You are a precise research assistant. Answer only from provided context. Never invent facts.")
    trace.append({"label": "✍️ Initial Answer Generated", "status": "generated",
                  "content": answer[:400] + ("…" if len(answer) > 400 else "")})

    # Step 3: Reflect + Revise loop
    for i in range(max_iter):
        reflect_prompt = f"""Review this RAG answer for quality.

Question: {question}
Answer: {answer}
Context used: {context[:800]}

Return EXACTLY:
#Issues:
- ...
#Verdict: PASS or FAIL
"""
        reflection = _claude(reflect_prompt, system="You are a strict answer quality reviewer. Be rigorous.")
        verdict = "PASS" if "PASS" in reflection.upper() else "FAIL"
        badge = "pass" if verdict == "PASS" else "fail"
        trace.append({"label": f"🪞 Reflection (iter {i+1}) — {verdict}", "status": badge,
                      "content": reflection[:400]})

        if verdict == "PASS":
            break

        # Revise
        revise_prompt = f"""Rewrite this answer to fix the issues. Stay grounded in context.

Question: {question}
Context: {context}
Issues found: {reflection}
Previous answer: {answer}

Format:
<your revised answer>

Recommended Research Papers:
• <title>. URL: <url>
"""
        answer = _claude(revise_prompt, system="You are a careful answer rewriter. Fix issues, stay factual.")
        trace.append({"label": f"🔧 Revised Answer (iter {i+1})", "status": "revised",
                      "content": answer[:400] + ("…" if len(answer) > 400 else "")})

    return answer, docs, trace

# ─── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <h2>🔬 Vertex RAG</h2>
        <p>Research Intelligence · Demo Mode</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Knowledge Base</div>', unsafe_allow_html=True)
    for doc in DEMO_DOCUMENTS:
        ptype = doc["paper_type"]
        color = "#a78bfa" if "AI" in ptype else "#f87171"
        st.markdown(
            f'<div class="sb-doc">{doc["source"]}<span class="sb-type" style="color:{color}">{ptype}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sb-section">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, 4)
    paper_filter_opt = st.selectbox("Filter by type", ["All", "AI / ML", "Security"])
    paper_filter = None if paper_filter_opt == "All" else paper_filter_opt
    max_iter = st.slider("Max reflect/revise iterations", 1, 3, 2)

    st.divider()
    st.caption("**Demo mode** — in-memory search, no Milvus needed.")
    st.caption("Uses `all-MiniLM-L6-v2` embeddings + Claude Sonnet for reasoning.")

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Vertex <span>RAG</span></h1>
    <p>Research Intelligence · Powered by Claude · 4 papers loaded</p>
</div>
""", unsafe_allow_html=True)

# Suggested queries
st.markdown("**💡 Try asking:**")
cols = st.columns(3)
suggestions = [
    "What is multi-head attention and why is it important?",
    "How does BERT differ from GPT in pretraining?",
    "What are the main LLM security attack vectors?",
    "How does RAG reduce hallucination in LLMs?",
    "Compare BERT and Transformer architectures",
    "What defense mechanisms exist for prompt injection?",
]
for i, col in enumerate(cols):
    for j in range(2):
        idx = i * 2 + j
        if col.button(suggestions[idx], key=f"sug_{idx}", use_container_width=True):
            st.session_state["pending_query"] = suggestions[idx]
            st.rerun()

st.divider()

# Chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div style="background:#1e1e1e;border:1px solid #2a2a2a;border-radius:10px;padding:12px 16px;margin:8px 0;font-size:0.93rem;">'
                    f'<span style="color:#666;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">You</span><br>'
                    f'{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="answer-block"><div class="answer-meta">Claude · Research Assistant</div>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔍 Claude Reasoning Trace", expanded=False):
                for step in msg["trace"]:
                    css = step.get("status", "ok")
                    st.markdown(
                        f'<div class="trace-step {css}">'
                        f'<div class="trace-step-label">{step["label"]}</div>'
                        f'<div class="trace-step-body">{step["content"]}</div>'
                        f'</div>', unsafe_allow_html=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)

# Query input
pending = st.session_state.pop("pending_query", None)
query = st.chat_input("Ask a question about the research papers…")
if pending:
    query = pending

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("Claude is reasoning over the knowledge base…"):
        try:
            answer, sources, trace = run_rag_pipeline(query, top_k=top_k, paper_filter=paper_filter, max_iter=max_iter)
            st.session_state.chat_history.append({
                "role": "assistant", "content": answer,
                "sources": sources, "trace": trace,
            })
        except Exception as e:
            st.error(f"Error: {e}")
            if "ANTHROPIC_API_KEY" in str(e) or "auth" in str(e).lower():
                st.info("💡 Set your API key: `export ANTHROPIC_API_KEY=sk-ant-...`")

    st.rerun()
