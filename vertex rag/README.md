
# 1. go into the folder
cd vertex rag  

# 2. install dependencies
pip install -r requirements.txt

# 3. set your Claude API key
export ANTHROPIC_API_KEY="your_key_here"   # Mac/Linux
setx ANTHROPIC_API_KEY "your_key_here"     # Windows

# 4. run the app
streamlit run demo_mode.py



We need to also create 2 more folders
rag_index and sample_docs



# 🔬 Vertex RAG — Powered by Claude

> **Research Intelligence Platform** — Upload academic papers, ask questions, get Claude-powered answers with multi-step reasoning, reflection, and revision.

Built for the **Claude Builder Club × CMUAI Hackathon** · April 18, 2025

---

## 🚀 What It Does

**Vertex RAG** is an AI research assistant that:

1. **Ingests** PDFs (research papers, reports, articles)
2. **Indexes** them into Milvus vector DB with BAAI/bge-large embeddings
3. **Answers** questions using a LangGraph pipeline powered by Claude:
   - `translate query → retrieve chunks → generate → reflect → revise → output`
4. **Shows** a full reasoning trace (what Claude did step by step)
5. **Benchmarks** HNSW vs IVF_PQ vs DiskANN index performance

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[LangGraph Pipeline]
    │
    ├─ translate_query   ← Claude Haiku (multilingual support)
    ├─ retrieve          ← Milvus cosine similarity search
    ├─ generate          ← Claude Sonnet (grounded answer)
    ├─ reflect           ← Claude Sonnet (quality check)
    ├─ revise            ← Claude Sonnet (fix issues)  ─┐
    └─ translate_output  ← Claude Haiku (output lang)  ◄─┘ (loop up to N times)
```

---

## 🧠 Problem Statement We're Solving

**"Researchers waste hours manually cross-referencing papers to answer specific questions."**

Vertex RAG turns a pile of PDFs into a queryable knowledge base with:
- Grounded, citation-backed answers
- Multi-step Claude reasoning (not just summarization)
- Visible AI reasoning trace (build trust)
- Paper recommendations with URL verification

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- Docker (for Milvus)
- Anthropic API key

### 1. Start Milvus (vector database)

```bash
# Pull and run Milvus standalone
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.4.0 \
  milvus run standalone
```

Or use the official docker-compose:
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

### 2. Install Python deps

```bash
pip install -r requirements.txt
```

### 3. Set API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🎮 Demo Flow (90-second walkthrough)

1. **Upload tab** → drag in a research PDF (e.g., an AI/ML paper)
2. Select paper type → click **Index**
3. **Chat tab** → ask: *"What is the main contribution of this paper?"*
4. Watch Claude reason, reflect, and revise in the **Reasoning Trace**
5. See source citations + recommended papers with verified URLs
6. **Stats tab** → run benchmark to compare HNSW vs IVF_PQ vs DiskANN

---

## 🔑 Key Claude Integration Points

| Step | Claude Model | What it does |
|------|-------------|--------------|
| Query translation | Claude Haiku | Translates non-English queries |
| Answer generation | Claude Sonnet | Grounded RAG answer |
| Reflection | Claude Sonnet | Quality gate — PASS/FAIL |
| Revision | Claude Sonnet | Fixes grounding + format |
| URL verification | Claude Sonnet | Verifies paper URLs |
| Output translation | Claude Haiku | Translates answer to user's language |

---

## 📁 Project Structure

```
vertex_rag/
├── app.py              ← Streamlit UI (3 tabs: Upload, Chat, Stats)
├── indexer.py          ← PDFIndexer class (Milvus + LangGraph + Claude)
├── requirements.txt    ← Python dependencies
├── .streamlit/
│   └── config.toml     ← Dark theme config
├── sample_docs/        ← Drop sample PDFs here for demo
└── README.md
```

---

## 🏆 Why This Wins

- **Claude is essential** — not just called once, but drives a multi-step reasoning loop
- **Visible AI reasoning** — judges can see Claude thinking, reflecting, improving
- **Real utility** — researchers/students can use this today
- **Index benchmarking** — technically impressive extra that takes 30 seconds to demo
- **Polished dark UI** — looks like a real product, not a notebook

---

## 👥 Team

Built at Claude Builder Club × CMUAI Hackathon, April 2025
