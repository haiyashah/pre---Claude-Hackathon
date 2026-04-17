# Devpost Submission — Vertex RAG

> Copy-paste this into Devpost at submission time. Fill in team names.

---

## Project Title
**Vertex RAG — Research Intelligence Powered by Claude**

## Tagline
*Turn a pile of PDFs into a queryable AI research assistant — with Claude reasoning every step of the way.*

## The Problem
Researchers, students, and analysts waste hours manually reading through stacks of papers to answer specific questions. Existing tools either summarize without grounding (hallucinations), or just do keyword search (no reasoning). There's no tool that combines deep retrieval with transparent, verifiable AI reasoning.

## What We Built
**Vertex RAG** is an AI research intelligence platform that:

1. **Ingests** PDF research papers and indexes them into a Milvus vector store
2. **Retrieves** semantically relevant chunks using BAAI/bge-large-en-v1.5 embeddings
3. **Reasons** over retrieved chunks using a multi-step Claude pipeline (LangGraph):
   - Generate → Reflect → Revise (loops until Claude self-certifies quality)
4. **Shows** a full reasoning trace — users see exactly how Claude thought through the answer
5. **Benchmarks** three index strategies (HNSW vs IVF_PQ vs DiskANN) for performance comparison

## How Claude is Used (Deep Integration)
Claude is the reasoning engine at every step:

| Step | Model | Purpose |
|------|-------|---------|
| Query translation | Claude Haiku | Handles non-English queries |
| Answer generation | Claude Sonnet | Grounded answer from retrieved context |
| Quality reflection | Claude Sonnet | Self-evaluates answer — PASS/FAIL verdict |
| Answer revision | Claude Sonnet | Fixes grounding issues identified in reflection |
| URL verification | Claude Sonnet | Verifies recommended paper URLs |
| Output translation | Claude Haiku | Translates answer to user's language |

This is NOT a "call Claude once to summarize" — Claude drives a stateful reasoning loop.

## Tech Stack
- **Frontend**: Streamlit (custom dark theme)
- **LLM**: Anthropic Claude Sonnet 4.5 + Haiku 4.5
- **Vector DB**: Milvus (HNSW / IVF_PQ / DiskANN)
- **Embeddings**: BAAI/bge-large-en-v1.5 (local)
- **Pipeline**: LangGraph (stateful multi-step reasoning)
- **PDF parsing**: pdfminer.six

## Demo Script (90 seconds)
1. Show 3 pre-indexed research papers in the sidebar
2. Ask: *"What is multi-head attention and why does it matter?"*
3. Claude generates answer, reflects on quality, revises if needed
4. Show the **Reasoning Trace** — judges see Claude's thinking process
5. Show source citations and paper recommendations with verified URLs
6. Optional: Run index benchmark (HNSW wins on recall, IVF_PQ on speed)

## Team
- [Your name] — Frontend, demo design, prompt engineering
- [Teammate] — RAG pipeline, Milvus integration, LangGraph

## Links
- GitHub: [add on day of]
- Demo video: [add on day of]
