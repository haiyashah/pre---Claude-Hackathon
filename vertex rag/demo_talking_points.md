# 🎤 Demo Script — Vertex RAG (90 seconds)

Use this verbatim or adapt it. Practice it 3 times before judging.

---

## 0:00–0:15 | The Problem

> "Researchers spend hours manually cross-referencing papers to answer one question.
> Existing AI tools either hallucinate facts, or just summarize — they don't reason.
> We built Vertex RAG to fix that."

*[Show the app loaded with 3+ papers in the sidebar]*

---

## 0:15–0:35 | The Demo

> "I'll ask a real research question."

*[Type in chat box:]*
```
What is multi-head attention and how does it compare to BERT's pretraining approach?
```

*[While it loads:]*
> "Claude is doing something most RAG systems don't — it's retrieving relevant chunks,
> generating an answer, then REFLECTING on its own answer to check quality,
> and revising if needed. This is a multi-step reasoning loop."

---

## 0:35–0:55 | The Wow Moment

*[Answer appears — click "Claude Reasoning Trace"]*

> "Here's what makes Vertex RAG different — full transparency.
> You can see every step Claude took:
> — what it retrieved
> — what it generated
> — how it reflected on quality
> — how it revised the answer"

*[Scroll through trace]*

> "Claude graded its own answer, found an issue, and fixed it.
> This isn't just summarization — it's reasoning."

---

## 0:55–1:15 | The Technical Depth

*[Point to sources section]*

> "Every claim is backed by a specific chunk from a specific paper —
> with page numbers and URLs Claude verified."

*[Switch to Stats tab — show benchmark if pre-run]*

> "We also benchmarked three vector index strategies:
> HNSW, IVF_PQ, and DiskANN —
> so researchers can pick the right speed/recall tradeoff for their use case."

---

## 1:15–1:30 | Close

> "Vertex RAG works with any PDF knowledge base —
> papers, reports, documentation, legal docs.
> Claude isn't just called once — it's the reasoning engine at every step.
> This is what AI-powered research actually looks like."

---

## Key Lines to Memorize
- *"Claude reflects on its own answer and revises it — that's the loop"*
- *"Not just summarization — full multi-step reasoning"*  
- *"Every claim is traceable to a source chunk"*
- *"Claude is the engine at every step, not just the output"*

---

## If Something Breaks
- Milvus down → switch to `demo_mode.py` (no Docker needed)
- Slow response → have a pre-recorded answer ready to paste in
- API error → have screenshots ready

## Pre-Demo Checklist
- [ ] `export ANTHROPIC_API_KEY=sk-ant-...`
- [ ] Milvus running (`docker ps | grep milvus`)
- [ ] 3+ PDFs already indexed
- [ ] Test query ran successfully once
- [ ] `demo_mode.py` as backup tab open
- [ ] Reasoning trace expanded and visible
