import os
import json
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import anthropic
import requests as http_requests

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextBox

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# ─────────────────────────────────────────────────────────────────────────────
# Claude config  (primary LLM for all generation / reflection / revision)
# ─────────────────────────────────────────────────────────────────────────────
CLAUDE_MODEL   = "claude-sonnet-4-5"
CLAUDE_HAIKU   = "claude-haiku-4-5"

# ─────────────────────────────────────────────────────────────────────────────
# Gemini — kept only for translation service URL fallback
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM   = 1024

GEN_MODELS = {
    "claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude-haiku-4-5":  "claude-haiku-4-5",
}
DEFAULT_GEN = "claude-sonnet-4-5"

INDEX_TYPES = ["HNSW", "IVF_PQ", "DiskANN"]

PAPER_TYPES = [
    "AI / ML",
    "Security",
    "Other",
]

# ─────────────────────────────────────────────────────────────────────────────
# Milvus config
# ─────────────────────────────────────────────────────────────────────────────
MILVUS_URI              = os.environ.get("MILVUS_URI",        "http://localhost:19530")
MILVUS_TOKEN            = os.environ.get("MILVUS_TOKEN",      "")
MILVUS_ALIAS            = "rag_conn"
DEFAULT_COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "papers_rag")

# ─────────────────────────────────────────────────────────────────────────────
# Output format — Claude must follow this exactly
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_FORMAT_INSTRUCTIONS = """
Your response MUST follow this exact structure — no deviations:

<Your answer here, 4-6 sentences, grounded only in the provided context.>

Recommended Research Papers:
• <Exact paper title 1>. URL: <url1>
• <Exact paper title 2>. URL: <url2>

Rules:
- Use bullet points (•) for the paper list, not dashes or numbers.
- The section header must be exactly: Recommended Research Papers:
- Only list papers from the provided allowed list. Do not invent papers.
- If a real URL is available from context, use it. Otherwise write: Searching for URL.
- Do NOT include any text after the second paper bullet.
- Do NOT include any heading like "Revised Answer" or any unrelated heading (##, #).
"""

SYSTEM_PROMPT_GENERATE = (
    "You are a precise research assistant powered by Claude. "
    "Answer questions using ONLY the provided document context. "
    "Never invent facts or cite papers not in the allowed list. "
    "Be concise, accurate, and follow the output format exactly."
)

SYSTEM_PROMPT_REFLECT = (
    "You are a strict RAG answer quality reviewer. "
    "Your only job is to check whether the answer is grounded, properly formatted, and follows all rules. "
    "Be rigorous — FAIL anything that deviates from the required format."
)

SYSTEM_PROMPT_REVISE = (
    "You are a careful RAG answer rewriter. "
    "Fix all issues identified in the reflection while staying strictly grounded in the provided context. "
    "Never invent papers or facts."
)

# ─────────────────────────────────────────────────────────────────────────────
# Translator config
# ─────────────────────────────────────────────────────────────────────────────
TRANSLATOR_URL = os.environ.get("TRANSLATOR_URL", "http://127.0.0.1:8080")


def _claude_translate(text: str, source_language: str, target_language: str) -> str:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=CLAUDE_HAIKU,
        max_tokens=1024,
        system="You are a precise translator. Return ONLY the translated text, nothing else.",
        messages=[{
            "role": "user",
            "content": f"Translate from {source_language} to {target_language}. Return ONLY the translation.\n\n{text}"
        }]
    )
    return resp.content[0].text.strip()


def call_translator(text: str, source_language: str, target_language: str) -> str:
    if source_language == target_language:
        return text
    try:
        resp = http_requests.post(
            f"{TRANSLATOR_URL}/translate",
            json={"text": text, "source_language": source_language, "target_language": target_language},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["translated_text"]
    except Exception:
        return _claude_translate(text, source_language, target_language)


# ─────────────────────────────────────────────────────────────────────────────
# Reflection / revision prompts
# ─────────────────────────────────────────────────────────────────────────────
_reflection_prompt = PromptTemplate.from_template(
"""#Role: You are a strict RAG answer reviewer.
#Task: Decide if the answer is grounded in the provided context and follows the required format.
#Format: Return EXACTLY:
#Issues:
- ...
#Fixes:
- ...
#Verdict: PASS or FAIL

#Rules:
- FAIL if the answer includes claims not supported by the provided Context.
- FAIL if the section header is not exactly: Recommended Research Papers:
- FAIL if the papers are not listed with bullet points (•), not dashes or numbers.
- FAIL if 'Recommended Research Papers' section is missing or includes papers not in the provided list.
- FAIL if the answer is unclear, contradictory, or ignores the user's question.
- FAIL if there are more than 2 recommended papers.
- FAIL if any URL is clearly wrong or unrelated to the paper title.
Question:
{question}

Context (snippets):
{context_block}

Allowed recommended papers (exact titles):
{papers_block}

Draft answer:
{answer}
"""
)

_revise_prompt = PromptTemplate.from_template(
"""#Role: You are a careful RAG answer rewriter.
#Task: Rewrite the answer to fix the issues while staying grounded in Context.
#Rules:
- Use ONLY the provided Context for factual claims.
- The section header must be exactly: Recommended Research Papers:
- List papers using bullet points (•), not dashes or numbers. Max 2 papers.
- List ONLY papers from the Allowed recommended papers list. Do NOT invent papers.
- Keep the answer concise (2-4 sentences) and directly responsive.
- If the original URL was not relevant to the paper title, remove it in the revision.
- After removing irrelevant URL, add a URL that is relevant to the paper title.
- If no URL is known, write: URL: Not available
- Do NOT include any heading like "Revised Answer" or any unrelated heading (##, #).

Question:
{question}

Context (snippets):
{context_block}

Allowed recommended papers (exact titles):
{papers_block}

Reflection:
{reflection}

Previous answer:
{answer}
"""
)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state
# ─────────────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:           str
    top_k:              int
    paper_filter:       Optional[str]
    documents:          List[Dict]
    answer:             str
    input_language:     str
    translated_query:   Optional[str]
    recommended_papers: Optional[List[str]]
    reflection:         Optional[str]
    status:             Optional[str]
    iteration:          Optional[int]
    max_iterations:     Optional[int]
    trace:              Optional[List[Dict]]


# ─────────────────────────────────────────────────────────────────────────────
# PDFIndexer (Milvus backend + Claude LLM)
# ─────────────────────────────────────────────────────────────────────────────
class PDFIndexer:
    """
    RAG indexer using Milvus as vector store and Claude as the LLM engine.
    LangGraph pipeline: translate → retrieve → generate → reflect → revise → translate output.
    """

    def __init__(
        self,
        chunk_size:          int  = 400,
        chunk_overlap:       int  = 50,
        model:               str  = DEFAULT_GEN,
        index_type:          str  = "HNSW",
        collection_name:     str  = DEFAULT_COLLECTION_NAME,
        drop_old_collection: bool = False,
        trace_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.chunk_size      = chunk_size
        self.chunk_overlap   = chunk_overlap
        self.model           = model
        self.index_type      = index_type
        self.embedding_dim   = EMBED_DIM
        self.collection_name = collection_name
        self.output_language = "English"
        self.trace_callback  = trace_callback

        self._collection: Optional[Collection] = None
        self._graph      = None
        self._embedder   = None
        self._client     = anthropic.Anthropic()

        self._connect_milvus()
        self._init_collection(drop_old=drop_old_collection)

    # ─── Claude LLM call ──────────────────────────────────────────────────────
    def _call_claude(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", max_tokens: int = 1200) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _emit_trace(self, event: Dict) -> None:
        if self.trace_callback:
            self.trace_callback(event)

    # ─── Utilities ────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_text(x: Any) -> str:
        return "" if x is None else str(x)

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            device = "mps"
            try:
                import torch
                if not torch.backends.mps.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
            self._embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
        return self._embedder

    def set_output_language(self, language: str = "English") -> None:
        self.output_language = language or "English"
        if self._collection is not None:
            self._build_graph()

    # ─── Milvus ───────────────────────────────────────────────────────────────
    def _connect_milvus(self):
        conn_kwargs = {"alias": MILVUS_ALIAS, "uri": MILVUS_URI}
        if MILVUS_TOKEN:
            conn_kwargs["token"] = MILVUS_TOKEN
        connections.connect(**conn_kwargs)

    def _make_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id",         dtype=DataType.INT64,       is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id",   dtype=DataType.INT64),
            FieldSchema(name="source",     dtype=DataType.VARCHAR,     max_length=512),
            FieldSchema(name="page",       dtype=DataType.INT64),
            FieldSchema(name="paper_type", dtype=DataType.VARCHAR,     max_length=128),
            FieldSchema(name="text",       dtype=DataType.VARCHAR,     max_length=65535),
            FieldSchema(name="embedding",  dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        ]
        return CollectionSchema(fields=fields, description="PDF RAG chunks")

    def _milvus_index_params(self) -> Dict:
        if self.index_type == "HNSW":
            return {"index_type": "HNSW",    "metric_type": "COSINE", "params": {"M": 32, "efConstruction": 200}}
        if self.index_type == "IVF_PQ":
            m = 8 if EMBED_DIM % 8 == 0 else 16
            return {"index_type": "IVF_PQ",  "metric_type": "COSINE", "params": {"nlist": 128, "m": m, "nbits": 8}}
        if self.index_type == "DiskANN":
            return {"index_type": "DISKANN", "metric_type": "COSINE", "params": {}}
        raise ValueError(f"Unknown index_type: {self.index_type}")

    def _milvus_search_params(self) -> Dict:
        if self.index_type == "HNSW":   return {"metric_type": "COSINE", "params": {"ef": 128}}
        if self.index_type == "IVF_PQ": return {"metric_type": "COSINE", "params": {"nprobe": 16}}
        return {"metric_type": "COSINE", "params": {}}

    def _init_collection(self, drop_old: bool = False):
        if drop_old and utility.has_collection(self.collection_name, using=MILVUS_ALIAS):
            utility.drop_collection(self.collection_name, using=MILVUS_ALIAS)
        if not utility.has_collection(self.collection_name, using=MILVUS_ALIAS):
            self._collection = Collection(
                name=self.collection_name, schema=self._make_schema(),
                using=MILVUS_ALIAS, shards_num=1,
            )
        else:
            self._collection = Collection(self.collection_name, using=MILVUS_ALIAS)
        self._build_graph()

    # ─── Extraction ───────────────────────────────────────────────────────────
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        pages    = []
        pdf_name = Path(pdf_path).name
        for page_num, layout in enumerate(extract_pages(pdf_path), start=1):
            parts = []
            for el in layout:
                if isinstance(el, (LTTextContainer, LTTextBox)):
                    t = el.get_text()
                    if t and t.strip():
                        parts.append(t)
            text = self._clean(" ".join(parts))
            if text:
                pages.append({"page": page_num, "text": text, "source": pdf_name})
        return pages

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x20-\x7E\n]", "", text)
        return text.strip()

    # ─── Chunking ─────────────────────────────────────────────────────────────
    def extract_and_chunk(self, pdf_path: str, paper_type: str = "Other") -> List[Dict]:
        pages    = self.extract_text_from_pdf(pdf_path)
        chunks   = []
        chunk_id = 0
        for page in pages:
            for text in self._chunk_text(page["text"]):
                chunks.append({
                    "chunk_id": chunk_id, "text": text,
                    "source": page["source"], "page": page["page"], "paper_type": paper_type,
                })
                chunk_id += 1
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, cur, cur_len = [], [], 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            wc = len(sent.split())
            if cur and (cur_len + wc > self.chunk_size):
                chunks.append(" ".join(cur))
                ov, ov_len = [], 0
                for s in reversed(cur):
                    sw = len(s.split())
                    if ov_len + sw <= self.chunk_overlap:
                        ov.insert(0, s); ov_len += sw
                    else:
                        break
                cur, cur_len = ov, ov_len
            cur.append(sent); cur_len += wc
        if cur:
            chunks.append(" ".join(cur))
        return [c for c in chunks if c.strip()]

    # ─── Embedding ────────────────────────────────────────────────────────────
    def embed_one(self, text: str, task: str = "retrieval_document") -> List[float]:
        embedder = self._get_embedder()
        prefix   = "Represent this sentence for searching relevant passages: " \
                   if task == "retrieval_query" else "Represent this passage for retrieval: "
        emb = embedder.encode(prefix + text, normalize_embeddings=True)
        return emb.tolist()

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        for chunk in chunks:
            chunk["embedding"] = self.embed_one(chunk["text"], "retrieval_document")
        return chunks

    # ─── Index building ───────────────────────────────────────────────────────
    def _drop_and_recreate_collection_for_index_type(self):
        if utility.has_collection(self.collection_name, using=MILVUS_ALIAS):
            utility.drop_collection(self.collection_name, using=MILVUS_ALIAS)
        self._collection = Collection(
            name=self.collection_name, schema=self._make_schema(),
            using=MILVUS_ALIAS, shards_num=1,
        )

    def build_index(self, embedded_chunks: List[Dict]) -> None:
        if not embedded_chunks:
            raise ValueError("No embedded chunks provided.")
        if "embedding" not in embedded_chunks[0]:
            raise ValueError("Chunks must contain 'embedding' before build_index().")

        self._drop_and_recreate_collection_for_index_type()
        self._collection.insert([
            [int(c["chunk_id"])                     for c in embedded_chunks],
            [str(c["source"])[:512]                 for c in embedded_chunks],
            [int(c["page"])                         for c in embedded_chunks],
            [str(c.get("paper_type","Other"))[:128] for c in embedded_chunks],
            [str(c["text"])[:65535]                 for c in embedded_chunks],
            [list(map(float, c["embedding"]))       for c in embedded_chunks],
        ])
        self._collection.flush()
        self._collection.create_index(field_name="embedding", index_params=self._milvus_index_params())
        self._collection.load()
        self._chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in embedded_chunks]
        self._build_graph()

    def add_to_index(self, embedded_chunks: List[Dict]) -> None:
        if self._collection is None:
            self._init_collection(drop_old=False)
        if not embedded_chunks:
            return
        self._collection.insert([
            [int(c["chunk_id"])                     for c in embedded_chunks],
            [str(c["source"])[:512]                 for c in embedded_chunks],
            [int(c["page"])                         for c in embedded_chunks],
            [str(c.get("paper_type","Other"))[:128] for c in embedded_chunks],
            [str(c["text"])[:65535]                 for c in embedded_chunks],
            [list(map(float, c["embedding"]))       for c in embedded_chunks],
        ])
        self._collection.flush()
        try:
            self._collection.load()
        except Exception:
            pass
        self._chunks.extend([{k: v for k, v in c.items() if k != "embedding"} for c in embedded_chunks])

    # ─── Search ───────────────────────────────────────────────────────────────
    def _search(self, q_vec: List[float], top_k: int, paper_filter: Optional[str] = None) -> List[Dict]:
        if self._collection is None:
            return []
        expr = f'paper_type == "{paper_filter.replace(chr(34), chr(92)+chr(34))}"' if paper_filter else None
        results = self._collection.search(
            data=[q_vec], anns_field="embedding", param=self._milvus_search_params(),
            limit=top_k, expr=expr,
            output_fields=["chunk_id","source","page","paper_type","text"],
            consistency_level="Strong",
        )
        docs = []
        if results and len(results) > 0:
            for hit in results[0]:
                e = hit.entity
                docs.append({
                    "chunk_id": e.get("chunk_id"), "source": e.get("source"),
                    "page": e.get("page"),         "paper_type": e.get("paper_type"),
                    "text": e.get("text"),         "score": float(hit.score),
                })
        return docs

    # ─── LangGraph pipeline ───────────────────────────────────────────────────
    def _build_graph(self) -> None:
        indexer = self

        def translate_query_node(state: RAGState) -> RAGState:
            lang = state.get("input_language", "English")
            q    = state["question"]
            translated = call_translator(text=q, source_language=lang, target_language="English")
            trace = list(state.get("trace") or [])
            trace.append({
                "step": "translate_query", "label": "🌐 Query Translation",
                "content": translated if translated != q else "(no translation needed)", "status": "ok",
            })
            indexer._emit_trace(trace[-1])
            return {**state, "translated_query": translated, "trace": trace}

        def retrieve_node(state: RAGState) -> RAGState:
            q     = state.get("translated_query") or state["question"]
            docs  = indexer._search(
                q_vec=indexer.embed_one(q, "retrieval_query"),
                top_k=state.get("top_k", 4),
                paper_filter=state.get("paper_filter"),
            )
            seen, recommended = set(), []
            for d in docs:
                t = d.get("source", "Unknown")
                if t not in seen:
                    seen.add(t); recommended.append(t)
            trace = list(state.get("trace") or [])
            trace.append({
                "step": "retrieve", "label": f"🔍 Retrieved {len(docs)} Chunks",
                "content": "\n".join(f"• [{d['source']} p{d['page']}] score={d['score']:.3f}" for d in docs),
                "status": "ok", "n_docs": len(docs),
            })
            indexer._emit_trace(trace[-1])
            return {**state, "documents": docs, "recommended_papers": recommended, "trace": trace}

        def generate_node(state: RAGState) -> RAGState:
            docs  = state.get("documents", [])
            recs  = state.get("recommended_papers", [])
            lang  = state.get("input_language", "English")
            q     = state.get("translated_query") or state["question"]
            ctx   = "\n\n".join(
                f"[{i}] {d['source']} (p{d['page']}, type={d.get('paper_type','?')}, score={d['score']:.3f})\n{d['text']}"
                for i, d in enumerate(docs, 1)
            ) if docs else "No relevant context found."
            top_2 = recs[:2]
            papers_block = "\n".join(f"- {t}" for t in top_2) if top_2 else "(none)"
            lang_note    = f"Answer in {lang}. " if lang != "English" else ""
            prompt = (
                f"{lang_note}{OUTPUT_FORMAT_INSTRUCTIONS}\n"
                "Answer using ONLY the provided context. Do NOT invent facts.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Allowed papers to recommend (use ONLY these, exact titles):\n{papers_block}\n\n"
                f"Question: {q}"
            )
            content = indexer._call_claude(prompt, system_prompt=SYSTEM_PROMPT_GENERATE)
            trace   = list(state.get("trace") or [])
            trace.append({
                "step": "generate", "label": "✍️ Claude Initial Draft",
                "content": content[:600] + ("…" if len(content) > 600 else ""), "status": "draft",
            })
            indexer._emit_trace(trace[-1])
            return {**state, "answer": indexer._safe_text(content), "trace": trace}

        def reflect_node(state: RAGState) -> RAGState:
            q     = state.get("translated_query") or state["question"]
            docs  = state.get("documents", [])
            recs  = state.get("recommended_papers", [])
            lang  = state.get("input_language", "English")
            ctx_b = "\n".join(f"- [{i}] {d['source']}: {d['text'][:300]}" for i, d in enumerate(docs, 1)) \
                    if docs else "(no context retrieved)"
            top_2 = recs[:2]
            pb    = "\n".join(f"- {t}" for t in top_2) if top_2 else "(none)"
            lang_note = f"\nNote: The answer should be in {lang}. FAIL if it is not." if lang != "English" else ""
            refl = indexer._call_claude(
                _reflection_prompt.format(
                    question=q + lang_note, context_block=ctx_b,
                    papers_block=pb, answer=state.get("answer", ""),
                ),
                system_prompt=SYSTEM_PROMPT_REFLECT,
            )
            m       = re.search(r"#Verdict:\s*(PASS|FAIL)", refl, re.IGNORECASE)
            status  = m.group(1).upper() if m else "FAIL"
            it      = int(state.get("iteration", 0))
            trace   = list(state.get("trace") or [])
            trace.append({
                "step": "reflect", "label": f"🪞 Reflection (iter {it + 1}) — {status}",
                "content": refl[:600] + ("…" if len(refl) > 600 else ""),
                "status": "pass" if status == "PASS" else "fail",
                "verdict": status, "iteration": it + 1,
            })
            indexer._emit_trace(trace[-1])
            return {**state, "reflection": refl, "status": status, "trace": trace}

        def revise_node(state: RAGState) -> RAGState:
            q     = state.get("translated_query") or state["question"]
            docs  = state.get("documents", [])
            recs  = state.get("recommended_papers", [])
            lang  = state.get("input_language", "English")
            ctx_b = "\n".join(f"- [{i}] {d['source']}: {d['text'][:300]}" for i, d in enumerate(docs, 1)) \
                    if docs else "(no context retrieved)"
            top_2     = recs[:2]
            pb        = "\n".join(f"- {t}" for t in top_2) if top_2 else "(none)"
            lang_rule = f"- Write the revised answer in {lang}.\n" if lang != "English" else ""
            revised   = indexer._call_claude(
                _revise_prompt.format(
                    question=q, context_block=ctx_b,
                    papers_block=pb + (f"\n{lang_rule}" if lang_rule else ""),
                    reflection=state.get("reflection", ""), answer=state.get("answer", ""),
                ),
                system_prompt=SYSTEM_PROMPT_REVISE,
            )
            it    = int(state.get("iteration", 0))
            trace = list(state.get("trace") or [])
            trace.append({
                "step": "revise", "label": f"🔧 Revision (iter {it + 1})",
                "content": revised[:600] + ("…" if len(revised) > 600 else ""),
                "status": "revised", "iteration": it + 1,
            })
            indexer._emit_trace(trace[-1])
            return {**state, "answer": revised, "trace": trace}

        def iterate_node(state: RAGState) -> RAGState:
            return {**state, "iteration": int(state.get("iteration", 0)) + 1}

        def translate_output_node(state: RAGState) -> RAGState:
            lang   = state.get("input_language", "English")
            ans    = state.get("answer", "")
            result = call_translator(text=ans, source_language="English", target_language=lang)
            trace  = list(state.get("trace") or [])
            if lang != "English":
                trace.append({
                    "step": "translate_output", "label": f"🌍 Output Translated → {lang}",
                    "content": result[:300], "status": "ok",
                })
                indexer._emit_trace(trace[-1])
            return {**state, "answer": result, "trace": trace}

        def should_continue(state: RAGState) -> str:
            if state.get("status") == "PASS":
                return "stop"
            if int(state.get("iteration", 0)) >= int(state.get("max_iterations", 2)):
                return "stop"
            return "continue"

        builder = StateGraph(RAGState)
        builder.add_node("translate_query",  translate_query_node)
        builder.add_node("retrieve",         retrieve_node)
        builder.add_node("generate",         generate_node)
        builder.add_node("reflect",          reflect_node)
        builder.add_node("revise",           revise_node)
        builder.add_node("iterate",          iterate_node)
        builder.add_node("translate_output", translate_output_node)

        builder.set_entry_point("translate_query")
        builder.add_edge("translate_query", "retrieve")
        builder.add_edge("retrieve",        "generate")
        builder.add_edge("generate",        "reflect")
        builder.add_edge("reflect",         "revise")
        builder.add_edge("revise",          "iterate")
        builder.add_conditional_edges("iterate", should_continue,
                                      {"continue": "reflect", "stop": "translate_output"})
        builder.add_edge("translate_output", END)
        self._graph = builder.compile()

    # ─── Public query ─────────────────────────────────────────────────────────
    def query(
        self,
        question:        str,
        top_k:           int           = 4,
        model:           Optional[str] = None,
        paper_filter:    Optional[str] = None,
        output_language: Optional[str] = None,
        max_iterations:  int           = 2,
    ) -> Tuple[str, List[Dict], List[Dict]]:
        """Returns (answer, sources, trace)."""
        if self._graph is None:
            self._build_graph()

        rebuild = False
        if model and model != self.model:
            self.model = model; rebuild = True
        if output_language and output_language != self.output_language:
            self.output_language = output_language; rebuild = True
        if rebuild:
            self._build_graph()

        init: RAGState = {
            "question": question, "top_k": top_k,
            "paper_filter": paper_filter, "documents": [],
            "answer": "", "input_language": output_language or "English",
            "translated_query": None, "recommended_papers": [],
            "reflection": None, "status": None,
            "iteration": 0, "max_iterations": max_iterations,
            "trace": [],
        }
        final = self._graph.invoke(init)
        return final["answer"], final["documents"], final.get("trace", [])

    # ─── Persistence ──────────────────────────────────────────────────────────
    def save_index(self, directory: str = "./rag_index") -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "milvus_index_meta.txt").write_text(
            f"collection_name={self.collection_name}\nindex_type={self.index_type}\nmilvus_uri={MILVUS_URI}\n",
            encoding="utf-8",
        )

    def load_index(self, directory: str = "./rag_index") -> None:
        self._connect_milvus()
        if not utility.has_collection(self.collection_name, using=MILVUS_ALIAS):
            raise FileNotFoundError(f"Milvus collection '{self.collection_name}' not found.")
        self._collection = Collection(self.collection_name, using=MILVUS_ALIAS)
        self._collection.load()
        self._build_graph()

    def clear_collection(self, drop: bool = False) -> None:
        if not utility.has_collection(self.collection_name, using=MILVUS_ALIAS):
            return
        if drop:
            utility.drop_collection(self.collection_name, using=MILVUS_ALIAS)
            self._collection = None
            return
        self._drop_and_recreate_collection_for_index_type()

    # ─── NPZ export / import ──────────────────────────────────────────────────
    def export_to_npz(self, npz_path: str) -> int:
        import numpy as np
        if self._collection is None:
            raise RuntimeError("No active collection to export.")
        self._collection.load()
        total = self._collection.num_entities
        if total == 0:
            raise ValueError("Collection is empty.")

        all_ids, all_src, all_pg, all_pt, all_tx, all_emb = [], [], [], [], [], []
        offset = 0
        while offset < total:
            rows = self._collection.query(
                expr="chunk_id >= 0",
                output_fields=["chunk_id","source","page","paper_type","text","embedding"],
                offset=offset, limit=1000,
            )
            for row in rows:
                all_ids.append(int(row["chunk_id"])); all_src.append(str(row["source"]))
                all_pg.append(int(row["page"]));      all_pt.append(str(row["paper_type"]))
                all_tx.append(str(row["text"]));      all_emb.append(row["embedding"])
            offset += 1000

        np.savez_compressed(
            npz_path,
            chunk_ids=np.array(all_ids, dtype=np.int64), sources=np.array(all_src, dtype=object),
            pages=np.array(all_pg, dtype=np.int64),      paper_types=np.array(all_pt, dtype=object),
            texts=np.array(all_tx, dtype=object),        embeddings=np.array(all_emb, dtype=np.float32),
            meta=np.array([json.dumps({
                "index_type": self.index_type, "collection_name": self.collection_name,
                "embed_model": EMBED_MODEL, "embed_dim": EMBED_DIM, "n_chunks": len(all_ids),
                "indexed_files": getattr(self, "_indexed_files", []),
                "file_url_map":  getattr(self, "_file_url_map",  {}),
            })], dtype=object),
        )
        return len(all_ids)

    def import_from_npz(self, npz_path: str, append: bool = False) -> dict:
        import numpy as np
        data = np.load(npz_path, allow_pickle=True)
        chunk_ids   = data["chunk_ids"].tolist()
        sources     = [str(s) for s in data["sources"].tolist()]
        pages       = data["pages"].tolist()
        paper_types = [str(p) for p in data["paper_types"].tolist()]
        texts       = [str(t)[:65535] for t in data["texts"].tolist()]
        embeddings  = data["embeddings"].tolist()
        meta = {}
        try:
            meta = json.loads(str(data["meta"][0]))
        except Exception:
            pass
        if not len(chunk_ids):
            raise ValueError("npz file contains no chunks.")
        if not append:
            self._drop_and_recreate_collection_for_index_type()
        elif self._collection is None:
            self._init_collection(drop_old=False)
        self._collection.insert([chunk_ids, sources, pages, paper_types, texts, embeddings])
        self._collection.flush()
        self._collection.create_index(field_name="embedding", index_params=self._milvus_index_params())
        self._collection.load()
        self._build_graph()
        self._chunks = [
            {"chunk_id": cid, "source": src, "page": pg, "paper_type": pt, "text": tx}
            for cid, src, pg, pt, tx in zip(chunk_ids, sources, pages, paper_types, texts)
        ]
        return {
            "n_chunks": len(chunk_ids), "index_type": meta.get("index_type", self.index_type),
            "embed_model": meta.get("embed_model", EMBED_MODEL), "collection_name": self.collection_name,
            "indexed_files": meta.get("indexed_files", []), "file_url_map": meta.get("file_url_map", {}),
        }
