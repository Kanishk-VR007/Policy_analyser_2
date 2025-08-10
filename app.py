import os
import io
import time
import json
import hashlib
import tempfile
import logging
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
import asyncio
import fitz  # PyMuPDF
import docx2txt
from bs4 import BeautifulSoup
import email
from email import policy as email_policy
import openai

# Environment Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "policy-index")
CACHE_DIR = os.environ.get("CACHE_DIR", "./policy_cache")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", 500))

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("policy-analyzer")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Embeddings/LLM calls will fail until provided.")
openai.api_key = OPENAI_API_KEY

os.makedirs(CACHE_DIR, exist_ok=True)

# Pinecone or FAISS
USE_PINECONE = bool(PINECONE_API_KEY and PINECONE_ENV)
USE_FAISS_FALLBACK = True

if USE_PINECONE:
    import pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if PINECONE_INDEX not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX, dimension=1536)
    pinecone_index = pinecone.Index(PINECONE_INDEX)
else:
    pinecone_index = None
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    FAISS_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

FAISS_INDEXES: Dict[str, Tuple] = {}

# FastAPI app
app = FastAPI(title="Policy Semantic Analyzer (Pinecone + OpenAI)")


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


async def download_bytes(url: str, timeout: int = 20) -> Tuple[bytes, str]:
    headers = {"User-Agent": "policy-analyzer/1.0"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            content = await resp.read()
            ctype = resp.headers.get("Content-Type", "")
            return content, ctype


def extract_text_from_pdf_bytes(b: bytes, max_pages: int = 300) -> str:
    texts: List[str] = []
    try:
        with fitz.open(stream=b, filetype="pdf") as doc:
            n = min(len(doc), max_pages)
            for i in range(n):
                page = doc.load_page(i)
                txt = page.get_text("text")
                if txt:
                    texts.append(txt)
    except Exception as e:
        logger.exception("PDF extraction failed: %s", e)
    return "\n".join(texts)


def extract_text_from_docx_bytes(b: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".docx")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b)
    try:
        text = docx2txt.process(path) or ""
    except Exception as e:
        logger.exception("DOCX extraction failed: %s", e)
        text = ""
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    return text


def extract_text_from_html_bytes(b: bytes) -> str:
    try:
        soup = BeautifulSoup(b, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except Exception as e:
        logger.exception("HTML extraction failed: %s", e)
        return ""


def extract_text_from_eml_bytes(b: bytes) -> str:
    try:
        msg = email.message_from_bytes(b, policy=email_policy.default)
        parts: List[str] = []
        if msg.is_multipart():
            for p in msg.walk():
                ct = p.get_content_type()
                if ct == "text/plain":
                    parts.append(p.get_content())
                elif ct == "text/html":
                    parts.append(BeautifulSoup(p.get_content(), "html.parser").get_text())
        else:
            ct = msg.get_content_type()
            if ct == "text/plain":
                parts.append(msg.get_content())
            elif ct == "text/html":
                parts.append(BeautifulSoup(msg.get_content(), "html.parser").get_text())
        header_parts: List[str] = []
        if msg.get("Subject"):
            header_parts.append("Subject: " + msg.get("Subject"))
        return "\n".join(header_parts + parts)
    except Exception as e:
        logger.exception("EML extraction failed: %s", e)
        return ""


def extract_text_from_bytes(b: bytes, content_type: str) -> str:
    c = (content_type or "").lower()
    try:
        if "pdf" in c:
            return extract_text_from_pdf_bytes(b)
        if "officedocument" in c or "word" in c:
            return extract_text_from_docx_bytes(b)
        if "html" in c:
            return extract_text_from_html_bytes(b)
        if "eml" in c or "message/rfc822" in c:
            return extract_text_from_eml_bytes(b)
        t = extract_text_from_pdf_bytes(b)
        if t.strip():
            return t
        return b.decode("utf-8", errors="replace")
    except Exception:
        return b.decode("utf-8", errors="replace")


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = cur + "\n\n" + p if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    if not chunks and text:
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i + max_chars])
    return chunks


def openai_embed_texts(texts: List[str]) -> List[List[float]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
    return [r["embedding"] for r in resp["data"]]


def upsert_to_pinecone(policy_id: str, chunks: List[str], embeddings: List[List[float]]) -> None:
    if not USE_PINECONE or pinecone_index is None:
        return
    to_upsert = []
    for i, (c, emb) in enumerate(zip(chunks, embeddings)):
        meta = {"policy_id": policy_id, "chunk_id": f"{policy_id}_c{i}", "text_preview": c[:300]}
        to_upsert.append((f"{policy_id}_c{i}", emb, meta))
    if to_upsert:
        pinecone_index.upsert(vectors=to_upsert)


def build_faiss_for_policy(policy_id: str, chunks: List[str]) -> None:
    if 'FAISS_MODEL' not in globals():
        return
    emb = FAISS_MODEL.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    import faiss
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(emb)
    FAISS_INDEXES[policy_id] = (idx, emb, chunks)


def save_policy_meta(policy_id: str, meta: Dict) -> None:
    path = os.path.join(CACHE_DIR, f"{policy_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_policy_meta(policy_id: str) -> Optional[Dict]:
    path = os.path.join(CACHE_DIR, f"{policy_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ProcessRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    policy_id: str
    question: str


@app.post("/process")
async def process(req: ProcessRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url required")
    policy_id = sha256(url)
    meta = load_policy_meta(policy_id)
    if meta:
        return {"policy_id": policy_id, "cached": True, "num_chunks": meta.get("num_chunks", 0)}
    b, ctype = await download_bytes(url)
    text = extract_text_from_bytes(b, ctype)
    chunks = chunk_text(text, max_chars=3000)
    embeddings = openai_embed_texts(chunks)
    if embeddings and USE_PINECONE:
        upsert_to_pinecone(policy_id, chunks, embeddings)
    elif USE_FAISS_FALLBACK:
        build_faiss_for_policy(policy_id, chunks)
    meta = {"policy_id": policy_id, "source_url": url, "num_chunks": len(chunks)}
    save_policy_meta(policy_id, meta)
    return {"policy_id": policy_id, "cached": False, "num_chunks": len(chunks)}


@app.post("/ask")
async def ask(req: AskRequest):
    meta = load_policy_meta(req.policy_id)
    if not meta:
        raise HTTPException(status_code=404, detail="policy_id not found; call /process first")
    question = req.question.strip()
    q_emb = openai.Embedding.create(model=EMBED_MODEL, input=[question])["data"][0]["embedding"]
    retrieved_texts: List[str] = []
    if q_emb and USE_PINECONE:
        matches = pinecone_index.query(vector=q_emb, top_k=3, filter={"policy_id": {"$eq": req.policy_id}}, include_metadata=True)
        for m in matches.get("matches", []):
            retrieved_texts.append(m.get("metadata", {}).get("text_preview", ""))
    elif req.policy_id in FAISS_INDEXES:
        idx, emb_arr, chunks = FAISS_INDEXES[req.policy_id]
        import numpy as np
        import faiss
        qv = np.array(q_emb, dtype=np.float32)
        faiss.normalize_L2(qv.reshape(1, -1))
        D, I = idx.search(qv.reshape(1, -1), 3)
        for idx_pos in I[0].tolist():
            retrieved_texts.append(chunks[idx_pos])
    prompt = "Answer this question using context:\n\n" + "\n\n".join(retrieved_texts)
    llm_out = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0
    ).choices[0].message.content
    return {"policy_id": req.policy_id, "question": question, "answer": llm_out}
