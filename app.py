import os
import io
import time
import json
import hashlib
import tempfile
import logging
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import aiohttp
import asyncio
import fitz
import docx2txt
from bs4 import BeautifulSoup
import email
from email import policy as email_policy
import openai

USE_PINECONE = bool(os.environ.get("PINECONE_API_KEY"))
if USE_PINECONE:
    import pinecone

USE_FAISS_FALLBACK = True
if USE_FAISS_FALLBACK:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("policy-analyzer")

app = FastAPI(title="Policy Semantic Analyzer (Pinecone + OpenAI)")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Embeddings/LLM calls will fail until provided.")
openai.api_key = OPENAI_API_KEY

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "policy-index")
CACHE_DIR = os.environ.get("CACHE_DIR", "./policy_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", 500))

FAISS_MODEL = None
FAISS_INDEXES: Dict[str, Tuple] = {}

if USE_PINECONE:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if PINECONE_INDEX not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX, dimension=1536)
    pinecone_index = pinecone.Index(PINECONE_INDEX)
else:
    pinecone_index = None
    FAISS_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


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
        except Exception as e:
            logger.debug("Could not remove temp file: %s", e)
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
        if "pdf" in c or c.endswith("/pdf"):
            return extract_text_from_pdf_bytes(b)
        if "officedocument" in c or "word" in c or c.endswith(".docx"):
            return extract_text_from_docx_bytes(b)
        if "html" in c or c.startswith("text/html"):
            return extract_text_from_html_bytes(b)
        if "eml" in c or "message/rfc822" in c:
            return extract_text_from_eml_bytes(b)
        t = extract_text_from_pdf_bytes(b)
        if t.strip():
            return t
        try:
            return b.decode("utf-8", errors="replace")
        except Exception:
            return b.decode("latin1", errors="replace")
    except Exception as e:
        logger.exception("Generic extraction failed: %s", e)
        # fallback to best-effort decode
        try:
            return b.decode("utf-8", errors="replace")
        except Exception:
            return b.decode("latin1", errors="replace")


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
    embeddings = [r["embedding"] for r in resp["data"]]
    return embeddings


def upsert_to_pinecone(policy_id: str, chunks: List[str], embeddings: List[List[float]]) -> None:
    if not USE_PINECONE or pinecone_index is None:
        return
    to_upsert = []
    try:
        for i, (c, emb) in enumerate(zip(chunks, embeddings)):
            meta = {"policy_id": policy_id, "chunk_id": f"{policy_id}_c{i}", "text_preview": c[:300]}
            to_upsert.append((f"{policy_id}_c{i}", emb, meta))
            if len(to_upsert) >= 100:
                pinecone_index.upsert(vectors=to_upsert)
                to_upsert = []
        if to_upsert:
            pinecone_index.upsert(vectors=to_upsert)
    except Exception as e:
        logger.exception("Pinecone upsert failed: %s", e)


def build_faiss_for_policy(policy_id: str, chunks: List[str]) -> None:
    if FAISS_MODEL is None:
        return
    try:
        emb = FAISS_MODEL.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(emb)
        dim = emb.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(emb)
        FAISS_INDEXES[policy_id] = (idx, emb, chunks)
    except Exception as e:
        logger.exception("FAISS build failed: %s", e)


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


def pinecone_query_top(policy_id: str, query_emb: List[float], top_k: int = 3):
    if not USE_PINECONE or pinecone_index is None:
        return []
    try:
        resp = pinecone_index.query(vector=query_emb, top_k=top_k, namespace=None, filter={"policy_id": {"$eq": policy_id}}, include_metadata=True)
        matches = resp.get("matches", [])
        return matches
    except Exception as e:
        logger.exception("Pinecone query failed: %s", e)
        return []


def faiss_query_top(policy_id: str, query_vec: List[float], top_k: int = 3):
    if policy_id not in FAISS_INDEXES:
        return []
    try:
        idx, emb_arr, chunks = FAISS_INDEXES[policy_id]
        q = np.array(query_vec, dtype=np.float32)
        q = q / np.linalg.norm(q)
        D, I = idx.search(np.expand_dims(q, 0), top_k)
        out = []
        for score, idx_pos in zip(D[0].tolist(), I[0].tolist()):
            if idx_pos < 0:
                continue
            out.append({"id": f"{policy_id}_c{idx_pos}", "score": float(score), "metadata": {"text_preview": chunks[idx_pos][:300], "chunk_index": idx_pos}})
        return out
    except Exception as e:
        logger.exception("FAISS query failed: %s", e)
        return []


def compose_llm_prompt(question: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = (
        "You are a careful insurance policy analyst. "
        "Given the following snippets extracted from an insurance policy and a user's question, answer whether the described treatment/benefit is covered or excluded, "
        "list the exact conditions, waiting periods, limits (amounts), and cite which snippet you used (by index). "
        "If uncertain, explicitly say so and provide the most relevant phrases.\n\n"
        f"QUESTION: {question}\n\n"
        "SNIPPETS (index: text):\n"
    )
    for i, c in enumerate(retrieved_chunks):
        prompt += f"[{i}] {c[:1200]}\n\n"
    prompt += (
        "\n\nAnswer in JSON with fields: answer (short), covered (true/false/likely/unknown), confidence (0-1), conditions (list), limits (list), sources (list of indices), explanation (short).\n"
        "Keep the JSON strictly parseable."
    )
    return prompt


def call_llm_chat(prompt: str, max_tokens: int = 600) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in insurance policies."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.0
    )
    return resp.choices[0].message.content


class ProcessRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    policy_id: str
    question: str


class BatchRunRequest(BaseModel):
    documents: str
    questions: List[str]


@app.post("/process")
async def process(req: ProcessRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url required")
    policy_id = sha256(url)
    meta = load_policy_meta(policy_id)
    if meta:
        return {"policy_id": policy_id, "cached": True, "num_chunks": meta.get("num_chunks", 0)}
    try:
        b, ctype = await download_bytes(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"download failed: {e}")

    text = extract_text_from_bytes(b, ctype)
    if not text or len(text.strip()) < 30:
        raise HTTPException(status_code=400, detail="no text extracted or too short")
    chunks = chunk_text(text, max_chars=3000)
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunks created")
    try:
        embeddings = openai_embed_texts(chunks)
    except Exception as e:
        embeddings = None
        logger.exception("openai embeddings failed")
    if embeddings and USE_PINECONE:
        upsert_to_pinecone(policy_id, chunks, embeddings)
    elif embeddings is None and USE_FAISS_FALLBACK:
        build_faiss_for_policy(policy_id, chunks)
    else:
        if USE_FAISS_FALLBACK:
            build_faiss_for_policy(policy_id, chunks)

    meta = {
        "policy_id": policy_id,
        "source_url": url,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_chunks": len(chunks),
        "content_type": ctype
    }
    save_policy_meta(policy_id, meta)
    return {"policy_id": policy_id, "cached": False, "num_chunks": len(chunks)}


@app.post("/ask")
async def ask(req: AskRequest):
    meta = load_policy_meta(req.policy_id)
    if not meta:
        raise HTTPException(status_code=404, detail="policy_id not found; call /process first")
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    try:
        q_emb = openai.Embedding.create(model=EMBED_MODEL, input=[question])["data"][0]["embedding"]
    except Exception as e:
        q_emb = None
        logger.exception("query embedding failed")
    top_k = 3
    retrieved_texts: List[str] = []
    candidates = []
    if q_emb and USE_PINECONE:
        try:
            matches = pinecone_index.query(vector=q_emb, top_k=top_k, filter={"policy_id": {"$eq": req.policy_id}}, include_metadata=True)
            matches = matches.get("matches", [])
        except Exception as e:
            logger.exception("Pinecone query failed inside ask: %s", e)
            matches = []
        for m in matches:
            txt_preview = m.get("metadata", {}).get("text_preview", "")
            candidates.append({"id": m["id"], "score": float(m["score"]), "preview": txt_preview})
            retrieved_texts.append(txt_preview)
    else:
        if req.policy_id in FAISS_INDEXES:
            idx, emb_arr, chunks = FAISS_INDEXES[req.policy_id]
            try:
                qv = FAISS_MODEL.encode([question], convert_to_numpy=True)[0]
                faiss.normalize_L2(qv.reshape(1, -1))
                D, I = idx.search(qv.reshape(1, -1), top_k)
                for score, idx_pos in zip(D[0].tolist(), I[0].tolist()):
                    if idx_pos < 0:
                        continue
                    candidates.append({"id": f"{req.policy_id}_c{idx_pos}", "score": float(score), "preview": chunks[idx_pos][:300]})
                    retrieved_texts.append(chunks[idx_pos])
            except Exception as e:
                logger.exception("FAISS query failed inside ask: %s", e)
        else:
            return {"policy_id": req.policy_id, "question": question, "answers": ["no_retrieval_possible"], "candidates": []}
    prompt = compose_llm_prompt(question, retrieved_texts)
    try:
        llm_out = call_llm_chat(prompt)
    except Exception as e:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")
    parsed = None
    try:
        parsed = json.loads(llm_out.strip())
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", llm_out)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = {"answer_text": llm_out.strip()}

    output = {
        "policy_id": req.policy_id,
        "question": question,
        "candidates": candidates,
        "raw_llm": llm_out,
        "answers": parsed if parsed is not None else {"answer_text": llm_out.strip()}
    }
    return output


@app.post("/hackrx/run")
async def batch_run(req: BatchRunRequest):
    """
    Accepts the sample upload structure:
    {
      "documents": "https://.../policy.pdf",
      "questions": [ "Q1", "Q2", ... ]
    }
    Returns:
    {
      "answers": [ "ans1", "ans2", ... ]
    }
    """
    proc = ProcessRequest(url=req.documents)
    resp = await process(proc)
    policy_id = resp["policy_id"]
    answers: List[str] = []
    for q in req.questions:
        ans_resp = await ask(AskRequest(policy_id=policy_id, question=q))
        out = ans_resp.get("answers")
        if isinstance(out, dict):
            if "answer" in out:
                answers.append(out["answer"])
            elif "answer_text" in out:
                answers.append(out["answer_text"])
            elif "explanation" in out:
                answers.append(out["explanation"])
            else:
                answers.append(json.dumps(out))
        elif isinstance(out, list):
            # join
            answers.append(" ".join(out))
        else:
            answers.append(ans_resp.get("raw_llm", "no_answer"))
    return {"answers": answers}
