import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import openai

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path("E:/New folder/cache/docs")  # folder with {COMPANY}/{YEAR}/10-K.html
INDEX_PATH = Path("cache/faiss.index")
CHUNKS_PATH = Path("cache/chunks.json")
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH = 64

openai.api_key = "sk-proj-L3UI21MY2YhOwdNRUQt9FotH85AfqMoEuHjTkFbgS2EtYFcNirnmwDz9Nuy4wLNyGMjey7u-6zT3BlbkFJFxuNzhy3CHo3R8vKGuKjVgQzEMIhOgg41PF1RDd8V-UFhHhxQqKCoTP-lWONULdjhq5wF2AqYAPENAI_API_KEY"  # replace with your key
LLM_MODEL = "gpt-3.5-turbo"  # or gpt-4o-mini if available
TOP_K = 5

# -------------------------
# UTILITIES
# -------------------------
def extract_text_from_html(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
            return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print(f"[ERROR] reading {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size_words: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size_words]))
        i += chunk_size_words - overlap
    return chunks

# -------------------------
# LOAD DOCS
# -------------------------
def load_docs(base_dir: Path) -> List[Dict[str,Any]]:
    all_chunks = []
    for comp_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        company = comp_dir.name
        for year_dir in sorted([y for y in comp_dir.iterdir() if y.is_dir()]):
            year = year_dir.name
            path = year_dir / "10-K.html"
            if not path.exists():
                candidates = list(year_dir.glob("*.htm*"))
                if candidates:
                    path = candidates[0]
                else:
                    print(f"[WARN] missing 10-K for {company} {year}")
                    continue
            text = extract_text_from_html(path)
            if not text:
                continue
            chunks = chunk_text(text)
            for i, c in enumerate(chunks, start=1):
                all_chunks.append({"company": company, "year": year, "page": i, "text": c})
    print(f"[INFO] Loaded {len(all_chunks)} chunks from {base_dir}")
    return all_chunks

# -------------------------
# FAISS INDEX
# -------------------------
def build_or_load_index(model: SentenceTransformer, all_chunks: List[Dict[str,Any]]):
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        print("[INFO] Loading cached FAISS index & chunks...")
        index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return index, saved

    print("[INFO] Building FAISS index...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=EMBED_BATCH, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    return index, all_chunks

# -------------------------
# RETRIEVAL
# -------------------------
def retrieve(query: str, model: SentenceTransformer, index, all_chunks: List[Dict[str,Any]], k:int=TOP_K) -> List[Dict[str,Any]]:
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(all_chunks):
            results.append(all_chunks[idx])
    return results

# -------------------------
# NUMERIC EXTRACTION
# -------------------------
NUM_RE = re.compile(r"(?:\$)?\s*([0-9]{1,3}(?:[,0-9]*)?(?:\.\d+)?)\s*(%|million|billion|bn|m|k)?", re.IGNORECASE)
def parse_number_from_text(text: str) -> Optional[float]:
    pct_matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if pct_matches:
        return float(pct_matches[0])
    m = NUM_RE.search(text)
    if not m: return None
    num_str, unit = m.group(1), m.group(2)
    try: num = float(num_str.replace(",", ""))
    except: return None
    if unit:
        u = unit.lower()
        if u in ("billion","bn"): num *= 1_000_000_000
        elif u in ("million","m"): num *= 1_000_000
        elif u == "k": num *= 1_000
    return num

# -------------------------
# QUERY TYPE DETECTION
# -------------------------
def detect_query_type(query: str) -> str:
    q = query.lower()
    if "compare" in q and "growth" in q: return "multi-aspect"
    if "compare" in q or "highest" in q or "which company had" in q: return "cross-company"
    if re.search(r"from\s+(\d{4})\s+to\s+(\d{4})", q): return "yoy"
    if "percentage" in q or "percent" in q: return "segment"
    return "basic"

# -------------------------
# SYSTEM PROMPT
# -------------------------
SYSTEM_PROMPT = """
You are a financial analyst assistant. Answer questions using ONLY the retrieved SEC 10-K excerpts provided. 
Do NOT hallucinate numbers or facts.

Rules:
1. Answer concisely and clearly.
2. Return the answer strictly in the following JSON format:

{
  "query": "<original query>",
  "answer": "<final concise answer based only on retrieved data.>",
  "reasoning": "<step-by-step reasoning showing how you used the sources to derive the answer.>",
  "sub_queries": ["<list of sub-queries you considered>"],
  "sources": [
    {
      "company": "<ticker>",
      "year": "<filing year>",
      "excerpt": "<relevant text snippet>",
      "page": <page number if available, else 0>
    }
  ]
}

Always cite exact wording from excerpts. JSON must be valid. Use only years and values in the retrieved text.
"""

def polish_with_llm(final_struct_local: Dict[str,Any], query:str) -> Dict[str,Any]:
    user_content = f"Local retrieved chunks:\n{json.dumps(final_struct_local, indent=2)}\n\nQuery: {query}\nReturn only JSON as specified."
    try:
        resp = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":user_content}
            ],
            temperature=0,
            max_tokens=600
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return json.loads(text)
    except Exception as e:
        return {
            "query": query,
            "answer": final_struct_local.get("answer_local"),
            "reasoning": final_struct_local.get("reasoning_local"),
            "sub_queries": final_struct_local.get("sub_queries", []),
            "sources": final_struct_local.get("sources", [])
        }

# -------------------------
# SIMPLE AGENT FOR BASIC / CROSS-COMPANY
# -------------------------
def prepare_sources(retrieved: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    sources = []
    for chunk in retrieved:
        sources.append({
            "company": chunk["company"],
            "year": chunk["year"],
            "excerpt": chunk["text"][:500]+"...",
            "page": chunk["page"]
        })
    return sources

def generate_answer(retrieved: List[Dict[str,Any]]) -> str:
    # Try to find first numeric value as an example
    for chunk in retrieved:
        num = parse_number_from_text(chunk["text"])
        if num:
            return f"{chunk['company']} reported {num} as per filing in {chunk['year']}."
    # fallback
    return "No clear numeric answer found in top retrieved chunks."

def agent_answer(query: str, model: SentenceTransformer, index, all_chunks: List[Dict[str,Any]]) -> Dict[str,Any]:
    retrieved = retrieve(query, model, index, all_chunks, k=TOP_K)
    sources = prepare_sources(retrieved)
    answer_local = generate_answer(retrieved)
    final_struct_local = {
        "query": query,
        "answer_local": answer_local,
        "reasoning_local": f"Retrieved top {len(retrieved)} chunks and extracted numbers where possible.",
        "sub_queries": [query],
        "sources": sources
    }
    return polish_with_llm(final_struct_local, query)

# -------------------------
# MAIN
# -------------------------
def main():
    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = load_docs(BASE_DIR)
    if not all_chunks:
        print("[ERROR] No documents found.")
        return
    index, all_chunks = build_or_load_index(model, all_chunks)
    print("[INFO] Ready. Ask queries (type 'exit' to quit).")

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if not q: continue
        if q.lower() in ("exit", "quit"): break
        try:
            out = agent_answer(q, model, index, all_chunks)
            print(json.dumps(out, indent=2))
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
