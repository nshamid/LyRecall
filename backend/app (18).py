"""API LyRecall (FastAPI) untuk Hugging Face Space (Docker)."""
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from engine import Engine

DATA_REPO = os.environ.get("DATA_REPO", "USERNAME/lyrecall-index")     # repo dataset privat
DATA_DIR = os.environ.get("DATA_DIR", "/home/user/data")
HF_TOKEN = os.environ.get("HF_TOKEN")                                  # token Read (secret Space)
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]

ENGINE = Engine()
SLOTS = threading.Semaphore(2)          # maksimal 2 pencarian berjalan bersamaan


def _load():
    try:
        from huggingface_hub import snapshot_download
        ENGINE.stage = "mengunduh indeks dari Hugging Face"
        snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=DATA_DIR, token=HF_TOKEN)
        ENGINE.load(DATA_DIR)
    except Exception as e:                                             # tampilkan di /health
        ENGINE.error = f"{type(e).__name__}: {e}"
        print("GAGAL MEMUAT:", ENGINE.error, flush=True)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_load, daemon=True).start()               # port tetap terbuka selama memuat
    yield


app = FastAPI(title="LyRecall API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"ready": ENGINE.ready, "stage": ENGINE.stage, "error": ENGINE.error}


@app.get("/search")
def search(q: str = Query(..., min_length=2, max_length=500),
           top: int = Query(10, ge=1, le=20),
           mode: str = Query("hybrid", pattern="^(hybrid|lexical)$")):
    if ENGINE.error:
        raise HTTPException(500, "Server gagal memuat indeks")
    if not ENGINE.ready:
        raise HTTPException(503, f"Indeks sedang dimuat ({ENGINE.stage})")
    t0 = time.perf_counter()
    with SLOTS:
        results = ENGINE.search(q, top=top, mode=mode)
    return {"query": q, "mode": mode, "latency_ms": round(1000 * (time.perf_counter() - t0), 1),
            "results": results}
