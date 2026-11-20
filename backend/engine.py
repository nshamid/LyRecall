"""Mesin pencarian LyRecall: BM25 + TF-IDF karakter + dense (MiniLM), RRF berbobot, agregasi max ke lagu.
Logikanya sama dengan notebook 06/06b (search_bm25, search_tfidf, dense_batch, fuse_w)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

K_CAND = 100
LABEL = {"bm25": "Exact", "tfidf": "Fuzzy", "dense": "Semantic"}
EMPTY = (np.array([], dtype=np.int64), np.array([], dtype=np.float32))


def score_postings(M, ids, weights, n_docs):
    s = np.zeros(n_docs, dtype=np.float32)
    ip, ix, dt = M.indptr, M.indices, M.data
    for j, w in zip(ids, weights):
        a, e = ip[j], ip[j + 1]
        s[ix[a:e]] += w * dt[a:e]
    return s


def topk(scores, k):
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    keep = scores[idx] > 0
    return idx[keep], scores[idx][keep]


class Engine:
    def __init__(self):
        self.ready = False
        self.error = None
        self.stage = "belum dimulai"

    # ------------------------------------------------------------------ pemuatan
    def load(self, data_dir, log=print):
        data_dir = Path(data_dir)
        sys.path.insert(0, str(data_dir))
        from lyrecall_text import clean_lyrics, normalize_lex   # modul yang sama dengan saat offline
        self.clean_lyrics, self.normalize_lex = clean_lyrics, normalize_lex

        cfg = json.load(open(data_dir / "final_config.json"))
        self.k, self.depth, self.w_dense = int(cfg["k"]), int(cfg["depth"]), float(cfg["w_dense"])

        self.stage = "memuat tabel chunk dan judul"
        chunks = pd.read_parquet(data_dir / "chunks_L3_S2.parquet", columns=["song_id", "text"])
        self.song_of = chunks["song_id"].to_numpy()
        self.N = len(chunks)
        self.snippet_of = chunks["text"].str.slice(0, 90).tolist()      # hanya cuplikan pendek
        del chunks
        songs = pd.read_parquet(data_dir / "songs_clean.parquet", columns=["song_id", "title", "artist"])
        self.title_of = dict(zip(songs["song_id"], songs["title"]))
        self.artist_of = dict(zip(songs["song_id"], songs["artist"]))
        del songs
        log(f"chunk: {self.N:,} | lagu: {len(self.title_of):,}")

        if self.w_dense > 0:
            self.stage = "memuat embedding dan model dense"
            self._load_dense(data_dir)
        self.stage = "memuat indeks leksikal"
        self._load_lexical(data_dir)
        self.stage = "siap"
        self.ready = True
        log(f"siap: k={self.k}, depth={self.depth}, w_dense={self.w_dense}")

    def _load_dense(self, data_dir):
        import torch
        from sentence_transformers import SentenceTransformer
        ep = json.load(open(data_dir / "embed_params.json"))
        assert ep["n_chunks"] == self.N, "embedding dan tabel chunk tidak cocok"
        self.model = SentenceTransformer(ep["model"], device="cpu")
        self.model.max_seq_length = ep["max_seq_length"]
        arr = np.load(data_dir / "chunk_emb_fp16.npy")
        self.E = torch.from_numpy(arr).float()           # fp16 -> fp32 (CPU lebih cepat dan stabil)
        del arr

    def _load_lexical(self, data_dir):
        lp = json.load(open(data_dir / "lexical_params.json"))
        assert lp["n_chunks"] == self.N, "indeks leksikal dan tabel chunk tidak cocok"
        self.bm25 = sp.load_npz(data_dir / "bm25_csc.npz")
        self.tfidf = sp.load_npz(data_dir / "tfidf_csc.npz")
        self.idf_c = np.load(data_dir / "tfidf_idf.npy")
        self.vocab = json.load(open(data_dir / "bm25_vocab.json"))
        self.sublinear = lp["sublinear_tf"]
        self.hv = HashingVectorizer(analyzer="char", ngram_range=tuple(lp["ngram"]),
                                    n_features=lp["hash_features"], alternate_sign=False, norm=None,
                                    lowercase=False, dtype=np.float32)

    # ------------------------------------------------------------------ retriever
    def search_bm25(self, q, k=K_CAND):
        ids = [self.vocab[t] for t in set(self.normalize_lex(q).split()) if t in self.vocab]
        if not ids:
            return EMPTY
        return topk(score_postings(self.bm25, ids, np.ones(len(ids), np.float32), self.N), k)

    def search_tfidf(self, q, k=K_CAND):
        qv = self.hv.transform([self.normalize_lex(q)])
        ids = qv.indices
        if len(ids) == 0:
            return EMPTY
        w = qv.data.astype(np.float32)
        if self.sublinear:
            w = 1 + np.log(w)
        w = w * self.idf_c[ids]
        n = np.linalg.norm(w)
        if n == 0:
            return EMPTY
        return topk(score_postings(self.tfidf, ids, w / n, self.N), k)

    def search_dense(self, q, k=K_CAND):
        import torch
        text = " ".join(self.clean_lyrics(q).split("\n"))
        with torch.no_grad():
            qe = self.model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
            _, ix = torch.topk(self.E @ qe[0].float(), k)
        return ix.cpu().numpy()

    # ------------------------------------------------------------------ fusi
    def search(self, q, top=10, mode="hybrid"):
        """RRF berbobot di tingkat chunk -> agregasi max ke lagu -> daftar hasil."""
        w_dense = self.w_dense if mode == "hybrid" else 0.0
        lists = [("bm25", self.search_bm25(q)[0], 1.0), ("tfidf", self.search_tfidf(q)[0], 1.0)]
        if w_dense > 0:
            lists.append(("dense", self.search_dense(q), w_dense))
        sc, rank_in = {}, {}
        for name, ids, wt in lists:
            for r in range(min(self.depth, len(ids))):
                c = int(ids[r])
                if c < 0:
                    break
                sc[c] = sc.get(c, 0.0) + wt / (self.k + r + 1)
                rank_in.setdefault(c, {})[name] = r + 1
        best = {}
        for c, s in sc.items():
            sid = int(self.song_of[c])
            if sid not in best or s > best[sid][0]:
                best[sid] = (s, c)
        out = []
        for pos, (sid, (s, c)) in enumerate(sorted(best.items(), key=lambda x: -x[1][0])[:top], 1):
            r = rank_in[c]
            out.append({
                "rank": pos, "song_id": sid,
                "title": self.title_of.get(sid, ""), "artist": self.artist_of.get(sid, ""),
                "score": round(s, 5), "match": LABEL[min(r, key=r.get)],
                "snippet": self.snippet_of[c].replace("\n", " / "),
            })
        return out
