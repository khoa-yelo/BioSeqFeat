import numpy as np
from typing import Literal
from Bio.Align import substitution_matrices
 
# ---------------------------------------------------------------------------
# Load BLOSUM62 from Biopython and build a numpy lookup matrix
# ---------------------------------------------------------------------------
_BLOSUM62_BIO = substitution_matrices.load("BLOSUM62")
 
# Use the 20 standard amino acids only (skip B, Z, X, *)
AA_ORDER = [aa for aa in _BLOSUM62_BIO.alphabet if aa in "ARNDCQEGHILKMFPSTWYV"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
EMBED_DIM = len(AA_ORDER)  # 20
 
# Build (20, 20) numpy matrix from the Biopython object
BLOSUM62 = np.array(
    [[_BLOSUM62_BIO[a, b] for b in AA_ORDER] for a in AA_ORDER],
    dtype=np.float32,
)
 
 
def _seq_to_embeddings(seq: str) -> np.ndarray:
    """Convert a protein sequence to an (L, 20) array of BLOSUM62 row vectors.
    Non-standard residues (X, B, Z, etc.) are skipped."""
    rows = []
    for aa in seq.upper():
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            rows.append(BLOSUM62[idx])
    if not rows:
        raise ValueError(f"No valid amino acids found in sequence: {seq[:30]}...")
    return np.stack(rows)  # (L, 20)
 
 
# -----------------------------------------------------------------------
# Function 1: Average embedding (mean pooling)
# -----------------------------------------------------------------------
def average_embedding(seq: str) -> np.ndarray:
    emb = _seq_to_embeddings(seq)  # (L, 20)
    return emb.mean(axis=0)        # (20,)
 
 
# -----------------------------------------------------------------------
# Function 2: Fixed-dimension compression
# -----------------------------------------------------------------------
def compress_sequence(
    seq: str,
    dim: int = 20,
    method: Literal["moving_avg", "dct", "adaptive_pool"] = "adaptive_pool",
) -> np.ndarray:
    emb = _seq_to_embeddings(seq)  # (L, 20)
    L = emb.shape[0]
 
    if L <= dim:
        padded = np.zeros((dim, 20), dtype=np.float32)
        padded[:L] = emb
        return padded.ravel()
 
    if method == "moving_avg":
        return _moving_avg_pool(emb, dim).ravel()
    elif method == "adaptive_pool":
        return _adaptive_avg_pool(emb, dim).ravel()
    elif method == "dct":
        return _dct_compress(emb, dim).ravel()
    else:
        raise ValueError(f"Unknown method: {method}")
 
 
# --- compression backends ---
 
def _moving_avg_pool(emb: np.ndarray, dim: int) -> np.ndarray:
    L = emb.shape[0]
    out = np.zeros((dim, 20), dtype=np.float32)
    for i in range(dim):
        start = int(i * L / dim)
        end = int((i + 1) * L / dim)
        out[i] = emb[start:end].mean(axis=0)
    return out
 
 
def _adaptive_avg_pool(emb: np.ndarray, dim: int) -> np.ndarray:
    L = emb.shape[0]
    out = np.zeros((dim, 20), dtype=np.float32)
    for i in range(dim):
        s = i * L / dim
        e = (i + 1) * L / dim
        si, ei = int(np.floor(s)), int(np.ceil(e))
        if si == ei:
            ei = si + 1
        window = emb[si:ei].copy()
        if s - si > 0:
            window[0] *= (1.0 - (s - si))
        if ei - e > 0 and ei - si > 0:
            window[-1] *= (1.0 - (ei - e))
        total_weight = e - s
        out[i] = window.sum(axis=0) / total_weight
    return out
 
 
def _dct_compress(emb: np.ndarray, dim: int) -> np.ndarray:
    try:
        from scipy.fft import dct
    except ImportError:
        from scipy.fftpack import dct
    coeffs = dct(emb, type=2, axis=0, norm="ortho")  # (L, 20)
    return coeffs[:dim].astype(np.float32)
