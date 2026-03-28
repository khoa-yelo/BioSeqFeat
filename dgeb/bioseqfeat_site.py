"""
functional_site_featurizer.py
─────────────────────────────
Reference-free functional site extraction and featurization
for BioSeqFeat. Designed to complement BLOSUM/AAC/DPC/PseAAC
by focusing on functionally decisive residue positions.

Architecture mirrors BioSeqFeat's Featurizer base class pattern.

Features produced (all reference-free):
  1. ProSite-style regex presence vector       (~40-dim binary)
  2. Metal / cofactor motif presence vector    (~15-dim binary)
  3. Active site physicochemical features      (15-dim float)
  4. Active site BLOSUM features               (20-dim float)
  5. Active site k-mer features                (variable, default 400-dim)
  6. BLOSUM-entropy site physicochemical       (15-dim float)
  7. Site density / coverage statistics        (5-dim float)

Total default output: ~510-dim
"""

from __future__ import annotations

import re
from itertools import product
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bioseqfeat.base import Featurizer

# ─────────────────────────────────────────────────────────────
# BLOSUM62 matrix (20 standard AAs)
# ─────────────────────────────────────────────────────────────

BLOSUM62: Dict[str, np.ndarray] = {
    "A": np.array([ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0]),
    "R": np.array([-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3]),
    "N": np.array([-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3]),
    "D": np.array([-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3]),
    "C": np.array([ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1]),
    "Q": np.array([-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2]),
    "E": np.array([-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2]),
    "G": np.array([ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3]),
    "H": np.array([-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3]),
    "I": np.array([-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3]),
    "L": np.array([-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1]),
    "K": np.array([-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2]),
    "M": np.array([-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1]),
    "F": np.array([-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1]),
    "P": np.array([-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2]),
    "S": np.array([ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2]),
    "T": np.array([ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0]),
    "W": np.array([-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3]),
    "Y": np.array([-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1]),
    "V": np.array([ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]),
}

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# ─────────────────────────────────────────────────────────────
# Physicochemical property scales
# ─────────────────────────────────────────────────────────────

HYDROPHOBICITY = {  # Kyte-Doolittle
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

CHARGE = {  # at pH 7
    "A": 0,  "R": 1,  "N": 0,  "D": -1, "C": 0,
    "Q": 0,  "E": -1, "G": 0,  "H": 0.1,"I": 0,
    "L": 0,  "K": 1,  "M": 0,  "F": 0,  "P": 0,
    "S": 0,  "T": 0,  "W": 0,  "Y": 0,  "V": 0,
}

VOLUME = {  # Å³ (Zimmerman)
    "A": 52.6,  "R": 109.1, "N": 75.7,  "D": 68.4,  "C": 68.3,
    "Q": 89.7,  "E": 84.7,  "G": 36.3,  "H": 91.9,  "I": 102.0,
    "L": 102.0, "K": 105.1, "M": 97.7,  "F": 113.9, "P": 73.6,
    "S": 54.9,  "T": 71.2,  "W": 135.4, "Y": 116.2, "V": 85.1,
}

FLEXIBILITY = {  # Vihinen
    "A": 0.360, "R": 0.530, "N": 0.460, "D": 0.510, "C": 0.350,
    "Q": 0.490, "E": 0.500, "G": 0.540, "H": 0.320, "I": 0.460,
    "L": 0.370, "K": 0.470, "M": 0.300, "F": 0.310, "P": 0.510,
    "S": 0.510, "T": 0.440, "W": 0.310, "Y": 0.420, "V": 0.390,
}

POLARITY = {  # Grantham
    "A": 8.1,  "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "Q": 10.5, "E": 12.3, "G": 9.0,  "H": 10.4, "I": 5.2,
    "L": 4.9,  "K": 11.3, "M": 5.7,  "F": 5.2,  "P": 8.0,
    "S": 9.2,  "T": 8.6,  "W": 5.4,  "Y": 6.2,  "V": 5.9,
}

PROPERTIES = {
    "hydrophobicity": HYDROPHOBICITY,
    "charge": CHARGE,
    "volume": VOLUME,
    "flexibility": FLEXIBILITY,
    "polarity": POLARITY,
}

# ─────────────────────────────────────────────────────────────
# Functional site patterns
# ─────────────────────────────────────────────────────────────

PROSITE_PATTERNS: Dict[str, str] = {
    # Nucleotide binding
    "p_loop_walker_a":      r"G.{4}GK[ST]",
    "walker_b":             r"[LIVMF]{4}D[EQ]",
    "atp_binding_1":        r"[LIVMF]G.G..G",
    "gtp_binding":          r"[AG]....GK[ST]",
    # Serine proteases
    "serine_protease_ser":  r"G[DST][ST][GAS][AS][STV][PAGV]",
    "serine_protease_his":  r"[LIVM][LIVM][DN].G[LIVM].H",
    # Cysteine proteases
    "cysteine_protease":    r"[WF].C[SA].A[ST]",
    # Kinases
    "protein_kinase_atp":   r"[LIV]G[^EQ]G[^DW].G",
    "protein_kinase_active": r"D[LIVMF]K",
    # NRPS adenylation domain (Stachelhaus-adjacent)
    "nrps_a_domain_core":   r"[LIVMF].GD[SA][SA][LI]",
    "nrps_condensation":    r"HH.{3}D[GA]",
    # PKS
    "pks_ks_active":        r"GH[ST][AG]",
    "pks_acyl_carrier":     r"GG[DE][ST][LIVM]",
    # Oxidoreductases
    "rossmann_fold":        r"[LIVM]{2}.G[STAGC].G[STAGC][STAGC]",
    "fad_binding_1":        r"GG.{2}[LIV].{4}[DEQNH]",
    "nad_binding":          r"[LIVMF][LIVMF].G.G..G",
    # Glycosylases / glycosidases
    "glycoside_hydrolase":  r"[LIVM][LIVM][NQ].EP",
    # RNase / DNase
    "rnase_his":            r"[LIVM][LIVM]H[DE][ST]",
    # Zinc finger (various)
    "zinc_finger_c2h2":     r"C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H",
    "zinc_finger_c4":       r"C.{2}C.{3}[LIVMF].{8}C.{2}C",
    # Helicase
    "helicase_atp":         r"[DE]EAH",
    # Thioredoxin
    "thioredoxin":          r"[LIV].CP.C",
    # Isomerases
    "tim_barrel_phospho":   r"[STAGC][STAGC][HKR][LIVMF]",
}

METAL_MOTIFS: Dict[str, str] = {
    "heme_cxxch":           r"C.{2}C.H",
    "iron_sulfur_c4":       r"C.{2}C.{2}C.{3}C",
    "iron_sulfur_c3":       r"C.{2}C.C",
    "zinc_his_3":           r"H.{2,4}H.{8,30}H",
    "zinc_cys_3":           r"C.{2,4}C.{8,30}C",
    "copper_type1":         r"H.{0,2}C.{2}[HY]",
    "calcium_ef":           r"[DNSEGQAVMI].{0,2}[DNSEGQAVMI].{3}[DNSEGQAVMI].{3}[DE]",
    "manganese_his":        r"H.{0,3}[DE].{0,3}H",
    "nickel_binding":       r"[HDE].{3,6}[HDE].{3,6}[HDE]",
    "molybdenum_binding":   r"C[GAS]..C",
    "selenium_sec":         r"U",  # selenocysteine
}

SIGNAL_PATTERNS: Dict[str, str] = {
    "signal_peptide_n":     r"^M[LIVMFYW]{2,5}[LIVMFYW]{7,15}[ASG]",
    "nls_monopartite":      r"K[RK][^DE]{0,1}[RK]",
    "nls_bipartite":        r"[RK]{2}.{10}[RK]{3}",
    "tat_signal":           r"[ST][ST]RR",
    "gpi_anchor_c":         r"[AGSTNQLIVMF]{3,8}$",
}

ALL_PATTERNS = {**PROSITE_PATTERNS, **METAL_MOTIFS, **SIGNAL_PATTERNS}

# ─────────────────────────────────────────────────────────────
# Core helper functions
# ─────────────────────────────────────────────────────────────

def _safe_aa(aa: str) -> str:
    """Return aa if standard, else 'A' as fallback."""
    return aa if aa in BLOSUM62 else "A"


def _prop(aa: str, prop: Dict[str, float], default: float = 0.0) -> float:
    return prop.get(aa, default)


def _blosum_entropy(aa: str) -> float:
    """
    Shannon entropy of softmax(BLOSUM62 row).
    Low entropy → strongly constrained → likely functional.
    """
    row = BLOSUM62.get(aa, np.zeros(20)).astype(float)
    shifted = row - row.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


 
def blosum_entropy_sites(seq: str, top_k: int = 20,
                         exclude_scaffold: bool = True) -> List[int]:
    """
    Predict functional positions by lowest BLOSUM entropy
    (most evolutionarily constrained residues).
 
    Excludes C, W, P by default — these are structurally conserved
    across almost all proteins (disulfide bonds, structural tryptophans,
    proline kinks) and their low entropy reflects scaffold constraints
    rather than functional chemistry. Excluding them pushes the selection
    toward H, D, E, S, K — the residues that actually do catalysis.
 
    Returns sorted list of position indices.
    """
    SCAFFOLD = set("CWP")
    entropies = []
    for i, aa in enumerate(seq):
        aa_c = _safe_aa(aa)
        if exclude_scaffold and aa_c in SCAFFOLD:
            ent = float("inf")  # deprioritize scaffold residues
        else:
            ent = _blosum_entropy(aa_c)
        entropies.append((ent, i))
    entropies.sort()  # ascending: lowest entropy first
    return [i for _, i in entropies[:top_k]]
 
 
def scan_patterns(seq: str, patterns: Dict[str, str]) -> Dict[str, List[int]]:
    """
    Scan sequence for all regex patterns.
    Returns dict of pattern_name → list of match start positions.
    """
    hits: Dict[str, List[int]] = {}
    for name, pattern in patterns.items():
        try:
            hits[name] = [m.start() for m in re.finditer(pattern, seq)]
        except re.error:
            hits[name] = []
    return hits
 
 
def get_site_positions(
    seq: str,
    use_prosite: bool = True,
    use_metal: bool = True,
    use_signal: bool = True,
    use_entropy_fallback: bool = True,
    entropy_top_k: int = 20,
    min_sites: int = 3,
) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Get functional site positions from all sources.
    Falls back to BLOSUM entropy if rule-based hits are sparse.
 
    Returns:
        positions: sorted unique list of functional positions
        hits: raw pattern hit dict for presence vector
    """
    patterns = {}
    if use_prosite:
        patterns.update(PROSITE_PATTERNS)
    if use_metal:
        patterns.update(METAL_MOTIFS)
    if use_signal:
        patterns.update(SIGNAL_PATTERNS)
 
    hits = scan_patterns(seq, patterns)
 
    # Collect all hit positions
    positions = set()
    for pos_list in hits.values():
        positions.update(pos_list)
 
    # Expand: include residues within 2 positions of each hit start
    expanded = set()
    for pos in positions:
        for offset in range(-2, 5):
            idx = pos + offset
            if 0 <= idx < len(seq):
                expanded.add(idx)
 
    # Fallback to entropy-based sites if coverage is low
    if len(expanded) < min_sites and use_entropy_fallback:
        entropy_positions = blosum_entropy_sites(seq, top_k=entropy_top_k)
        expanded.update(entropy_positions)
 
    return sorted(expanded), hits
 
 
# ─────────────────────────────────────────────────────────────
# Feature extraction functions
# ─────────────────────────────────────────────────────────────
 
def presence_vector(hits: Dict[str, List[int]]) -> np.ndarray:
    """
    Binary vector: 1 if pattern has ≥1 hit, 0 otherwise.
    Order matches ALL_PATTERNS key order.
    """
    keys = list(ALL_PATTERNS.keys())
    return np.array([
        1.0 if hits.get(k) else 0.0
        for k in keys
    ])
 
 
def site_physicochemical(seq: str, positions: List[int]) -> np.ndarray:
    """
    15-dim physicochemical summary of residues at functional sites.
    """
    if not positions:
        return np.zeros(15)

    site_aas = [seq[i] for i in positions if i < len(seq)]
    if not site_aas:
        return np.zeros(15)

    features = []
    for prop_name, prop_dict in PROPERTIES.items():
        vals = [_prop(aa, prop_dict) for aa in site_aas]
        features.extend([
            np.mean(vals),
            np.std(vals),
            np.max(vals) - np.min(vals),  # range
    ])

    return np.array(features, dtype=float)  # 5 props × 3 stats = 15-dim


def site_blosum(seq: str, positions: List[int]) -> np.ndarray:
    """
    20-dim mean BLOSUM62 encoding restricted to functional site residues.
    Removes scaffold noise; encodes substitutability of active site.
    """
    if not positions:
        return np.zeros(20)

    site_aas = [seq[i] for i in positions if i < len(seq)]
    if not site_aas:
        return np.zeros(20)

    rows = np.array([BLOSUM62.get(_safe_aa(aa), np.zeros(20)) for aa in site_aas])
    return rows.mean(axis=0).astype(float)


def site_kmer(
    seq: str,
    positions: List[int],
    k: int = 2,
    window: int = 4,
) -> np.ndarray:
    """
    k-mer frequency vector computed from windows around functional sites.
    Default k=2 (dipeptide-level, 400-dim) for speed.
    Uses window around each site position to capture local context.
    """
    aas = AA_ORDER
    kmer_idx = {"".join(p): i for i, p in enumerate(product(aas, repeat=k))}
    counts = np.zeros(len(kmer_idx))

    for pos in positions:
        start = max(0, pos - window)
        end = min(len(seq), pos + window + 1)
        subseq = seq[start:end]
        for j in range(len(subseq) - k + 1):
            kmer = subseq[j:j+k]
            if kmer in kmer_idx:
                counts[kmer_idx[kmer]] += 1

    total = counts.sum()
    if total > 0:
        counts /= total

    return counts


def site_conservation_stats(seq: str, positions: List[int]) -> np.ndarray:
    """
    5-dim statistics about functional site coverage and conservation.
    """
    L = len(seq)
    if L == 0:
        return np.zeros(5)

    site_entropies = [_blosum_entropy(_safe_aa(seq[i])) for i in positions if i < L]
    n_sites = len(positions)

    return np.array([
        n_sites / L,                                        # site density
        n_sites,                                            # raw count
        np.mean(site_entropies) if site_entropies else 0.0, # mean entropy at sites
        np.std(site_entropies) if site_entropies else 0.0,  # entropy variation
        np.min(site_entropies) if site_entropies else 0.0,  # most conserved site
    ], dtype=float)


# ─────────────────────────────────────────────────────────────
# Main Featurizer class
# ─────────────────────────────────────────────────────────────

class FunctionalSiteFeaturizer(Featurizer):
    """
    Reference-free functional site featurizer for BioSeqFeat.

    Produces a fixed-length feature vector by:
      1. Detecting functional sites via regex patterns (ProSite-style,
         metal motifs, signal patterns) + BLOSUM entropy fallback
      2. Computing features restricted to those sites

    Compatible with BioSeqFeat's Featurizer / Pipeline architecture.

    Parameters
    ----------
    use_prosite : bool
        Include ProSite-style catalytic/binding patterns.
    use_metal : bool
        Include metal coordination motifs.
    use_signal : bool
        Include signal peptide / sorting signal patterns.
    use_entropy_fallback : bool
        Fall back to BLOSUM-entropy site prediction when pattern
        coverage is low. Ensures all proteins get non-zero site features.
    entropy_top_k : int
        Number of top conserved positions to use as fallback sites.
    site_window : int
        Window around each hit position to include as "site" context.
    kmer_k : int
        k for site k-mer features (default 2 = dipeptide, 400-dim).
    min_sites : int
        Minimum number of sites before triggering entropy fallback.
    """

    name: str = "functional_site"

    def __init__(
        self,
        use_prosite: bool = True,
        use_metal: bool = True,
        use_signal: bool = True,
        use_entropy_fallback: bool = True,
        entropy_top_k: int = 20,
        site_window: int = 4,
        kmer_k: int = 2,
        min_sites: int = 3,
    ):
        self.use_prosite = use_prosite
        self.use_metal = use_metal
        self.use_signal = use_signal
        self.use_entropy_fallback = use_entropy_fallback
        self.entropy_top_k = entropy_top_k
        self.site_window = site_window
        self.kmer_k = kmer_k
        self.min_sites = min_sites

        # Pre-compute output dimension
        self._n_patterns = len(ALL_PATTERNS)
        self._kmer_dim = 20 ** kmer_k
        self.output_dim = (
            self._n_patterns   # presence vector
            + 15               # site physicochemical
            + 20               # site BLOSUM
            + self._kmer_dim   # site k-mer
            + 15               # entropy-site physicochemical
            + 5                # coverage stats
        )

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        """
        Extract functional site features from a protein sequence.

        Parameters
        ----------
        seq : str
            Protein sequence (standard 1-letter AA codes).

        Returns
        -------
        np.ndarray of shape (output_dim,)
        """
        seq = seq.upper().strip()
        if not seq:
            return np.zeros(self.output_dim)

        # ── 1. Detect functional sites ──────────────────────────
        site_positions, hits = get_site_positions(
            seq,
            use_prosite=self.use_prosite,
            use_metal=self.use_metal,
            use_signal=self.use_signal,
            use_entropy_fallback=self.use_entropy_fallback,
            entropy_top_k=self.entropy_top_k,
            min_sites=self.min_sites,
        )

        # ── 2. Entropy-based sites (always, independently) ──────
        entropy_positions = blosum_entropy_sites(seq, top_k=self.entropy_top_k)

        # ── 3. Compute sub-features ─────────────────────────────
        f_presence = presence_vector(hits)
        f_site_pc  = site_physicochemical(seq, site_positions)
        f_site_bl  = site_blosum(seq, site_positions)
        f_site_km  = site_kmer(seq, site_positions, k=self.kmer_k, window=self.site_window)
        f_entr_pc  = site_physicochemical(seq, entropy_positions)
        f_stats    = site_conservation_stats(seq, site_positions)

        return np.concatenate([
            f_presence,
            f_site_pc,
            f_site_bl,
            f_site_km,
            f_entr_pc,
            f_stats,
        ]).astype(np.float32)

    def feature_names(self) -> List[str]:
        """Return human-readable feature names for interpretability."""
        names = []

        # Presence vector
        for k in ALL_PATTERNS:
            names.append(f"pattern_{k}")

        # Site physicochemical
        for prop in PROPERTIES:
            for stat in ["mean", "std", "range"]:
                names.append(f"site_{prop}_{stat}")

        # Site BLOSUM
        for aa in AA_ORDER:
            names.append(f"site_blosum_{aa}")

        # Site k-mer
        for p in product(AA_ORDER, repeat=self.kmer_k):
            names.append(f"site_kmer_{''.join(p)}")

        # Entropy-site physicochemical
        for prop in PROPERTIES:
            for stat in ["mean", "std", "range"]:
                names.append(f"entropy_{prop}_{stat}")

        # Coverage stats
        for stat in ["density", "count", "mean_entropy", "std_entropy", "min_entropy"]:
            names.append(f"site_{stat}")

        return names

    def __repr__(self) -> str:
        return (
            f"FunctionalSiteFeaturizer("
            f"prosite={self.use_prosite}, "
            f"metal={self.use_metal}, "
            f"signal={self.use_signal}, "
            f"entropy_fallback={self.use_entropy_fallback}, "
            f"kmer_k={self.kmer_k}, "
            f"output_dim={self.output_dim})"
        )


# ─────────────────────────────────────────────────────────────
# Specialized subclasses for specific task domains
# ─────────────────────────────────────────────────────────────

class BGCFeaturizer(FunctionalSiteFeaturizer):
    """
    Specialized for BGC / MIBiG classification.
    Upweights NRPS/PKS patterns and adds Stachelhaus-proxy features.
    """

    BGC_PATTERNS = {
        "nrps_a_core":      r"[LIVMF].GD[SA][SA][LI]",
        "nrps_c_domain":    r"HH.{3}D[GA]",
        "nrps_t_domain":    r"GG[DE][ST]",
        "pks_ks":           r"GH[ST][AG]",
        "pks_at_motif1":    r"GH[GS].G",
        "pks_at_motif2":    r"[LIVMF]FP",
        "pks_acp":          r"GG[DE][ST][LIVM]",
        "terpene_cyclase":  r"DD.{2}D",
        "halogenase":       r"[WF][WF].GG",
        "methyltransf":     r"[VL]LD[VL]G[GA]G",
        "beta_lactam":      r"[STA][STA][KR].{3}[KR]",
        "aminoglycoside":   r"[DE][LIVMF]{3}G[GA]",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add BGC-specific patterns to scanning
        ALL_PATTERNS.update(self.BGC_PATTERNS)
        self._n_patterns = len(ALL_PATTERNS)
        self.output_dim = (
            self._n_patterns + 15 + 20 + self._kmer_dim + 15 + 5
        )


class ConvergentECFeaturizer(FunctionalSiteFeaturizer):
    """
    Specialized for convergent evolution / EC retrieval.
    Uses entropy-heavy weighting — prioritizes physicochemical
    features at the most constrained positions.
    """

    def __init__(self, entropy_top_k: int = 15, **kwargs):
        super().__init__(
            use_entropy_fallback=True,
            entropy_top_k=entropy_top_k,
            **kwargs,
        )

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        """
        For convergent EC: weight site features by inverse entropy.
        Most constrained residues dominate the embedding.
        """
        base = super().extract_one(seq)

        seq = seq.upper().strip()
        if not seq:
            return base

        # Compute per-residue entropy weights
        entropies = np.array([_blosum_entropy(_safe_aa(aa)) for aa in seq])
        # Invert and normalize: low entropy → high weight
        weights = 1.0 / (entropies + 1e-3)
        weights /= weights.sum()

        # Weighted BLOSUM encoding (replaces simple mean)
        weighted_blosum = np.zeros(20)
        for i, aa in enumerate(seq):
            weighted_blosum += weights[i] * BLOSUM62.get(_safe_aa(aa), np.zeros(20))

        # Append as additional features
        return np.concatenate([base, weighted_blosum]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# DGEB-compatible wrapper
# ─────────────────────────────────────────────────────────────

class _BioSeqFeatEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, *args, **kwargs):
        return None


class _BioSeqFeatTokenizer:
    """Dummy tokenizer that stores raw sequences for use in encoding."""

    def __init__(self):
        self._last_sequences: list[str] = []

    def __call__(self, sequences, max_length=None, padding=True, truncation=True):
        self._last_sequences = list(sequences)
        batch_size = len(sequences)
        seq_len = max_length if max_length is not None else 10
        return {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


class BioSeqFeatSiteTransformer(BioSeqTransformer):
    """DGEB-compatible model that produces embeddings from functional site features.

    Uses FunctionalSiteFeaturizer (~494-dim by default): pattern presence vector,
    site physicochemical features, site BLOSUM encoding, site k-mer frequencies,
    entropy-site physicochemical features, and site coverage statistics.
    No HuggingFace model is loaded.

    Example
    -------
    >>> model = BioSeqFeatSiteTransformer()
    >>> evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    >>> results = evaluation.run(model, output_folder="results/bioseqfeat_site")
    """

    MODEL_NAMES = ["bioseqfeat-site"]

    def __init__(self, **kwargs):
        self._featurizer = FunctionalSiteFeaturizer()
        self._feat_dim = self._featurizer.output_dim
        self._tokenizer_instance = _BioSeqFeatTokenizer()
        kwargs.setdefault("num_processes", 0)
        super().__init__(model_name="bioseqfeat-site", **kwargs)

    def _load_model(self, model_name):
        return _BioSeqFeatEncoder(self._feat_dim)

    def _get_tokenizer(self, model_name):
        return self._tokenizer_instance

    def _encode_single_batch(self, batch_dict):
        sequences = self._tokenizer_instance._last_sequences
        features = self._featurizer.extract_batch(sequences)  # (batch_size, feat_dim)
        features_tensor = torch.from_numpy(features).float().to(self.device)
        # DGEB expects shape (batch_size, num_layers, embed_dim)
        return features_tensor.unsqueeze(1).expand(-1, len(self.layers), -1).contiguous()

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def embed_dim(self) -> int:
        return self._feat_dim

    @property
    def modality(self):
        return Modality.PROTEIN


# ─────────────────────────────────────────────────────────────
# Quick test / demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = BioSeqFeatSiteTransformer()
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    results = evaluation.run(model, output_folder="results/bioseqfeat_site")
    print(results)