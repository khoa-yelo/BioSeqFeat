"""
MMseqs2 landmark embedding featurizer.

Each protein is embedded as a (n_anchors,) vector of MMseqs2 bit-scores
against a fixed set of anchor/landmark proteins.  Anchors cover the target
sequence space (e.g. bacterial SwissProt representatives), so the embedding
encodes alignment-based similarity to that space — directly relevant for
cross-kingdom retrieval tasks.

Compared to HMM profiles this approach:
  - Produces position-aware alignment scores (not just family membership)
  - Can detect remote homologs that fall outside known Pfam families
  - Scales to any anchor set without retraining

Requires MMseqs2 to be installed and on PATH.

Typical usage
-------------
1. Prepare an anchor FASTA (e.g. bacterial SwissProt cluster representatives):

    mmseqs easy-cluster swissprot_bacteria.fasta clusters tmp --min-seq-id 0.5
    # use clusters_rep_seq.fasta as anchor_fasta

2. Build the featurizer (builds the MMseqs2 DB once):

    feat = MMseqsLandmark("anchors/bacteria_reps.fasta")
    vec = feat.extract_one("MKTLLLTLVVV...")   # shape (n_anchors,)

3. Use in a Pipeline:

    from bioseqfeat import Pipeline, NormalizedFeaturizer
    pipe = Pipeline(
        featurizers=[NormalizedFeaturizer(feat)],
        weights=[1.0],
    )
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ..base import Featurizer


def _parse_fasta_ids(fasta_path: str) -> list[str]:
    """Return ordered list of sequence IDs from a FASTA file."""
    ids = []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                # take first whitespace-delimited token after ">"
                ids.append(line[1:].split()[0])
    return ids


def _build_mmseqs_db(fasta_path: str, db_path: str) -> None:
    """Run mmseqs createdb to build a sequence database."""
    cmd = ["mmseqs", "createdb", fasta_path, db_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs createdb failed:\n{result.stderr}")


def _run_mmseqs_search(
    query_seq: str,
    target_db: str,
    e_thresh: float,
    sensitivity: float,
    cpu: int,
) -> dict[str, float]:
    """
    Search one query sequence against the target MMseqs2 DB.

    Returns {target_id: -log10(evalue)} for all hits passing e_thresh.
    Scores are continuous and graded: strong hits score high, weak hits
    score near zero, no-hits are absent (will be filled as 0.0 by the caller).
    Uses mmseqs easy-search (handles createdb + search + convertalis internally).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        query_fasta = os.path.join(tmpdir, "query.fasta")
        result_file = os.path.join(tmpdir, "results.tsv")
        tmp_path = os.path.join(tmpdir, "tmp")
        os.makedirs(tmp_path)

        clean = "".join(c for c in str(query_seq) if c.isalpha())
        with open(query_fasta, "w") as fh:
            fh.write(f">query\n{clean}\n")

        cmd = [
            "mmseqs", "easy-search",
            query_fasta,
            target_db,
            result_file,
            tmp_path,
            "-e", str(e_thresh),
            "-s", str(sensitivity),
            "--threads", str(cpu),
            "--format-output", "target,evalue",
            "-v", "0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mmseqs easy-search failed:\n{result.stderr}")

        scores: dict[str, float] = {}
        if os.path.exists(result_file):
            with open(result_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    target_id = parts[0]
                    try:
                        evalue = float(parts[1])
                    except ValueError:
                        continue
                    # -log10(evalue): strong hit → high score, weak hit → ~0
                    # clamp evalue to a minimum to avoid log(0)
                    evalue = max(evalue, 1e-300)
                    score = -np.log10(evalue)
                    # keep best (lowest evalue = highest score) per target
                    if target_id not in scores or score > scores[target_id]:
                        scores[target_id] = score

    return scores


class MMseqsLandmark(Featurizer):
    """Fixed-dim landmark embedding from MMseqs2 alignment scores.

    Each dimension corresponds to one anchor/landmark protein.  The value
    is ``-log10(evalue)`` of the best alignment between the query and that
    anchor (0.0 if no hit passes ``e_thresh``).  This gives a continuous,
    graded score: strong homologs score high, weak/distant ones score near
    zero, and true no-hits are zero — avoiding the sparse bit-score vectors
    that result from hard E-value cutoffs.

    The anchor FASTA is converted to a MMseqs2 sequence database on first
    construction (``mmseqs createdb``).  Subsequent calls to ``extract_one``
    run ``mmseqs easy-search`` for each query.

    Parameters
    ----------
    anchor_fasta : str or Path
        Path to a FASTA file of anchor/landmark proteins.  Sequence IDs
        must be unique (only the part before the first whitespace is used).
    db_dir : str or Path, optional
        Directory where the MMseqs2 database files are written.  Defaults
        to ``<anchor_fasta_dir>/mmseqs_db/``.  Re-used on subsequent runs
        if the DB already exists.
    e_thresh : float
        MMseqs2 E-value threshold (default 1e6 — very loose, captures even
        weak similarities; scores are -log10(evalue) so weak hits land near
        zero rather than being discarded entirely).
    sensitivity : float
        MMseqs2 sensitivity (-s flag, default 7.5 — high sensitivity mode,
        comparable to PSI-BLAST).  Range 1.0 (fast) to 7.5 (sensitive).
    cpu : int
        Number of threads (default 4).

    Example
    -------
    >>> feat = MMseqsLandmark("anchors/bacteria_reps.fasta")
    >>> vec = feat.extract_one("MKTLLLTLVVVTIVCLDLG...")
    >>> vec.shape
    (500,)   # depends on number of anchors
    """

    name = "mmseqs_landmark"

    def __init__(
        self,
        anchor_fasta: str | Path,
        db_dir: str | Path | None = None,
        e_thresh: float = 1e6,
        sensitivity: float = 7.5,
        cpu: int = 4,
    ):
        self.anchor_fasta = str(anchor_fasta)
        self.e_thresh = e_thresh
        self.sensitivity = sensitivity
        self.cpu = cpu

        # Determine DB directory
        if db_dir is None:
            db_dir = os.path.join(os.path.dirname(os.path.abspath(self.anchor_fasta)),
                                  "mmseqs_db")
        self._db_dir = str(db_dir)
        self._db_path = os.path.join(self._db_dir, "anchors")

        # Load anchor IDs (defines vector dimension and ordering)
        self._anchor_ids: list[str] = _parse_fasta_ids(self.anchor_fasta)
        if not self._anchor_ids:
            raise ValueError(f"No sequences found in anchor FASTA: {self.anchor_fasta}")
        self._id_to_idx: dict[str, int] = {
            aid: i for i, aid in enumerate(self._anchor_ids)
        }

        # Build MMseqs2 DB if not already present
        os.makedirs(self._db_dir, exist_ok=True)
        if not os.path.exists(self._db_path):
            _build_mmseqs_db(self.anchor_fasta, self._db_path)

    @property
    def n_anchors(self) -> int:
        return len(self._anchor_ids)

    @property
    def anchor_ids(self) -> list[str]:
        return list(self._anchor_ids)

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        """Return a (n_anchors,) bit-score vector for one protein sequence."""
        vec = np.zeros(self.n_anchors, dtype=np.float32)
        scores = _run_mmseqs_search(
            seq, self._db_path, self.e_thresh, self.sensitivity, self.cpu
        )
        for target_id, score in scores.items():
            idx = self._id_to_idx.get(target_id)
            if idx is not None:
                vec[idx] = score
        return vec

    def extract_batch(self, seqs: Sequence[str], **kwargs) -> np.ndarray:
        return np.stack([self.extract_one(s, **kwargs) for s in seqs])
