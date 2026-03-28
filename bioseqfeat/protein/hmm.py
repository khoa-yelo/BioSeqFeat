"""
HMM profile scoring featurizer using hmmscan against a pre-built HMM database.

The output is a fixed-length vector (one entry per HMM profile in the database)
containing the best per-sequence bit score, or 0.0 for profiles with no hit.

Requires HMMER3 (hmmscan, hmmpress) to be installed and on PATH.
The HMM database must be pressed first:
    hmmpress Pfam-A.subset.hmm
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ..base import Featurizer


def _get_hmm_names(hmm_db: str) -> list[str]:
    """Return ordered list of profile NAME fields from an HMM file."""
    names = []
    open_fn = __import__("gzip").open if hmm_db.endswith(".gz") else open
    with open_fn(hmm_db, "rt") as fh:
        for line in fh:
            if line.startswith("NAME"):
                names.append(line.split()[1])
    return names


def _run_hmmscan(
    seq: str,
    hmm_db: str,
    e_thresh: float,
    cpu: int,
    report_all_scores: bool = False,
) -> dict[str, float]:
    """
    Run hmmscan for one sequence; return {hmm_name: best_bit_score}.

    Uses --tblout to get per-sequence scores, then takes the max bit score
    per profile.  Only hits with E-value <= e_thresh are kept, unless
    report_all_scores=True, in which case all profiles are reported with
    their raw bit scores (E-value threshold is effectively disabled).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "query.fasta")
        tblout_path = os.path.join(tmpdir, "out.tblout")

        with open(fasta_path, "w") as f:
            clean = ''.join(c for c in str(seq) if c.isprintable() and c != '-')
            f.write(f">query\n{clean}\n")

        effective_e = "1e6" if report_all_scores else str(e_thresh)
        cmd = [
            "hmmscan",
            "--tblout", tblout_path,
            "--noali",
            "-E", effective_e,
            "--domE", effective_e,
            "--cpu", str(cpu),
            hmm_db,
            fasta_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"hmmscan failed:\n{result.stderr}")

        scores: dict[str, float] = {}
        with open(tblout_path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.split()
                # tblout columns: target_name, accession, query_name, accession,
                #                 E-value, score, bias, ...
                hmm_name = cols[0]
                e_value = float(cols[4])
                bit_score = float(cols[5])
                if report_all_scores or e_value <= e_thresh:
                    if hmm_name not in scores or bit_score > scores[hmm_name]:
                        scores[hmm_name] = bit_score

    return scores


class HMMFeaturizer(Featurizer):
    """Score a protein sequence against a subset HMM database.

    Each position in the output vector corresponds to one HMM profile (ordered
    by their appearance in the .hmm file). The value is the best per-sequence
    bit score from hmmscan, or 0.0 if no hit was found at the given E-value
    threshold.

    The HMM database must be pressed with ``hmmpress`` before use.

    Parameters
    ----------
    hmm_db : str or Path
        Path to the pressed HMM database (e.g. ``Pfam-A.subset.hmm``).
    e_thresh : float
        Per-sequence E-value threshold passed to hmmscan (default: 10.0,
        keeps weak hits so the vector is not too sparse for short sequences).
        Ignored when ``report_all_scores=True``.
    cpu : int
        Number of CPU threads for hmmscan (default: 1).
    report_all_scores : bool
        If True, disable the E-value filter so that every profile in the
        database receives its raw bit score (instead of 0.0 for sub-threshold
        profiles). Produces a denser, fully continuous feature vector but
        increases runtime. Default: False.

    Example
    -------
    >>> feat = HMMFeaturizer("db/Pfam-A.subset.hmm")
    >>> vec = feat.extract_one("MKTLLLTLVVVTIVCLDLGYTPVS...")
    >>> vec.shape
    (1011,)
    """

    name = "hmm_profile"

    def __init__(
        self,
        hmm_db: str | Path,
        e_thresh: float = 10.0,
        cpu: int = 1,
        report_all_scores: bool = False,
    ):
        self.hmm_db = str(hmm_db)
        self.e_thresh = e_thresh
        self.cpu = cpu
        self.report_all_scores = report_all_scores

        # Load and cache the ordered profile names once
        self._profile_names: list[str] = _get_hmm_names(self.hmm_db)
        self._name_to_idx: dict[str, int] = {
            n: i for i, n in enumerate(self._profile_names)
        }

    @property
    def n_profiles(self) -> int:
        return len(self._profile_names)

    @property
    def profile_names(self) -> list[str]:
        return list(self._profile_names)

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        """Return a (n_profiles,) bit-score vector for one protein sequence."""
        vec = np.zeros(self.n_profiles, dtype=np.float32)
        scores = _run_hmmscan(seq, self.hmm_db, self.e_thresh, self.cpu, self.report_all_scores)
        for name, score in scores.items():
            idx = self._name_to_idx.get(name)
            if idx is not None:
                vec[idx] = score
        return vec


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Score a protein sequence against an HMM subset database."
    )
    parser.add_argument("sequence", help="Protein sequence string (single-letter AA)")
    parser.add_argument(
        "--hmm-db",
        default=os.path.join(os.path.dirname(__file__), "../../../db/Pfam-A.subset.hmm"),
        help="Path to pressed HMM database (default: db/Pfam-A.subset.hmm)",
    )
    parser.add_argument("--e-thresh", type=float, default=10.0,
                        help="hmmscan E-value threshold (default: 10.0)")
    parser.add_argument("--cpu", type=int, default=1,
                        help="Number of CPU threads for hmmscan (default: 1)")
    parser.add_argument("--output", choices=["vector", "named", "numpy"],
                        default="named",
                        help="Output format: 'vector' (plain scores), "
                             "'named' (JSON dict of nonzero hits), "
                             "'numpy' (save .npy to stdout path)")
    parser.add_argument("--out-file", default=None,
                        help="Save numpy vector to this .npy file (with --output numpy)")
    args = parser.parse_args()

    hmm_db = os.path.realpath(args.hmm_db)
    feat = HMMFeaturizer(hmm_db, e_thresh=args.e_thresh, cpu=args.cpu)
    vec = feat.extract_one(args.sequence)

    if args.output == "vector":
        print(" ".join(f"{v:.4f}" for v in vec))
    elif args.output == "named":
        hits = {feat.profile_names[i]: float(vec[i])
                for i in np.nonzero(vec)[0]}
        print(json.dumps(hits, indent=2))
    elif args.output == "numpy":
        if args.out_file is None:
            parser.error("--out-file required with --output numpy")
        np.save(args.out_file, vec)
        print(f"Saved {vec.shape} vector to {args.out_file}")


if __name__ == "__main__":
    main()
