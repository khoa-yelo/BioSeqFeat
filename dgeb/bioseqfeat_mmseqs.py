"""
MMseqs2 landmark embedding pipeline for DGEB evaluation.

Each protein is embedded as a (n_anchors,) vector of MMseqs2 bit-scores
against a fixed set of bacterial anchor proteins, then L2-normalised.

The anchor set should be a representative sample of the target retrieval
space (bacterial proteins).  A good default is cluster representatives
from SwissProt bacteria at 50% sequence identity:

    # 1. Download bacterial SwissProt sequences
    #    (filter uniprot_sprot.fasta by taxonomy — see prepare_anchors.py)

    # 2. Cluster at 50% identity to get ~500-1000 representatives
    mmseqs easy-cluster bacteria_sprot.fasta clusters tmp --min-seq-id 0.5 -c 0.8
    cp clusters_rep_seq.fasta db/bacteria_anchors.fasta

    # MMseqsLandmark builds the MMseqs2 DB automatically on first run.

Usage
-----
    python bioseqfeat_mmseqs.py --anchor-fasta db/bacteria_anchors.fasta

Or combined with BLOSUM+HMM (recommended):
    python bioseqfeat_mmseqs.py --anchor-fasta db/bacteria_anchors.fasta --combined
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from types import SimpleNamespace
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality

from bioseqfeat import Pipeline, NormalizedFeaturizer, BlosumAvg, BlosumCompress
from bioseqfeat.protein.hmm import HMMFeaturizer
from bioseqfeat.protein.mmseqs import MMseqsLandmark

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db", "Pfam-A.subset.hmm",
)
_DEFAULT_ANCHOR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db", "bacteria_anchors.fasta",
)
_E_THRESH = 10.0
_CPU = 4


class _BioSeqFeatEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, *args, **kwargs):
        return None


class _BioSeqFeatTokenizer:
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


class BioSeqFeatMMseqsTransformer(BioSeqTransformer):
    """DGEB-compatible model using MMseqs2 landmark embeddings.

    Two modes controlled by ``combined``:

    False (default) — MMseqs2 landmarks only
        Embedding = L2-normalised (n_anchors,) bit-score vector.
        Fast baseline to test whether alignment-based embeddings beat
        composition/BLOSUM features on retrieval.

    True — BLOSUM + HMM + MMseqs2 landmarks
        Full retrieval-optimised pipeline combining evolutionary signals:
          BlosumAvg (w=5) + BlosumCompress (w=5) + HMM (w=3) + MMseqs2 (w=4)
        MMseqs2 weight is set highest-alongside-BLOSUM because landmark
        alignment directly encodes bacterial-space similarity.

    Parameters
    ----------
    anchor_fasta : str
        Path to the anchor FASTA file.
    combined : bool
        Whether to combine with BLOSUM+HMM features (default False).
    hmm_db : str
        Path to the pressed Pfam HMM database (used only when combined=True).
    e_thresh : float
        E-value threshold for MMseqs2 and hmmscan (default 10.0).
    sensitivity : float
        MMseqs2 sensitivity (default 7.5).
    cpu : int
        Number of CPU threads (default 4).
    """

    MODEL_NAMES = ["bioseqfeat-mmseqs", "bioseqfeat-mmseqs-combined"]

    def __init__(
        self,
        anchor_fasta: str = _DEFAULT_ANCHOR,
        combined: bool = False,
        hmm_db: str = _DB_PATH,
        e_thresh: float = _E_THRESH,
        sensitivity: float = 7.5,
        cpu: int = _CPU,
        **kwargs,
    ):
        self._combined = combined
        _mmseqs_raw = MMseqsLandmark(
            anchor_fasta, e_thresh=e_thresh, sensitivity=sensitivity, cpu=cpu
        )
        _mmseqs_norm = NormalizedFeaturizer(_mmseqs_raw)

        if combined:
            _hmm_raw = HMMFeaturizer(hmm_db, e_thresh=e_thresh, cpu=cpu)
            _hmm_norm = NormalizedFeaturizer(_hmm_raw)
            self._feat_dim = 20 + 400 + _hmm_raw.n_profiles + _mmseqs_raw.n_anchors
            self._pipeline = Pipeline(
                featurizers=[
                    NormalizedFeaturizer(BlosumAvg()),
                    NormalizedFeaturizer(BlosumCompress(dim=20, method="dct")),
                    _hmm_norm,
                    _mmseqs_norm,
                ],
                weights=[5.0, 5.0, 3.0, 4.0],
            )
            model_name = "bioseqfeat-mmseqs-combined"
        else:
            self._feat_dim = _mmseqs_raw.n_anchors
            self._pipeline = _mmseqs_norm
            model_name = "bioseqfeat-mmseqs"

        self._tokenizer_instance = _BioSeqFeatTokenizer()
        kwargs.setdefault("num_processes", 0)
        super().__init__(model_name=model_name, **kwargs)

    def _load_model(self, model_name):
        return _BioSeqFeatEncoder(self._feat_dim)

    def _get_tokenizer(self, model_name):
        return self._tokenizer_instance

    def _encode_single_batch(self, batch_dict):
        sequences = self._tokenizer_instance._last_sequences
        if self._combined:
            features = self._pipeline.extract_batch(sequences)
        else:
            features = np.stack([self._pipeline.extract_one(s) for s in sequences])
        features_tensor = torch.from_numpy(features).float().to(self.device)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMseqs2 landmark embedding DGEB evaluation")
    parser.add_argument(
        "--anchor-fasta",
        default=_DEFAULT_ANCHOR,
        help="Path to anchor FASTA file (default: db/bacteria_anchors.fasta)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Combine MMseqs2 landmarks with BLOSUM+HMM features",
    )
    parser.add_argument(
        "--hmm-db", default=_DB_PATH,
        help="Path to pressed Pfam HMM DB (used with --combined)",
    )
    parser.add_argument("--e-thresh", type=float, default=_E_THRESH)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--cpu", type=int, default=_CPU)
    args = parser.parse_args()

    if not os.path.exists(args.anchor_fasta):
        print(f"ERROR: anchor FASTA not found: {args.anchor_fasta}")
        print("Prepare it with:")
        print("  mmseqs easy-cluster bacteria_sprot.fasta clusters tmp --min-seq-id 0.5 -c 0.8")
        print("  cp clusters_rep_seq.fasta db/bacteria_anchors.fasta")
        sys.exit(1)

    model = BioSeqFeatMMseqsTransformer(
        anchor_fasta=args.anchor_fasta,
        combined=args.combined,
        hmm_db=args.hmm_db,
        e_thresh=args.e_thresh,
        sensitivity=args.sensitivity,
        cpu=args.cpu,
    )
    out_dir = "results/bioseqfeat_mmseqs_combined" if args.combined else "results/bioseqfeat_mmseqs"
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    results = evaluation.run(model, output_folder=out_dir)
    print(results)
