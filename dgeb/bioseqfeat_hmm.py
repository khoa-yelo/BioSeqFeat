"""
HMM profile score vector for DGEB evaluation.

Runs hmmscan against the 1011-profile Pfam clan-representative subset.
Each dimension in the output vector is the best per-sequence bit score for
one HMM profile (0.0 for no hit).  The vector is L2-normalized before use.

Requires HMMER3 (hmmscan) on PATH and the pressed HMM database:
    hmmpress db/Pfam-A.subset.hmm
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from types import SimpleNamespace
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality

from bioseqfeat.protein.hmm import HMMFeaturizer
from bioseqfeat import NormalizedFeaturizer

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db", "Pfam-A.subset.hmm",
)

# Determined at runtime from the HMM file (1011 profiles in the current subset)
_HMM_DB = _DB_PATH
_E_THRESH = 10.0   # loose threshold — keeps weak hits, avoids all-zero rows
_CPU = 4


class _BioSeqFeatEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, *args, **kwargs):
        return None


class _BioSeqFeatTokenizer:
    """Dummy tokenizer that caches raw sequences for use in encoding."""

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


class BioSeqFeatHMMTransformer(BioSeqTransformer):
    """DGEB-compatible model that produces embeddings from HMM profile scores.

    For each sequence, runs ``hmmscan`` against the 1011-profile Pfam
    clan-representative subset and uses the L2-normalized bit-score vector
    as the embedding.  Sequences with no hits produce a zero vector (rare at
    E-value 10.0, but possible for very short or highly unusual sequences).

    Parameters
    ----------
    hmm_db : str
        Path to the pressed HMM database.  Defaults to ``db/Pfam-A.subset.hmm``
        relative to the repository root.
    e_thresh : float
        hmmscan E-value threshold (default 10.0).
    cpu : int
        Number of CPU threads for hmmscan (default 4).

    Example
    -------
    >>> model = BioSeqFeatHMMTransformer()
    >>> evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    >>> results = evaluation.run(model, output_folder="results/bioseqfeat_hmm")
    """

    MODEL_NAMES = ["bioseqfeat-hmm"]

    def __init__(self, hmm_db: str = _HMM_DB, e_thresh: float = _E_THRESH,
                 cpu: int = _CPU, report_all_scores: bool = False, **kwargs):
        _raw = HMMFeaturizer(hmm_db, e_thresh=e_thresh, cpu=cpu,
                             report_all_scores=report_all_scores)
        self._pipeline = NormalizedFeaturizer(_raw)
        self._feat_dim = _raw.n_profiles
        self._tokenizer_instance = _BioSeqFeatTokenizer()
        self._report_all_scores = report_all_scores
        kwargs.setdefault("num_processes", 0)
        model_name = "bioseqfeat-hmm-all" if report_all_scores else "bioseqfeat-hmm"
        super().__init__(model_name=model_name, **kwargs)

    def _load_model(self, model_name):
        return _BioSeqFeatEncoder(self._feat_dim)

    def _get_tokenizer(self, model_name):
        return self._tokenizer_instance

    def _encode_single_batch(self, batch_dict):
        sequences = self._tokenizer_instance._last_sequences
        # extract_batch runs hmmscan per-sequence (inherits from Featurizer)
        features = np.stack([
            self._pipeline.extract_one(s) for s in sequences
        ])  # (batch_size, n_profiles)
        features_tensor = torch.from_numpy(features).float().to(self.device)
        # DGEB expects (batch_size, num_layers, embed_dim)
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
    tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
    evaluation = dgeb.DGEB(tasks=tasks)

    for report_all in (False, True):
        model = BioSeqFeatHMMTransformer(report_all_scores=report_all)
        out_dir = "results/bioseqfeat_hmm_all" if report_all else "results/bioseqfeat_hmm"
        print(f"\n=== Running with report_all_scores={report_all} -> {out_dir} ===")
        results = evaluation.run(model, output_folder=out_dir)
        print(results)
