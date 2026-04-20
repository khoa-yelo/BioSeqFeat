"""
Retrieval-optimised BioSeqFeat pipeline for DGEB evaluation.

Drops all composition-based featurizers (AAC, DPC, PseAAC, CTD,
GlobalDescriptors) that encode kingdom-level amino-acid biases and
hurt cross-kingdom retrieval by making functionally homologous
archaeal/eukaryotic-bacterial pairs look MORE different.

Only evolutionary signal is kept:
  - BlosumAvg        (20-d)  residue-level substitution composition
  - BlosumCompress   (400-d) position-aware BLOSUM compression
  - HMM profiles     (1011-d) Pfam family membership

Weight rationale
----------------
BlosumAvg        20    5.0   (38 %)   proven best single feature on retrieval
BlosumCompress  400    5.0   (38 %)   adds sequence-order to evolutionary signal
HMMProfile     1011    3.0   (23 %)   domain/family membership
                      ----
Total dim: 1431        13.0

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

from bioseqfeat import Pipeline, NormalizedFeaturizer, BlosumAvg, BlosumCompress
from bioseqfeat.protein.hmm import HMMFeaturizer

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db", "Pfam-A.subset.hmm",
)
_E_THRESH = 10.0
_CPU = 4

# Dim: 20 + 400 = 420 static; HMM dim determined at runtime
_STATIC_DIM = 20 + 400


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


class BioSeqFeatRetrievalTransformer(BioSeqTransformer):
    """DGEB-compatible retrieval-optimised pipeline.

    Uses only evolutionary features (BLOSUM + HMM); strips composition
    features that add cross-kingdom kingdom-identity noise.

    Parameters
    ----------
    hmm_db : str
        Path to the pressed HMM database.
    e_thresh : float
        hmmscan E-value threshold (default 10.0).
    cpu : int
        Number of CPU threads for hmmscan (default 4).
    """

    MODEL_NAMES = ["bioseqfeat-retrieval"]

    def __init__(self, hmm_db: str = _DB_PATH, e_thresh: float = _E_THRESH,
                 cpu: int = _CPU, **kwargs):
        _hmm_raw = HMMFeaturizer(hmm_db, e_thresh=e_thresh, cpu=cpu)
        self._feat_dim = _STATIC_DIM + _hmm_raw.n_profiles
        self._pipeline = Pipeline(
            featurizers=[
                NormalizedFeaturizer(BlosumAvg()),
                NormalizedFeaturizer(BlosumCompress(dim=20, method="dct")),
                NormalizedFeaturizer(_hmm_raw),
            ],
            weights=[5.0, 5.0, 3.0],
        )
        self._tokenizer_instance = _BioSeqFeatTokenizer()
        kwargs.setdefault("num_processes", 0)
        super().__init__(model_name="bioseqfeat-retrieval", **kwargs)

    def _load_model(self, model_name):
        return _BioSeqFeatEncoder(self._feat_dim)

    def _get_tokenizer(self, model_name):
        return self._tokenizer_instance

    def _encode_single_batch(self, batch_dict):
        sequences = self._tokenizer_instance._last_sequences
        features = self._pipeline.extract_batch(sequences)
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
    model = BioSeqFeatRetrievalTransformer()
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    results = evaluation.run(model, output_folder="results/bioseqfeat_retrieval")
    print(results)
