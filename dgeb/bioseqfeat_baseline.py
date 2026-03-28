import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from types import SimpleNamespace
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality

from bioseqfeat import Pipeline, BlosumAvg, BlosumCompress

# BlosumAvg -> (20,), BlosumCompress(dim=20) -> (400,), total = 420
_FEAT_DIM = 20 + 20 * 20


class _BioSeqFeatEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=_FEAT_DIM)

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


class BioSeqFeatTransformer(BioSeqTransformer):
    """DGEB-compatible model that produces embeddings from the bioseqfeat pipeline.

    Uses BlosumAvg (20-d) + BlosumCompress(dim=20, method="dct") (400-d)
    concatenated into a 420-d vector. No HuggingFace model is loaded.

    Example
    -------
    >>> model = BioSeqFeatTransformer()
    >>> evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    >>> results = evaluation.run(model, output_folder="results/bioseqfeat")
    """

    MODEL_NAMES = ["bioseqfeat-blosum"]

    def __init__(self, **kwargs):
        self._pipeline = Pipeline(
            featurizers=[BlosumAvg(), BlosumCompress(dim=20, method="dct")],
        )
        self._tokenizer_instance = _BioSeqFeatTokenizer()
        kwargs.setdefault("num_processes", 0)
        super().__init__(model_name="bioseqfeat-blosum", **kwargs)

    def _load_model(self, model_name):
        return _BioSeqFeatEncoder()

    def _get_tokenizer(self, model_name):
        return self._tokenizer_instance

    def _encode_single_batch(self, batch_dict):
        sequences = self._tokenizer_instance._last_sequences
        features = self._pipeline.extract_batch(sequences)  # (batch_size, feat_dim)
        features_tensor = torch.from_numpy(features).float().to(self.device)
        # DGEB expects shape (batch_size, num_layers, embed_dim)
        return features_tensor.unsqueeze(1).expand(-1, len(self.layers), -1).contiguous()

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def embed_dim(self) -> int:
        return _FEAT_DIM

    @property
    def modality(self):
        return Modality.PROTEIN


if __name__ == "__main__":
    model = BioSeqFeatTransformer()
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    results = evaluation.run(model, output_folder="results/bioseqfeat_blosum")
    print(results)
