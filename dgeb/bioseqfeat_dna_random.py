"""
Random baseline for DNA DGEB evaluation.

Generates fixed-dimensional random Gaussian embeddings per sequence.
The random seed is derived from the sequence itself so that the same
sequence always produces the same embedding (deterministic), but
embeddings for distinct sequences are uncorrelated.

This establishes a lower bound for DNA task performance.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib

import torch
import numpy as np
from types import SimpleNamespace
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality

_EMBED_DIM = 512


def _seq_to_embedding(seq: str, dim: int) -> np.ndarray:
    """Deterministic random embedding: seed from MD5 of sequence."""
    digest = hashlib.md5(seq.encode()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class _RandomEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, *args, **kwargs):
        return None


class _DummyTokenizer:
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


class RandomDNATransformer(BioSeqTransformer):
    """DGEB-compatible random baseline for DNA tasks.

    Each sequence receives a deterministic random unit-norm embedding
    derived from the MD5 hash of the sequence string.  This cannot
    learn any sequence signal, so its performance represents pure chance.

    Example
    -------
    >>> model = RandomDNATransformer()
    >>> evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.DNA))
    >>> results = evaluation.run(model, output_folder="results/dna_random")
    """

    MODEL_NAMES = ["bioseqfeat-dna-random"]

    def __init__(self, embed_dim: int = _EMBED_DIM, **kwargs):
        self._feat_dim = embed_dim
        self._tokenizer_instance = _DummyTokenizer()
        kwargs.setdefault("num_processes", 0)
        super().__init__(model_name="bioseqfeat-dna-random", **kwargs)

    def _load_model(self, model_name):
        return _RandomEncoder(self._feat_dim)

    def _get_tokenizer(self, model_name):
        return self._tokenizer_instance

    def _encode_single_batch(self, batch_dict):
        sequences = self._tokenizer_instance._last_sequences
        features = np.stack([
            _seq_to_embedding(seq, self._feat_dim)
            for seq in sequences
        ])  # (batch_size, feat_dim)
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
        return Modality.DNA


if __name__ == "__main__":
    model = RandomDNATransformer()
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.DNA))
    results = evaluation.run(model, output_folder="results/dna_random")
    print(results)
