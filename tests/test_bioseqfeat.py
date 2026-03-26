"""
Tests for BioSeqFeat: feature extraction for protein sequences.

Run with: pytest tests/test_bioseqfeat.py -v
"""

import numpy as np
import pytest

from bioseqfeat import BlosumAvg, BlosumCompress, Featurizer, Pipeline
from bioseqfeat.protein.blosum import (
    AA_ORDER,
    AA_TO_IDX,
    BLOSUM62,
    EMBED_DIM,
    _adaptive_avg_pool,
    _dct_compress,
    _moving_avg_pool,
    _seq_to_embeddings,
    average_embedding,
    compress_sequence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHORT_SEQ = "ACDE"          # 4 residues — shorter than default dim=20
STANDARD_SEQ = "ACDEFGHIKLMNPQRSTVWY"   # all 20 standard amino acids, length=20
LONG_SEQ = "ACDEFGHIKLMNPQRSTVWY" * 5  # length=100


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_aa_order_length(self):
        assert len(AA_ORDER) == 20

    def test_aa_order_only_standard(self):
        for aa in AA_ORDER:
            assert aa in "ARNDCQEGHILKMFPSTWYV"

    def test_aa_to_idx_consistent(self):
        for aa, idx in AA_TO_IDX.items():
            assert AA_ORDER[idx] == aa

    def test_blosum62_shape(self):
        assert BLOSUM62.shape == (20, 20)

    def test_blosum62_dtype(self):
        assert BLOSUM62.dtype == np.float32

    def test_embed_dim(self):
        assert EMBED_DIM == 20


# ---------------------------------------------------------------------------
# _seq_to_embeddings
# ---------------------------------------------------------------------------

class TestSeqToEmbeddings:
    def test_standard_sequence_shape(self):
        emb = _seq_to_embeddings(STANDARD_SEQ)
        assert emb.shape == (20, 20)

    def test_lowercase_accepted(self):
        emb_upper = _seq_to_embeddings("ACDE")
        emb_lower = _seq_to_embeddings("acde")
        np.testing.assert_array_equal(emb_upper, emb_lower)

    def test_nonstandard_residues_skipped(self):
        # X, B, Z are non-standard — they should be skipped
        emb_clean = _seq_to_embeddings("ACDE")
        emb_mixed = _seq_to_embeddings("AXBCZDZE")  # X, B, Z interspersed
        np.testing.assert_array_equal(emb_clean, emb_mixed)

    def test_each_row_matches_blosum62(self):
        emb = _seq_to_embeddings("AC")
        np.testing.assert_array_equal(emb[0], BLOSUM62[AA_TO_IDX["A"]])
        np.testing.assert_array_equal(emb[1], BLOSUM62[AA_TO_IDX["C"]])

    def test_all_invalid_raises(self):
        with pytest.raises(ValueError, match="No valid amino acids"):
            _seq_to_embeddings("XXXBBBZZZ")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="No valid amino acids"):
            _seq_to_embeddings("")


# ---------------------------------------------------------------------------
# average_embedding
# ---------------------------------------------------------------------------

class TestAverageEmbedding:
    def test_output_shape(self):
        vec = average_embedding(STANDARD_SEQ)
        assert vec.shape == (20,)

    def test_single_residue_equals_blosum_row(self):
        vec = average_embedding("A")
        np.testing.assert_array_equal(vec, BLOSUM62[AA_TO_IDX["A"]])

    def test_average_of_two(self):
        expected = (BLOSUM62[AA_TO_IDX["A"]] + BLOSUM62[AA_TO_IDX["C"]]) / 2
        np.testing.assert_allclose(average_embedding("AC"), expected, rtol=1e-5)

    def test_output_dtype_float(self):
        vec = average_embedding(STANDARD_SEQ)
        assert np.issubdtype(vec.dtype, np.floating)


# ---------------------------------------------------------------------------
# compress_sequence
# ---------------------------------------------------------------------------

class TestCompressSequence:
    @pytest.mark.parametrize("method", ["moving_avg", "adaptive_pool", "dct"])
    def test_output_shape(self, method):
        dim = 10
        vec = compress_sequence(LONG_SEQ, dim=dim, method=method)
        assert vec.shape == (dim * 20,)

    @pytest.mark.parametrize("method", ["moving_avg", "adaptive_pool", "dct"])
    def test_output_shape_default_dim(self, method):
        vec = compress_sequence(LONG_SEQ, method=method)
        assert vec.shape == (20 * 20,)

    def test_short_sequence_zero_padded(self):
        # SHORT_SEQ has length 4 < dim=20; trailing values should be zero
        vec = compress_sequence(SHORT_SEQ, dim=20)
        # First 4 bins filled, rest zero-padded
        mat = vec.reshape(20, 20)
        np.testing.assert_array_equal(mat[4:], np.zeros((16, 20)))

    @pytest.mark.parametrize("method", ["moving_avg", "adaptive_pool", "dct"])
    def test_output_is_1d(self, method):
        vec = compress_sequence(LONG_SEQ, dim=5, method=method)
        assert vec.ndim == 1

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            compress_sequence(LONG_SEQ, method="unknown")

    @pytest.mark.parametrize("method", ["moving_avg", "adaptive_pool", "dct"])
    def test_output_dtype_float(self, method):
        vec = compress_sequence(LONG_SEQ, dim=5, method=method)
        assert np.issubdtype(vec.dtype, np.floating)


# ---------------------------------------------------------------------------
# Compression backends
# ---------------------------------------------------------------------------

class TestMovingAvgPool:
    def test_output_shape(self):
        emb = _seq_to_embeddings(LONG_SEQ)
        out = _moving_avg_pool(emb, dim=10)
        assert out.shape == (10, 20)

    def test_single_bin_equals_mean(self):
        emb = _seq_to_embeddings(STANDARD_SEQ)
        out = _moving_avg_pool(emb, dim=1)
        np.testing.assert_allclose(out[0], emb.mean(axis=0), rtol=1e-5)


class TestAdaptiveAvgPool:
    def test_output_shape(self):
        emb = _seq_to_embeddings(LONG_SEQ)
        out = _adaptive_avg_pool(emb, dim=10)
        assert out.shape == (10, 20)

    def test_single_bin_equals_mean(self):
        emb = _seq_to_embeddings(STANDARD_SEQ)
        out = _adaptive_avg_pool(emb, dim=1)
        np.testing.assert_allclose(out[0], emb.mean(axis=0), rtol=1e-5)


class TestDctCompress:
    def test_output_shape(self):
        emb = _seq_to_embeddings(LONG_SEQ)
        out = _dct_compress(emb, dim=10)
        assert out.shape == (10, 20)

    def test_output_dtype(self):
        emb = _seq_to_embeddings(LONG_SEQ)
        out = _dct_compress(emb, dim=10)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# BlosumAvg featurizer
# ---------------------------------------------------------------------------

class TestBlosumAvg:
    def test_extract_one_shape(self):
        feat = BlosumAvg()
        vec = feat.extract_one(STANDARD_SEQ)
        assert vec.shape == (20,)

    def test_extract_one_matches_average_embedding(self):
        feat = BlosumAvg()
        np.testing.assert_array_equal(feat.extract_one(STANDARD_SEQ), average_embedding(STANDARD_SEQ))

    def test_extract_batch_shape(self):
        feat = BlosumAvg()
        seqs = [STANDARD_SEQ, SHORT_SEQ, LONG_SEQ]
        mat = feat.extract_batch(seqs)
        assert mat.shape == (3, 20)

    def test_name(self):
        assert BlosumAvg.name == "blosum_avg"

    def test_repr(self):
        assert "BlosumAvg" in repr(BlosumAvg())


# ---------------------------------------------------------------------------
# BlosumCompress featurizer
# ---------------------------------------------------------------------------

class TestBlosumCompress:
    def test_extract_one_shape_default(self):
        feat = BlosumCompress()
        vec = feat.extract_one(LONG_SEQ)
        assert vec.shape == (20 * 20,)

    def test_extract_one_shape_custom_dim(self):
        feat = BlosumCompress(dim=5)
        vec = feat.extract_one(LONG_SEQ)
        assert vec.shape == (5 * 20,)

    @pytest.mark.parametrize("method", ["moving_avg", "adaptive_pool", "dct"])
    def test_methods(self, method):
        feat = BlosumCompress(dim=10, method=method)
        vec = feat.extract_one(LONG_SEQ)
        assert vec.shape == (10 * 20,)

    def test_extract_batch_shape(self):
        feat = BlosumCompress(dim=5)
        seqs = [STANDARD_SEQ, SHORT_SEQ, LONG_SEQ]
        mat = feat.extract_batch(seqs)
        assert mat.shape == (3, 5 * 20)

    def test_name(self):
        assert BlosumCompress.name == "blosum_compress"

    def test_repr(self):
        assert "BlosumCompress" in repr(BlosumCompress())


# ---------------------------------------------------------------------------
# Featurizer abstract base class
# ---------------------------------------------------------------------------

class TestFeaturizer:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Featurizer()

    def test_subclass_must_implement_extract_one(self):
        class Incomplete(Featurizer):
            name = "incomplete"
        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_works(self):
        class Const(Featurizer):
            name = "const"
            def extract_one(self, seq, **kwargs):
                return np.zeros(5)

        feat = Const()
        np.testing.assert_array_equal(feat.extract_one("ACDE"), np.zeros(5))

    def test_extract_batch_default_impl(self):
        class Const(Featurizer):
            name = "const"
            def extract_one(self, seq, **kwargs):
                return np.ones(3)

        mat = Const().extract_batch(["A", "B", "C"])
        assert mat.shape == (3, 3)
        np.testing.assert_array_equal(mat, np.ones((3, 3)))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_single_featurizer_no_weights(self):
        pipe = Pipeline([BlosumAvg()])
        vec = pipe.extract_one(STANDARD_SEQ)
        # weight=1 → scale=1 → no change
        np.testing.assert_array_equal(vec, BlosumAvg().extract_one(STANDARD_SEQ))

    def test_two_featurizers_output_shape(self):
        pipe = Pipeline([BlosumAvg(), BlosumCompress(dim=5)])
        vec = pipe.extract_one(LONG_SEQ)
        assert vec.shape == (20 + 5 * 20,)

    def test_weights_scale_output(self):
        feat = BlosumAvg()
        pipe_w2 = Pipeline([feat], weights=[4.0])
        pipe_w1 = Pipeline([feat], weights=[1.0])
        expected = BlosumAvg().extract_one(STANDARD_SEQ)
        np.testing.assert_allclose(pipe_w2.extract_one(STANDARD_SEQ), expected * 2.0, rtol=1e-5)
        np.testing.assert_allclose(pipe_w1.extract_one(STANDARD_SEQ), expected * 1.0, rtol=1e-5)

    def test_default_weights_all_ones(self):
        pipe = Pipeline([BlosumAvg(), BlosumAvg()])
        np.testing.assert_array_equal(pipe.weights, [1.0, 1.0])

    def test_extract_batch_shape(self):
        pipe = Pipeline([BlosumAvg(), BlosumCompress(dim=5)])
        seqs = [STANDARD_SEQ, SHORT_SEQ, LONG_SEQ]
        mat = pipe.extract_batch(seqs)
        assert mat.shape == (3, 20 + 5 * 20)

    def test_empty_featurizers_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Pipeline([])

    def test_weight_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            Pipeline([BlosumAvg(), BlosumAvg()], weights=[1.0])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Pipeline([BlosumAvg()], weights=[-1.0])

    def test_names_property(self):
        pipe = Pipeline([BlosumAvg(), BlosumCompress()])
        assert pipe.names == ["blosum_avg", "blosum_compress"]

    def test_repr(self):
        pipe = Pipeline([BlosumAvg()], weights=[2.0])
        r = repr(pipe)
        assert "Pipeline" in r
        assert "blosum_avg" in r

    def test_zero_weight_allowed(self):
        # zero weight should not raise; that featurizer contributes nothing
        pipe = Pipeline([BlosumAvg(), BlosumCompress(dim=5)], weights=[1.0, 0.0])
        vec = pipe.extract_one(STANDARD_SEQ)
        compress_block = vec[20:]
        np.testing.assert_array_equal(compress_block, np.zeros(5 * 20))
