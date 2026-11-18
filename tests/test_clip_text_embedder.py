import mlx.core as mx

from mlx_shap_e.models.embeddings.clip_text_embedder import MlxClipTextEmbedder


def test_select_pool_indices_prefers_eot_over_max_token():
    eot = 49407
    tokens = mx.array(
        [
            [1, 50000, eot, 2, 0, 0],
            [1, eot, 50000, 2, 0, 0],
        ],
        dtype=mx.int32,
    )
    idx = MlxClipTextEmbedder._select_pool_indices(tokens, eot_token=eot)
    assert idx.tolist() == [2, 1]


def test_select_pool_indices_falls_back_to_last_nonzero_when_missing_eot():
    eot = 49407
    tokens = mx.array(
        [
            [3, 4, 5, 0, 0],
            [0, 0, 7, 8, 9],
        ],
        dtype=mx.int32,
    )
    idx = MlxClipTextEmbedder._select_pool_indices(tokens, eot_token=eot)
    assert idx.tolist() == [2, 4]
