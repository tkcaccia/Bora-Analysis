import numpy as np

from benchmark.prepare_cmpb_transfer_tiles import tile_features, tile_targets


def test_tile_feature_weighting_respects_token_overlap():
    tokens = np.arange(4, dtype=np.float32).reshape(2, 2, 1)
    features = tile_features(tokens, (28, 28), 21)
    assert features.shape == (2, 2, 1)
    # First tile covers a full first token plus half the second on both axes.
    expected = (196 * 0 + 98 * 1 + 98 * 2 + 49 * 3) / 441
    assert np.isclose(features[0, 0, 0], expected)


def test_source_targets_exclude_mixed_tiles():
    reference = np.ones((20, 20), np.uint8)
    reference[:10, 10:] = 2
    reference[10:, 10:15] = 2
    support = np.ones_like(reference)
    labels, accepted = tile_targets(reference, support, 10, {1: 1, 2: 3}, .8)
    assert accepted.tolist() == [[True, True], [True, False]]
    assert labels[0, 1] == 3
