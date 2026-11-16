"""CPU-only exact token-coordinate regression for rowwise sparse persistence."""
import numpy as np


def test_selected_tiles_reconstruct_frozen_dense_tokens():
    rng = np.random.default_rng(20261114)
    columns = np.array([0, 3, 7], dtype=np.int32)
    dense = rng.normal(size=(16, 8 * 16, 1280)).astype(np.float16)
    blocks = np.ascontiguousarray(dense.reshape(16, 8, 16, 1280)[:, columns]
                                  .transpose(1, 0, 2, 3))
    assert blocks.shape == (3, 16, 16, 1280)
    for tile_index, tile_column in enumerate(columns):
        for local_y, local_x in [(0, 0), (8, 11), (15, 15)]:
            np.testing.assert_array_equal(
                blocks[tile_index, local_y, local_x],
                dense[local_y, tile_column * 16 + local_x])


if __name__ == '__main__':
    test_selected_tiles_reconstruct_frozen_dense_tokens()
    print('Sparse tile blocks preserve exact row and token coordinates')
