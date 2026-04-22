from __future__ import annotations

import numpy as np

from plume.models.convlstm_contract import (
    CONVLSTM_CHANNEL_MANIFEST,
    CONVLSTM_GRID_HEIGHT,
    CONVLSTM_GRID_WIDTH,
    CONVLSTM_INPUT_CHANNELS,
    CONVLSTM_NORMALIZATION_MODE,
    CONVLSTM_SEQUENCE_LENGTH,
    CONVLSTM_STORED_INPUT_SHAPE,
    CONVLSTM_STORED_TARGET_SHAPE,
    CONVLSTM_TARGET_POLICY,
    CONVLSTM_PLUME_TARGET_CHANNEL,
    plume_to_model_space,
    plume_to_physical_space,
)


def test_canonical_convlstm_contract_constants():
    assert CONVLSTM_SEQUENCE_LENGTH == 3
    assert CONVLSTM_INPUT_CHANNELS == 10
    assert (CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH) == (64, 64)
    assert len(CONVLSTM_CHANNEL_MANIFEST) == 10
    assert CONVLSTM_CHANNEL_MANIFEST[0] == "plume_concentration"
    assert CONVLSTM_STORED_INPUT_SHAPE == (3, 10, 64, 64)
    assert CONVLSTM_STORED_TARGET_SHAPE == (1, 10, 64, 64)
    assert CONVLSTM_NORMALIZATION_MODE == "none"
    assert CONVLSTM_TARGET_POLICY == "stored_future_full_target_plume_supervision_first"
    assert CONVLSTM_PLUME_TARGET_CHANNEL == 0


def test_plume_transform_round_trip_and_clamp():
    raw = np.array([[0.0, 1e-12, 2e-12]])
    model = plume_to_model_space(raw)
    restored = plume_to_physical_space(model)
    np.testing.assert_allclose(restored, raw)

    negative_restored = plume_to_physical_space(np.array([-1.0]), clamp_non_negative=True)
    assert float(negative_restored[0]) == 0.0
