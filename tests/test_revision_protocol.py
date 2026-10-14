"""Tests for revision-specific protocol and reviewer-requested controls."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from eca_ldnet.data import preprocess_hazy  # noqa: E402
from eca_ldnet.losses import LOSS_PHASE_WEIGHTS, make_loss  # noqa: E402
from eca_ldnet.train import _split_grouped_pairs  # noqa: E402


def test_scene_variants_stay_in_one_split(tmp_path):
    pairs = []
    for scene in range(10):
        clean = tmp_path / f"clean_{scene}.bin"
        clean.write_bytes(f"scene-{scene}".encode())
        for variant in range(2):
            hazy = tmp_path / f"hazy_{scene}_{variant}.bin"
            hazy.write_bytes(f"hazy-{scene}-{variant}".encode())
            pairs.append((str(hazy), str(clean)))
    train_idx, val_idx, groups = _split_grouped_pairs(pairs, val_fraction=0.2, seed=17)
    assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
    assert len(train_idx) + len(val_idx) == len(pairs)


def test_radiometric_preprocessing_preserves_values():
    image = np.asarray([[[0.1, 0.3, 0.9], [0.2, 0.5, 0.7]]], dtype=np.float32)
    output = preprocess_hazy(image, protocol="radiometric")
    np.testing.assert_allclose(output, image)


def test_loss_ablation_removes_only_selected_term():
    truth = tf.zeros([1, 32, 32, 3], dtype=tf.float32)
    pred = tf.ones_like(truth) * 0.2
    full = make_loss("primary")(truth, pred).numpy()
    no_pixel = make_loss("primary", exclude=["charbonnier"])(truth, pred).numpy()
    assert full > no_pixel > 0


def test_recovered_stage_objective_weights():
    assert LOSS_PHASE_WEIGHTS["primary"] == {
        "charbonnier": 0.35, "ssim": 0.20, "edge": 0.10,
        "frequency": 0.10, "perceptual": 0.25,
    }
    assert LOSS_PHASE_WEIGHTS["stage3"] == {
        "charbonnier": 0.40, "ssim": 0.20, "edge": 0.10,
        "frequency": 0.05, "perceptual": 0.25,
    }
    assert LOSS_PHASE_WEIGHTS["stage4"]["mse"] == 0.40
    assert LOSS_PHASE_WEIGHTS["stage4"]["charbonnier"] == 0.40
    assert LOSS_PHASE_WEIGHTS["stage4"]["ssim"] == 0.05
    assert LOSS_PHASE_WEIGHTS["stage4"]["ms_ssim"] == 0.10
    assert LOSS_PHASE_WEIGHTS["stage4"]["vgg19_perceptual"] == 0.05
