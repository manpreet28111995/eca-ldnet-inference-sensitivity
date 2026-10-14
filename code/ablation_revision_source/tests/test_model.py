"""Architecture tests: parameter count, I/O shape, output range, layer presence."""
import numpy as np
import pytest

from eca_ldnet.config import IMG_SIZE

# The published model has exactly 1,481,871 parameters (1.482M).
EXPECTED_PARAMS = 1_481_871


def test_param_count(model):
    assert model.count_params() == EXPECTED_PARAMS


def test_io_shape(model):
    assert model.input_shape == (None, None, None, 3)
    assert model.output_shape == (None, None, None, 3)


def test_forward_pass_range(model):
    x = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    y = model.predict(x, verbose=0)
    assert y.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert np.isfinite(y).all()
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_dual_attention_and_physics_present(model):
    names = [type(layer).__name__ for layer in model.layers]
    assert "ECABlock" in names
    assert "PixelAttention" in names
    assert "PhysicsCorrectionLayer" in names


def test_physics_module_outputs(model):
    layer_names = [layer.name for layer in model.layers]
    assert "atm_light" in layer_names      # estimated atmospheric light A
    assert "transmission" in layer_names   # estimated transmission map t


def test_new_model_uses_observed_input_for_asm(model):
    assert len(model.get_layer("physics_output").input) == 4


def test_fully_convolutional_resolution_shape(model):
    x = np.random.default_rng(9).random((1, 32, 48, 3), dtype=np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == x.shape


def test_physics_correction_legacy_and_observed_modes():
    import tensorflow as tf

    from eca_ldnet.layers import PhysicsCorrectionLayer

    layer = PhysicsCorrectionLayer(eps=0.1, blend=0.5)
    hazy = tf.fill([1, 2, 2, 3], 0.8)
    raw = tf.fill([1, 2, 2, 3], 0.2)
    a = tf.fill([1, 3], 0.1)
    t = tf.fill([1, 2, 2, 1], 0.5)
    legacy = layer([raw, a, t]).numpy()
    observed = layer([hazy, raw, a, t]).numpy()
    assert np.allclose(legacy, 0.23333333, atol=1e-6)
    assert np.allclose(observed, 0.6, atol=1e-6)


def test_learnable_blend_is_bounded_and_initialized_at_eight_percent():
    import tensorflow as tf

    from eca_ldnet.layers import PhysicsCorrectionLayer

    layer = PhysicsCorrectionLayer(blend=0.08, learnable_blend=True, blend_max=0.20)
    hazy = tf.fill([1, 2, 2, 3], 0.8)
    raw = tf.fill([1, 2, 2, 3], 0.2)
    a = tf.fill([1, 3], 0.1)
    t = tf.fill([1, 2, 2, 1], 0.5)
    _ = layer([hazy, raw, a, t])
    assert np.isclose(float(layer.current_blend.numpy()), 0.08, atol=1e-6)
    assert len(layer.trainable_variables) == 1
    with tf.GradientTape() as tape:
        objective = tf.reduce_sum(layer([hazy, raw, a, t]))
    assert tape.gradient(objective, layer.trainable_variables)[0] is not None


@pytest.mark.parametrize("variant", ["se", "cbam", "pa_eca", "parallel", "standard", "group", "residual"])
def test_attention_comparison_variants_build_and_run(variant):
    import numpy as np

    from eca_ldnet.model import build_eca_ldnet

    if variant in {"standard", "group"}:
        model = build_eca_ldnet(conv_type=variant, use_physics=False)
    elif variant == "residual":
        model = build_eca_ldnet(correction_mode="learned_residual")
        assert abs(model.count_params() - FULL) < 100
    else:
        model = build_eca_ldnet(attention_variant=variant, use_physics=False)
    x = np.random.default_rng(5).random((1, 256, 256, 3), dtype=np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (1, 256, 256, 3)
    assert np.isfinite(y).all()


def test_roundtrip_save_load(model, tmp_path):
    import tensorflow as tf

    from eca_ldnet.layers import CUSTOM_OBJECTS

    path = tmp_path / "rt.keras"
    model.save(path)
    reloaded = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
    assert reloaded.count_params() == EXPECTED_PARAMS
