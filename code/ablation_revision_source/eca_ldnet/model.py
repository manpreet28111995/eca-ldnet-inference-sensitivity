"""Full ECA-LDNet architecture assembly.

A U-Net-style encoder-decoder (depths 32/64/128/256, bottleneck 512) with dual
attention in every residual block, an embedded physics-guidance module that
estimates atmospheric light ``A`` and a transmission map ``t`` from the
bottleneck, transmission-derived haze-density gating in the decoder, and a final
:class:`~eca_ldnet.layers.PhysicsCorrectionLayer`. The assembled model has
1,481,871 parameters (1.482M).
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model, layers

from .blocks import decoder_block, dws_conv_block, encoder_block, residual_block
from .config import BOTTLENECK_DROP, CHANNELS, DEC_FILTERS, ENC_FILTERS, IMG_SIZE, PHYS_BLEND, PHYS_EPS
from .layers import PhysicsCorrectionLayer, ResidualRefinementLayer


def build_eca_ldnet(
    img_size: Optional[int] = None,
    use_eca: bool = True,
    use_pa: bool = True,
    attention_variant: Optional[str] = None,
    conv_type: str = "dws",
    use_physics: bool = True,
    physics_blend: float = PHYS_BLEND,
    physics_eps: float = PHYS_EPS,
    learnable_physics_blend: bool = False,
    physics_blend_max: float = 0.20,
    physics_uses_observed_input: bool = True,
    correction_mode: str = "asm",
    width_multiplier: float = 1.0,
    name: str = "ECA_LDNet",
) -> Model:
    """Build and return the ECA-LDNet Keras functional model.

    The three boolean flags exist for controlled ablation and all default to the
    full configuration (which reproduces the released 1,481,871-parameter model):

    * ``use_eca``     - include the ECA channel-attention branch.
    * ``use_pa``      - include the pixel (spatial) attention branch.
    * ``use_physics`` - include the ASM-parameterized guidance module (light /
      transmission heads, haze-density decoder gating, and the ASM correction
      head). When ``False`` the network emits the raw sigmoid output directly.
    * ``physics_uses_observed_input`` - apply the stabilized inversion to the
      observed hazy input (the paper's stated equation). Set False only to
      reproduce the legacy checkpoint's raw-prediction correction.
    """
    if conv_type not in {"dws", "standard", "group"}:
        raise ValueError("conv_type must be 'dws', 'standard', or 'group'")
    if correction_mode not in {"asm", "learned_residual"}:
        raise ValueError("correction_mode must be 'asm' or 'learned_residual'")
    if width_multiplier <= 0:
        raise ValueError("width_multiplier must be positive")
    enc_filters = [max(8, int(round(f * width_multiplier / 8)) * 8) for f in ENC_FILTERS]
    dec_filters = [max(8, int(round(f * width_multiplier / 8)) * 8) for f in DEC_FILTERS]
    ab = dict(
        use_eca=use_eca, use_pa=use_pa, attention_variant=attention_variant,
        conv_type=conv_type,
    )
    spatial_size = (img_size, img_size) if img_size is not None else (None, None)
    inputs = layers.Input(shape=(*spatial_size, CHANNELS), name="hazy_input")

    # ---- ENCODER ----
    x, skip1 = encoder_block(inputs, enc_filters[0], **ab)   # 256 -> 128
    x, skip2 = encoder_block(x, enc_filters[1], **ab)        # 128 -> 64
    x, skip3 = encoder_block(x, enc_filters[2], **ab)        # 64  -> 32
    x, skip4 = encoder_block(x, enc_filters[3], **ab)        # 32  -> 16

    # ---- BOTTLENECK (16x16x512, no pooling) ----
    x = dws_conv_block(x, enc_filters[4], conv_type=conv_type)
    x = residual_block(x, enc_filters[4], drop_rate=0, **ab)
    x = layers.Dropout(BOTTLENECK_DROP)(x)
    bottleneck = x

    if use_physics:
        # ---- PHYSICS GUIDANCE MODULE ----
        a = layers.GlobalAveragePooling2D()(bottleneck)
        a = layers.Dense(64, activation="relu")(a)
        a = layers.Dropout(0.1)(a)
        a = layers.Dense(32, activation="relu")(a)
        a = layers.Dense(3, activation="sigmoid", dtype="float32", name="atm_light")(a)

        t = layers.Conv2D(enc_filters[0], 1, padding="same")(bottleneck)
        t = layers.BatchNormalization()(t)
        t = layers.Activation("relu")(t)
        t = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", padding="same", name="transmission")(t)

        t_16 = layers.UpSampling2D(2, interpolation="bilinear")(t)   # 32x32x1
        t_32 = layers.UpSampling2D(4, interpolation="bilinear")(t)   # 64x64x1

    # ---- DECODER (with haze-density gating when physics is enabled) ----
    x = decoder_block(x, skip4, dec_filters[0], **ab)        # 16 -> 32
    if use_physics:
        x = layers.Multiply(name="phys_d4")([x, 1.0 - t_16])
    x = decoder_block(x, skip3, dec_filters[1], **ab)        # 32 -> 64
    if use_physics:
        x = layers.Multiply(name="phys_d3")([x, 1.0 - t_32])
    x = decoder_block(x, skip2, dec_filters[2], **ab)        # 64 -> 128
    x = decoder_block(x, skip1, dec_filters[3], **ab)        # 128 -> 256

    # ---- FINAL REFINEMENT + OUTPUT ----
    x = dws_conv_block(x, max(8, int(round(16 * width_multiplier / 8)) * 8), conv_type=conv_type)
    raw = layers.Conv2D(3, 1, padding="same", use_bias=False, dtype="float32", name="raw_conv")(x)
    raw = layers.Activation("sigmoid", dtype="float32", name="raw_sigmoid")(raw)

    if use_physics:
        # The encoder downsamples by 16. A fixed factor preserves dynamic
        # spatial dimensions for fully convolutional native-resolution tests.
        t_full = layers.UpSampling2D(16, interpolation="bilinear", name="t_full")(t)
        if correction_mode == "asm":
            output = PhysicsCorrectionLayer(
                eps=physics_eps, blend=physics_blend,
                learnable_blend=learnable_physics_blend,
                blend_max=physics_blend_max,
                dtype="float32", name="physics_output"
            )(
                [inputs, raw, a, t_full] if physics_uses_observed_input else [raw, a, t_full]
            )
        else:
            output = ResidualRefinementLayer(
                blend=physics_blend, dtype="float32", name="residual_refinement"
            )([raw, a, t_full])
    else:
        output = raw

    return Model(inputs=inputs, outputs=output, name=name)


def load_pretrained(path: str, compile: bool = False) -> Model:
    """Load a released ``.keras`` checkpoint with the required custom objects."""
    from .layers import CUSTOM_OBJECTS
    from .callbacks import WarmupCosineDecay
    from .losses import make_loss, psnr_metric, ssim_metric, mae_metric

    # Compiled training checkpoints serialize this schedule in the AdamW
    # configuration. It is not a layer, so it is registered separately from
    # the architecture objects in layers.CUSTOM_OBJECTS.
    custom_objects = {
        **CUSTOM_OBJECTS,
        "WarmupCosineDecay": WarmupCosineDecay,
        "psnr_metric": psnr_metric,
        "ssim_metric": ssim_metric,
        "mae_metric": mae_metric,
    }
    # Training checkpoints serialize the named closure returned by make_loss.
    # Recreate the phase/protocol variants so compiled same-stage resumes can
    # restore the loss as well as AdamW slots and its LR schedule.
    for phase in ("primary", "stage3", "stage4"):
        for protocol in ("legacy", "imagenet"):
            loss_fn = make_loss(phase, perceptual_protocol=protocol)
            custom_objects[loss_fn.__name__] = loss_fn

    return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=compile)
