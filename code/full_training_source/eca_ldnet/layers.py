"""Custom Keras layers for ECA-LDNet.

Three building blocks define the network's identity:

* :class:`ECABlock`            - Efficient Channel Attention (1-D conv over the
  channel descriptor; parameter-free up to a tiny ``k``-tap kernel).
* :class:`PixelAttention`      - lightweight spatial attention that learns
  *where* in the feature map to focus.
* :class:`PhysicsCorrectionLayer` - blends the network output with an
  atmospheric-scattering-model (ASM) reconstruction ``J = (I - A) / t + A``.

These definitions are byte-for-byte compatible with the layers serialised into
the released ``.keras`` checkpoints, so saved models load with
``custom_objects=CUSTOM_OBJECTS``.
"""
from __future__ import annotations

import math

import tensorflow as tf
from tensorflow.keras import layers


class ECABlock(layers.Layer):
    """Efficient Channel Attention (Wang et al., CVPR 2020).

    Squeeze with global average pooling, model cross-channel interaction with a
    single 1-D convolution of size ``kernel_size``, and gate the input with the
    resulting per-channel weights.
    """

    def __init__(self, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channels = int(input_shape[-1])
        self.conv = layers.Conv1D(1, self.kernel_size, padding="same", use_bias=False)
        self.gap = layers.GlobalAveragePooling2D()
        super().build(input_shape)

    def call(self, x):
        b = tf.shape(x)[0]
        gap = self.gap(x)                                  # (B, C)
        gap = tf.reshape(gap, [b, self.channels, 1])       # (B, C, 1)
        attn = tf.sigmoid(self.conv(gap))                  # (B, C, 1)
        attn = tf.reshape(attn, [b, 1, 1, self.channels])  # (B, 1, 1, C)
        return x * attn

    def get_config(self):
        cfg = super().get_config()
        cfg["kernel_size"] = self.kernel_size
        return cfg


class PixelAttention(layers.Layer):
    """Spatial (pixel-wise) attention.

    A 1x1 bottleneck conv (C -> C/4) followed by a 1x1 conv to a single-channel
    sigmoid map that re-weights every spatial location.
    """

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.conv1 = layers.Conv2D(max(8, c // 4), 1, padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(1, 1, padding="same", activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        attn = tf.nn.relu(self.conv1(x))
        attn = self.conv2(attn)            # (B, H, W, 1)
        return x * attn

    def get_config(self):
        return super().get_config()


class SEBlock(layers.Layer):
    """Squeeze-and-Excitation channel attention baseline."""

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(1, channels // 16)
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.reduce = layers.Conv2D(hidden, 1, activation="relu")
        self.expand = layers.Conv2D(channels, 1, activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        return x * self.expand(self.reduce(self.pool(x)))


class CBAMBlock(layers.Layer):
    """Sequential channel-plus-spatial CBAM-style attention baseline."""

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(1, channels // 16)
        self.pool_avg = layers.GlobalAveragePooling2D(keepdims=True)
        self.pool_max = layers.GlobalMaxPooling2D(keepdims=True)
        self.shared_reduce = layers.Conv2D(hidden, 1, activation="relu")
        self.shared_expand = layers.Conv2D(channels, 1)
        self.spatial = layers.Conv2D(1, 7, padding="same", activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        avg = self.shared_expand(self.shared_reduce(self.pool_avg(x)))
        maximum = self.shared_expand(self.shared_reduce(self.pool_max(x)))
        channel = tf.sigmoid(avg + maximum)
        x = x * channel
        spatial = tf.concat(
            [tf.reduce_mean(x, axis=-1, keepdims=True), tf.reduce_max(x, axis=-1, keepdims=True)],
            axis=-1,
        )
        return x * self.spatial(spatial)


class ResidualRefinementLayer(layers.Layer):
    """Capacity-matched learned correction control using the same A/t heads."""

    def __init__(self, blend: float = 0.08, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        self.blend = float(blend)
        self.refine = layers.Conv2D(
            3, 1, padding="same", activation="tanh",
            kernel_initializer="zeros", bias_initializer="zeros",
        )

    def call(self, inputs):
        raw, a, t = inputs
        a = tf.reshape(a, [-1, 1, 1, 3])
        a = tf.broadcast_to(a, tf.shape(raw))
        residual = self.refine(tf.concat([raw, a, t], axis=-1))
        return tf.clip_by_value(raw + self.blend * residual, 0.0, 1.0)

    def get_config(self):
        cfg = super().get_config()
        cfg["blend"] = self.blend
        return cfg


class PhysicsCorrectionLayer(layers.Layer):
    """Atmospheric-scattering-model guided refinement.

    Given the raw network output ``img``, an estimated atmospheric light ``A``
    (B, 3) and a transmission map ``t`` (B, H, W, 1), reconstruct the scene
    radiance ``J = (I - A) / (t + eps) + A`` and blend it lightly into the
    output so the physics acts as a soft prior rather than a hard constraint.
    """

    def __init__(self, eps: float = 0.1, blend: float = 0.08,
                 learnable_blend: bool = False, blend_max: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if not 0.0 < blend_max <= 1.0 or blend > blend_max:
            raise ValueError("blend_max must be in (0, 1] and >= blend")
        self.eps = eps
        self.blend = blend
        self.learnable_blend = bool(learnable_blend)
        self.blend_max = float(blend_max)

    def build(self, input_shape):
        if self.learnable_blend:
            ratio = min(max(self.blend / self.blend_max, 1e-6), 1.0 - 1e-6)
            initial_logit = math.log(ratio / (1.0 - ratio))
            self.blend_logit = self.add_weight(
                name="blend_logit", shape=(), trainable=True,
                initializer=tf.keras.initializers.Constant(initial_logit),
            )
        super().build(input_shape)

    @property
    def current_blend(self):
        if self.learnable_blend:
            return self.blend_max * tf.sigmoid(self.blend_logit)
        return tf.constant(self.blend, dtype=tf.float32)

    def call(self, inputs):
        # Three inputs preserve compatibility with the submitted checkpoint
        # (whose correction was applied to the raw CNN estimate). New models
        # pass [hazy_input, raw_prediction, A, t] and implement the equation
        # stated in the manuscript using the observed hazy image.
        if len(inputs) == 3:
            raw, a, t = inputs
            observed = raw
        elif len(inputs) == 4:
            observed, raw, a, t = inputs
        else:
            raise ValueError("Expected [raw, A, t] or [hazy, raw, A, t]")
        a_bc = tf.reshape(a, [-1, 1, 1, 3])
        a_bc = tf.broadcast_to(a_bc, tf.shape(raw))
        t_bc = tf.broadcast_to(t, tf.shape(raw))
        j = (observed - a_bc) / (t_bc + self.eps) + a_bc
        j = tf.clip_by_value(j, 0.0, 1.0)
        blend = tf.cast(self.current_blend, raw.dtype)
        return raw * (1.0 - blend) + j * blend

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "eps": self.eps, "blend": self.blend,
            "learnable_blend": self.learnable_blend, "blend_max": self.blend_max,
        })
        return cfg


# Used when loading released checkpoints with tf.keras.models.load_model(...)
CUSTOM_OBJECTS = {
    "ECABlock": ECABlock,
    "PixelAttention": PixelAttention,
    "SEBlock": SEBlock,
    "CBAMBlock": CBAMBlock,
    "ResidualRefinementLayer": ResidualRefinementLayer,
    "PhysicsCorrectionLayer": PhysicsCorrectionLayer,
}
