"""Composite building blocks: depthwise-separable conv, residual, enc/dec.

The attention flags (``use_eca`` / ``use_pa``) make the dual-attention modules
individually switchable so that controlled ablation variants can be built from
exactly the same code path as the full model.
"""
from __future__ import annotations

from tensorflow.keras import layers

from .config import ECA_KERNEL, SPATIAL_DROP
from .layers import CBAMBlock, ECABlock, PixelAttention, SEBlock


def dws_conv_block(x, filters, kernel_size=3, strides=1, conv_type="dws"):
    """Convolution block selectable for matched-budget backbone comparisons."""
    if conv_type == "dws":
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu6")(x)
    elif conv_type == "standard":
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    elif conv_type == "group":
        in_channels = int(x.shape[-1])
        groups = min(8, in_channels, filters)
        while in_channels % groups or filters % groups:
            groups -= 1
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same", use_bias=False,
            groups=groups,
        )(x)
    else:
        raise ValueError("conv_type must be 'dws', 'standard', or 'group'")
    if conv_type != "dws":
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu6")(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    return x


def residual_block(x, filters, use_eca=True, use_pa=True, drop_rate=SPATIAL_DROP,
                   attention_variant=None, conv_type="dws"):
    """Two DWS conv blocks + optional dual attention (ECA then PixelAttention) + residual.

    ``use_eca`` and ``use_pa`` toggle the channel- and spatial-attention branches
    independently, which is what the ablation driver flips.
    """
    skip = x
    if x.shape[-1] != filters:
        skip = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        skip = layers.BatchNormalization()(skip)
    x = dws_conv_block(x, filters, conv_type=conv_type)
    x = dws_conv_block(x, filters, conv_type=conv_type)
    if attention_variant is None:
        if use_eca:
            x = ECABlock(kernel_size=ECA_KERNEL)(x)
        if use_pa:
            x = PixelAttention()(x)
    elif attention_variant == "none":
        pass
    elif attention_variant == "eca":
        x = ECABlock(kernel_size=ECA_KERNEL)(x)
    elif attention_variant == "pixel":
        x = PixelAttention()(x)
    elif attention_variant == "se":
        x = SEBlock()(x)
    elif attention_variant == "cbam":
        x = CBAMBlock()(x)
    elif attention_variant == "eca_pa":
        x = PixelAttention()(ECABlock(kernel_size=ECA_KERNEL)(x))
    elif attention_variant == "pa_eca":
        x = ECABlock(kernel_size=ECA_KERNEL)(PixelAttention()(x))
    elif attention_variant == "parallel":
        channel = ECABlock(kernel_size=ECA_KERNEL)(x)
        spatial = PixelAttention()(x)
        x = layers.Average()([channel, spatial])
    else:
        raise ValueError(f"Unknown attention_variant: {attention_variant}")
    if drop_rate > 0:
        x = layers.SpatialDropout2D(drop_rate)(x)
    x = layers.Add()([x, skip])
    return x


def encoder_block(x, filters, use_eca=True, use_pa=True, attention_variant=None, conv_type="dws"):
    """Residual block (kept as skip) followed by 2x max-pool downsample."""
    x = residual_block(
        x, filters, use_eca=use_eca, use_pa=use_pa, drop_rate=SPATIAL_DROP,
        attention_variant=attention_variant,
        conv_type=conv_type,
    )
    skip = x
    x = layers.MaxPooling2D(2)(x)
    return x, skip


def decoder_block(x, skip, filters, use_eca=True, use_pa=True, attention_variant=None, conv_type="dws"):
    """Bilinear upsample, concat skip, fuse with 1x1 conv, then a residual block."""
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = residual_block(
        x, filters, use_eca=use_eca, use_pa=use_pa, drop_rate=SPATIAL_DROP / 2,
        attention_variant=attention_variant,
        conv_type=conv_type,
    )
    return x
