"""Loss functions and metrics for ECA-LDNet.

The Stage 1/2 objective is a weighted sum of Charbonnier, SSIM, edge (Sobel),
log-FFT frequency, and VGG16 perceptual losses. Stage 3 re-weights toward
Charbonnier + perceptual for final polishing. A VGG16 feature extractor is built
lazily on first use so that importing this module never forces a network
download.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model

from .config import IMG_SIZE, W_EDGE, W_FREQ, W_MAE, W_PERCEP, W_SSIM

_PERCEPTUAL_MODEL = None
_VGG19_MODEL = None


def _get_perceptual_model() -> Model:
    """Lazily build a frozen VGG16 feature extractor (adds 0 trainable params)."""
    global _PERCEPTUAL_MODEL
    if _PERCEPTUAL_MODEL is None:
        from tensorflow.keras.applications import VGG16

        base = VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
        base.trainable = False
        _PERCEPTUAL_MODEL = Model(
            inputs=base.input,
            outputs=[
                base.get_layer("block1_conv2").output,
                base.get_layer("block2_conv2").output,
                base.get_layer("block3_conv3").output,
            ],
            name="vgg_perceptual",
        )
        _PERCEPTUAL_MODEL.trainable = False
    return _PERCEPTUAL_MODEL


def _get_vgg19_model() -> Model:
    """Frozen VGG19 features used by the submitted Stage 4 refinement loss."""
    global _VGG19_MODEL
    if _VGG19_MODEL is None:
        from tensorflow.keras.applications import VGG19

        base = VGG19(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
        base.trainable = False
        _VGG19_MODEL = Model(
            inputs=base.input,
            outputs=[base.get_layer(name).output for name in (
                "block1_conv2", "block2_conv2", "block3_conv4",
                "block4_conv4", "block5_conv4",
            )],
            name="vgg19_perceptual",
        )
        _VGG19_MODEL.trainable = False
    return _VGG19_MODEL


def _cast(a, b):
    return tf.cast(a, tf.float32), tf.cast(b, tf.float32)


def _vgg_imagenet_input(image):
    """Differentiable Keras VGG Caffe-style preprocessing for RGB [0,1] tensors."""
    image = tf.reverse(image * 255.0, axis=[-1])  # RGB -> BGR
    mean = tf.constant([103.939, 116.779, 123.68], shape=[1, 1, 1, 3], dtype=tf.float32)
    return image - mean


def mae_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def charbonnier_loss(y_true, y_pred, eps=1e-3):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + eps * eps))


def ssim_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def edge_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.abs(tf.image.sobel_edges(y_true) - tf.image.sobel_edges(y_pred)))


def frequency_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)

    def log_fft(img):
        gray = tf.reduce_mean(img, axis=-1)
        return tf.math.log1p(tf.abs(tf.signal.rfft2d(gray)))

    return tf.reduce_mean(tf.abs(log_fft(y_true) - log_fft(y_pred)))


def perceptual_loss(y_true, y_pred, protocol="legacy"):
    y_true, y_pred = _cast(y_true, y_pred)
    model = _get_perceptual_model()
    if protocol == "imagenet":
        true_input, pred_input = _vgg_imagenet_input(y_true), _vgg_imagenet_input(y_pred)
    elif protocol == "legacy":
        # Faithfully reproduce the submitted notebook's RGB-minus-mean input.
        mean = tf.constant([123.68, 116.779, 103.939], shape=[1, 1, 1, 3], dtype=tf.float32)
        true_input, pred_input = y_true * 255.0 - mean, y_pred * 255.0 - mean
    else:
        raise ValueError("perceptual protocol must be 'legacy' or 'imagenet'")
    true_feats = model(true_input, training=False)
    pred_feats = model(pred_input, training=False)
    loss = 0.0
    for t_f, p_f in zip(true_feats, pred_feats):
        loss += tf.reduce_mean(tf.abs(t_f - p_f))
    return loss / 3.0


def stage4_perceptual_loss(y_true, y_pred):
    """VGG19 multi-layer loss using the source notebook's ImageNet preprocessing."""
    y_true, y_pred = _cast(y_true, y_pred)
    model = _get_vgg19_model()
    true_feats = model(_vgg_imagenet_input(y_true), training=False)
    pred_feats = model(_vgg_imagenet_input(y_pred), training=False)
    weights = (0.1, 0.15, 0.3, 0.3, 0.15)
    return sum(w * tf.reduce_mean(tf.abs(a - b)) for a, b, w in zip(true_feats, pred_feats, weights))


def ms_ssim_loss(y_true, y_pred):
    """Reproduce Stage 4's five-scale average-pooled SSIM loss."""
    y_true, y_pred = _cast(y_true, y_pred)
    weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    total = tf.constant(0.0, tf.float32)
    for weight in weights:
        total += weight * (1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)))
        y_true = tf.nn.avg_pool2d(y_true, ksize=2, strides=2, padding="VALID")
        y_pred = tf.nn.avg_pool2d(y_pred, ksize=2, strides=2, padding="VALID")
    return total


def mse_loss(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.square(y_true - y_pred))


def combined_loss(y_true, y_pred):
    """Stage 1 / Stage 2 objective."""
    return (
        W_MAE * charbonnier_loss(y_true, y_pred)
        + W_SSIM * ssim_loss(y_true, y_pred)
        + W_EDGE * edge_loss(y_true, y_pred)
        + W_FREQ * frequency_loss(y_true, y_pred)
        + W_PERCEP * perceptual_loss(y_true, y_pred)
    )


def combined_loss_s3(y_true, y_pred):
    """Stage 3 final-polish objective (Charbonnier + perceptual heavy)."""
    return (
        0.40 * charbonnier_loss(y_true, y_pred)
        + 0.20 * ssim_loss(y_true, y_pred)
        + 0.10 * edge_loss(y_true, y_pred)
        + 0.05 * frequency_loss(y_true, y_pred)
        + 0.25 * perceptual_loss(y_true, y_pred)
    )


LOSS_TERMS = {
    "charbonnier": charbonnier_loss,
    "ssim": ssim_loss,
    "edge": edge_loss,
    "frequency": frequency_loss,
    "perceptual": perceptual_loss,
}
PRIMARY_WEIGHTS = {"charbonnier": W_MAE, "ssim": W_SSIM, "edge": W_EDGE, "frequency": W_FREQ, "perceptual": W_PERCEP}
STAGE3_WEIGHTS = {"charbonnier": 0.40, "ssim": 0.20, "edge": 0.10, "frequency": 0.05, "perceptual": 0.25}
STAGE4_WEIGHTS = {
    "mse": 0.40, "charbonnier": 0.40, "ssim": 0.05, "ms_ssim": 0.10,
    "vgg19_perceptual": 0.05, "edge": 0.0, "frequency": 0.0,
    "color_consistency": 0.0, "gradient": 0.0,
}
LOSS_PHASE_WEIGHTS = {"primary": PRIMARY_WEIGHTS, "stage3": STAGE3_WEIGHTS, "stage4": STAGE4_WEIGHTS}


def make_loss(phase="primary", exclude=None, perceptual_protocol="legacy"):
    """Build a loss callable; exclusions support controlled term ablations."""
    # `polish` remains a compatibility alias for the original Stage 3 objective.
    if phase == "polish":
        phase = "stage3"
    if phase not in {"primary", "stage3", "stage4"}:
        raise ValueError("phase must be 'primary', 'stage3', or 'stage4'")
    weights = dict(LOSS_PHASE_WEIGHTS[phase])
    terms = dict(LOSS_TERMS)
    terms.update({"mse": mse_loss, "ms_ssim": ms_ssim_loss, "vgg19_perceptual": stage4_perceptual_loss})
    terms["perceptual"] = lambda y, p: perceptual_loss(y, p, protocol=perceptual_protocol)
    for term in (exclude or ()):
        if term not in weights:
            raise ValueError(f"Unknown loss term: {term}")
        weights[term] = 0.0

    def loss(y_true, y_pred):
        total = tf.constant(0.0, dtype=tf.float32)
        for name, weight in weights.items():
            if weight:
                total += weight * terms[name](y_true, y_pred)
        return total

    loss.__name__ = f"{phase}_loss_without_{'_'.join(exclude or ()) or 'none'}"
    return loss


# ---- Metrics -------------------------------------------------------------
def psnr_metric(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def mae_metric(y_true, y_pred):
    y_true, y_pred = _cast(y_true, y_pred)
    return tf.reduce_mean(tf.abs(y_true - y_pred))
