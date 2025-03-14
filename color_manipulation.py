from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array

__all__ = ["get_bayer_mask", "jax_bayer2GRAY", "jax_bayer2RGB", "jax_RGB2bayer"]


def get_bayer_mask(shape: Tuple[int, int]) -> Tuple[Array, Array, Array]:
    """
    Makes a mask for GBRG or GR2 bayer pattern.

    Returns a tuple of boolean matrices which are True where pixel values exist for the
    given color in the bayer pattern.

    Args:
        shape: 2D shape of the desired output masks.
    """
    channels = {channel: np.zeros(shape, dtype="bool") for channel in "rgb"}
    for channel, (y, x) in zip("grbg", [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return jnp.array(tuple(channels.values()))


def jax_bayer2GRAY(image: Array, mask: Tuple[Array, Array, Array]) -> Array:
    """
    Demosiacs a Bayer image and outputs a grayscale image using Bilinear interpolation.
    The arrangement of colour filters on the pixels is determined by the provided mask
    array. First demosiacs into RGB colourspace and makes Grayscale image using the
    relation ``G = 0.299 * R + 0.587 * G + 0.114 * B``.

    Returns a grayscale image.

    Args:
        image: 2D Bayer CFA.
        mask: Tuple of 3 boolean arrays, ordered RGB, that are True where pixel values
            exist for the corresponding color in the bayer pattern. Obtained using the
            function ``get_bayer_mask``.
    """
    interpolation_g = (
        jnp.array(
            [
                [0, 1, 0],
                [1, 4, 1],
                [0, 1, 0],
            ]
        )
        / 4
    )

    interpolation_rb = (
        jnp.array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]
        )
        / 4
    )
    R = jax.scipy.signal.convolve2d(
        jnp.pad(image * mask[0], pad_width=1, mode="reflect"),
        interpolation_rb,
        mode="valid",
    )

    G = jax.scipy.signal.convolve2d(
        jnp.pad(image * mask[1], pad_width=1, mode="reflect"),
        interpolation_g,
        mode="valid",
    )
    B = jax.scipy.signal.convolve2d(
        jnp.pad(image * mask[2], pad_width=1, mode="reflect"),
        interpolation_rb,
        mode="valid",
    )

    return 0.299 * R + 0.587 * G + 0.114 * B


def jax_bayer2RGB(image: Array, mask: Tuple[Array, Array, Array]) -> Array:
    """
    Demosiacs a Bayer image and outputs an RGB image using Bilinear interpolation.
    The arrangement of colour filters on the pixels is determined by the provided mask
    array. Interpolates each color channel separately using bilinear interpolation.

    Returns an RGB image with shape (H, W, 3).

    Args:
        image: 2D Bayer CFA.
        mask: Tuple of 3 boolean arrays, ordered RGB, that are True where pixel values
            exist for the corresponding color in the bayer pattern. Obtained using the
            function ``get_bayer_mask``.
    """
    interpolation_g = (
        jnp.array(
            [
                [0, 1, 0],
                [1, 4, 1],
                [0, 1, 0],
            ]
        )
        / 4
    )

    interpolation_rb = (
        jnp.array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]
        )
        / 4
    )
    R = jax.scipy.signal.convolve2d(
        jnp.pad(image * mask[0], pad_width=1, mode="reflect"),
        interpolation_rb,
        mode="valid",
    )

    G = jax.scipy.signal.convolve2d(
        jnp.pad(image * mask[1], pad_width=1, mode="reflect"),
        interpolation_g,
        mode="valid",
    )
    B = jax.scipy.signal.convolve2d(
        jnp.pad(image * mask[2], pad_width=1, mode="reflect"),
        interpolation_rb,
        mode="valid",
    )

    return jnp.stack([R, G, B], axis=-1)


def jax_RGB2bayer(rgb_image: Array, mask: Tuple[Array, Array, Array]) -> Array:
    """
    Converts an RGB image to Bayer pattern using the provided mask pattern.

    Args:
        rgb_image: RGB image with shape (H, W, 3)
        mask: Tuple of 3 boolean arrays, ordered RGB, that are True where pixel values
            exist for the corresponding color in the bayer pattern. Obtained using the
            function ``get_bayer_mask``.

    Returns:
        2D Bayer pattern image
    """
    # Split the RGB channels
    R = rgb_image[..., 0] * mask[0]
    G = rgb_image[..., 1] * mask[1]
    B = rgb_image[..., 2] * mask[2]

    # Combine them into bayer pattern
    return R + G + B


def jax_RGB2HSV(rgb_image):
    """
    Convert RGB to HSV.

    Parameters:
      rgb_image: RGB image with components in the range [0, 255].

    Returns:
      h: Hue in degrees [0, 360)
      s: Saturation in [0, 1]
      v: Value in [0, 1]
    """
    # Normalize to [0,1]
    rgb_image = rgb_image / 255.0
    r, g, b = rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]

    # Compute the maximum and minimum of r, g, b
    max_val = jnp.maximum(jnp.maximum(r, g), b)
    min_val = jnp.minimum(jnp.minimum(r, g), b)
    delta = max_val - min_val

    # Compute hue
    h = jnp.where(
        delta == 0,
        0.0,
        jnp.where(
            max_val == r,
            60.0 * jnp.mod((g - b) / delta, 6.0),
            jnp.where(
                max_val == g,
                60.0 * (((b - r) / delta) + 2.0),
                60.0 * (((r - g) / delta) + 4.0),
            ),
        ),
    )

    # Compute saturation
    s = jnp.where(max_val == 0, 0.0, delta / max_val)
    # Value is the maximum
    v = max_val
    return jnp.stack([h, s, v], axis=-1)
