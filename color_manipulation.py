from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array

__all__ = ["get_bayer_mask", "jax_bayer2GRAY", "jax_bayer2RGB", "RGB2bayer"]


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


def RGB2bayer(rgb_image: Array, mask: Tuple[Array, Array, Array]) -> Array:
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


if __name__ == "__main__":
    import os
    from time import time

    import cv2

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    image = np.random.randint(255, size=(80, 3072, 3072), dtype=np.uint8)
    print(len(image))

    bayer_mask = get_bayer_mask(image.shape[-2:])
    print(bayer_mask.shape)

    strt = time()
    debayer = []
    for i in range(image.shape[0]):
        debayer.append(cv2.cvtColor(image[i], cv2.COLOR_BayerGR2GRAY))
    debayer = np.array(debayer)
    end = time()
    print("cv2: ", end - strt)
    print(debayer.shape)
    del debayer

    test = np.random.randint(255, size=(32, 3072, 3072))
    vmap_debayer = jax.vmap(jax_bayer2GRAY, in_axes=(0, None))
    jit_debayer = jax.jit(vmap_debayer)
    jit_debayer(test, bayer_mask)
    del test

    strt = time()
    out = []
    for i in range(image.shape[0] // 32):
        zslice = jit_debayer(image[i * 32 : (i + 1) * 32], bayer_mask)
        out.append(zslice)
    end = time()
    print("jax: ", end - strt)

    out = np.concatenate(out, axis=0)
    print(out.shape)
