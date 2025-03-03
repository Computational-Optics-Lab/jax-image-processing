from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from color_manipulation import get_bayer_mask, jax_bayer2GRAY

from scipy.signal.windows import gaussian


def _gkern(kernlen=25, nsig=13):
    """Returns a 2D Gaussian kernel."""
    kern1d = gaussian(kernlen, std=nsig)
    kern2d = jnp.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


@partial(jax.vmap, in_axes=(0, None, None, None))
def calculate_sharpness(
    image: Array,
    gaussian_kernel: Array,
    debayering: bool,
    bayer_mask: Tuple[Array, Array, Array],
) -> Array:
    """
    Calclates contrast measure of the image using Kevin's method after debayering the
    image.

    Divides the original image by its gaussian blurred version. Applies gradient filters
    and smoothing to create the final contrast measure.

    Returns pixel-wise contrast measure array.

    Args:
        image: 2D, single channel image
        gaussian_kernel: 2D gaussian kernel to smooth the measure
        bayer_mask: Tuple of 3 boolean arrays, ordered RGB, that are True where pixel
        values exist for the corresponding color in the bayer pattern. Obtained using
        the function ``get_bayer_mask``.
    """
    if debayering:
        image = jax_bayer2GRAY(image, bayer_mask)
    hpf = jnp.divide(
        image, jax.scipy.signal.fftconvolve(image, _gkern(9, 8), mode="same")
    )
    # kernels in x- and y -direction for Laplacian
    kx = jnp.array(
        [
            [2, 0, -2],
            [8, 0, -8],
            [2, 0, -2],
        ]
    )
    ky = jnp.array(
        [
            [2, 8, 2],
            [0, 0, 0],
            [-2, -8, -2],
        ]
    )
    im_dx = jax.scipy.signal.correlate2d(hpf, kx, mode="same")
    im_dy = jax.scipy.signal.correlate2d(hpf, ky, mode="same")
    grad = jnp.sqrt(jnp.square(im_dx) + jnp.square(im_dy))
    
    return jax.scipy.signal.fftconvolve(grad, gaussian_kernel, mode="same")


@partial(jax.vmap, in_axes=(0, None, None, None))
def calculate_sharpness_sml(
    image: Array,
    gaussian_kernel: Array,
    debayering: bool,
    bayer_mask: Tuple[Array, Array, Array],
) -> Array:
    """
    Calculates contrast measure using the sum modified Laplacian method.

    Returns pixel-wise contrast measure array.

    Args:
        image: 2D, single channel image
        gaussian_kernel: 2D gaussian kernel to smooth the measure
        bayer_mask: Tuple of 3 boolean arrays, ordered RGB, that are True where pixel
        values exist for the corresponding color in the bayer pattern. Obtained using
        the function ``get_bayer_mask``.
    """
    if debayering:
        image = jax_bayer2GRAY(image, bayer_mask)
    # kernels in x- and y -direction for Laplacian
    kx = jnp.array(
        [
            [2, 0, -2],
            [8, 0, -8],
            [2, 0, -2],
        ]
    )
    ky = jnp.array(
        [
            [2, 8, 2],
            [0, 0, 0],
            [-2, -8, -2],
        ]
    )
    # add absolute of image convolved with kx to absolute
    # of image convolved with ky (modified laplacian)
    ml_img = jnp.abs(jax.scipy.signal.convolve2d(image, kx, mode="same")) + jnp.abs(
        jax.scipy.signal.convolve2d(image, ky, mode="same")
    )
    # apply a 2d box filter
    # ml_img = jax.scipy.signal.convolve2d(ml_img, jnp.ones((7, 7)), mode="same")
    return jax.scipy.signal.fftconvolve(ml_img, gaussian_kernel, mode="same")


# @partial(jax.vmap, in_axes=(1, None, None, None))
# @partial(jax.vmap, in_axes=(1, None, None, None))
def all_in_focus_one_stack(
    imstack: Array,
    gaussian_kernel: Array,
    debayering: bool,
    bayer_mask: Tuple[Array, Array, Array],
) -> Array:
    """
    TODO

    Args:
        imstack: z-stack of 2D images
        gaussian_kernel: 2D Gaussian kernel
        bayer_mask: Tuple of 3 boolean arrays, ordered RGB, that are True where pixel
        values exist for the corresponding color in the bayer pattern. Obtained using
        the function ``get_bayer_mask``.
    """
    contrast_measures = calculate_sharpness(
        imstack, gaussian_kernel, debayering, bayer_mask
    )
    indices = jnp.argmax(contrast_measures, axis=0)
    return (
        indices,
        jnp.take_along_axis(imstack, indices[jnp.newaxis, ...], axis=0).squeeze(),
    )


# TODO docs
class all_in_focus:
    def __init__(
        self,
        gkernel_size: int,
        gkernel_sig: float,
        debayering: bool = True,
        data_shape: Tuple[int, int, int, int, int] = (5, 3, 6, 3072, 3072),
        batch_size_x: int = 3,
        batch_size_y: int = 3,
    ) -> None:
        self.debayering = debayering
        if debayering:
            self.bayer_mask = get_bayer_mask(data_shape[-2:])
        self.batch_size_x = batch_size_x
        self.batch_size_y = batch_size_y
        self.gausskernel = _gkern(gkernel_size, gkernel_sig)
        self.calculator = jax.jit(
            jax.vmap(
                jax.vmap(
                    partial(
                        all_in_focus_one_stack,
                        gaussian_kernel=self.gausskernel,
                        debayering=self.debayering,
                        bayer_mask=self.bayer_mask,
                    ),
                    (1,),
                ),
                (1,),
            )
        )
        warmup = np.zeros(
            (
                data_shape[0],
                batch_size_x,
                batch_size_y,
                data_shape[3],
                data_shape[4],
            ),
            dtype=np.uint8,
        )
        a, b = self.calculator(
            warmup,  # self.gausskernel, self.debayering, self.bayer_mask
        )
        del warmup, a, b

    def get_all_in_focus(self, input_stack: Array) -> Array:
        aif_im = np.zeros(input_stack.shape[1:], dtype=jnp.uint8)
        indices = np.zeros(input_stack.shape[1:], dtype=jnp.uint8)
        # batch on the 2nd axis
        for i in range(input_stack.shape[1] // self.batch_size_x):
            for j in range(input_stack.shape[2] // self.batch_size_y):
                ind, im = self.calculator(
                    input_stack[
                        :,
                        i * self.batch_size_x : (i + 1) * self.batch_size_x,
                        j * self.batch_size_y : (j + 1) * self.batch_size_y,
                    ],
                )
                aif_im[
                    i * self.batch_size_x : (i + 1) * self.batch_size_x,
                    j * self.batch_size_y : (j + 1) * self.batch_size_y,
                ] = im
                indices[
                    i * self.batch_size_x : (i + 1) * self.batch_size_x,
                    j * self.batch_size_y : (j + 1) * self.batch_size_y,
                ] = ind
        return indices, aif_im
