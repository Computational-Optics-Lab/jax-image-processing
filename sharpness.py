from functools import partial

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array

__all__ = ["get_opencv_laplace_kernel", "calculate_laplacian", "laplace_engine"]


def get_opencv_laplace_kernel(ksize: int = 1) -> Array:
    """
    Obtains the Laplace kernel used by ``cv2.Laplacian`` for the given kernel size. This
    way we can keep high fidelity with OpenCV calculated Laplacian values.

    Returns a 2D Laplace kernel of shape (ksize, ksize).

    Args:
        ksize: size of the desired kernel. Must be an odd integer.
    """
    if ksize != 1:
        blocksize = ksize
    else:
        blocksize = 3
    impulse = np.zeros((blocksize + 2, blocksize + 2), dtype=np.uint8)
    impulse[blocksize // 2 + 1, blocksize // 2 + 1] = 1
    kernel = cv2.Laplacian(impulse, cv2.CV_32F, ksize=ksize)
    # crop this down to blocksize and return
    return jnp.array(kernel[1:-1, 1:-1])


def calculate_laplacian(image: Array, ksize: int) -> Array:
    """
    Calclates the laplcian matrix of the image.

    Returns laplcian.

    Args:
        image: 2D, single channel image
        ksize: Laplace kernel size (openCV kernels are used)
    """
    kernel = get_opencv_laplace_kernel(ksize=ksize)
    return jax.scipy.signal.convolve2d(image, kernel, mode="same")


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(0, None))
def downsample_bayer_laplacian(bayer_image: Array, kernel: Array) -> Array:
    """
    Calculates mean laplacian of a GBRG Bayer image by selecting green only pixels. Uses
    2x2 downsampling on the Bayer mask. The kernel should be generated using the
    ``get_opencv_laplace_kernel`` method.

    Returns the mean value of the laplacian of the image.

    Args:
        bayer_image: 2D, GBRG bayer image
        kernel: 2D laplace kernel
    """
    metric = jnp.abs(
        jax.scipy.signal.convolve2d(bayer_image[1::2, ::2], kernel, mode="same")
    )
    metric = metric * (metric > 100)
    return jnp.mean(jax.scipy.signal.convolve2d(metric, jnp.ones((3, 3)), mode="same"))


@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(0, None, None))
def downsample_debayer_laplacian(
    bayer_image: Array, bayer_mask: Array, kernel: Array
) -> Array:
    """
    Calculates mean laplacian of a debayered gray image. Uses
    2x2 downsampling on the Bayer mask. The kernel should be generated using the
    ``get_opencv_laplace_kernel`` method.

    Returns the mean value of the laplacian of the image.

    Args:
        bayer_image: 2D, GBRG bayer image
        kernel: 2D laplace kernel
    """
    gray_im = jax_bayer2GRAY(bayer_image, bayer_mask)
    return jnp.var(jax.scipy.signal.convolve2d(gray_im, kernel, mode="same"))


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(0, None))
def downsample_bayer_nv(bayer_image: Array, kernel: Array) -> Array:
    """
    Calculates mean laplacian of a GBRG Bayer image by selecting green only pixels. Uses
    2x2 downsampling on the Bayer mask. The kernel should be generated using the
    ``get_opencv_laplace_kernel`` method.

    Returns the mean value of the laplacian of the image.

    Args:
        bayer_image: 2D, GBRG bayer image
        kernel: 2D laplace kernel
    """
    return jnp.var(bayer_image[511:2559:2, 512:2560:2]) / jnp.mean(
        bayer_image[511:2559:2, 512:2560:2]
    )


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(0, None))
def downsample_bayer_sml(bayer_image: Array, kernel: Array) -> Array:
    """
    Calculates mean laplacian of a GBRG Bayer image by selecting green only pixels. Uses
    2x2 downsampling on the Bayer mask. The kernel should be generated using the
    ``get_opencv_laplace_kernel`` method.

    Returns the mean value of the laplacian of the image.

    Args:
        bayer_image: 2D, GBRG bayer image
        kernel: 2D laplace kernel
    """
    # kernels in x- and y -direction for Laplacian
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
    # add absoulte of image convolved with kx to absolute
    # of image convolved with ky (modified laplacian)
    ml_img = jnp.abs(
        jax.scipy.signal.convolve2d(
            bayer_image[255:2815:2, 256:2816:2], kx, mode="same"
        )
    ) + jnp.abs(
        jax.scipy.signal.convolve2d(
            bayer_image[255:2815:2, 256:2816:2], ky, mode="same"
        )
    )
    # threhold the modified laplacian image
    ml_img = ml_img * (ml_img > 1)
    # apply 2d box filter to the thresholded image
    return jnp.mean(jax.scipy.signal.convolve2d(ml_img, jnp.ones((3, 3)), mode="same"))


# try a laplacian class to A. have a single callable method for calculating the laplcian
# in the acquistion code rather than setting up loops in there, and B. to see if this
# allows for a warmup (which is not happening, probably because of lax.scan)


# TODO write docs
class laplace_engine:
    def __init__(self, ksize, data_shape=(50, 6, 6, 3072, 3072), batch_size=3):
        self.kernel = get_opencv_laplace_kernel(ksize)
        self.batch_size = batch_size  # 3 works best with 50 slices on the DUMC system
        self.calculator = jax.jit(downsample_bayer_laplacian)
        zero_slice = jnp.zeros(
            (data_shape[0], self.batch_size, data_shape[3], data_shape[4]),
        )
        k = self.calculator(zero_slice, self.kernel)
        del k, zero_slice

    def get_metric(self, data):
        datak = np.reshape(data.values, (data.shape[0], -1, 3072, 3072))
        metrics = []
        for j in range(datak.shape[1] // self.batch_size):
            metrics.append(
                self.calculator(
                    datak[:, j * self.batch_size : (j + 1) * self.batch_size],
                    self.kernel,
                )
            )

        return (
            jnp.hstack(metrics)
            .reshape((data.shape[0], data.shape[1], data.shape[2]))
        )

    def find_best_z(self, data):
        metrics = self.get_metric(data)
        return jnp.argmax(metrics, axis=0)

def _process_one_zslice(data_slice, ksize, batch_size):
    batches = jnp.arange(0, data_slice.shape[0], batch_size)
    # length = data_slice.shape[0]

    def body(_, i):
        return None, downsample_bayer_laplacian(
            jax.lax.dynamic_slice(data_slice, (i, 0, 0), (batch_size, 3072, 3072)),
            ksize,
        )

    return jnp.hstack(jax.lax.scan(body, None, batches)[1]).squeeze()