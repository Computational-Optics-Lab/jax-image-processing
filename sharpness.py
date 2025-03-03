from functools import partial

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array

__all__ = ["get_opencv_laplace_kernel", "calculate_laplacian", "laplace_engine"]


def get_opencv_laplace_kernel(ksize: int = 1) -> Array:
    """
    Obtains the Laplace kernel used by ``cv2.Laplacian`` for the given kernel size.

    Args:
        ksize: size of the desired kernel. Must be an odd integer.

    Returns:
        Array: 2D Laplace kernel of shape (ksize, ksize) or (3, 3) if ksize=1
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
    Calculates the Laplacian matrix of the image using OpenCV-style kernels.

    Returns a 2D array containing the Laplacian values for each pixel.

    Args:
        image: 2D, single channel image array
        ksize: Laplace kernel size (openCV kernels are used). Must be an odd integer.

    Returns:
        Array: 2D array of same shape as input containing Laplacian values
    """
    kernel = get_opencv_laplace_kernel(ksize=ksize)
    return jax.scipy.signal.convolve2d(image, kernel, mode="same")


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(0, None))
def downsample_bayer_laplacian(bayer_image: Array, kernel: Array) -> Array:
    """
    Calculates mean Laplacian of a GBRG Bayer image by selecting green pixels.

    Uses 2x2 downsampling on the Bayer mask. Applies thresholding to focus on
    significant edges and performs local averaging using a 3x3 window.

    Args:
        bayer_image: 2D array containing GBRG Bayer pattern image
        kernel: 2D Laplace kernel from get_opencv_laplace_kernel()

    Returns:
        float: Mean value of the thresholded and locally averaged Laplacian
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
    Calculates variance of Laplacian of a debayered gray image.

    Uses 2x2 downsampling on the Bayer mask. The kernel should be generated using the
    ``get_opencv_laplace_kernel`` method.

    Args:
        bayer_image: 2D GBRG bayer image
        bayer_mask: 2D array defining the Bayer pattern mask
        kernel: 2D Laplace kernel

    Returns:
        float: Variance of the Laplacian values in the debayered image
    """
    gray_im = jax_bayer2GRAY(bayer_image, bayer_mask)
    return jnp.var(jax.scipy.signal.convolve2d(gray_im, kernel, mode="same"))


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(0, None))
def downsample_bayer_nv(bayer_image: Array, kernel: Array) -> Array:
    """
    Calculates normalized variance metric on R/B channels of Bayer image.

    Uses 2x2 downsampling and specific ROI selection (511:2559, 512:2560).
    The kernel parameter is unused but kept for API consistency.

    Args:
        bayer_image: 2D GBRG bayer image
        kernel: Unused parameter, kept for API consistency

    Returns:
        float: Normalized variance (variance/mean) of the selected ROI
    """
    return jnp.var(bayer_image[511:2559:2, 512:2560:2]) / jnp.mean(
        bayer_image[511:2559:2, 512:2560:2]
    )


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(0, None))
def downsample_bayer_sml(bayer_image: Array, kernel: Array) -> Array:
    """
    Calculates Sum-Modified-Laplacian (SML) focus measure on Bayer image.

    Uses 2x2 downsampling and combines horizontal and vertical edge detection.
    The kernel parameter is unused but kept for API consistency.

    Args:
        bayer_image: 2D GBRG bayer image
        kernel: Unused parameter, kept for API consistency

    Returns:
        float: Mean of thresholded and box-filtered SML values
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
    """
    Engine for calculating Laplacian-based focus metrics on batches of images.

    Handles batched processing of microscopy image data to compute focus metrics
    and find optimal focus positions.

    Args:
        ksize (int): Size of the Laplacian kernel to use
        data_shape (tuple): Expected shape of input data (slices, y, x, height, width)
        batch_size (int): Number of images to process simultaneously

    Attributes:
        kernel (Array): Pre-computed Laplacian kernel
        batch_size (int): Batch size for processing
        calculator (Callable): JIT-compiled Laplacian calculation function
    """

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
        """
        Calculates focus metrics for a batch of images.

        Args:
            data: Array-like object containing image data to process

        Returns:
            Array: Focus metrics reshaped to match input data dimensions
        """
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
        """
        Finds the optimal focus position for each XY position.

        Args:
            data: Array-like object containing Z-stack image data

        Returns:
            Array: Indices of best focus positions for each XY position
        """
        metrics = self.get_metric(data)
        return jnp.argmax(metrics, axis=0)

def _process_one_zslice(data_slice, ksize, batch_size):
    """
    Internal helper function to process a single Z-slice of data in batches.

    Args:
        data_slice: 3D array containing a single Z-slice of image data
        ksize: 2D Laplace kernel to use for processing
        batch_size: Number of images to process simultaneously

    Returns:
        Array: 1D array of concatenated Laplacian metrics for the Z-slice
    """
    batches = jnp.arange(0, data_slice.shape[0], batch_size)
    # length = data_slice.shape[0]

    def body(_, i):
        return None, downsample_bayer_laplacian(
            jax.lax.dynamic_slice(data_slice, (i, 0, 0), (batch_size, 3072, 3072)),
            ksize,
        )

    return jnp.hstack(jax.lax.scan(body, None, batches)[1]).squeeze()