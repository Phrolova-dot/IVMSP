import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import cv2
from pylab import *
from PIL import Image
import scipy.ndimage as ndim
from typing import Tuple, Union
import os

# (Reusing sobel_filter from Ex 5.3 for comparison)
def sobel_filter(np_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fm = ndim.sobel(np_img, axis=1, output=float)
    fn = ndim.sobel(np_img, axis=0, output=float)
    return fm, fn

# ---------------------------------------------------------
# Function: gauss_filter
# ---------------------------------------------------------
def gauss_filter(
    np_img: np.ndarray, sigma: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:

    assert isinstance(np_img, np.ndarray) and isinstance(sigma, (int, float))

    # TODO: Gaussian gradient using 'order' parameter [cite: 506-510]
    # order=(0,1) corresponds to 0th derivative in y (smooth) and 1st in x (diff) -> Fm
    fm = ndim.gaussian_filter(np_img, sigma=sigma, order=(0, 1), output=np.float64)
    # order=(1,0) corresponds to 1st derivative in y (diff) and 0th in x (smooth) -> Fn
    fn = ndim.gaussian_filter(np_img, sigma=sigma, order=(1, 0), output=np.float64)

    return fm, fn

# ---------------------------------------------------------
# Function: laplace_filter
# ---------------------------------------------------------
def laplace_filter(np_img: np.ndarray) -> np.ndarray:

    assert isinstance(np_img, np.ndarray)

    # TODO: Laplace filter (approximation of 2nd derivative) [cite: 519]
    fmn = ndim.laplace(np_img.astype(float))

    return fmn

# ---------------------------------------------------------
# Function: laplace_gauss_filter
# ---------------------------------------------------------
def laplace_gauss_filter(np_img: np.ndarray, sigma: Union[int, float]) -> np.ndarray:

    assert isinstance(np_img, np.ndarray) and isinstance(sigma, (int, float))

    # TODO: Laplacian of Gaussian (LoG) [cite: 528]
    fmn = ndim.gaussian_laplace(np_img.astype(float), sigma=sigma)

    return fmn

# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load Image
    try:
        img_path = os.path.join(os.environ.get("HOME", "."), "chessboard.png")
        gray_img = np.array(Image.open(img_path).convert("L"), "uint8") # [cite: 534-538]
    except FileNotFoundError:
        print("Warning: chessboard.png not found. Creating dummy image.")
        gray_img = np.zeros((200, 200), dtype="uint8")

    sigma = 5 # [cite: 539]

    # TODO: Calculate all features [cite: 543-549]
    fm1, fn1 = sobel_filter(gray_img) # Sobel
    fm2, fn2 = gauss_filter(gray_img, sigma) # Gaussian Gradient
    fmn1 = laplace_filter(gray_img) # Laplace
    fmn2 = laplace_gauss_filter(gray_img, sigma) # LoG

    # TODO: Plotting according to Figure 5.11 [cite: 550-567]
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 2, 1), plt.axis("off")
    plt.imshow(fm1, "gray"), plt.title("Sobel m")

    plt.subplot(3, 2, 2), plt.axis("off")
    plt.imshow(fn1, "gray"), plt.title("Sobel n")

    plt.subplot(3, 2, 3), plt.axis("off")
    plt.imshow(fm2, "gray"), plt.title("Gauss m")

    plt.subplot(3, 2, 4), plt.axis("off")
    plt.imshow(fn2, "gray"), plt.title("Gauss n")

    plt.subplot(3, 2, 5), plt.axis("off")
    plt.imshow(fmn1, "gray"), plt.title("Laplace mn")

    plt.subplot(3, 2, 6), plt.axis("off")
    plt.imshow(fmn2, "gray"), plt.title("Laplace Gauss mn")

    plt.tight_layout()
    plt.savefig("laplace_filtered.png")
