import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
import numpy as np
import cv2
import scipy.ndimage as ndim
import os
from typing import Tuple

# ---------------------------------------------------------------
# Function Definition
# ---------------------------------------------------------------
def edge_detector_1d(
    np_img: np.ndarray, dm: np.ndarray, dn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert all(isinstance(arr, np.ndarray) for arr in (np_img, dm, dn))

    # TODO: Calculate gradients
    # Fm represents vertical edges (changes in horizontal direction), so we calculate along axis 1
    # Fn represents horizontal edges (changes in vertical direction), so we calculate along axis 0
    # We use float64 output to prevent uint8 overflow (which causes the "zigzag" artifact)
    fm = ndim.correlate1d(np_img, dm, axis=1, output=np.float64) # TODO
    fn = ndim.correlate1d(np_img, dn, axis=0, output=np.float64) # TODO

    # TODO: Calculate Magnitude
    # L = sqrt(Fm^2 + Fn^2) [cite: 140]
    magnitude = np.sqrt(fm**2 + fn**2) # TODO

    # TODO: Calculate Phase
    # phi = arctan(Fm / Fn) [cite: 140]
    # Using arctan2 to handle division by zero and correct quadrant
    phase = np.arctan2(fm, fn) # TODO

    return fm, fn, magnitude, phase

# ---------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------

# Check if image exists, otherwise create a synthetic one (Self-check helper)
img_path = os.path.join(os.environ.get("HOME", "."), "chessboard.pbm")
if not os.path.exists(img_path):
    # Create a synthetic chessboard if file is missing, to ensure code runs
    gray_img = np.zeros((100, 100), dtype="uint8")
    for i in range(100):
        for j in range(100):
            if ((i // 10) + (j // 10)) % 2 == 0:
                gray_img[i, j] = 255
else:
    # Load the image as specified in the document [cite: 226-231]
    gray_img = np.array(
        Image.open(img_path).convert("L"),
        "uint8"
    )

# Define kernels based on Eqs 5.11 - 5.16 [cite: 174-185]
# Gradient 1 (Backward): [-1, 1, 0]
dm1 = np.array([-1, 1, 0])
dn1 = np.array([-1, 1, 0])

# Gradient 2 (Symmetrical): [-1, 0, 1]
dm2 = np.array([-1, 0, 1])
dn2 = np.array([-1, 0, 1])

# Gradient 3 (Forward): [0, -1, 1]
dm3 = np.array([0, -1, 1])
dn3 = np.array([0, -1, 1])

# Execute edge detection for all 3 gradients [cite: 236-245]
fm1, fn1, magnitude1, phase1 = edge_detector_1d(gray_img, dm1, dn1) # TODO
fm2, fn2, magnitude2, phase2 = edge_detector_1d(gray_img, dm2, dn2) # TODO
fm3, fn3, magnitude3, phase3 = edge_detector_1d(gray_img, dm3, dn3) # TODO

# ---------------------------------------------------------------
# Plotting (Matching Figure 5.6) [cite: 246-257]
# ---------------------------------------------------------------
plt.figure(figsize=(10, 15))

# Row 1: Original Image with Titles
plt.subplot(5, 3, 1), plt.axis("off"), plt.imshow(gray_img, "gray"), plt.title("Gradient 1")
plt.subplot(5, 3, 2), plt.axis("off"), plt.imshow(gray_img, "gray"), plt.title("Gradient 2")
plt.subplot(5, 3, 3), plt.axis("off"), plt.imshow(gray_img, "gray"), plt.title("Gradient 3")

# Row 2: Fm (Vertical Edges) - Centered Title
plt.subplot(5, 3, 4), plt.axis("off"), plt.imshow(fm1, "gray")
plt.subplot(5, 3, 5), plt.axis("off"), plt.imshow(fm2, "gray"), plt.title("Fm vertical edges") # Title centered
plt.subplot(5, 3, 6), plt.axis("off"), plt.imshow(fm3, "gray")

# Row 3: Fn (Horizontal Edges)
plt.subplot(5, 3, 7), plt.axis("off"), plt.imshow(fn1, "gray")
plt.subplot(5, 3, 8), plt.axis("off"), plt.imshow(fn2, "gray"), plt.title("Fn horizontal edges")
plt.subplot(5, 3, 9), plt.axis("off"), plt.imshow(fn3, "gray")

# Row 4: Absolute Value
plt.subplot(5, 3, 10), plt.axis("off"), plt.imshow(magnitude1, "gray")
plt.subplot(5, 3, 11), plt.axis("off"), plt.imshow(magnitude2, "gray"), plt.title("Absolute value")
plt.subplot(5, 3, 12), plt.axis("off"), plt.imshow(magnitude3, "gray")

# Row 5: Phase
plt.subplot(5, 3, 13), plt.axis("off"), plt.imshow(phase1, "gray")
plt.subplot(5, 3, 14), plt.axis("off"), plt.imshow(phase2, "gray"), plt.title("Phase")
plt.subplot(5, 3, 15), plt.axis("off"), plt.imshow(phase3, "gray")

plt.tight_layout()
plt.savefig("filtered.png") # TODO