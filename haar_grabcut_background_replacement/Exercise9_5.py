import cv2
import numpy as np
import time

def add_alpha_channel(img, mask):
    """Build alpha from a segmentation mask, preserving black foreground pixels."""
    if mask.shape != img.shape[:2]:
        raise ValueError("Mask and image dimensions must match")
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = (mask != 0).astype(np.uint8) * 255
    return result
