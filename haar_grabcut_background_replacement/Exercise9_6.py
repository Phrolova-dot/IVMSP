import cv2
import numpy as np
import time

def paste_image(ag_img, b_img, x, y, w, h):
    """Composite matching full-frame foreground and background arrays."""
    if ag_img.shape[:2] != b_img.shape[:2]:
        raise ValueError("Foreground and background dimensions must match")
    height, width = b_img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return b_img
    fg_crop = ag_img[y0:y1, x0:x1]
    roi = b_img[y0:y1, x0:x1]
    alpha = fg_crop[:, :, 3:4].astype(np.float64) / 255.0
    b_img[y0:y1, x0:x1] = np.rint(alpha * fg_crop[:, :, :3] + (1-alpha) * roi).astype(np.uint8)
    return b_img
