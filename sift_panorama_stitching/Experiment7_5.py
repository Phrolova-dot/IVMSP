from pathlib import Path
import cv2
import select
import sys
import time
from typing import List, Tuple, Optional
import numpy as np
import threading
import psutil
import math



def draw_matches(
    img1: np.ndarray,
    kp1: Tuple[cv2.KeyPoint, ...],
    img2: np.ndarray,
    kp2: Tuple[cv2.KeyPoint, ...],
    matches: List[cv2.DMatch],
) -> np.ndarray:
    assert (
        all(isinstance(arr, np.ndarray) for arr in (img1, img2))
        and all(
            isinstance(t, tuple)
            and all(isinstance(element, cv2.KeyPoint) for element in t)
            for t in (kp1, kp2)
        )
        and isinstance(matches, list)
        and all(isinstance(element, cv2.DMatch) for element in matches)
    )

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]


    matches_img = np.zeros((max([h1, h2]), w1 + w2, 3), dtype="uint8")

    matches_img[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1[:, :, :3]
    matches_img[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2[:, :, :3]

    for i in matches:
        img1_idx = i.queryIdx
        img2_idx = i.trainIdx

        (m1, n1) = kp1[img1_idx].pt
        (m2, n2) = kp2[img2_idx].pt

        cv2.circle(matches_img, (int(m1), int(n1)), 5, (0, 255, 0), 1)
        cv2.circle(matches_img, (int(m2) + w1, int(n2)), 5, (0, 255, 0), 1)

        cv2.line(
            matches_img,
            (int(m1), int(n1)),
            (int(m2) + w1, int(n2)),
            (0, 255, 0), 1
        )

    return matches_img

def main():
    img1 = cv2.imread(str(Path(__file__).with_name("imgNBG1.png")), 0)
    img2 = cv2.imread(str(Path(__file__).with_name("imgNBG2.png")), 0)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Place imgNBG1.png and imgNBG2.png beside this script")

    FEATURES = 400
    sift = cv2.SIFT_create(FEATURES)


    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("No usable features in one or both images")

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue
        match1, match2 = pair
        if match1.distance < 0.8 * match2.distance:
            good_matches.append(match1)

    print(f"Good matches with Brute-Force found: {len(good_matches)}")


    feat_img_a = cv2.drawKeypoints(img1, kp1, None, (0, 0, 255), 4)
    feat_img_b = cv2.drawKeypoints(img2, kp2, None, (0, 0, 255), 4)
    cv2.imwrite("featImgA.png", feat_img_a)
    cv2.imwrite("featImgB.png", feat_img_b)


    result_matches = draw_matches(img1, kp1, img2, kp2, good_matches)
    cv2.imwrite("matches.png", result_matches)


if __name__ == "__main__":
    main()
