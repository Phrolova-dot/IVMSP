
import cv2
import numpy as np
import math
from typing import Tuple, List, Optional

# Exercise 7.3
gray = cv2.imread("mandrill.png", 0) 
scale_levels = 5
sigma = 1.6
k = math.sqrt(2)


scale_space_gauss = np.zeros(
    (gray.shape[0], gray.shape[1], scale_levels), dtype=np.uint8
)

for i in range(scale_levels):
    k_sigma = sigma * k**i

    scale_space_gauss[:, :, i] = cv2.GaussianBlur(
        gray, ksize=(0, 0), sigmaX=k_sigma, sigmaY=k_sigma
    )
    
    cv2.imwrite(
        "imgGauss_Oct1_" + str(i) + ".png",
        scale_space_gauss[:, :, i],
    )


scale_space_dog = np.zeros(
    (gray.shape[0], gray.shape[1], scale_levels - 1), dtype=np.uint8
)

for i in range(scale_levels - 1):
    scale_space_dog[:, :, i] = cv2.absdiff(scale_space_gauss[:, :, i+1], scale_space_gauss[:, :, i]) # TODO: Compute difference
    
    cv2.imwrite(
        "imgDoG_Oct1_" + str(i + 1) + ".png",
        scale_space_dog[:, :, i],
    )

next_oct_gray = cv2.pyrDown(gray)

scale_space_gauss_2 = np.zeros(
    (next_oct_gray.shape[0], next_oct_gray.shape[1], scale_levels), dtype=np.uint8
)


for i in range(scale_levels):
    k_sigma = sigma * k**i
    scale_space_gauss_2[:, :, i] = cv2.GaussianBlur( 
        next_oct_gray, ksize=(0, 0), sigmaX=k_sigma, sigmaY=k_sigma
    )
    
    cv2.imwrite(
        "imgGauss_Oct2_" + str(i) + ".png",
        scale_space_gauss_2[:, :, i],
    )

scale_space_dog_2 = np.zeros(
    (next_oct_gray.shape[0], next_oct_gray.shape[1], scale_levels - 1), dtype=np.uint8
)

for i in range(scale_levels - 1):
    scale_space_dog_2[:, :, i] = cv2.absdiff(
        scale_space_gauss_2[:, :, i+1], scale_space_gauss_2[:, :, i]
    )
    cv2.imwrite(
        "imgDoG_Oct2_" + str(i + 1) + ".png",
        scale_space_dog_2[:, :, i],
    )

