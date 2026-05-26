# IVMSP

Image and video processing coursework scripts and generated experiment results.

## Contents

- `exercise_4_gray_histogram_processing/`  
  Grayscale conversion, histogram analysis, histogram equalization, gamma correction, and normalization.
- `exercise_5_filtering_edges/`  
  Image filtering, Sobel edge detection, and Laplace filtering exercises.
- `experiment_6_harris_corner_detection/`  
  Harris corner detection experiments and generated result images.
- `experiment_7_sift_panorama/`  
  Gaussian/DoG pyramids, SIFT keypoints, feature matching, and panorama stitching scripts.
- `exercise_8/`  
  Exercise 8 scripts.
- `exercise_9/`  
  Exercise 9 scripts.

## Setup

```bash
pip install -r requirements.txt
```

For Raspberry Pi camera scripts:

```bash
pip install -r requirements-rpi.txt
```

## Notes

- Generated images are stored alongside their related scripts for easier review.
- Some scripts expect input files in their working directory. Run scripts from the directory that contains the script and its images unless the script says otherwise.
- `experiment_7_sift_panorama/Experiment7.py` and `Experiment7_5.py` reference `imgNBG1.png` and `imgNBG2.png`; these files were not present in the local source folder.
