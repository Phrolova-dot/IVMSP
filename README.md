# IVMSP

Image and video processing coursework scripts and generated experiment results.

## Contents

- `grayscale_histogram_processing/`  
  Grayscale conversion, histogram analysis, histogram equalization, gamma correction, and normalization.
- `filtering_edge_detection/`  
  Image filtering, Sobel edge detection, and Laplace filtering exercises.
- `harris_corner_detection/`  
  Harris corner detection experiments and generated result images.
- `sift_panorama_stitching/`  
  Gaussian/DoG pyramids, SIFT keypoints, feature matching, and panorama stitching scripts.
- `raspberry_pi_camera_pen_detection/`  
  Raspberry Pi camera preview, video playback, FPS measurement, and pen detection scripts.
- `haar_grabcut_background_replacement/`  
  Haar cascade object detection, GrabCut segmentation, and background replacement scripts.

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
- `sift_panorama_stitching/Experiment7.py` and `Experiment7_5.py` reference `imgNBG1.png` and `imgNBG2.png`; these files were not present in the local source folder.
