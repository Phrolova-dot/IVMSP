# Image and Video Processing Lab

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

## 验证与运行

在仓库根目录执行不依赖硬件的回归测试：

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖有符号梯度、分割掩码、彩色全景拼接、匹配失败、取消采集、角度显示和视频资源释放。
相机驱动、GPIO、检测准确率及树莓派实际帧率仍需真机验证。

- `Experiment7.py` 进入交互拍照流程，采集 2–6 张图；取消采集会正常退出。
  `Experiment7_5.py` 从脚本所在目录读取 `imgNBG1.png` 和 `imgNBG2.png`。
- 将笔分类器 XML 放在笔检测脚本旁；仓库尚未包含这些训练文件。
- `python3 raspberry_pi_camera_pen_detection/Exercise8_3.py --video INPUT.mp4`
  使用视频输入，无需加载树莓派硬件库，按 `q` 停止。不传 `--video` 时使用相机和 GPIO 6 红键。
- `Exercise8_5.py` 显示图像坐标系中的笔方向：向右为 0°，向下为 90°，以 180° 为周期。
  检测始终读取未绘制标记的原图。
- GrabCut 使用 `cutout, mask = grab_cut(..., return_mask=True)` 返回二值掩码。
  将掩码传给 `add_alpha_channel(cutout, mask)`，保留目标内部的黑色像素。
  `paste_image` 接收尺寸一致的整帧前景和背景。
- 背景替换运行方式：

```bash
python3 haar_grabcut_background_replacement/Exercise9_7.py foreground.mp4 background.mp4 classifier.xml --output output_final.mp4
```

输出采用前景视频的尺寸和帧率；背景缩放到相同尺寸，按帧序号配对，任一路结束即停止。
需要时间同步时，应使用相同帧率的两路输入。
