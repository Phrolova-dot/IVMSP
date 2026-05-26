import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from PIL import Image
from typing import Tuple

def normal_cdf() -> Tuple[np.ndarray, np.ndarray]:
    # 使用 50 个采样点并返回 ndarray
    x = np.linspace(-4, 4, 50)
    y = norm.cdf(x)
    return x, y

def convert_rgb_to_gray(file_name: str) -> np.ndarray:
    assert isinstance(file_name, str)
    # 明确返回 uint8
    return np.array(Image.open(file_name).convert("L"), dtype=np.uint8)

def histogram_equalize(img_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(img_gray, np.ndarray) and img_gray.ndim == 2

    # 直方图与 CDF
    hist = np.bincount(img_gray.ravel(), minlength=256)
    cdf = hist.cumsum().astype(np.float64)

    # 归一化到 [0, 255]，忽略前导零（避免除零）
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_norm = (cdf_masked - cdf_masked.min()) * 255.0 / (cdf_masked.max() - cdf_masked.min())
    cdf_lut = np.ma.filled(cdf_norm, 0).astype(np.uint8)  # 作为 LUT

    # 应用 LUT
    img_equal = cdf_lut[img_gray]
    return img_equal.astype(img_gray.dtype), cdf_lut

# —— 主程序 —— #
mandrill_1 = convert_rgb_to_gray("mandrill.png")
mandrill_2, cdf = histogram_equalize(mandrill_1)
x, y = normal_cdf()

plt.figure(figsize=(12, 8))

# (1) 原图
plt.subplot(231), plt.title("Before"), plt.axis("off")
plt.imshow(mandrill_1, cmap="gray")

# (2) 正态 CDF
plt.subplot(232), plt.title("Normal cdf")
plt.plot(x, y)
plt.xlabel("x"), plt.ylabel("Probability")

# (3) 均衡化后
plt.subplot(233), plt.title("After"), plt.axis("off")
plt.imshow(mandrill_2, cmap="gray")

# (4) 原图直方图
plt.subplot(234), plt.title("Histogram 1"), plt.xlim(0, 255)
plt.hist(mandrill_1.ravel(), bins=256, range=(0, 256), color="steelblue")

# (5) 转换函数（CDF 映射）
plt.subplot(235), plt.title("Transformation")
plt.plot(np.arange(256), cdf, color="darkorange")
plt.xlabel("Input intensity"), plt.ylabel("Equalized intensity")
plt.xlim(0, 255), plt.ylim(0, 255)

# (6) 均衡化后直方图
plt.subplot(236), plt.title("Histogram 2"), plt.xlim(0, 255)
plt.hist(mandrill_2.ravel(), bins=256, range=(0, 256), color="gray")

plt.tight_layout()
plt.savefig("mandrill_equalized.png")
print("直方图均衡化完成并已保存为 mandrill_equalized.png")