import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

# 基础函数
def convert_rgb_to_gray(path: str) -> np.ndarray:
    # 读入并转灰度
    return np.array(Image.open(path).convert("L"))

def histogram_equalize(img_gray: np.ndarray):
    # 直方图均衡化：返回均衡图与 LUT
    hist = np.bincount(img_gray.ravel(), minlength=256)
    cdf = hist.cumsum().astype(np.float64)
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_norm = (cdf_masked - cdf_masked.min()) * 255.0 / (cdf_masked.max() - cdf_masked.min())
    lut = np.ma.filled(cdf_norm, 0).astype(np.uint8)
    equal = lut[img_gray]
    return equal, lut

# Gamma 校正：返回校正图与传递函数 (0..255)
def gamma_correction(img_gray: np.ndarray, gamma: float):
    # Vout = 255 * (Vin/255)^(1/gamma)
    assert img_gray.ndim == 2
    inv_gamma = 1.0 / gamma
    norm = img_gray.astype(np.float32) / 255.0
    out = np.clip(norm ** inv_gamma, 0.0, 1.0)
    out_img = (out * 255.0 + 0.5).astype(np.uint8)
    vin = np.arange(256, dtype=np.float32) / 255.0
    tf = (vin ** inv_gamma * 255.0 + 0.5).astype(np.uint8)
    return out_img, tf

# 归一化 (对比度拉伸)
def normalize_image(img_gray: np.ndarray) -> np.ndarray:
    vmin = int(img_gray.min())
    vmax = int(img_gray.max())
    if vmax == vmin:
        return img_gray.copy()
    out = (img_gray.astype(np.float32) - vmin) * 255.0 / (vmax - vmin)
    return (out + 0.5).astype(np.uint8)

def normal_cdf():
    # 生成正态 CDF 曲线（使用 math.erf 避免依赖 numpy.erf）
    x = np.linspace(-4, 4, 200)
    y = 0.5 * (1.0 + np.array([math.erf(xx / math.sqrt(2.0)) for xx in x]))
    return x, y

if __name__ == "__main__":
    img = convert_rgb_to_gray("mandrill.png")

    eq_img, eq_lut = histogram_equalize(img)
    gamma_04_img, gamma_04_tf = gamma_correction(img, 0.4)
    gamma_24_img, gamma_24_tf = gamma_correction(img, 2.4)
    norm_img = normalize_image(img)
    x, y = normal_cdf()

    Image.fromarray(gamma_04_img).save("mandrill_gamma_0.4.png")
    Image.fromarray(gamma_24_img).save("mandrill_gamma_2.4.png")
    Image.fromarray(norm_img).save("mandrill_normalized.png")

    plt.figure(figsize=(14, 8))
    plt.subplot(231); plt.title("Original"); plt.axis("off"); plt.imshow(img, cmap="gray")
    plt.subplot(232); plt.title("Gamma 0.4"); plt.axis("off"); plt.imshow(gamma_04_img, cmap="gray")
    plt.subplot(233); plt.title("Gamma 2.4"); plt.axis("off"); plt.imshow(gamma_24_img, cmap="gray")
    plt.subplot(234); plt.title("Equalized"); plt.axis("off"); plt.imshow(eq_img, cmap="gray")
    plt.subplot(235); plt.title("Normalized"); plt.axis("off"); plt.imshow(norm_img, cmap="gray")
    plt.subplot(236); plt.title("Transfer functions")
    plt.plot(gamma_04_tf, label="gamma=0.4")
    plt.plot(gamma_24_tf, label="gamma=2.4")
    plt.plot(np.arange(256), eq_lut, label="Equalize LUT", alpha=0.6)
    plt.plot([0, 255], [0, 255], "k--", label="Identity")
    plt.xlim(0, 255); plt.ylim(0, 255); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("mandrill_gamma_normalize_equalize.png")
    print("Done: generated gamma, normalization and equalization results.")