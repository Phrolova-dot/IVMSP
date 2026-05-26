import numpy as np
import matplotlib
matplotlib.use("agg") # 非交互式后端，防止报错
import matplotlib.pyplot as plt
import scipy.ndimage as ndim
from PIL import Image
import os
from typing import Tuple

# ==========================================
# 辅助：如果没有图片，生成一个带噪声的棋盘格
# ==========================================
def check_and_create_image():
    img_path = os.path.join(os.environ.get("HOME", "."), "chessboard.png")
    if not os.path.exists(img_path):
        # 生成 200x200 棋盘格并添加噪声
        img = np.zeros((200, 200), dtype=np.float64)
        for i in range(200):
            for j in range(200):
                if ((i // 25) + (j // 25)) % 2 == 0:
                    img[i, j] = 255
        # 添加高斯噪声以测试 Sobel 的抗噪性
        noise = np.random.normal(0, 20, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(img_path)
    return img_path

# ==========================================
# 1. Sobel Filter (SciPy Built-in)
# ==========================================
def sobel_filter(
    np_img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    assert isinstance(np_img, np.ndarray)

    # TODO: Use ndim.sobel
    # axis=1 检测列方向变化 (垂直边缘 Fm)
    # axis=0 检测行方向变化 (水平边缘 Fn)
    # 使用 float64 防止 uint8 溢出导致负数截断
    fm = ndim.sobel(np_img, axis=1).astype(np.float64) # [cite: 752]
    fn = ndim.sobel(np_img, axis=0).astype(np.float64) # [cite: 755]

    # TODO: Magnitude
    magnitude = np.sqrt(fm**2 + fn**2) # [cite: 758]

    # TODO: Phase
    phase = np.arctan2(fm, fn) # [cite: 761]

    return fm, fn, magnitude, phase

# ==========================================
# 2. General Edge Detector (Manual Correlation)
# ==========================================
def edge_detector(
    np_img: np.ndarray, dm: np.ndarray, dn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert all(isinstance(arr, np.ndarray) for arr in (np_img, dm, dn))

    # TODO: 2D Correlation
    fm = ndim.correlate(np_img.astype(np.float64), dm) # [cite: 774]
    fn = ndim.correlate(np_img.astype(np.float64), dn) # [cite: 776]

    # TODO: Magnitude
    magnitude = np.sqrt(fm**2 + fn**2) # [cite: 778]

    # TODO: Phase
    phase = np.arctan2(fm, fn) # [cite: 781]

    return fm, fn, magnitude, phase

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    img_path = check_and_create_image()
    
    # 加载图像 [cite: 797-801]
    gray_img = np.array(
        Image.open(img_path).convert("L"),
        "uint8"
    )

    # TODO: Define Sobel Kernels (Eq 5.17 - 5.20)
    # Standard Sobel Dm (Vertical Edges) [cite: 709]
    dm = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    # Standard Sobel Dn (Horizontal Edges) [cite: 709]
    dn = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]])

    # Rotated Sobel Dm (Diagonal) [cite: 724]
    ddm = np.array([[-2, -1, 0],
                    [-1,  0, 1],
                    [0,   1, 2]])

    # Rotated Sobel Dn (Diagonal) [cite: 724]
    ddn = np.array([[0, -1, -2],
                    [1,  0, -1],
                    [2,  1,  0]])

    # TODO: Apply Filters [cite: 802-808]
    # Sobel 1: SciPy Built-in
    fm1, fn1, magnitude1, phase1 = sobel_filter(gray_img)

    # Sobel 2: Manual Standard Sobel
    fm2, fn2, magnitude2, phase2 = edge_detector(gray_img, dm, dn)

    # Sobel 3: Manual Rotated Sobel
    fm3, fn3, magnitude3, phase3 = edge_detector(gray_img, ddm, ddn)

    # TODO: Plotting (Figure 5.9) [cite: 809-825]
    results = [
        (fm1, fn1, magnitude1, phase1, "Sobel 1"),
        (fm2, fn2, magnitude2, phase2, "Sobel 2"),
        (fm3, fn3, magnitude3, phase3, "Sobel 3")
    ]
    
    row_titles = ["", "Fm Vertical Edges", "Fn Horizontal Edges", "Absolute Value", "Phase"]
    
    plt.figure(figsize=(10, 15))
    
    for col in range(3):
        fm, fn, mag, pha, title = results[col]
        imgs = [gray_img, fm, fn, mag, pha]
        
        for row in range(5):
            plt.subplot(5, 3, row * 3 + col + 1)
            plt.imshow(imgs[row], cmap="gray")
            plt.axis("off")
            
            # 设置标题
            if row == 0:
                plt.title(title)
            elif col == 1: # 仅在中间列显示行标题，模拟文档排版
                plt.title(row_titles[row])

    plt.tight_layout()
    plt.savefig("sobel_filtered.png")