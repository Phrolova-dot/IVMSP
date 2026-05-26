import cv2
import numpy as np
import time

def grab_cut(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    start_time = time.time() # Task: Save time before

    # (1) 创建 mask (uint8)
    mask = np.zeros(frame.shape[:2], np.uint8)

    # (2) 创建背景和前景模型 (1, 65, float64)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # (3) 定义矩形 [cite: 1009, 1028]
    rect = (x, y, w, h)

    # (4) 执行 GrabCut，迭代 5 次
    # 模式为 GC_INIT_WITH_RECT，因为我们有 Haar Cascade 提供的矩形
    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # (5) 转换 mask
    # mask 中 0(BG) 和 2(Prob. BG) 设为 0，其余 (1, 3) 设为 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 将 mask 应用于图像 
    frame = frame * mask2[:, :, np.newaxis]

    end_time = time.time() # Task: Save time after
    # print(f"GrabCut Runtime: {end_time - start_time} seconds") 

    return frame