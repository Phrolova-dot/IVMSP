def paste_image(ag_img: np.ndarray, b_img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    # ag_img 是带有 Alpha 通道的前景 (Augmented Image)
    # b_img 是背景图像 (Background)

    # 提取 ROI (Region of Interest) 的背景部分
    # 注意：需要确保坐标不超出图像边界，这里简化处理
    roi = b_img[y:y+h, x:x+w]
    
    # 调整 ag_img 大小以匹配 ROI (GrabCut 后的大小通常即为 frame 大小，
    # 但如果是裁剪过的图像需要 resize。假设 grab_cut 返回的是全帧大小但被 mask 遮罩过，
    # 或者我们需要裁剪出 rect 部分。根据 Exercise 9.7 的逻辑，
    # 我们应该只粘贴 rect 部分。这里假设 ag_img 已经被裁剪为 (w, h) 大小，
    # 或者我们需要从 ag_img 中切片。)
    
    # 更稳妥的逻辑是：ag_img 传入的是全帧大小的 GrabCut 结果
    # 但根据 GrabCut 的输出，非 rect 区域已经是黑色的。
    # 为了简化，我们这里假设传入的 ag_img 已经被裁剪为 (h, w) 大小，
    # 或者我们在函数内部进行切片：
    fg_crop = ag_img[y:y+h, x:x+w]
    
    # (1) 归一化 Alpha 通道 (0.0 - 1.0) 
    alpha_ag = fg_crop[:, :, 3] / 255.0
    alpha_b = 1.0 - alpha_ag

    # (2) 混合通道
    for c in range(0, 3):
        # 逐像素混合：Out = Alpha * FG + (1 - Alpha) * BG
        b_img[y:y+h, x:x+w, c] = (alpha_ag * fg_crop[:, :, c] +
                                  alpha_b * roi[:, :, c])

    return b_img