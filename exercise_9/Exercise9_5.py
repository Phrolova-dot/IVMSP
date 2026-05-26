def add_alpha_channel(img: np.ndarray) -> np.ndarray:
    start_time = time.time()

    # (1) 颜色空间转换 BGR -> BGRA 
    # 这会自动添加全不透明 (255) 的 Alpha 通道 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # 找到黑色像素 [0, 0, 0] 并将其 Alpha 通道设为 0 (透明) 
    # 检查前三个通道是否都为 0
    black_pixels = np.all(img[:, :, :3] == 0, axis=2)
    img[black_pixels, 3] = 0

    end_time = time.time()
    # print(f"Add Alpha Runtime: {end_time - start_time} seconds")

    return img