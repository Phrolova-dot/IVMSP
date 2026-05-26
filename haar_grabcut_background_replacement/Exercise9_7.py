import cv2
import numpy as np
import time

# --- 步骤 1: 放入之前写的辅助函数 ---

def grab_cut(frame, x, y, w, h):
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

def add_alpha_channel(img):
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
def paste_image(ag_img, b_img, x, y, w, h):
    for c in range(0, 3):
        # 逐像素混合：Out = Alpha * FG + (1 - Alpha) * BG
        b_img[y:y+h, x:x+w, c] = (alpha_ag * fg_crop[:, :, c] +
                                  alpha_b * roi[:, :, c])

    return b_img

# --- 步骤 2: 编写主函数 (Exercise 9.7) ---

def background_replacement(input_video_path, background_video_path, weightsfile):
    
    # (1) 初始化视频捕获对象 [cite: 1107, 1156]
    cap = cv2.VideoCapture(input_video_path)
    background_cap = cv2.VideoCapture(background_video_path)

    # (2) 初始化分类器 [cite: 1109, 1157]
    object_classifier = cv2.CascadeClassifier(weightsfile)

    frame_counter = 0

    # (3) 初始化视频写入器 (VideoWriter) [cite: 1118-1120, 1158]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 注意：如果背景视频尺寸不同，可能需要 resize，或者在这里以背景视频尺寸为准
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    out = cv2.VideoWriter("output_final.mp4", fourcc, 10.0, (width, height))

    # (4) 确定最大帧数 [cite: 1123, 1159]
    # 输出视频长度受限于两个输入视频中较短的那个
    len_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    len_bg = int(background_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(len_input, len_bg)

    print(f"Starting processing for {max_frames} frames...")

    # --- 步骤 3: 主循环处理 ---
    while frame_counter < max_frames:
        
        # 读取两路视频的帧 [cite: 1129-1130]
        ret, frame = cap.read()
        success, background_frame = background_cap.read()

        if not ret or not success:
            break

        # 转换为灰度图用于 Haar 检测 [cite: 1131]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测对象 (人脸/人) [cite: 1139]
        detected_objects = object_classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # (5) 核心整合步骤：对检测到的对象执行替换流程 [cite: 1140, 1161]
        for x, y, w, h in detected_objects:
            # A. 使用 GrabCut 提取前景
            cutout = grab_cut(frame, x, y, w, h)
            
            # B. 添加 Alpha 通道 (变透明)
            cutout_alpha = add_alpha_channel(cutout)
            
            # C. 粘贴到背景图上
            # 注意：这里直接修改了 background_frame
            background_frame = paste_image(cutout_alpha, background_frame, x, y, w, h)

        # 写入合成后的帧 [cite: 1150]
        out.write(background_frame)
        
        frame_counter += 1
        if frame_counter % 10 == 0:
            print(f"Processed {frame_counter}/{max_frames} frames")

    # (6) 释放所有资源 [cite: 1152-1154, 1164]
    cap.release()
    background_cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done.")

# --- 步骤 4: 执行程序 ---
# background_replacement('foreground.mp4', 'background.mp4', 'haarcascade_frontalface_default.xml')