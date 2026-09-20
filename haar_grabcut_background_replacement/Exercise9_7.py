import cv2
import numpy as np
import time

# --- 步骤 1: 放入之前写的辅助函数 ---

def grab_cut(frame, x, y, w, h, return_mask=False):
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

    return (frame, mask2) if return_mask else frame

def add_alpha_channel(img, mask):
    """Build alpha from a segmentation mask, preserving black foreground pixels."""
    if mask.shape != img.shape[:2]:
        raise ValueError("Mask and image dimensions must match")
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = (mask != 0).astype(np.uint8) * 255
    return result


def paste_image(ag_img, b_img, x, y, w, h):
    """Composite matching full-frame foreground and background arrays."""
    if ag_img.shape[:2] != b_img.shape[:2]:
        raise ValueError("Foreground and background dimensions must match")
    height, width = b_img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return b_img
    fg_crop = ag_img[y0:y1, x0:x1]
    roi = b_img[y0:y1, x0:x1]
    alpha = fg_crop[:, :, 3:4].astype(np.float64) / 255.0
    b_img[y0:y1, x0:x1] = np.rint(alpha * fg_crop[:, :, :3] + (1-alpha) * roi).astype(np.uint8)
    return b_img


def background_replacement(input_video_path, background_video_path, weightsfile,
                           output_path="output_final.mp4"):
    classifier = cv2.CascadeClassifier(str(weightsfile))
    if classifier.empty():
        raise ValueError(f"Cannot load classifier: {weightsfile}")
    cap = cv2.VideoCapture(str(input_video_path))
    background_cap = cv2.VideoCapture(str(background_video_path))
    out = None
    try:
        if not cap.isOpened() or not background_cap.isOpened():
            raise OSError("Cannot open foreground or background video")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid input video dimensions or frame rate")
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (width, height))
        if not out.isOpened():
            raise OSError(f"Cannot open output video: {output_path}")
        # Pair frames by index and stop at the shorter input.
        while True:
            ok, frame = cap.read()
            bg_ok, background = background_cap.read()
            if not ok or not bg_ok:
                break
            background = cv2.resize(background, (width, height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, y, w, h in classifier.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
                cutout, mask = grab_cut(frame, x, y, w, h, return_mask=True)
                background = paste_image(add_alpha_channel(cutout, mask),
                                         background, x, y, w, h)
            out.write(background)
    finally:
        cap.release()
        background_cap.release()
        if out is not None:
            out.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replace a video background with GrabCut")
    parser.add_argument("foreground")
    parser.add_argument("background")
    parser.add_argument("classifier")
    parser.add_argument("--output", default="output_final.mp4")
    args = parser.parse_args()
    background_replacement(args.foreground, args.background, args.classifier, args.output)
