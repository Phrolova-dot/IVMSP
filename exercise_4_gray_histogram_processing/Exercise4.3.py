import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

# --- Task 1: Load image as uint8 RGB array ---
def load_image_to_array(file_name: str) -> np.ndarray:
    try:
        img = Image.open(file_name).convert("RGB")
        return np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    except (FileNotFoundError, OSError) as e:
        print(f"Load failed: {file_name} ({e})")
        return np.empty((0, 0, 3), dtype=np.uint8)

# --- Task 2: Save grayscale array as image ---
def save_array_to_gray_img(np_array_img: np.ndarray, file_name: str):
    assert isinstance(np_array_img, np.ndarray), "Input is not an ndarray"
    assert np_array_img.dtype == np.uint8, "Array dtype must be uint8"
    assert np_array_img.ndim == 2, "Must be a 2D grayscale array"
    Image.fromarray(np_array_img, mode="L").save(file_name)

# --- Task 3: Grayscale conversion ---
def _check_rgb(img_rgb: np.ndarray):
    assert isinstance(img_rgb, np.ndarray), "Input must be an ndarray"
    assert img_rgb.dtype == np.uint8, "dtype must be uint8"
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, "Shape must be (H,W,3)"

def convert_rgb_equal(img_rgb: np.ndarray) -> np.ndarray:
    _check_rgb(img_rgb)
    # (R+G+B)/3 按题意，使用 NumPy 平均并四舍五入
    return np.rint(img_rgb.mean(axis=2)).astype(np.uint8)

def convert_rgb_weighted_1(img_rgb: np.ndarray) -> np.ndarray:
    _check_rgb(img_rgb)
    r = img_rgb[..., 0].astype(np.float32)
    g = img_rgb[..., 1].astype(np.float32)
    b = img_rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.rint(y).astype(np.uint8)

def convert_rgb_weighted_2(img_rgb: np.ndarray) -> np.ndarray:
    _check_rgb(img_rgb)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # 按题意：使用 dot() 完成分量点积，输出 (H,W)
    y = np.dot(img_rgb.astype(np.float32), weights)
    return np.rint(y).astype(np.uint8)

# 可选：纯整数加权（更快，非题目强制）
def convert_rgb_weighted_int(img_rgb: np.ndarray) -> np.ndarray:
    _check_rgb(img_rgb)
    img16 = img_rgb.astype(np.uint16)
    y = (77 * img16[..., 0] + 150 * img16[..., 1] + 29 * img16[..., 2] + 128) >> 8
    return y.astype(np.uint8)

# --- Task 4: Plot histogram (custom + matplotlib) ---
def plot_histogram(img_gray: np.ndarray, plot_path: str):
    assert isinstance(img_gray, np.ndarray), "Input is not an ndarray"
    if img_gray.ndim == 3 and img_gray.shape[2] == 3:
        img_gray = convert_rgb_weighted_1(img_gray)
    assert img_gray.ndim == 2 and img_gray.dtype == np.uint8, "Must be uint8 grayscale"

    flat = img_gray.flatten()  # 降维至 1D
    hist_custom = np.bincount(flat, minlength=256)

    # 对照：np.histogram
    hist_np, _ = np.histogram(flat, bins=256, range=(0, 256))
    assert np.array_equal(hist_custom, hist_np), "Histogram mismatch between bincount and numpy.histogram"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(np.arange(256), hist_custom, width=1.0, color="steelblue")
    axes[0].set_title("Custom histogram")
    axes[0].set_xlim(0, 255)

    axes[1].hist(flat, bins=np.arange(257), color="gray")
    axes[1].set_title("matplotlib histogram")
    axes[1].set_xlim(0, 255)

    for ax in axes:
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")

    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    return hist_custom

# --- Runtime comparison ---
# 参考 4.2：PIL 循环实现（用于对比，不用于高效处理）
def convert_pil_equal_loop(img_rgb: np.ndarray) -> np.ndarray:
    _check_rgb(img_rgb)
    pil = Image.fromarray(img_rgb, mode="RGB")
    w, h = pil.size
    src = pil.load()
    out = Image.new("L", (w, h))
    dst = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            dst[x, y] = (r + g + b) // 3
    return np.array(out, dtype=np.uint8)


def convert_pil_weighted_loop(img_rgb: np.ndarray) -> np.ndarray:
    _check_rgb(img_rgb)
    pil = Image.fromarray(img_rgb, mode="RGB")
    w, h = pil.size
    src = pil.load()
    out = Image.new("L", (w, h))
    dst = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            dst[x, y] = int(0.299 * r + 0.587 * g + 0.114 * b + 0.5)
    return np.array(out, dtype=np.uint8)


def convert_loop_weighted(img_rgb: np.ndarray) -> np.ndarray:
    _check_rgb(img_rgb)
    h, w, _ = img_rgb.shape
    out = np.empty((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = img_rgb[y, x]
            out[y, x] = int(0.299 * r + 0.587 * g + 0.114 * b + 0.5)
    return out


def runtime_compare(img_rgb: np.ndarray, repeat: int = 3):
    funcs = [
        ("pil_equal_loop", convert_pil_equal_loop),
        ("pil_weighted_loop", convert_pil_weighted_loop),
        ("np_loop_weighted", convert_loop_weighted),
        ("np_equal", convert_rgb_equal),
        ("np_weighted_1", convert_rgb_weighted_1),
        ("np_weighted_2_dot", convert_rgb_weighted_2),
        ("np_weighted_int", convert_rgb_weighted_int),
    ]
    # 预热
    for _, f in funcs:
        _ = f(img_rgb)
    # 计时（使用 time.time()）
    times = []
    for name, f in funcs:
        t0 = time.time()
        for _ in range(repeat):
            _ = f(img_rgb)
        dt = (time.time() - t0) / repeat
        times.append((name, dt))
    print("Runtime (s/run, ascending):")
    for name, dt in sorted(times, key=lambda x: x[1]):
        print(f"  {name:18s} {dt:.6f}")

# --- Example main ---
if __name__ == "__main__":
    imgs = ["img_0.png", "img_50.png", "img_100.png", "mandrill.png"]
    out_dir = "out_np"
    os.makedirs(out_dir, exist_ok=True)

    for p in imgs:
        arr = load_image_to_array(p)
        if arr.size == 0:
            continue
        g_eq = convert_rgb_equal(arr)
        g_w1 = convert_rgb_weighted_1(arr)
        g_w2 = convert_rgb_weighted_2(arr)
        g_wi = convert_rgb_weighted_int(arr)

        base = os.path.splitext(os.path.basename(p))[0]
        save_array_to_gray_img(g_eq, os.path.join(out_dir, f"{base}_gray_equal.png"))
        save_array_to_gray_img(g_w1, os.path.join(out_dir, f"{base}_gray_w1.png"))
        save_array_to_gray_img(g_w2, os.path.join(out_dir, f"{base}_gray_w2.png"))
        save_array_to_gray_img(g_wi, os.path.join(out_dir, f"{base}_gray_w_int.png"))
        plot_histogram(g_w1, os.path.join(out_dir, f"{base}_hist.png"))

        print(f"Done: {p}")
        runtime_compare(arr)