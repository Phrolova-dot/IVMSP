from PIL import Image
from typing import List

def convert_to_gray_equal(rgb_img: Image.Image) -> Image.Image:
    img = rgb_img.convert("RGB")
    w, h = img.size
    src = img.load()
    out = Image.new("L", (w, h))
    dst = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            dst[x, y] = (r + g + b) // 3
    return out

def convert_to_gray_weighted(rgb_img: Image.Image) -> Image.Image:
    img = rgb_img.convert("RGB")
    w, h = img.size
    src = img.load()
    out = Image.new("L", (w, h))
    dst = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            # 四舍五入
            dst[x, y] = round(0.299 * r + 0.587 * g + 0.114 * b)
    return out

def histogram(img_gray: Image.Image) -> List[int]:
    if img_gray.mode != "L":
        raise ValueError("需要灰度图 (mode 'L')")
    hist = [0] * 256
    for v in img_gray.getdata():
        hist[v] += 1
    return hist

if __name__ == "__main__":
    try:
        mandrill = Image.open("mandrill.png")
        g_eq = convert_to_gray_equal(mandrill)
        g_wt = convert_to_gray_weighted(mandrill)
        g_eq.save("mandrill_gray_equal.png")
        g_wt.save("mandrill_gray_weighted.png")
        h_wt = histogram(g_wt)
        print("完成。像素总数:", sum(h_wt))
    except FileNotFoundError:
        print("未找到 mandrill.png")