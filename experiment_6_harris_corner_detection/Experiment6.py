import numpy as np
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from typing import Union, Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import gridspec


def to_rgb(img_gray):

    img_u8 = np.clip(img_gray, 0, 255).astype(np.uint8)
    return np.stack((img_u8,)*3, axis=-1)

def create_test_img() -> np.ndarray:

    img = np.zeros((800, 800), dtype=np.float64)
    img.fill(255)
    img[0:400, 0:400] = 0
    return img

def add_noise(img: np.ndarray, dev: Union[int, float]) -> np.ndarray:

    np.random.seed(42)
    n_img = img + np.random.normal(0, dev, img.shape)
    return n_img

def compute_derivatives(
    img: np.ndarray, sigma: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
    
    fm = gaussian_filter(img, sigma=sigma, order=(0, 1))
    fn = gaussian_filter(img, sigma=sigma, order=(1, 0))
    return fm, fn

def window_img(
    img_window: Tuple[int, int, int, int],
    img: np.ndarray,
    fm: np.ndarray,
    fn: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    frame_size = 20
    red = np.array([255, 0, 0])
    m0, n0, m1, n1 = img_window[0], img_window[1], img_window[2], img_window[3]
    
    img_with_frame = to_rgb(img)
    
    img_with_frame[m0:m0+frame_size, n0:n1] = red
    img_with_frame[m1-frame_size:m1, n0:n1] = red
    img_with_frame[m0:m1, n0:n0+frame_size] = red
    img_with_frame[m0:m1, n1-frame_size:n1] = red

    w_fm = fm[m0:m1, n0:n1]
    w_fn = fn[m0:m1, n0:n1]
    
    return w_fm, w_fn, img_with_frame

#key code starts here
img = create_test_img()
n_img = add_noise(img, 30)

fm, fn = compute_derivatives(img, 3)
fm_n, fn_n = compute_derivatives(n_img, 3)

flat = (600, 200, 700, 300)
edge = (350, 100, 450, 200)
corner = (350, 350, 450, 450)


w_fm1, w_fn1, img1 = window_img(flat, img, fm, fn)
w_fm3, w_fn3, img3 = window_img(edge, img, fm, fn)
w_fm5, w_fn5, img5 = window_img(corner, img, fm, fn)


w_fm2, w_fn2, img2 = window_img(flat, n_img, fm_n, fn_n)
w_fm4, w_fn4, img4 = window_img(edge, n_img, fm_n, fn_n)
w_fm6, w_fn6, img6 = window_img(corner, n_img, fm_n, fn_n)


plt.figure() 
plt.subplot(4, 3, 1), plt.title("noiseless"), plt.axis("off")
plt.imshow(img, cmap='gray')


plt.subplot(4, 3, 4), plt.title("flat"), plt.axis("off")
plt.imshow(img1) 
plt.subplot(4, 3, 5), plt.title("Fm"), plt.axis("off")
plt.imshow(w_fm1, cmap='gray')
plt.subplot(4, 3, 6), plt.title("Fn"), plt.axis("off")
plt.imshow(w_fn1, cmap='gray')


plt.subplot(4, 3, 7), plt.title("edge"), plt.axis("off")
plt.imshow(img3) 
plt.subplot(4, 3, 8), plt.axis("off")
plt.imshow(w_fm3, cmap='gray')
plt.subplot(4, 3, 9), plt.axis("off")
plt.imshow(w_fn3, cmap='gray')


plt.subplot(4, 3, 10), plt.title("corner"), plt.axis("off")
plt.imshow(img5) 
plt.subplot(4, 3, 11), plt.axis("off")
plt.imshow(w_fm5, cmap='gray')
plt.subplot(4, 3, 12), plt.axis("off")
plt.imshow(w_fn5, cmap='gray')

plt.tight_layout()
plt.savefig("noiseless_harris.png")

plt.figure(figsize=(10, 12))
plt.subplot(4, 3, 1), plt.title("noisy"), plt.axis("off")
plt.imshow(n_img, cmap='gray')

plt.subplot(4, 3, 4), plt.title("noisy flat"), plt.axis("off")
plt.imshow(img2) 
plt.subplot(4, 3, 5), plt.title("Fm"), plt.axis("off")
plt.imshow(w_fm2, cmap='gray')
plt.subplot(4, 3, 6), plt.title("Fn"), plt.axis("off")
plt.imshow(w_fn2, cmap='gray')

plt.subplot(4, 3, 7), plt.title("noisy edge"), plt.axis("off")
plt.imshow(img4) 
plt.subplot(4, 3, 8), plt.axis("off")
plt.imshow(w_fm4, cmap='gray')
plt.subplot(4, 3, 9), plt.axis("off")
plt.imshow(w_fn4, cmap='gray')

plt.subplot(4, 3, 10), plt.title("noisy corner"), plt.axis("off")
plt.imshow(img6) 
plt.subplot(4, 3, 11), plt.axis("off")
plt.imshow(w_fm6, cmap='gray')
plt.subplot(4, 3, 12), plt.axis("off")
plt.imshow(w_fn6, cmap='gray')

plt.tight_layout()
plt.savefig("noisy_harris.png")

#Exercise 6.2
plt.figure()
plt.subplot(3, 2, 1), plt.title("flat")
plt.ylabel("Fn")
plt.axis([-10, 10, -10, 10])
plt.grid(True)
plt.plot(w_fm1[:], w_fn1[:], "rx") 

plt.subplot(3, 2, 2), plt.title("noisy flat")
plt.axis([-10, 10, -10, 10])
plt.grid(True)
plt.plot(w_fm2.flatten(), w_fn2.flatten(), "rx") 


plt.subplot(3, 2, 3), plt.title("edge")
plt.axis([-40, 40, -40, 40]) 
plt.grid(True)
plt.ylabel("Fn")
plt.plot(w_fm3.flatten(), w_fn3.flatten(), "rx")


plt.subplot(3, 2, 4), plt.title("noisy edge")
plt.axis([-40, 40, -40, 40])
plt.grid(True)
plt.plot(w_fm4.flatten(), w_fn4.flatten(), "rx")


plt.subplot(3, 2, 5), plt.title("corner")
plt.axis([-40, 40, -40, 40])
plt.grid(True)
plt.xlabel("Fm"), plt.ylabel("Fn")
plt.plot(w_fm5.flatten(), w_fn5.flatten(), "rx")


plt.subplot(3, 2, 6), plt.title("noisy corner")
plt.axis([-40, 40, -40, 40])
plt.grid(True)
plt.xlabel("Fm")
plt.plot(w_fm6.flatten(), w_fn6.flatten(), "rx")

plt.tight_layout()
plt.savefig("eigen.png")

#Exercise 6.3
def compute_harris_corners(
    fm: np.ndarray,
    fn: np.ndarray,
    k: Union[int, float] = 0.04,
    sigma: Union[int, float] = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    assert all(isinstance(arr, np.ndarray) for arr in (fm, fn)) and all(
        isinstance(num, (int, float)) for num in (k, sigma)
    )
    
    mmm = gaussian_filter(fm**2, sigma)
    mmn = gaussian_filter(fm * fn, sigma)
    mnm = mmn
    mnn = gaussian_filter(fn**2, sigma)
    det_m = mmm * mnn - mmn**2
    tra_m = mmm + mnn 
    r = det_m - k * (tra_m**2)
    
    return mmm, mmn, mnm, mnn, det_m, tra_m, r

_, _, _, _, _, _, r   = compute_harris_corners(fm,   fn)
_, _, _, _, _, _, r_n = compute_harris_corners(fm_n, fn_n)

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection="3d")

x = np.arange(r.shape[1])
y = np.arange(r.shape[0])
X, Y = np.meshgrid(x, y)
Z = r

ax1.plot_surface(X, Y, Z, rstride=20, cstride=20, alpha=0.3, cmap="viridis")
ax1.contour(X, Y, Z, zdir="z", offset=np.min(Z), cmap="viridis")
ax1.contour(X, Y, Z, zdir="x", offset=np.min(X), cmap="viridis")
ax1.contour(X, Y, Z, zdir="y", offset=np.max(Y), cmap="viridis")

ax1.set_xlabel("m")
ax1.set_ylabel("n")
ax1.set_zlabel("R(m,n)")
ax1.set_title("Plot for R")

plt.tight_layout()
plt.savefig("noiseless_harris_response.png")

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(projection="3d")

Z_n = r_n

ax2.plot_surface(X, Y, Z_n, rstride=20, cstride=20, alpha=0.3, cmap="viridis")
ax2.contour(X, Y, Z_n, zdir="z", offset=np.min(Z_n), cmap="viridis")
ax2.contour(X, Y, Z_n, zdir="x", offset=np.min(X), cmap="viridis")
ax2.contour(X, Y, Z_n, zdir="y", offset=np.max(Y), cmap="viridis")

ax2.set_xlabel("m")
ax2.set_ylabel("n")
ax2.set_zlabel("R(m,n)")
ax2.set_title("Plot for R (Noisy)")

plt.tight_layout()
plt.savefig("noisy_harris_response.png")

def detect_corners(r: np.ndarray, threshold: Union[int, float] = 0.5) -> np.ndarray:
    assert isinstance(r, np.ndarray) and isinstance(threshold, (int, float))

    corners_candidates = r > threshold

    corner_pos = np.argwhere(corners_candidates)

    return corner_pos

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, height_ratios=[8, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

#key changes here
corners_candidates    = detect_corners(r,   threshold=20.0)
corners_candidates_n  = detect_corners(r_n, threshold=20.0)

ax1.set_title("corners in image")
ax1.imshow(img, cmap="gray")
if corners_candidates.size > 0:
    ys, xs = corners_candidates[:, 0], corners_candidates[:, 1]
    ax1.plot(xs, ys, "r+", markersize=5)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

ax2.set_title("corners in noisy image")
ax2.imshow(n_img, cmap="gray")
if corners_candidates_n.size > 0:
    ys_n, xs_n = corners_candidates_n[:, 0], corners_candidates_n[:, 1]
    ax2.plot(xs_n, ys_n, "r+", markersize=5)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

ax3.text(0.2, 0.8, f"{len(corners_candidates)} corners found", fontsize=14)
ax3.axis("off")
ax4.text(0.2, 0.8, f"{len(corners_candidates_n)} corners found", fontsize=14)
ax4.axis("off")
plt.tight_layout()
plt.savefig("corner_detection.png")

# Exercise 6.4
def rotate_img(
    img: np.ndarray, deg: Union[int, float] = 45, scale: Union[int, float] = 1.0
) -> np.ndarray:
    assert isinstance(img, np.ndarray) and all(
        isinstance(num, (int, float)) for num in (deg, scale)
    )
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    return cv2.warpAffine(img, m, (w, h))

img = create_test_img()
img_rot = rotate_img(img, 45)

fm_rot, fn_rot = compute_derivatives(img_rot, 3)
_, _, _, _, _, _, r_rot = compute_harris_corners(fm_rot, fn_rot)

corners = detect_corners(r_rot, threshold=r_rot.max() * 0.01)

plt.figure()
plt.title("Rotated Image Corners")
plt.imshow(img_rot, cmap='gray')
plt.plot(corners[:, 1], corners[:, 0], 'r.')
plt.axis("off")
plt.savefig("rotated.png")

# Exercise 6.5
def import_and_convert_img(img_path: str, img_size: int = 800) -> np.ndarray:
    assert isinstance(img_path, str) and isinstance(img_size, int)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    
    h, w = img.shape[:2]
    
    if h != img_size:

        scale_factor = img_size / h
        
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)
        
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


image_path = "chessboard.png"  


img_real = import_and_convert_img(image_path, img_size=800)

fm_real, fn_real = compute_derivatives(img_real, sigma=3)
_, _, _, _, det_m, tra_m, r_real = compute_harris_corners(fm_real, fn_real, k=0.04, sigma=3)
    

thresh = r_real.max() * 0.01
corners_real = detect_corners(r_real, threshold=thresh)
    

plt.figure()
    
plt.subplot(3, 2, 1)
plt.title("Image")
plt.imshow(img_real, cmap="gray")
plt.axis("off")
    
plt.subplot(3, 2, 2)
plt.title("Corner Response R")
plt.imshow(r_real, cmap="gray")
plt.axis("off")
    
plt.subplot(3, 2, 3)
plt.title("det(M)")
plt.imshow(det_m, cmap="gray")
plt.axis("off")
    
plt.subplot(3, 2, 4)
plt.title("trace(M)")
plt.imshow(tra_m, cmap="gray")
plt.axis("off")
    
plt.subplot(3, 2, 5)
plt.title(f"Corners in Image")
plt.imshow(img_real, cmap="gray")
plt.plot(corners_real[:, 1], corners_real[:, 0], "r.", markersize=8)
plt.axis("off")

plt.subplot(3, 2, 6)
plt.text(0.5, 0.5, f"({len(corners_real)} found)", fontsize=14, ha='center', va='center')
plt.axis("off")
plt.tight_layout()
plt.savefig("chessboard_pa.png")