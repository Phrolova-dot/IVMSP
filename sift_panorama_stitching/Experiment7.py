import cv2
import select
import sys
import time
from typing import List, Tuple, Optional
import numpy as np
import threading
import psutil
import math
from gpiozero import Button
from picamera2 import Picamera2 


DEBUG = True 


# Exercise 7.2


INFO = {
    "start": "Starting panorama program.",
    "init": "Initializing camera and buttons...",
    "init_ok": "Initialization succeeded.",
    "init_fail": "A critical error occurred. Restarting session...",
    "settings_ok": "Settings confirmed. Starting capture session.",
    "cam_error": "A camera error occurred. Restarting session...",
    "exit_ok": "Camera closed successfully.",
    "take_img": "Press the GREEN button to take a picture. Press RED or 'q' to exit.",
    "res_fail": "Invalid resolution format. Please use 'width, height' (e.g., 1920, 1080).",
    "frame_count_fail": "Invalid number. Defaulting to 4 frames.",
}

USER_INPUT = {
    "res": "Enter resolution as 'width, height': ",
    "frame_count": "How many frames to capture (max 6)? ",
}


def get_user_settings() -> Tuple[Tuple[int, int], int]:
    while True:
        try:
            res_input = input(USER_INPUT["res"]) 
            width, height = res_input.split(",")
            resolution = (int(width), int(height))
            break
        except ValueError:
            print(INFO["res_fail"])

    try:
        frame_number = int(input(USER_INPUT["frame_count"])) 
        if not 0 < frame_number <= 6:
            raise ValueError
    except (ValueError, TypeError):
        print(INFO["frame_count_fail"]) 
        frame_number = 4
        
    print(f"Setting frames to {frame_number}")     
    return resolution, frame_number

def capture_sequence(
    resolution: Tuple[int, int], frame_count: int
) -> Optional[List[np.ndarray]]:
    
    print(INFO["init"]) 
    frames = []
    
    with Picamera2() as picam2:
        try:
            config = picam2.create_preview_configuration(
                main={"format": "XRGB8888", "size": resolution}
            )
            picam2.configure(config)
            
            green_button = Button(5)
            red_button = Button(6)
            
            picam2.start()
            print(INFO["init_ok"]) 
            print(INFO["take_img"]) 
            
            cv2.namedWindow("panorama", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "panorama", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            time.sleep(1)


        except Exception as e:
            print(f"\nERROR: {e}")
            print(INFO["init_fail"]) 
            return None
        
        is_pressed = False 
        frames_remaining = frame_count
        
        
        try:
            print(f"Frames to capture: {frames_remaining}") 
            while frames_remaining > 0:
                bgr_frame = picam2.capture_array()
                cv2.imshow("panorama", bgr_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q") or red_button.is_pressed:
                    print("Capture cancelled by user.")
                    return []


                if select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.readline().strip().lower() == "q":
                        print("Capture cancelled by user.")
                        return []
    
                if green_button.is_pressed: 
                    if not is_pressed:
                        frames.append(bgr_frame[..., :3]) 
                        frames_remaining -= 1
                        print(f"Picture taken! {frames_remaining} remaining.") 
                        is_pressed = True
                
                elif not green_button.is_pressed: 
                    is_pressed = False 
            
            return frames

        except Exception as e:
            print(f"\nERROR: {e}")
            print(INFO["cam_error"]) 
            return None
        finally:
            cv2.destroyAllWindows()
            print(INFO["exit_ok"]) 

def interface():
    print(INFO["start"]) 
    resolution, frame_number = get_user_settings()
    while True:
        captured_frames = capture_sequence(resolution, frame_number)
        
        if captured_frames is None:
            time.sleep(2)
            continue
            
        if captured_frames:
            print(f"\nSuccessfully captured {len(captured_frames)} frames. Exiting.")
            return captured_frames
        else:
            print("\nCapture session cancelled by user. Exiting.")
            break

def draw_matches(
    img1: np.ndarray,
    kp1: Tuple[cv2.KeyPoint, ...],
    img2: np.ndarray,
    kp2: Tuple[cv2.KeyPoint, ...],
    matches: List[cv2.DMatch],
) -> np.ndarray:
    assert (
        all(isinstance(arr, np.ndarray) for arr in (img1, img2))
        and all(
            isinstance(t, tuple)
            and all(isinstance(element, cv2.KeyPoint) for element in t)
            for t in (kp1, kp2)
        )
        and isinstance(matches, list)
        and all(isinstance(element, cv2.DMatch) for element in matches)
    )

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    

    matches_img = np.zeros((max([h1, h2]), w1 + w2, 3), dtype="uint8")

    matches_img[:h1, :w1] = np.dstack([img1,img1,img1]) 
    matches_img[:h2, w1:] = np.dstack([img2,img2,img2])

    for i in matches:
        img1_idx = i.queryIdx
        img2_idx = i.trainIdx
        
        (m1, n1) = kp1[img1_idx].pt
        (m2, n2) = kp2[img2_idx].pt
        
        cv2.circle(matches_img, (int(m1), int(n1)), 5, (0, 255, 0), 1)
        cv2.circle(matches_img, (int(m2) + w1, int(n2)), 5, (0, 255, 0), 1)
        
        cv2.line(
            matches_img, 
            (int(m1), int(n1)), 
            (int(m2) + w1, int(n2)), 
            (0, 255, 0), 1
        )
        
    return matches_img



# Exercise 7.6
def stitch_2_images(img_a: np.ndarray, img_b: np.ndarray, h: np.ndarray) -> np.ndarray:
    assert all(isinstance(arr, np.ndarray) for arr in (img_a, img_b, h))
    
    h_a, w_a = img_a.shape[:2] 
    h_b, w_b = img_b.shape[:2]

    pts_a = np.float32([[0, 0], [0, h_a], [w_a, h_a], [w_a, 0]]).reshape(-1, 1, 2)
    pts_b = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
    
    pts_b_pt = cv2.perspectiveTransform(pts_b, h)
    
    pts = np.concatenate((pts_a, pts_b_pt), axis=0)

    [m_min, n_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [m_max, n_max] = np.int32(pts.max(axis=0).ravel() + 0.5)

    ht = np.array([[1, 0, -m_min], [0, 1, -n_min], [0, 0, 1]])

    result = cv2.warpPerspective(img_b, ht.dot(h), (m_max - m_min, n_max - n_min))
    
    result[-n_min : -n_min + h_a, -m_min : -m_min + w_a] = img_a
    
    return result

def create_panorama(
    img1: np.ndarray, 
    img2: np.ndarray, 
    features: int = 400, 
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    assert (
        all(isinstance(arr, np.ndarray) for arr in (img1, img2))
        and isinstance(features, int)
        and (mask is None or isinstance(mask, np.ndarray))
    )

    sift = cv2.SIFT_create(features)
    kp1, des1 = sift.detectAndCompute(img1, mask)
    kp2, des2 = sift.detectAndCompute(img2, mask)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for match1, match2 in matches:
        if match1.distance < 0.8 * match2.distance: 
            good_matches.append(match1)

    feat_img1 = cv2.drawKeypoints(img1, kp1, None, (0, 0, 255), 4)
    feat_img2 = cv2.drawKeypoints(img2, kp2, None, (0, 0, 255), 4)

    if len(good_matches) >= 4:

        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) 

        h, mask_homography = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
        
        result12 = stitch_2_images(img1, img2, h)
        
        result_matches = draw_matches(img1, kp1, img2, kp2, good_matches) 
    else:
        print("Not enough matches to stitch.")
        result12 = img1
        result_matches = img1

    return feat_img1, feat_img2, result_matches, result12

img1 = cv2.imread("imgNBG1.png", 0)
img2 = cv2.imread("imgNBG2.png", 0)


feat_img1, feat_img2, result_matches, result12 = create_panorama(img1, img2)
cv2.imwrite("featImgA.png", feat_img1)
cv2.imwrite("featImgB.png", feat_img2)
cv2.imwrite("matches.png", result_matches)
cv2.imwrite("ImgAB.png", result12)

# Exercise 7.7
def show_free_memory(stop: threading.Event):
    while not stop.is_set():
        mem = psutil.virtual_memory()
        print(f"free memory: {mem.available / (1024**3):.2f} GB") 
        time.sleep(0.5)



if DEBUG:
    try:
        stop_event = threading.Event()
        thread_id = threading.Thread(target=show_free_memory, args=(stop_event,))
        print("DEBUG MODE ON")
    except:
        sys.exit(0)
        print("thread creation failed!")
            
imgs = interface()
assert len(imgs) > 1

_, _, _, result = create_panorama(imgs[0], imgs[1])
for next_img in imgs[2:]:
    _, _, _, result = create_panorama(result, next_img)
cv2.imwrite("panorama.png", result)

if DEBUG:
    stop_event.set()
    thread_id.join()