
import cv2
from gpiozero import Button
from picamera2 import Picamera2
import select
import sys
import time

red_button = Button(6)
num_frames = 200
cnt = 0

with Picamera2() as picam2: 

    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)} 
    )
    picam2.configure(config)
    picam2.start()
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    time.sleep(1)
    start = time.time()
    
    while True:
        cnt += 1
        if cnt == num_frames:

            end = time.time()
            seconds = end- start
            print(f"Time taken = {seconds} seconds.")
            fps = num_frames / seconds
            print(f"Estimated FPS = {fps} fps.")
            cnt = 0
            start = time.time()
        bgr_frame = picam2.capture_array()
        cv2.imshow("window", bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or red_button.is_pressed:
            break

        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() == "q":
                break
            
        time.sleep(1 / 10)
    picam2.stop()
    
cv2.destroyAllWindows()