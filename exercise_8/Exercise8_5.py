import cv2
from gpiozero import Button
from picamera2 import Picamera2
import select
import sys
import time
import numpy as np

red_button = Button(6)
pencasc_vert = cv2.CascadeClassifier("pen_vertical_classifier.xml")


def hough_transform(x: int, y: int, w: int, h: int, image: np.ndarray):
    height, width = image.shape[:2]

    pad_x = 20 
    pad_y = 10
    
    x_start = max(0, x - pad_x)
    y_start = max(0, y - pad_y)
    x_end = min(width, x + w + pad_x)
    y_end = min(height, y + h + pad_y)
    
    cropped_image = image[y_start:y_end, x_start:x_end]
    
    gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)


    edge_image = cv2.Canny(gray_cropped, 50, 150, apertureSize=3)


    lines = cv2.HoughLines(edge_image, 1, np.pi / 180, 100)

    if lines is not None:
        print("Line found!")
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            x1_c = int(x0 + 1000 * (-b))
            y1_c = int(y0 + 1000 * (a))
            x2_c = int(x0 - 1000 * (-b))
            y2_c = int(y0 - 1000 * (a))
            
   
            pt1 = (x1_c + x_start, y1_c + y_start)
            pt2 = (x2_c + x_start, y2_c + y_start)

            cv2.line(image, pt1, pt2, (0, 0, 255), 2)

            angle_deg = theta * 180 / np.pi

            if 0 <= angle_deg <= 180:
                 cv2.putText(image, f"{angle_deg:.2f}", (10, 50), 
                             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


with Picamera2() as picam2:
    config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    time.sleep(1)

    while True:
        bgr_frame = picam2.capture_array()
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        pens_vert = pencasc_vert.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=25,
            minSize=(25, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for x, y, w, h in pens_vert:
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            hough_transform(x, y, w, h, bgr_frame)

        cv2.imshow("window", bgr_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q") or red_button.is_pressed:
            break
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() == "q":
                break
        time.sleep(1/10)

    picam2.stop()
    cv2.destroyAllWindows()