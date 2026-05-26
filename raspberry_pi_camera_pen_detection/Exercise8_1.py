import cv2
from gpiozero import Button
import select
import sys


video_capture = cv2.VideoCapture("out.avi") 
red_button = Button(6)

while True:
    ret, gray = video_capture.read()

    if not ret:
        break

    cv2.imshow("Video", gray)

    if cv2.waitKey(1) & 0xFF == ord("q") or red_button.is_pressed:
        break

    if select.select([sys.stdin], [], [], 0)[0]:
        if sys.stdin.readline().strip().lower() == "q":
            break

video_capture.release()
cv2.destroyAllWindows()