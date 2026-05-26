import cv2
from gpiozero import Button
from picamera2 import Picamera2
import select
import sys
import time
import itertools

red_button = Button(6)

pencasc_vert = cv2.CascadeClassifier("pen_vertical_classifier.xml")
pencasc_hor = cv2.CascadeClassifier("pen_horizontal_classifier.xml")


with Picamera2() as picam2:
    config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
    
    if video_capture is None:
        picam2.configure(config)
        picam2.start()

    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    time.sleep(1)

    while True:
        if video_capture is None:
            bgr_frame = picam2.capture_array()
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        else:
            ret, gray = video_capture.read()
            
            if not ret:
                break

            bgr_frame = gray 
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        pens_vert = pencasc_vert.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=25,
            minSize=(25, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        pens_hor = pencasc_hor.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=30,
            minSize=(80, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for x, y, w, h in itertools.chain(pens_vert, pens_hor):
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#Exercise8_4
        # if len(pens_vert) > 0 or len(pens_hor) > 0:
            

        #     if frame_count % 10 == 0: 
        #         filename = f"output/detected_{frame_count}.jpg"
        #         cv2.imwrite(filename, bgr_frame)

        # frame_count += 1

        cv2.imshow("window", bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or red_button.is_pressed:
            break
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() == "q":
                break

    if video_capture is None:
        picam2.stop()
    else:
        video_capture.release()
    cv2.destroyAllWindows()