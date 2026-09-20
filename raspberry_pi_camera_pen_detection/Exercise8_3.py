import argparse
from contextlib import ExitStack
from pathlib import Path
import cv2
import itertools


def main(video=None):
    classifiers = []
    for filename in ("pen_vertical_classifier.xml", "pen_horizontal_classifier.xml"):
        path = Path(__file__).with_name(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Missing pen classifier: {path}")
        classifier = cv2.CascadeClassifier(str(path))
        if classifier.empty():
            raise ValueError(f"Invalid classifier: {path}")
        classifiers.append(classifier)

    try:
        with ExitStack() as stack:
            red_button = None
            if video is None:
                from gpiozero import Button
                from picamera2 import Picamera2
                red_button = stack.enter_context(Button(6))
                camera = stack.enter_context(Picamera2())
                camera.configure(camera.create_preview_configuration(
                    main={"format": "XRGB8888", "size": (640, 480)}))
                camera.start()
            else:
                capture = cv2.VideoCapture(str(video))
                stack.callback(capture.release)
                if not capture.isOpened():
                    raise OSError(f"Cannot open video: {video}")

            while True:
                if video is None:
                    frame = camera.capture_array()[:, :, :3].copy()
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vertical = classifiers[0].detectMultiScale(
                    gray, scaleFactor=1.7, minNeighbors=25, minSize=(25, 80))
                horizontal = classifiers[1].detectMultiScale(
                    gray, scaleFactor=1.8, minNeighbors=30, minSize=(80, 35))
                for x, y, w, h in itertools.chain(vertical, horizontal):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow("Pen detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q") or (red_button and red_button.is_pressed):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect pens from a Pi camera or video")
    parser.add_argument("--video", type=Path, help="Use a video file instead of Pi hardware")
    main(parser.parse_args().video)
