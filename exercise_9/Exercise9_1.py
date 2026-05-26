# haar_detector.py
import cv2
import sys

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 加载人脸分类器 (确保该xml文件在同目录下，或者使用OpenCV自带的路径)
# Haar Cascades 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error loading cascade file")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Haar 检测需要灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    # scaleFactor 和 minNeighbors 需要微调
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 绘制矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#rpicam-vid -t 10000 --width 640 --height 480 --framerate 10 --codec libav --libav-format mp4 -o foreground.mp4