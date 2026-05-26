# Exercise 7.4
import cv2

FEATURES = 400
MASK = None

img = cv2.imread("mandrill.png", 0)

sift = cv2.SIFT_create(FEATURES)

kp, des = sift.detectAndCompute(img, MASK)

print(f"Number of features found: {len(kp)}")
if des is not None:
    print(f"Descriptor size per feature: {des.shape[1]}")


feat_img = cv2.drawKeypoints(img, kp, None, (0, 0, 255), 4)

cv2.imwrite("mandrillKP.png", feat_img)
