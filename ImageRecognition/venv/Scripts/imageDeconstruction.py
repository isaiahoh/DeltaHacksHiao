import cv2
import numpy as py
import matplotlib as plt

img = cv2.imread('C:\\Users\\turtl\\git\\DeltaHacksHiao\\ImageRecognition\\image_4.png', cv2.IMREAD_COLOR)
blur = cv2.GaussianBlur(img, (15, 15), 0)

retval, threshold = cv2.threshold(blur, 117, 255, cv2.THRESH_BINARY)

cv2.imshow('image', img)
cv2.imshow("threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

