import cv2
import numpy as np
import matplotlib as plt

img = cv2.imread('C:\\Users\\turtl\\git\\DeltaHacksHiao\\ImageRecognition\\image_4.png', cv2.IMREAD_COLOR)
#Input image location
def imagePercentProcess(img):
    blur = cv2.medianBlur(img, 17)
    retval, threshold = cv2.threshold(blur, 117, 255, cv2.THRESH_BINARY)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 15)
    print(threshold)
    cv2.imshow('image', blur)
    cv2.imshow("adapt", adapt)
    cv2.imshow("threshold", threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return threshold
def imageNumberProcess(img):
    retval2, threshold2 = cv2.threshold(img, 172, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,contours, -1, 1, None, cv2.LINE_8, hierarchy, 1, None)

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    cv2.imshow("contours", img)
    return area

def colorPercent(threshold):
    img = threshold
    color = [255, 255, 255]  # RGB
    diff = 0
    boundaries = [([color[2]-diff, color[1]-diff, color[0]-diff],
                   [color[2]+diff, color[1]+diff, color[0]+diff])]
    # in order BGR as opencv represents images as numpy arrays in reverse order

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        ratio_color = cv2.countNonZero(mask)/(img.size/3)
        return (print('white pixel percentage:', np.round(ratio_color*100, 2)))


colorPercent(imagePercentProcess(img))


willitwork = imageNumberProcess(img)

print(willitwork)

cv2.contour