import cv2
import numpy as np
import matplotlib as plt

img = cv2.imread('C:\\Users\\ashar\\Documents\\git\\DeltaHacksHiao\\ImageRecognition\\image_4.png', cv2.IMREAD_COLOR)
#Input image location
def imageProcess(img):
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    retval, threshold = cv2.threshold(blur, 117, 255, cv2.THRESH_BINARY)
    print(threshold)
    cv2.imshow('image', img)
    cv2.imshow("threshold", threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return threshold

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


colorPercent(imageProcess(img))