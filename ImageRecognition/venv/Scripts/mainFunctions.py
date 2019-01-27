from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('C:\\Users\\ashar\\Documents\\git\\DeltaHacksHiao\\ImageRecognition\\Test3.png', cv2.IMREAD_COLOR)


# Image filteration methods

# For Image Type 1 and Type 2
def imagePercent(img):
    blur = cv2.medianBlur(img, 19)
    retval, threshold = cv2.threshold(blur, 137, 255, cv2.THRESH_BINARY)

    newimg = threshold
    color = [0, 0, 0]  # RGB
    diff = 0
    boundaries = [
        ([color[2] - diff, color[1] - diff, color[0] - diff], [color[2] + diff, color[1] + diff, color[0] + diff])]
    # in order BGR as opencv represents images as numpy arrays in reverse order
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(newimg, lower, upper)
        output = cv2.bitwise_and(newimg, newimg, mask=mask)

        ratio_color = cv2.countNonZero(mask) / (newimg.size / 3)

    cv2.imshow("Original", img)
    cv2.imshow("Masked", newimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (print('Black Pixel Percentage:', np.round(ratio_color * 100, 2),"%"))


def imagePercent3(img):
    blur = cv2.medianBlur(img, 15)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    finalgrey = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 255, 1)
    retval, threshold = cv2.threshold(finalgrey, 137, 255, cv2.THRESH_BINARY)


    newimg = threshold
    color = [0, 0, 0]  # RGB
    diff = 0
    boundaries = [
        ([color[2] - diff, color[1] - diff, color[0] - diff], [color[2] + diff, color[1] + diff, color[0] + diff])]
    # in order BGR as opencv represents images as numpy arrays in reverse order
    ratio_color = 0
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(newimg, lower, upper)
        output = cv2.bitwise_and(newimg, newimg, mask=mask)

        ratio_color = cv2.countNonZero(mask) / (newimg.size / 3)



    cv2.imshow('Original', img)
    cv2.imshow('Masked', finalgrey)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return (print('Black Pixel Percentage:', np.round(ratio_color * 100, 2),"%"))

imagePercent3(img)

















def lacapianMethod(img):
    # Set of filters
    filtered = cv2.Laplacian(img, 0, cv2.CV_64F, 1, 16, 0)
    filtered = (255 - filtered)
    retval, threshold = cv2.threshold(filtered, 230, 255, cv2.THRESH_BINARY)
    imgrey = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)

    # Contours define the edges of the object
    _, contours, _ = cv2.findContours(imgrey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(filtered, contours, -1, (0, 0, 255), 5)
    cv2.imshow("Filtered", filtered)
    cv2.imshow("Real",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    the_data = []
    histo_data = []
    numberofshapes = 0
    for i in range(len(contours)):

        if cv2.contourArea(contours[i]) > 5:
            numberofshapes += 1
            histo_data.append(cv2.contourArea(contours[i]))
            the_data.append(cv2.contourArea(contours[i]))

    x= np.asarray(the_data)
    plt.hist(x, bins=50)

    plt.xlabel("Grain Area in pixels")
    plt.ylabel("Frequency")
    plt.axvline(x.mean(), color ='k', linestyle = 'dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(x.mean()+x.mean()/numberofshapes +7500, max_-max_/numberofshapes -1, 'Mean Area: {:.2f}'.format(x.mean()))
    plt.text(15000,8, "There are %d shapes"%numberofshapes)
    plt.show()

    return numberofshapes








