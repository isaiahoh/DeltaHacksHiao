import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\ashar\\Documents\\git\\DeltaHacksHiao\\ImageRecognition\\Test3.png', cv2.IMREAD_COLOR)


filtered = cv2.Laplacian(img, 0,  cv2.CV_64F , 1, 16, 0)
filtered = (255-filtered)
retval, threshold = cv2.threshold(filtered, 230, 255, cv2.THRESH_BINARY)


imgrey = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)

_,contours,_ = cv2.findContours(imgrey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

numberofshapes = 0

the_data = []
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 5:
        numberofshapes +=1
        the_data.append([i, cv2.contourArea(contours[i])])

cv2.drawContours(filtered, contours, -1, (0, 0, 255), 5)
print(numberofshapes)


data_file = open("data.csv", "w")
for i in (the_data):
        stringer = str(i[0])
        stringer += ","
        stringer += str(i[1])
        stringer += "\n"
        data_file.write(stringer)

data_file.close()



print(the_data)






#hist = cv2.calcHist([filtered], [0], None, [256], [0, 256])
#plt.hist(filtered.ravel(),256,[0, 256])
#plt.show()

#cv2.imshow("contours", contours)
cv2.imshow("CHicken",img)
cv2.imshow("filtered",filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(the_data, normed=True)
plt.show()