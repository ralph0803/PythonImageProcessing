import cv2
import numpy as np

# read and scale down image
# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png
img = cv2.pyrDown(cv2.imread('Calc-Training_P_00112_LEFT_MLO.png', cv2.IMREAD_UNCHANGED))

# threshold image
# ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
#                                   127, 255, cv2.THRESH_BINARY)
ret, threshed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# find contours and get the external one
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)

# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 5)

print(len(contours))

img = cv2.resize(img,(480,480))
cv2.imshow("contours", img)


"""
windows
"""
cv2.waitKey(0) # here the window waits until user manually closes
# while True:
#     if cv2.waitKey(0) & 0xFF == ord('esc'):
#         break
# cv2.destroyAllWindows()

"""
linux
"""
# ESC = 27
# while True:
#     keycode = cv2.waitKey()
#     if keycode != -1:
#         keycode &amp;= 0xFF
#         if keycode == ESC:
#             break
# cv2.destroyAllWindows()
