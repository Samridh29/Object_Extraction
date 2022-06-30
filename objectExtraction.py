import cv2
import numpy as np

def display_image(image, text):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread('./NMLB_Original-1.png')
display_image(image, 'Original Image')

(h, w) = image.shape[:2]
print("Height: {} Width: {}".format(h, w))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image(gray, 'Gray Image')

edge = cv2.Canny(image, 100, 350)
display_image(edge, 'Edge Image')

print(image[0, ])

points = []
for i in range(0, h):
    for j in range(0, w):
        if edge[i, j] == 255:
            points.append((i, j))

display_image(image, 'Black Image')
# thresh = cv2.threshold(gray, 224, 255, cv2.THRESH_BINARY_INV)[1]
# display_image(thresh, 'Threshold')

# t =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#     cv2.THRESH_BINARY,11,2)
# display_image(t, 'Adaptive Threshold')
