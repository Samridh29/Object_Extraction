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

points=[]
for i in range(h):
    for j in range(w-1):
        if edge[i][j] == 255:
            if((edge[k][j] != 255 for k in range(0,i-1)) or (edge[k][j] != 255 for k in range(i+1,h))):
                points.append([i,j])
            #


# form image from points
black_image = np.zeros((h, w), dtype=np.uint8)
for point in points:
    black_image[point[0]][point[1]] = 255
display_image(black_image, 'con Image')


# thresh = cv2.threshold(gray, 224, 255, cv2.THRESH_BINARY_INV)[1]
# display_image(thresh, 'Threshold')

# t =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#     cv2.THRESH_BINARY,11,2)
# display_image(t, 'Adaptive Threshold')
