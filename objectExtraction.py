import cv2
import numpy as np
import matplotlib.pyplot as plt

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

points = []
ap = []

for i in range(h):
    for j in range(w-1):
        if edge[i][j] == 255:
            # points.append([i,j])
             for k in range(0,i-1):
               if edge[k][j] == 255:
                  points.append([k,j])
               if edge[k][j+1] == 255:
                  points.append([k,j+1])
                  break
             for k in reversed(range(h)) :
               if edge[k][j] == 255:
                  points.append([k,j])
                  break

black_image = np.zeros((h, w), dtype=np.uint8)
for i in points:
    black_image[i[0]][i[1]] = 255
display_image(black_image, 'Black Image')

# print(ap)
mask = np.zeros(image.shape, dtype=np.uint8)
#roi_corners = np.array([i[0]][i[1]], dtype=np.int32)
roi_corners = np.array(points)
# fill the ROI so it doesn't get wiped out when the mask is applied
#channel_count = image.shape[2] # i.e. 3 or 4 depending on your image
#ignore_mask_color = (255,)*channel_count
#cv2.fillPoly(mask, roi_corners, ignore_mask_color)

cv2.fillConvexPoly(mask, np.array(points, 'int32'), 255)
# from Masterfool: use cv2.fillConvexPoly if you know it's convex

# apply the mask
masked_image = cv2.bitwise_and(image, mask)

# save the result
cv2.imwrite('image_masked.png', masked_image)
plt.imshow(masked_image)
plt.show()