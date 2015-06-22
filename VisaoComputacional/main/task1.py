'''
Created on Jun 22, 2015

@author: lrmneves
'''
import cv2
from cv2 import waitKey

def get_points(event,x,y,flags,param):
    global count, coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        if count < 4:
            coordinates[count] = (x,y)
        count+=1;
        print coordinates
    

image = cv2.imread("images/image1.jpg")
coordinates = [(0,0),(0,0),(0,0),(0,0)]
count = 0;

image_name = "Image"
cv2.namedWindow(image_name,1)

cv2.setMouseCallback(image_name, get_points);
cv2.imshow(image_name,image)

cv2.waitKey(0)



# if __name__ == '__main__':
#     pass