'''
Created on Jun 22, 2015

@author: lrmneves
'''
import cv2
from cv2 import waitKey
from numpy.linalg import solve,inv
import numpy as np
import math
#mous callback to get points
def get_points(event,x,y,flags,param):
    global count, b
    if event == cv2.EVENT_LBUTTONDOWN:
        if count < 9:
            b[count] = x
            b[count+1] = y
        count+=2;

def calculate_boundaries(A,inv_h):
    points = [np.array([x,y,1]) for (x,y) in A]
    ret = []
    for t in points:
        ret.append(np.dot(t,inv_h))
        

    return ret
         

image = cv2.imread("images/image1.jpg")

#initialize b and save chosen coordinates to it
b = np.zeros((8,1))
delta_x = 81.9
delta_y = 61.3 
real_world = np.array([50,200,50+delta_x,200,50+delta_x,200+delta_y,50,200+delta_y])
count = 0;

image_name = "Image"
cv2.namedWindow(image_name,1)

cv2.setMouseCallback(image_name, get_points);
cv2.imshow(image_name,image)
#get b
while count < 4:
    print "Click all 4 b and press enter"
    waitKey(0)
    

    
#initialize A
A = np.zeros((8,8))
row_count = 0

#create A matrix
for row in A:
    if row_count%2 == 0:
        row[0] = real_world[row_count]
        row[1] = real_world[row_count + 1]
        row[2] = 1
        row[6] = -real_world[row_count]*b[row_count]
        row[7] = -real_world[row_count+1]*b[row_count]
    else:
        row[3] = real_world[row_count-1]
        row[4] = real_world[row_count]
        row[5] = 1
        row[6] = -real_world[row_count-1]*b[row_count]
        row[7] = -real_world[row_count]*b[row_count]
    row_count+=1

h = solve(A,b)
h = np.append(h,[1])
h = np.reshape(h,(3,3))
inv_h = inv(h)
x_min = 0
x_max = 800
y_min = 0
y_max = 600
boundary_points =  calculate_boundaries([(x_min, y_min),(x_max,y_min),(x_max,y_max),(x_min,y_max)],inv_h)
width = abs(max([x[0] for x in boundary_points])) + abs(min(x[0] for x in boundary_points))
height = abs(max([x[1] for x in boundary_points])) + abs(min(x[1] for x in boundary_points))

final_image = np.zeros((height,width,3), np.uint8)



cv2.imshow("Result",final_image)
# if __name__ == '__main__':
#     pass