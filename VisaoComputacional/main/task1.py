'''
Created on Jun 22, 2015

@author: lrmneves
'''
import cv2
from cv2 import waitKey
from numpy.linalg import solve,inv
import numpy as np

def get_points(event,x,y,flags,param):
    '''mouse callback to get the 4 points in counterclockwise order'''
    global count, real_world
    if event == cv2.EVENT_LBUTTONDOWN:
        if count < 9:
            real_world[count] = x
            real_world[count+1] = y
        count+=2;

def calculate_boundaries(A,h):
    '''function to calculate the boundaries by applying transformation to the boundary points'''
    points = [np.array([x,y,1]) for (x,y) in A]
    ret = []
    for t in points:
        ret.append(np.dot(t,h))

    return ret

def calculate_projection(real_world,image,h):
    height, width = real_world.shape[:2]
    for y in range(height):
        for x in range(width):
            try:
                new_point = calculate_new_point(x, y, h)
                real_world[y,x] = image[new_point[1],new_point[0]]
                
            except IndexError:
                pass
           
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2)

def calculate_new_point(real_world_x,real_world_y,inv_h):
    A = np.array([real_world_x,real_world_y,1])
    new_point = np.dot(A,inv_h)
    return (new_point[0]/new_point[2], new_point[1]/new_point[2])

image = cv2.imread("images/image1.jpg")

#initialize b and save chosen coordinates to it
#real_world are the points from the image given, points x and y on the equation
real_world = np.zeros((8,1))
delta_x = 819
delta_y = 613 

x_0 = 0
y_0 = 0
#b are the points from the given rectangle, matrix [x'0 y'0 ... x'n y'n] n = [0..3] 
b = np.array([x_0,y_0,x_0+delta_x,y_0,x_0+delta_x,y_0+delta_y,x_0,y_0+delta_y])
count = 0;

image_name = "Image"
cv2.namedWindow(image_name,1)

cv2.setMouseCallback(image_name, get_points);
cv2.imshow(image_name,image)
#get b
while count < 4:
    print "Click all 4 points in counter clockwise and press enter"
    waitKey(0)
    

    
#initialize A
A = np.zeros((8,8))
row_count = 0

#create A matrix, 8x8 matrix
for row in A:
    if row_count%2 == 0:
        row[0] = real_world[row_count]#xi
        row[1] = real_world[row_count + 1]#yi
        row[2] = 1
        #row[3-5] = 0
        row[6] = -real_world[row_count]*b[row_count] #-xi*x'i
        row[7] = -real_world[row_count+1]*b[row_count]#-yi*y'i
    else:
        #row[0-2] = 0
        row[3] = real_world[row_count-1]#xi
        row[4] = real_world[row_count]#yi
        row[5] = 1
        row[6] = -real_world[row_count-1]*b[row_count]#-xi*y'i
        row[7] = -real_world[row_count]*b[row_count]#-yi*y'i
    row_count+=1
#Solve linear system and save 3x3 H matrix, H3,3 = 1 WLOG
#Ah = b
h = solve(A,b)
h = np.append(h,[1])
h = np.reshape(h,(3,3))
#Calculate the inverse of the matrix to set boundaries
inv_h = inv(h)

#max_width is the max distance between topleft-toprigth points and bottowmleft-bottomright
width_A = calculate_distance((b[0],b[1]),(b[2],b[3]))
width_B = calculate_distance((b[4],b[5]),(b[6],b[7]))
new_width = max(int(width_A), int(width_B))

height_A = calculate_distance((b[0],b[1]),(b[6],b[7]))
height_B = calculate_distance((b[4],b[5]),(b[2],b[3]))
new_height = max(int(height_A), int(height_B))

warped = cv2.warpPerspective(image, h, (new_width, new_height))

# final_image = np.zeros((new_height,new_width,3), np.uint8)
# 
# height,width = image.shape[:2]
# 
# # 
# boundary_points =  calculate_boundaries([(0, 0),(width,0),(width,height),(0,height)],h)
# # 
# dx = abs((max([x[0]/x[2] for x in boundary_points]) - min(x[0]/x[2] for x in boundary_points))/new_width)
# dy = abs((max([x[1]/x[2] for x in boundary_points]) - min(x[1]/x[2] for x in boundary_points))/new_height)
#  
# for x in range(new_width):
#     for y in range(new_height):
#         new_point = calculate_new_point(x+100*dx, y+100*dy, inv_h)
#         final_image[y,x] = image[new_point[1],new_point[0]]
#  
cv2.imshow("Result",warped)        
cv2.waitKey(0)
# 

# 
# width = abs(max([x[0]/x[2] for x in boundary_points])) + abs(min(x[0]/x[2] for x in boundary_points)) 
# 
# height = abs(max([x[1]/x[2] for x in boundary_points])) + abs(min(x[1]/x[2] for x in boundary_points))
# 
# final_image = np.zeros((height,width,3), np.uint8)
# 
# calculate_projection(final_image,image,h)
# 
# cv2.imshow("Result",final_image)
# cv2.waitKey(0)
# if __name__ == '__main__':
#     pass