# %%
import numpy as np
import cv2 as cv

# %% Helper function to show the image
def showim(image,label = 'image'):
    cv.imshow(label, image)
    cv.waitKey()
    cv.destroyAllWindows() 

# %% Import the image 
image = cv.imread('./Input_images/Q1image.png')
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
## Construct a structuring element that is circular with  Diameter of 50
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(50,50))
## Using the kernel perform opening
closed = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
## Threshold the image to remove any other noise
_,closed =  cv.threshold(closed,127,255, cv.THRESH_BINARY)
showim(closed)

# %% Detect all the circles with the given parameters
circles = cv.HoughCircles(closed, cv.HOUGH_GRADIENT, 1,20, param1 = 70, param2 = 15, minRadius = 22,maxRadius = 50)

# %% Draw all the circles on to the image
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(closed,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(closed,(i[0],i[1]),2,(0,0,255),3)

# %%Print the number of circles
print("Number of Coins - ",circles.shape[1])

# %% Show the final image
showim(closed)


