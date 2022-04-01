# %%
import numpy as np
import cv2 as cv

# %% Helper function
def showim(image,label = 'image'):
    cv.imshow(label, image)
    cv.waitKey()
    cv.destroyAllWindows() 

# %% Import the images and convert it to grayscale for processing
image_a = cv.imread('./Input_images/Q2imageA.png')
gray_a = cv.cvtColor(image_a,cv.COLOR_BGR2GRAY)
image_b = cv.imread('./Input_images/Q2imageB.png')
gray_b = cv.cvtColor(image_b,cv.COLOR_BGR2GRAY)
showim(image_a,"Image A")
showim(image_b,"Image B")

# %%
sift = cv.xfeatures2d.SIFT_create()
## Use Sift to detect and computer PoI in both the images
kp_a, des_a = sift.detectAndCompute(gray_a,None)
kp_b, des_b = sift.detectAndCompute(gray_b,None)

# %%
bf = cv.BFMatcher()
## Use Brute force fitting algorithm to find the list of matches 
matches = bf.knnMatch(des_b,des_a,k=2)

# %%
good_points = []
## for all the matches 
for m,n in matches:
    ## Apply ratio test
    if m.distance<0.75*n.distance:
        i = m.trainIdx
        j = m.queryIdx
        ## Story the Points of those that fit the ratio
        good_points.append([kp_a[i].pt,kp_b[j].pt])

# %%
## The list of points of the left image
des_pts= np.asarray(good_points)[:,0]
## The list of points of the right image
src_pts  = np.asarray(good_points)[:,1]

# %%
## Find the homography that transforms the right image points to the left image
matrix,_ = cv.findHomography(src_pts,des_pts,cv.RANSAC,4.0)

# %%
## Warp the right image into using the homography into a bigger frame
dst = cv.warpPerspective(image_b, matrix, (image_a.shape[1]+image_b.shape[1],image_a.shape[0]))

# %%
## Replaces portion of the image with left image
dst[:image_a.shape[0],:image_a.shape[1]] = image_a
## Remove all the columns that are black bars
dst = dst[:,:np.argwhere(dst>0)[-1][1]]
showim(dst)


