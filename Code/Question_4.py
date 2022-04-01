# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# %%
def showim(image,label = 'image'):
    cv.imshow(label, image)
    cv.waitKey()
    cv.destroyAllWindows() 

# %%
def calculate_centroid(cluster,POI,features,K=4):
    centroids = np.zeros((K,features))
    for idx,c in enumerate(cluster):
        ## For every cluster list, calculate the new centroid as the mean of all the feature values (mean of BGR values)
        centroids[idx] = np.mean(POI[c],axis=0)
    return centroids
    pass

# %%
def assign_clusters(POI,centroids,K=4):
    clusters = [[] for _ in range(K)]
    for i,point in enumerate(POI):
        ## For every point in the list of points, check its distance with the list off centroids
        distances = [np.linalg.norm(center-point,axis=0) for center in centroids]
        closest_index = np.argmin(distances)
        ## Assign the point to the centroid which is closest
        clusters[closest_index].append(i)
    return clusters

# %%
def kmeans_clustring(POI,K = 4,max_itr = 100):
    ## Create a list of empty clusters and centroids
    cluster_idx = [[] for _ in range(K)]
    list_centroids = []
    POI = np.float32(POI)
    samples, features = POI.shape
    
    ## randomly choose K values from within the POI 
    random_idx = np.random.choice(samples,K,replace=False)
    list_centroids = [POI[i] for i in random_idx]

    
    for step in range(max_itr):
        print("Iteration number = ",step)
        ## Assign every point in the image to a specific cluster based on distanceec
        cluster_idx = assign_clusters(POI,list_centroids,K)
        old_centroid = list_centroids
        ## Calculate the new centroids using the new cluster list, as the mean of its feature values
        list_centroids = calculate_centroid(cluster_idx,POI,features,K)
        ## Check if the previous centroid values are equal to the new ones ==> convergence
        diff = sum([np.linalg.norm(i-j) for i,j in zip(old_centroid,list_centroids)])
        if(diff == 0):
            break

    list_label = np.zeros(samples)
    ## The cluster list is a list of index corresponding to the points position in the flattened image
    for idx, cluster in enumerate(cluster_idx):
        ## Assign each point its corresponding label ie, idx
        list_label[cluster] = idx
    return list_label,list_centroids
            

# %%
image = cv.imread('./Input_images/Q4image.png')
showim(image)
## Read the image and reshape to ((nxm)x3) where the 3 featrues are the BGR values
points = np.reshape(image,(-1,3))

# %%
np.random.seed(690420)
## Obtain the list of centroids and labels
labels,centers = kmeans_clustring(points)
## Normalize the BGR values of the centroid
centers = centers/255

# %%
## assign a new image with the values of the centroid color for every point based on the label
segmented_img = centers[labels.flatten().astype(np.int32)]
## Reshape back into an image
segmented_img = segmented_img.reshape(image.shape)
# segmented_img = cv.cvtColor(segmented_img,cv.COLOR_BGR2RGB)
showim(np.float32(segmented_img))
# plt.imshow(segmented_img)


