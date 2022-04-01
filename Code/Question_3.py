# %%
import numpy as np

# %%
def generate_A_matrix(world_corners,image_corner):
    ##  points in the world frame
    Xw = world_corners
    ##  points in the camera frame
    xp = image_corner
    ## A matrix
    matrix_A = np.matrix(np.zeros((Xw.shape[0]*2,12)))
    ## Populate the matrix
    for a,b in zip(enumerate(Xw),enumerate(xp)):
        # m[0] = u | m[1] = v
        # n[0] = x | n[1] = y | n[2] = z
        i,j,n,m = a[0],b[0],a[1],b[1]
        matrix_A[i+j,:] = n[0],n[1],n[2],1,0,0,0,0,-m[0]*n[0],-m[0]*n[1],-m[0]*n[2],-m[0]
        matrix_A[(i+j)+1,:] = 0,0,0,0,n[0],n[1],n[2],1,-m[1]*n[0],-m[1]*n[1],-m[1]*n[2],-m[1]

    return matrix_A

# %%
## Create a list of given world and image points
world_corners = np.array([[0,0,0],[0,3,0],[0,7,0],[0,11,0],[7,1,0],[0,11,7],[7,9,0],[0,1,7]],dtype=np.int32)
image_corners = np.array([[757,213],[758,415],[758,686],[759,966],[1190,172],[329,1041],[1204,850],[340,159]],dtype=np.int32)
A = generate_A_matrix(world_corners,image_corners)

# %%
## Find the Eigen vector and value of A.T * A
eigenval,eigenvec =  np.linalg.eig(np.matmul(A.T,A))
## p is the vector corresponding to teh least eigen value
p = eigenvec[:,np.argsort(eigenval)[0]]
## Split the column vector and rearrange to get P 3x4
p1,p2,p3 = np.array_split(p,3,axis=0)
Projection_matrix = np.vstack((p1.T,p2.T,p3.T))

# %%
## Using QR factorization find the rotation and intrinsic matrix
Rotation_matrix, intrinsic_matrix = np.linalg.qr(Projection_matrix[0:,0:-1])
intrinsic_matrix = intrinsic_matrix/intrinsic_matrix[-1,-1] ## Normalize the matrix such that [-1,-1] is 1
translation_vector = np.linalg.inv(intrinsic_matrix)@Projection_matrix[0:,-1]# Find the translation vector

# %%
print("Projection_matrix \n", Projection_matrix)
print("Normalized Intrinsic Matrix \n", intrinsic_matrix)
print("Rotation Matrix \n", Rotation_matrix)
print("Translation Vector \n", translation_vector)
