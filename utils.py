import numpy as np
import os

def OBB(pointArray):
    # pointArray = np.array(points)
    ca = np.cov(pointArray,y = None,rowvar = 0,bias = 1)
    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    # use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    ar = np.dot(pointArray,np.linalg.inv(tvect))
    
    # get the minimum and maximum x and y 
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff = (maxa - mina)*0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    
    # get the 8 corners by subtracting and adding half the bounding boxes height and width to the center
    pointShape = pointArray.shape
    if pointShape[1] == 2:
        corners = np.array([center+[-diff[0],-diff[1]],
                        center+[diff[0],-diff[1]],
                        center+[diff[0],diff[1]],
                        center+[-diff[0],diff[1]],
                        center+[-diff[0],-diff[1]]])
    if pointShape[1] == 3:
        corners = np.array([center+[-diff[0],-diff[1],-diff[2]],
                    center+[diff[0],-diff[1],-diff[2]],                    
                    center+[diff[0],diff[1],-diff[2]],
                    center+[-diff[0],diff[1],-diff[2]],
                    center+[-diff[0],diff[1],diff[2]],
                    center+[diff[0],diff[1],diff[2]],                    
                    center+[diff[0],-diff[1],diff[2]],
                    center+[-diff[0],-diff[1],diff[2]],
                    center+[-diff[0],-diff[1],-diff[2]]])   
    
    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the centerback
    corners = np.dot(corners,tvect)
    center = np.dot(center,tvect)
    radius = diff
    if pointShape[1] == 2:
        array0,array1 = np.abs(vect[0,:]),np.abs(vect[1,:])
        index0,index1 = np.argmax(array0),np.argmax(array1)
        radius[index0],radius[index1] = diff[0],diff[1]
    if pointShape[1] == 3:
        array0,array1,array2 = np.abs(vect[0,:]),np.abs(vect[1,:]),np.abs(vect[2,:])
        index0,index1,index2 = np.argmax(array0),np.argmax(array1),np.argmax(array2)
        radius[index0],radius[index1],radius[index2] = diff[0],diff[1],diff[2]
    eigenvalue = v
    eigenvector = vect
    # return corners, center, radius, eigenvalue, eigenvector

    return center, radius, mina, maxa, eigenvector

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("mkdir success")
    else:
        print("dir exists")