
import numpy as np

# bilayer 
# if want for single disk,  provide single gvector
# if want for single pixel, provide single uvector in shape 1,1,2
def cos2_sincos_bl_fitting(u, coefs, g):
	ndisks = len(g)
	nx, ny, = u.shape[0], u.shape[1]
	I = np.zeros((ndisks, nx, ny))
	for disk in range(ndisks):
	    for x in range(nx):
	        for y in range(ny):
	            I[disk,x,y] =  coefs[disk, 0] * np.cos(np.pi*np.dot(g[disk], u[x,y,:])) ** 2 
	            I[disk,x,y] += coefs[disk, 1] * np.cos(np.pi*np.dot(g[disk], u[x,y,:])) * np.sin(np.pi*np.dot(g[disk], u[x,y,:]))
	            I[disk,x,y] += coefs[disk, 2]
	return I

# returns dIfit/dparam for param in parameters
# for use in optimizations that need deriv info
def deriv_cos2_sincos_bl(u, coefs, g):
	return None

# bilayer
# if want for single disk,  provide single gvector
# if want for single pixel, provide single uvector in shape 1,1,2
def cos2_bl_fitting(u, coefs, g):
	ndisks = len(g)
	nx, ny, = u.shape[0], u.shape[1]
	I = np.zeros((ndisks, nx, ny))
	for disk in range(ndisks):
	    for x in range(nx):
	        for y in range(ny):
	            I[disk,x,y] =  coefs[disk, 0] * np.cos(np.pi*np.dot(g[disk], u[x,y,:])) ** 2 
	            I[disk,x,y] += coefs[disk, 1]
	return I

