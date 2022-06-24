

from visualization import displacement_colorplot, displacement_colorplot_lvbasis 
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from basis_utils import cartesian_to_latticevec, latticevec_to_cartesian, lv_to_rzlv, cartesian_to_rzcartesian, cartesian_to_rz_WZ 
from utils import normalize
from unwrap_utils import getAdjacencyMatrix
from diskset import DiskSet

gvecs = [ [0,1], [1,0], [1,-1], [-1,1], [-1,-1], [2,-1], [-2,1], [1,-2], [-1,2], [-1,0], [0,-1], [1,1] ]
gvecs = np.array(gvecs)
ndisks = len(gvecs)
nx, ny = 100,100 # small test want it fast
I = np.zeros((ndisks, nx, ny))
u_cart = np.zeros((nx, ny, 2))

# in units of a_0
X, Y = np.meshgrid(np.arange(-2.9, 14.7, 0.05), np.arange(-2.9, 14.7, 0.0)) # might not be realisitic resolution but not testing median fit 
for i in range(nx):
    for j in range(ny):
        u_cart[i,j,:] = [-Y[i,j], X[i,j]]

u_zcart = cartesian_to_rz_WZ(u_cart.copy(), sign_wrap=False)
u_lv = cartesian_to_latticevec(u_zcart)
coefs = np.zeros((ndisks,3))
for disk in range(ndisks):
    for i in range(nx):
        for j in range(ny):
            gdotu = np.dot(gvecs[disk], np.array([u_lv[i,j,0], u_lv[i,j,1]]))
            noise = 0.01 * np.random.normal(0,1)
            I[disk, i, j] = np.cos(np.pi * gdotu)**2 
            coefs[disk,:] = 1, 0, 0

for disk in range(ndisks): I[disk,:,:] = normalize(I[disk,:,:]) # Ci=0, Ai=1 when Bi=0

u_unwrap = u_cart.copy()
centerdist, boundary_val, delta_val, combine_crit, spdist = 0.01, 0.3, 0.3, 0.0, 2.0
centers, adjacency_type = getAdjacencyMatrix(u_zcart, boundary_val, delta_val, combine_crit, spdist)
unwrapsavepath = os.path.join("..", "data", "Test_Rigid", "unwrap.pkl")
fitsavepath    = os.path.join("..", "data", "Test_Rigid", "fit.pkl")
disksavepath   = os.path.join("..", "data", "Test_Rigid", "diskset.pkl")

with open(unwrapsavepath, 'wb') as f: pickle.dump( [u_cart, centers, adjacency_type, None], f )
with open(fitsavepath,    'wb') as f: pickle.dump( [None, coefs[:,0], coefs[:,1], coefs[:,2], u_lv], f)
