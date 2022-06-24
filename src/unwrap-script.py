
import os
import numpy as np
import pickle
from visualization import displacement_colorplot, plot_adjacency
from utils import read_excel, trim_to_equal_dim, bin, merge_u, nan_gaussian_filter
from basis_utils import rotate_uvecs
from unwrap_utils import neighborDistFilter, normNeighborDistance
import matplotlib.pyplot as plt  
from strain import unwrap, plot_rotated_gradients, strain
import scipy.ndimage as ndimage
from new_utils import dump_matrix

def strain_NK_disps(ds, ses, method2=True, mydat=False):
    
    if mydat:
        uvecs, prefix, dsnum, _ = import_uvector()
        my_u = latticevec_to_cartesian(uvecs)

    data_path = os.path.join('..','unittest-data','NK-tblg-uvecs')
    if method2: savepath = os.path.join(data_path, 'DS{}_session{}_unwrapped_method2.pkl'.format(ds, ses))
    with open(savepath, 'rb') as f: 
        v = pickle.load(f)
        u, centers, adjacency_type = v[:]
        u = u[0]

    if ds == 4 and ses == 1: offset, pad, sigma, nan_filter, nan_thresh = -0.05, 25, 2, False, None # unwrap perfect
    if ds == 4 and ses == 1 and method2: offset, pad, sigma, nan_filter, nan_thresh = -0.05, 5, 2, False, None # unwrap perfect

    if ds == 4 and ses == 2: offset, pad, sigma, nan_filter, nan_thresh = -0.05, 25, 2, True, 0.05  # small unwrap issue areas
    if ds == 8 and ses == 1: offset, pad, sigma, nan_filter, nan_thresh = -0.05, 25, 2, False, None # small unwrap issue areas
    if ds == 2 and ses == 1: offset, pad, sigma, nan_filter, nan_thresh = -0.05, 25, 2, False, None # small unwrap issue areas
    

    if mydat:
        f, ax = plt.subplots(2,2)
        img = displacement_colorplot(ax[0,0], u); 
        d = normNeighborDistance(u, norm=False)
        ax[0,1].imshow(d, origin='lower')
        img = displacement_colorplot(ax[1,0], my_u); 
        d = normNeighborDistance(my_u, norm=False)
        ax[1,1].imshow(d, origin='lower')
        plt.show()

        if nan_filter: u = neighborDistFilter(u, thresh=nan_thresh)
        u = u[pad:-pad, pad:-pad, :]
        ux = bin(u[:,:,0], bin_w=2, size_retain=True, method=np.nanmedian)
        uy = bin(u[:,:,1], bin_w=2, size_retain=True, method=np.nanmedian)
        ux = nan_gaussian_filter(ux, sigma) #ndimage.gaussian_filter(ux, sigma=sigma)  
        uy = nan_gaussian_filter(uy, sigma) #ndimage.gaussian_filter(uy, sigma=sigma)  
        u = merge_u(ux, uy)

        f, ax = plt.subplots(2,2)
        img = displacement_colorplot(ax[0,0], u); 
        d = normNeighborDistance(u, norm=False)
        ax[0,1].imshow(d, origin='lower')

        if nan_filter: my_u = neighborDistFilter(my_u, thresh=nan_thresh)
        my_u = my_u[pad:-pad, pad:-pad, :]
        my_ux = bin(my_u[:,:,0], bin_w=2, size_retain=True, method=np.nanmedian)
        my_uy = bin(my_u[:,:,1], bin_w=2, size_retain=True, method=np.nanmedian)
        my_ux = nan_gaussian_filter(my_ux, sigma) #ndimage.gaussian_filter(ux, sigma=sigma)  
        my_uy = nan_gaussian_filter(my_uy, sigma) #ndimage.gaussian_filter(uy, sigma=sigma)  
        my_u = merge_u(my_ux, my_uy)

        f, ax = plt.subplots(2,2)
        img = displacement_colorplot(ax[1,0], my_u); 
        d = normNeighborDistance(my_u, norm=False)
        ax[1,1].imshow(d, origin='lower')

        plt.show()

        f, ax = plt.subplots(3,7)    
        plot_rotated_gradients(my_ux, my_uy, offset, ax[0,:])
        plot_rotated_gradients(my_ux, my_uy, offset + 2/3*np.pi, ax[1,:])
        plot_rotated_gradients(my_ux, my_uy, offset + 4/3*np.pi, ax[2,:])
        plt.show()

        f, ax = plt.subplots(3,7)    
        plot_rotated_gradients(ux, uy, offset, ax[0,:])
        plot_rotated_gradients(ux, uy, offset + 2/3*np.pi, ax[1,:])
        plot_rotated_gradients(ux, uy, offset + 4/3*np.pi, ax[2,:])
        plt.show()
        
        exit()


    else:
        f, ax = plt.subplots(1,2)
        img = displacement_colorplot(ax[0], u); 
        d = normNeighborDistance(u, norm=False)
        ax[1].imshow(d, origin='lower')
        if ds == 4 and ses == 1: # perfect
            from heterostrain import hetstrain_from_adjacencies 
            from unwrap_utils import getAdjacencyMatrix
            centers, adjacency_type = getAdjacencyMatrix(u, boundary_val=0.4, delta_val=0.3, combine_crit=5.0, spdist=5.0) 
        
        plt.show()

        if nan_filter: u = neighborDistFilter(u, thresh=nan_thresh)
        u = u[pad:-pad, pad:-pad, :]
        ux = bin(u[:,:,0], bin_w=2, size_retain=True, method=np.nanmedian)
        uy = bin(u[:,:,1], bin_w=2, size_retain=True, method=np.nanmedian)
        ux = nan_gaussian_filter(ux, sigma) #ndimage.gaussian_filter(ux, sigma=sigma)  
        uy = nan_gaussian_filter(uy, sigma) #ndimage.gaussian_filter(uy, sigma=sigma)  
        u = merge_u(ux, uy)
        
        f, ax = plt.subplots(3,8)  
        thetam, hetstrain, a, ss = hetstrain_from_adjacencies(centers, (adjacency_type > 0), ax[1,0], ax[2,0])  
        img = displacement_colorplot(None, ux, uy)
        plot_adjacency(img, centers, adjacency_type, ax=ax[0,0], colored=False)
        ax[0,0].axis('off')
        plot_rotated_gradients(ux, uy, offset, ax[0,1:], centers, adjacency_type, thetam, a, ss)
        plot_rotated_gradients(ux, uy, offset + 2/3*np.pi, ax[1,1:], centers, adjacency_type, thetam, a, ss)
        plot_rotated_gradients(ux, uy, offset + 4/3*np.pi, ax[2,1:], centers, adjacency_type, thetam, a, ss)
        plt.show()
        exit()

    exx, exy, eyx, eyy, _, _ = strain(u)   
    gxy = 0.5*(exy+eyx)
    dump_matrix(exx, os.path.join(data_path, 'DS{}_session{}_exx.txt'.format(ds, ses)))
    dump_matrix(gxy, os.path.join(data_path, 'DS{}_session{}_gxy.txt'.format(ds, ses)))
    dump_matrix(eyy, os.path.join(data_path, 'DS{}_session{}_eyy.txt'.format(ds, ses)))
    f, ax = plt.subplots(1,3)
    ax[0].imshow(exx); ax[1].imshow(eyy); ax[2].imshow(gxy)
    plt.show()


def unwrap_NK_disps(ds, ses):

    data_path = os.path.join('..','unittest-data','NK-tblg-uvecs')
    fx = os.path.join(data_path, 'DS{}_session{}_xdisplacements_beforeunwrapping.xlsx'.format(ds, ses))
    fy = os.path.join(data_path, 'DS{}_session{}_ydisplacements_beforeunwrapping.xlsx'.format(ds, ses))
    savepath = os.path.join(data_path, 'DS{}_session{}_unwrapped_redo.pkl'.format(ds, ses))

    ux, uy = read_excel(fx), read_excel(fy)
    ux, uy = trim_to_equal_dim(ux, even=True), trim_to_equal_dim(uy, even=True)
    u = np.zeros((ux.shape[0], ux.shape[1], 2))
    # NK code had displacement vectors in angstroms, this code expects displacement vectors in units of a_0, a_0 for graphene is 2.46 angs
    u[:,:,0], u[:,:,1] = ux, uy 
    u = u / 2.46
        
    if ds == 4 and ses == 1: # perfect
        boundary_val = 0.45
        delta_val = 0.2
        combine_crit = 10.0
        spdist = 5.0 # decrease if stuff too connected
        centerdist = 0.1
        manual = False

    if ds == 4 and ses == 2: # small issues
        boundary_val = 0.4
        delta_val = 0.2
        combine_crit = 13.0
        spdist = 10.0 # decrease if stuff too connected
        centerdist = 0.1
        manual = False

    if ds == 8 and ses == 1: # small issues
        boundary_val = 0.4
        delta_val = 0.2
        combine_crit = 10.0
        spdist = 10.0 # decrease if stuff too connected
        centerdist = 0.1
        manual = False

    if ds == 2 and ses == 1: # small issues
        boundary_val = 0.4
        delta_val = 0.2
        combine_crit = 10.0
        spdist = 15.0 # decrease if stuff too connected
        centerdist = 0.1
        manual = False
        
    u, c, adjt = unwrap(u, manual=manual, ip=False, centerdist=centerdist, boundary_val=boundary_val, delta_val=delta_val, combine_crit=combine_crit, spdist=spdist)
    with open(savepath, 'wb') as f: pickle.dump([u, c, adjt], f) # save unwrapped

unwrap_NK_disps(ds=4, ses=1)
#exit()
#unwrap_NK_disps(ds=8, ses=1)
strain_NK_disps(ds=4, ses=1, mydat=False)
strain_NK_disps(ds=2, ses=1)
strain_NK_disps(ds=4, ses=2)

