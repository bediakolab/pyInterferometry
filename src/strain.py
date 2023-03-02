from utils import *
from visualization import displacement_colorplot
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import gc
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.colors as mplc
import colorsys
from scipy import ndimage
from strain_utils import differentiate
from unwrap_utils import neighborDistFilter, getAdjacencyMatrix, geometric_unwrap, normDistToNearestCenter, BFS_from_center_uwrap, refit_outliers, automatic_sp_rotation 
from unwrap_utils import refit_by_region, refit_convex_regions, geometric_unwrap_tri  
from visualization import plot_adjacency, plot_contour, displacement_colorplot  
from denoise import denoise, wrap
from new_utils import import_uvector, dump_matrix
from skimage.restoration import denoise_tv_chambolle
from basis_utils import latticevec_to_cartesian, cartesian_to_latticevec, lv_to_rzcartesian, flip_to_reference, rotate_uvecs, get_nearby_equivs, getclosestequiv, cartesian_to_rz_WZ  
from new_utils import project_down, import_unwrap_uvector, normNeighborDistance, wrapNeighborDistance, wrapNeighborStrain, anom_filter, anom_nan_filter
from utils import manual_cropping, tic, toc
#from heterostrain import hetstrain_from_adjacencies
from new_utils import crop_displacement

def unwrap_multigrid(u, params):

    if params is not None:
        centerdist=params['centerdist']
        boundary_val=params['boundary_val']
        delta_val=params['delta_val']
        combine_crit=params['combine_crit'] 
        ndist_crit=params['ndist_crit'] 
        spdist=params['spdist']

    u[:,:,0], u[:,:,1] = -u[:,:,0], u[:,:,1]
    u = cartesian_to_rz_WZ(u, sign_wrap=False)
    centers, adjacency_type = getAdjacencyMatrix(u, boundary_val, delta_val, combine_crit, spdist)
    points = [ [c[1], c[0]] for c in centers ]
    u, ang, adjacency_type = automatic_sp_rotation(u, centers, adjacency_type, transpose=True) # rotate so sp closest to vertical is sp1, gvector choice degenerate under 2pi/3 rotations so arbitrary sp1/sp2/sp3
    
    def project_help(u, cents):
        ufilt = project_down(u, 2, method='L2')
        cents = [ [int(round(c[0]/2)), int(round(c[1]/2))] for c in cents ]
        cents = [ [np.min([c[0], ufilt.shape[0]-1]), np.min([c[1], ufilt.shape[1]-1])] for c in cents]
        pts = [ [c[1], c[0]] for c in cents ]
        return ufilt, cents, pts

    ufilt2, centers2, points2 = project_help(u, centers)
    ufilt4, centers4, points4 = project_help(ufilt2, centers2)

    centers4, adjacency_type = getAdjacencyMatrix(ufilt4, boundary_val, delta_val, combine_crit/4, spdist/4)
    u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, verts = geometric_unwrap(centers4, adjacency_type, ufilt4, voronibool=True, plotting=True) 
    print('click to remove centers not confident about before refitting')
    img = displacement_colorplot(None, u_unwrapped);
    centers4, _ = manual_remove_AA(img, [ [c[1], c[0]] for c in centers4 ], adjacency_type.copy())
    centers4 = [ [c[1], c[0]] for c in centers4 ]
    dists = normDistToNearestCenter(ufilt4.shape[0], ufilt4.shape[1], centers4)
    variable_region = (dists > centerdist/4).astype(int)
    f, ax = plt.subplots(2,2)
    plot_contour(dists, centerdist/4, ax=ax[0,0], plotflag=False)
    ax[0,1].quiver(u_unwrapped[:,:,0], u_unwrapped[:,:,1])
    ax[1,0].quiver(u_signalign[:,:,0], u_signalign[:,:,1])
    ax[1,1].quiver(u_adjusts[:,:,0], u_adjusts[:,:,1])
    plt.show()
    u = strain_method_3(u_unwrapped, points4, variable_region)
    f, ax = plt.subplots(); img = displacement_colorplot(ax, ufilt4); plt.show();
    ux = unbin(ufilt4[:,:,0], 2)
    uy = unbin(ufilt4[:,:,1], 2)
    uref = np.zeros((ux.shape[0], ux.shape[1], 2))
    uref[:,:,0], uref[:,:,1] = ux, uy
    f, ax = plt.subplots(); img = displacement_colorplot(ax, ufilt2); plt.show();
    ufilt2 = flip_to_reference(ufilt2.copy(), uref)
    f, ax = plt.subplots(); img = displacement_colorplot(ax, ufilt2); plt.show();
    exit()

    dists = normDistToNearestCenter(ufilt2.shape[0], ufilt2.shape[1], centers2)
    variable_region = (dists > centerdist/2).astype(int)
    ufilt2 = strain_method_3(ufilt2, points2, variable_region)
    f, ax = plt.subplots(); img = displacement_colorplot(ax, ufilt2); plt.show();
    ux = unbin(ufilt2[:,:,0], 2)
    uy = unbin(ufilt2[:,:,1], 2)
    uref = np.zeros((ux.shape[0], ux.shape[1], 2))
    uref[:,:,0], uref[:,:,1] = ux, uy
    u = flip_to_reference(u, uref)
    f, ax = plt.subplots(); img = displacement_colorplot(ax, u); plt.show();

def strain_method_3(u, points, variable_region, anomtol=0.05):
    def plthelp(u, wd=None, tol=0.05, plotbool=False):
        if plotbool: f, ax = plt.subplots(1,2)
        d = normNeighborDistance(u, norm=False)
        if wd is None: wd = wrapNeighborDistance(u, extend=True)
        if plotbool: ax[0].imshow(d, origin='lower', vmax=0.05)
        refitbool = ((np.abs(d-wd))>tol)
        if plotbool: ax[1].imshow(refitbool, origin='lower', vmax=0.05)
        if plotbool: plt.show();
        obj = np.sum(refitbool.flatten())
        print('objective is ', obj, 'w/ tol ', tol)
        return wd, refitbool, obj
    def pick_local_optimal(u, refit):
        for x in range(u.shape[0]):
            for y in range(u.shape[1]):
                if refit[x,y] and y<u.shape[1]-1 and not refit[x,y+1]:
                    u[x,y,:] = getclosestequiv(u[x,y], u[x,y+1])
                    refit[x,y] = False
                elif refit[x,y] and x<u.shape[0]-1 and not refit[x+1,y]:
                    u[x,y,:] = getclosestequiv(u[x+1,y], u[x,y])
                    refit[x,y] = False
                elif refit[x,y] and y>0 and not refit[x,y-1]:
                    u[x,y,:] = getclosestequiv(u[x,y], u[x,y-1])
                    refit[x,y] = False
                elif refit[x,y] and x>0 and not refit[x-1,y]:
                    u[x,y,:] = getclosestequiv(u[x,y], u[x-1,y])
                    refit[x,y] = False
        return u, refit
    def pick_local_optimal_cycle(u, wd, refitbool):
        wd, refitbool, obj = plthelp(u, wd, tol=0.005)
        while True:
            uprev = u.copy()
            u, refitbool = pick_local_optimal(u.copy(), refitbool) 
            obj_prev = obj
            wd, refitbool, obj = plthelp(u, wd, tol=0.005)
            if obj >= obj_prev: 
                wd, refitbool, obj = plthelp(uprev, wd, tol=0.005)
                print('revert to objective of {}'.format(obj))
                return uprev, refitbool
        return u, refitbool

    tic()
    u, wd = anom_nan_filter(u, None)
    plt.close('all')
    gc.collect()
    neighdist = wd
    wd, refitbool, obj = plthelp(u, wd, tol=0.01)
    u_new, reg = BFS_from_center_uwrap(1, variable_region.copy(), u.copy(), centers=points, xwid=2, ywid=2, ovlp=1, fittype='median', debug=False, extend=True)  
    wd, refitbool, obj_new = plthelp(u, wd, tol=0.01)
    if obj_new < obj: u = u_new
    u, refitbool = pick_local_optimal_cycle(u, wd, refitbool)
    wd, refitbool, obj = plthelp(u, wd=wd, tol=0.01)
    toc('some prelims')
    counter = 1
    while True:
        wd, refitbool, obj = plthelp(u, wd=wd, tol=0.005)
        tic()
        u = refit_convex_regions(u, refitbool, maxsize=np.inf, minsize=5, fittype='median')
        toc('refit convex regions {}'.format(counter))
        wd, refitbool, nobj = plthelp(u, wd=wd, tol=0.005)
        u, refitbool = pick_local_optimal_cycle(u, wd, refitbool)
        if nobj >= obj: break
        counter += 1
    counter = 1
    while True:
        wd, refitbool, obj = plthelp(u, wd=wd, tol=0.005)
        oldu = u.copy()
        tic()
        u = refit_convex_regions(u, refitbool, maxsize=np.inf, minsize=5, fittype='ip')
        toc('refit convex regions ip {}'.format(counter))
        wd, refitbool, nobj = plthelp(u, wd=wd, tol=0.005)
        u, refitbool = pick_local_optimal_cycle(u, wd, refitbool)
        if nobj >= obj: 
            u = oldu
            nobj = obj
            print('reverting to {}'.format(obj))
            break 
        counter += 1    
    tic()
    u, reg = refit_by_region(u, refitbool, maxsize=5, width=3, fittype='ip')
    wd, refitbool, nobj = plthelp(u, wd=wd, tol=0.005, plotbool=True)
    toc('refit small regions ip')    
    return u   

# unwraps and gets strain from a simple differentiation
def unwrap_old(u, params=None, manual=False, plotbool=False, voronibool=True, centerdist=0.05, ndist_crit=0.05, refit_offset=5, boundary_val=None, 
    delta_val=None, combine_crit=None, spdist=None, ip=True, nan_filter=False, nan_thresh=None, flip=False, wz=True, L1=False):

    while True:
        methodid = input("Method? \n1: voronoi (good for large twist data, P or AP) \n2: watershed (good for most AP data unless very large twist) \n3: triangulate (good for small-intermediate twist P data)\n").lower().strip()[0] 
        if int(methodid) == 1: 
            voronibool = True 
            tribool = False
            print('using voronoi')
            break
        elif int(methodid) == 2: 
            voronibool = False 
            tribool = False
            print('using watershed')
            break
        elif int(methodid) == 3:     
            voronibool = False 
            tribool = True
            print('using triangulate')
            print('ERROR: not completely implemented yet sorry try something else')
        else:
            print('unrecognized/unimplemented method please try again'.format(methodid))

    img = displacement_colorplot(None, u)
    crop_displacement(img, u)
    
    if params is not None:
        centerdist=params['centerdist']
        boundary_val=params['boundary_val']
        delta_val=params['delta_val']
        combine_crit=params['combine_crit'] 
        ndist_crit=params['ndist_crit'] 
        spdist=params['spdist']
        flip=params['flip'] 
        ip=params['ip']
        refit_offset=params['refit_offset']
        remove_filter_crit = params['remove_filter_crit']
        L1 = params['L1']
        wz = params['wz']

    # get AA centers and connectivity (type of SP connecting them)
    if flip: u[:,:,0], u[:,:,1] = -u[:,:,0], u[:,:,1]
    if wz: u = cartesian_to_rz_WZ(u, sign_wrap=False)
    centers, adjacency_type = getAdjacencyMatrix(u, boundary_val, delta_val, combine_crit, spdist)
    points = [ [c[1], c[0]] for c in centers ]
    u, ang, adjacency_type = automatic_sp_rotation(u, centers, adjacency_type, transpose=True) # rotate so sp closest to vertical is sp1, gvector choice degenerate under 2pi/3 rotations so arbitrary sp1/sp2/sp3
    
    # first plot is of the adjacency matrix to make sure things look alright
    #img = displacement_colorplot(None, u)
    #plot_adjacency(img, centers, adjacency_type)
    #plt.show(); 
    
    # next is of the voroni regions and anticipated offsets
    if not tribool: 
        u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, vertices = geometric_unwrap(centers, adjacency_type, u, voronibool, plotting=True) 
    else: 
        u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, vertices = geometric_unwrap_tri(centers, adjacency_type, u) 
        
    # third plot is of the regions deemed close enough to AA centers to be fixed in the unwrapping
    if plotbool: f, ax = plt.subplots(2,3)
    dists = normDistToNearestCenter(u.shape[0], u.shape[1], centers)
    variable_region = (dists > centerdist).astype(int)
    if plotbool: 
        plot_contour(dists, centerdist, ax=ax[0,0], plotflag=False)
        ax[0,1].imshow(u_unwrapped[:,:,0], origin='lower')
        ax[0,2].imshow(u_unwrapped[:,:,1], origin='lower')
        ax[1,1].imshow(u_adjusts[:,:,0], origin='lower')
        ax[1,2].imshow(u_adjusts[:,:,1], origin='lower')
        #ax[1,0].quiver(u_signalign[:,:,0], u_signalign[:,:,1])
    #if not voronibool and plotbool:
    #    ax[1,0].imshow(regions[:,:], origin='lower')
    #elif plotbool:
    #    for i in range(len(regions)):
    #        region = regions[i]
    #        polygon = vertices[region]
    #        ax[1,0].fill(*zip(*polygon), alpha=0.4)  
    if plotbool: plt.show()
    u = strain_method_3(u_unwrapped, points, variable_region)
    if nan_filter: u = neighborDistFilter(u, thresh=nan_thresh)
    return u, centers, adjacency_type       


# stepsize ss is nm per pixel
# coef : piezoelectric coefficent for this material in C per m
# returns piezo-charge in C per m^2 for top layer, piezo_bot = piezo_top if P, = -piezo_top if AP
def unscreened_piezocharge(u, sample_angle=0, smoothfunc=(lambda u: u), ss=1, coef=1):

    Ux = smoothfunc(u[:,:,0]) * 0.5 # want intralayer, with u = utop - ubottom = 2 * utop
    Uy = smoothfunc(u[:,:,1]) * 0.5 # measured u want utop so divide by 2 
    rotation_correction = sample_angle * np.pi/180
    def dxfunc(func):
        return np.cos(rotation_correction) * np.gradient(func, axis=1) - np.sin(rotation_correction) * np.gradient(func, axis=0) 
    def dyfunc(func):
        return np.sin(rotation_correction) * np.gradient(func, axis=1) + np.cos(rotation_correction) * np.gradient(func, axis=0) 

    # rho_piezo_top is e_11_top * [ d_xx u_y + d_xy u_x + d_xy u_x - d_yy u_y ]
    # THIS ASSUMES HOMOBILAYER utot = utop - ubot = utop - (-utop) so ----> utop = utot/2
    # u provided is in units of pixels, dx grid in pixels, first derivative unitless
    # second derivative is in inverse pixels
    d_xx_u_y = dxfunc(dxfunc(Uy))
    d_yy_u_y = dyfunc(dyfunc(Uy))
    d_xy_u_y = dxfunc(dyfunc(Uy))
    d_xy_u_x = dxfunc(dyfunc(Ux))
   
    piezo_unscaled = (d_xx_u_y + d_xy_u_x + d_xy_u_x - d_yy_u_y) #in inverse pixels
    piezo_unscaled = piezo_unscaled * 1e9 * 1/ss #in inverse m
    piezo_top = coef * piezo_unscaled # top layer piezocharge in C m^-2
    piezo_top = piezo_top * (6.241)  # top layer piezocharge in e nm^-2
    # piezo_top = piezo_top * (6.241e14)  # top layer piezocharge in e cm^-2
    return piezo_top

# stepsize ss is nm per pixel
# coef : piezoelectric coefficent for this material in C per m
# returns strain induced polarization in C per m for top layer
def strain_induced_polarization(u, sample_angle=0, smoothfunc=(lambda u: u), ss=1, coef=1):

    Ux = smoothfunc(u[:,:,0]) * 0.5 # want intralayer, with u = utop - ubottom = 2 * utop
    Uy = smoothfunc(u[:,:,1]) * 0.5 # measured u want utop so divide by 2 
    rotation_correction = sample_angle * np.pi/180
    def dxfunc(func):
        return np.cos(rotation_correction) * np.gradient(func, axis=1) - np.sin(rotation_correction) * np.gradient(func, axis=0) 
    def dyfunc(func):
        return np.sin(rotation_correction) * np.gradient(func, axis=1) + np.cos(rotation_correction) * np.gradient(func, axis=0) 
    # u provided is in units of pixels, dx grid in pixels, first derivative unitless
    # second derivative is in inverse pixels
    d_x_u_y = dxfunc(Uy)
    d_y_u_y = dyfunc(Uy)
    d_x_u_x = dxfunc(Ux)
    d_y_u_x = dyfunc(Ux)
    P_unscaled = np.array([0.5 * (d_x_u_y + d_y_u_x), d_x_u_x - d_y_u_y]) #unitless
    P_top = coef * P_unscaled # top layer piezocharge in C m^-1
    P_top = P_top * (6.241e9) # top layer piezocharge in e nm^-1
    # P_top * (6.241e16) # top layer piezocharge in e cm^-1
    return P_top    

# strain from simple differentiation, asume unwrapped
def strain(u, sample_angle=0, smoothfunc=(lambda u: u), ax=None, plotbool=False, norm=False):

    Ux = smoothfunc(u[:,:,0]) # want intralayer, with u = utop - ubottom = 2 * utop
    Uy = smoothfunc(u[:,:,1]) # measured u want utop so divide by 2 

    dux_dx = np.gradient(Ux * 0.5, axis=1) 
    dux_dy = np.gradient(Ux * 0.5, axis=0) 
    duy_dx = np.gradient(Uy * 0.5, axis=1)
    duy_dy = np.gradient(Uy * 0.5, axis=0)
    rotation_correction = sample_angle * np.pi/180
    print('accounting for sample rotation of {} degrees, {} rad '.format(sample_angle, rotation_correction))
    cor_dux_dx  = np.cos(rotation_correction) * dux_dx - np.sin(rotation_correction) * dux_dy 
    cor_dux_dy  = np.sin(rotation_correction) * dux_dx + np.cos(rotation_correction) * dux_dy 
    cor_duy_dx  = np.cos(rotation_correction) * duy_dx - np.sin(rotation_correction) * duy_dy 
    cor_duy_dy  = np.sin(rotation_correction) * duy_dx + np.cos(rotation_correction) * duy_dy

    exx = cor_dux_dx # dx ux
    exy = cor_dux_dy # dy ux
    eyx = cor_duy_dx # dx uy
    eyy = cor_duy_dy # dy uy

    #f,ax = plt.subplots(2,2)
    #ax[0,0].imshow(exx)
    #ax[0,1].imshow(exy)
    #ax[1,0].imshow(eyx)
    #ax[1,1].imshow(eyy)
    #plt.show()

    # u provided is in units of pixels, dx grid in pixels, first derivative unitless
    e_off = 0.5 * (exy + eyx) # this was g_xy in previous code, symmetrized off-diag (dxuy + dyux)/2

    gamma = np.zeros((exx.shape[0], exx.shape[1]))
    dil = np.zeros((exx.shape[0], exx.shape[1]))
    theta_p = np.zeros((exx.shape[0], exx.shape[1]))
    theta_t = np.zeros((exx.shape[0], exx.shape[1]))
    for i in range(exx.shape[0]):
        for j in range(exx.shape[1]):
            if np.isnan(exx[i,j]) or np.isnan(e_off[i,j]) or np.isnan(eyy[i,j]):
                gamma[i,j] = np.nan
                dil[i,j] = np.nan
                #theta_p[i,j] = np.nan
                theta_t[i,j] = np.nan
            else:
                e = np.matrix([[exx[i,j], e_off[i,j]], [e_off[i,j], eyy[i,j]]])
                v, u = np.linalg.eig(e)
                emax, emin = np.max(v), np.min(v)
                gamma[i,j] = emax - emin 
                dil[i,j] = emax + emin 
                #theta_p[i,j] = np.arctan(2*e_off[i,j]/(exx[i,j] - eyy[i,j])) * 1/2 * 180/np.pi
                theta_t[i,j] = ( eyx[i,j] - exy[i,j] ) * 180/np.pi # the curl!
    if norm:
        exx = exx/np.max(np.abs(exx.flatten()))
        exy = exy/np.max(np.abs(exy.flatten()))
        eyx = eyx/np.max(np.abs(eyx.flatten()))
        eyy = eyy/np.max(np.abs(eyy.flatten()))
    if plotbool:
        if ax is None: f, ax = plt.subplots(1,5)
        displacement_colorplot(ax[0], ux, uy)
        ax[1].imshow(exx, origin='lower')
        ax[2].imshow(exy, origin='lower')
        ax[3].imshow(eyx, origin='lower')
        ax[4].imshow(eyy, origin='lower')
    return exx, exy, eyx, eyy, gamma, theta_p, theta_t, dil

def rotate_strain(ux, uy, ang):
    R = np.matrix([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]])
    n = ux.shape[0]
    uxnew, uynew = np.zeros(ux.shape), np.zeros(uy.shape)
    for i in range(n):
        for j in range(n):
            c = R @ np.array([ux[i,j], uy[i,j]])
            uxnew[i,j], uynew[i,j] = c[0,0], c[0,1]
    return uxnew, uynew

def rotated_gradient(ux, uy, ang, eps=1e-4):
    ang += eps
    ux, uy = rotate_strain(ux, uy, ang)
    exx = np.gradient(ux, axis=1)
    exy = np.gradient(ux, axis=0)
    eyx = np.gradient(uy, axis=1)
    eyy = np.gradient(uy, axis=0)
    # elementwise multiply dU/dx * dx/dx' = dU/dx', want x'
    # dx'/dx = cos(ang) since x' = xcos(ang) - ysin(ang)
    # dx'/dy = sin(ang) since x' = xcos(ang) - ysin(ang)
    new_exx = np.multiply(exx, 1/np.cos(ang)) + np.multiply(exy, -1/np.sin(ang))
    new_eyx = np.multiply(eyx, 1/np.cos(ang)) + np.multiply(eyy, -1/np.sin(ang))
    # same, y' = xsin(ang) + ycos(ang)
    new_exy = np.multiply(exx, 1/np.sin(ang)) + np.multiply(exy, 1/np.cos(ang))
    new_eyy = np.multiply(eyx, 1/np.sin(ang)) + np.multiply(eyy, 1/np.cos(ang))
    return new_exx, new_exy, new_eyx, new_eyy

def plot_rotated_gradients(ux, uy, ang, axlist, centers=None, adjacency_type=None, thetam=0, a=1, ss=1):
    ux = ux*a/ss #ux*a: units of nm , ux*a/ss: units of ss
    uy = uy*a/ss
    exx, exy, eyx, eyy = rotated_gradient(ux, uy, ang)
    _, _, _, _, gamma, thetap, theta = strain(merge_u(ux, uy))
    print('subtracting off overall average twist angle of {} degrees = {} radians'.format(thetam, np.pi/180 * thetam))
    theta = theta - thetam  
    d = normNeighborDistance(merge_u(ux,uy), norm=False)
    data = [100*exx, 100*exy, 100*eyx, 100*eyy, 100*gamma, theta, 2*d]
    title = ['% exx (x=red)', '% exy', '% eyx', '% eyy (y=blue)', '% gamma', 'theta_r (deg)', 'mean neigh. dist (nm)']
    cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'inferno', 'RdBu_r', 'inferno']
    for i in range(len(data)):
        if cmaps[i] == 'RdBu_r': 
            lim = np.max(np.abs(data[i].flatten()))
            im = axlist[i].imshow(data[i], origin='lower', cmap=cmaps[i], vmin=-lim, vmax=lim)
        else: 
            im = axlist[i].imshow(data[i], origin='lower', cmap=cmaps[i])
        if centers is not None: plot_adjacency(None, centers, adjacency_type, ax=axlist[i], colored=False)
        plt.colorbar(im, ax=axlist[i], orientation='horizontal')
        axlist[i].set_title(title[i])
        axlist[i].axis('off')
    R = np.matrix([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]])
    for i in range(4):
        c = R @ np.array([20,0])
        axlist[i].arrow(20, 20, c[0,0], c[0,1], ec='r')
        c = R @ np.array([0,20])
        axlist[i].arrow(20, 20, c[0,0], c[0,1], ec='b')


def plot_grad(ux, uy, axlist, centers=None, adjacency_type=None, thetam=0, a=1, ss=1):
    ux = ux*a/ss 
    uy = uy*a/ss
    _, _, _, _, gamma, thetap, theta, dil = strain(merge_u(ux, uy))
    print('subracting off theta moire of ', thetam)
    theta = theta - thetam  
    data =  [100*gamma, theta]
    title = ['% gamma', 'theta_r (deg)']
    cmaps = ['inferno', 'RdBu_r']
    for i in range(len(data)):
        if cmaps[i] == 'RdBu_r': 
            lim = np.max(np.abs(data[i].flatten()))
            im = axlist[i].imshow(data[i], origin='lower', cmap=cmaps[i], vmin=-lim, vmax=lim)
        else: 
            im = axlist[i].imshow(data[i], origin='lower', cmap=cmaps[i])
        if centers is not None: plot_adjacency(None, centers, adjacency_type, ax=axlist[i], colored=False)
        plt.colorbar(im, ax=axlist[i], orientation='horizontal')
        axlist[i].set_title(title[i])
        axlist[i].axis('off')
    return theta, 100*gamma, 100*dil

def save_strain(prefix, dsnum, exx, eyy, exy, eyx, gamma, theta_t, theta_p):
    os.makedirs(os.path.join('..', 'results', prefix), exist_ok=True)
    print("saving strain to text files in ../results/{}".format(prefix))
    print("these can be imported into matlab for instance with M = readmatrix('ds1_exx.txt')")
    dump_matrix(exx,     os.path.join('..', 'results', prefix, 'ds{}_exx.txt'.format(dsnum)))
    dump_matrix(exy,     os.path.join('..', 'results', prefix, 'ds{}_exy.txt'.format(dsnum)))
    dump_matrix(eyy,     os.path.join('..', 'results', prefix, 'ds{}_eyy.txt'.format(dsnum)))
    dump_matrix(eyx,     os.path.join('..', 'results', prefix, 'ds{}_eyx.txt'.format(dsnum)))
    dump_matrix(gamma,   os.path.join('..', 'results', prefix, 'ds{}_gamma.txt'.format(dsnum)))
    dump_matrix(theta_t, os.path.join('..', 'results', prefix, 'ds{}_theta_t.txt'.format(dsnum)))
    dump_matrix(theta_p, os.path.join('..', 'results', prefix, 'ds{}_theta_p.txt'.format(dsnum)))

if __name__ == '__main__':

    sbool = boolquery("unwrap a saved dataset?")
    while sbool:
        uvecs, prefix, dsnum, isbinned = import_uvector(all=True)
        uvecs = latticevec_to_cartesian(uvecs)
        #if boolquery('crop test?'): uvecs = uvecs[0:50, 0:50, :]
        
        params = dict()
        params['remove_filter_crit']=0.1 #0.1 flag for how to nan out little erroneous chunks.
        params['centerdist']=0.01 #dont change me, flag for how much of data around the u=0 centroids to hold fix when doing the median/ip filters outwards. (does bfs on them)
        
        # CHANGE ME
        params['boundary_val']=0.3#0.4 #see below
        params['delta_val']=0.3 #P:0.3 #see below
        params['combine_crit']=0.0#5.0 #see below
        params['spdist']=2.0#5.0 #P:10.0 #see below

        params['ip']=True # dont change me, flag to use integer program.
        params['flip']=True # dont change me, flag to flip a sign in orrientation to make it spiral
        params['L1'] = False # dont change me, flag to use either L2 or L1 norms in the integer program cost function
        params['wz'] = True # dont change me, flag to work in a weigner seitz unit cell for the zones. otherwise uses conventional.
        params['ndist_crit']=0.2 #0.1 unused
        params['refit_offset']=2 #1  unuseds

        print('*****************************************************************************************************************************************')
        print('Using the following hyperparameters to find soliton walls in the data. Please edit them if automatic detection behaving poorly.')
        print("They're at lines 397-400 of strain.py.")
        print('   boundaryval={} --- threshold to find u=0 (colored black) centroids. '.format(params['boundary_val']))
        print('   deltaval={} --- threshold to find soliton walls. SPs found if angle of u is within this many radians of an expected angle'.format(params['delta_val']))
        print('   combinecrit={} --- threshold to combine nearby centroids. centers within this many pixels of eachother are merged '.format(params['combine_crit']))
        print("   spdist={} --- controls how close the u=0 centers need to be to the soliton walls (in pixels) in order to register them as 'connecting' to it. Increase for AP. ".format(params['spdist']))
        print('*****************************************************************************************************************************************')

        sanityplot = True
        if not isbinned: savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_unwrap'.format(dsnum))
        else: savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_binunwrap'.format(dsnum))
        counter = 1
        while os.path.exists(savepath):
            if not isbinned: savepath = os.path.join('..','results', prefix, 'datredo{}_ds{}.pkl_unwrap'.format(counter, dsnum))
            else: savepath = os.path.join('..','results', prefix, 'datredo{}_ds{}.pkl_binunwrap'.format(counter, dsnum))
            counter += 1
        tic()
        #u, centers, adjacency_type = unwrap_multigrid(uvecs.copy(), params=params) 
        u, centers, adjacency_type = unwrap(uvecs.copy(), params=params, plotbool=sanityplot)
        toc('unwrap')
        print("saving as {}...".format(savepath))
        with open(savepath, 'wb') as f: pickle.dump([u, centers, adjacency_type, params], f) # save unwrapped
        sbool = boolquery("unwrap another saved dataset?")
    
    sbool = boolquery("differentiate a saved unwrapped dataset?")
    while sbool:

        if False:
            def gammaf(exx, exy, eyx, eyy):
                e_off = 0.5*(exy+eyx)
                nx, ny = exx.shape[0], exx.shape[1]
                gamma = np.zeros((nx, ny))
                theta = np.zeros((nx, ny))
                for i in range(exx.shape[0]):
                    for j in range(exx.shape[1]):
                        e = np.matrix([[exx[i,j], e_off[i,j]], [e_off[i,j], eyy[i,j]]])
                        v, u = np.linalg.eig(e)
                        emax, emin = np.max(v), np.min(v)
                        gamma[i,j] = emax - emin 
                        theta[i,j] = ( exy[i,j] - eyx[i,j] ) * 180/np.pi
                return gamma, theta
            def nanfilt(m):
                mean = np.mean(m)
                s = np.std(m)
                z = np.abs(m - mean)/s
                m[z > 1] = np.nan 
                return m
            sigma = 1
            u, prefix, dsnum, centers, adjacency_type = import_unwrap_uvector()
            exx, exy, eyx, eyy, _ = wrapNeighborStrain(u, ss=1, extend=True, eno=False, forward=True)
            exx, exy, eyx, eyy = -exx, exy,-eyx, eyy
            gamma, theta = gammaf(exx, exy, eyx, eyy)
            gamma, theta = nanfilt(gamma), nanfilt(theta)
            f, ax = plt.subplots(2,2); 
            lim = np.max(np.abs(exx.flatten()))
            ax[0,1].imshow(theta, origin='lower', cmap='RdBu_r');
            gamma = nan_gaussian_filter(gamma, sigma)  
            theta = nan_gaussian_filter(theta, sigma) 
            ax[1,1].imshow(theta, origin='lower', cmap='RdBu_r');
            plt.show(); 
            exit();
        
        u, prefix, dsnum, centers, adjacency_type = import_unwrap_uvector()
        d = normNeighborDistance(u, norm=False)
        xmin, xmax, ymin, ymax = manual_cropping(d, vmax=0.1)
        u = u[xmin:xmax, ymin:ymax, :]
        centers = [[c[0]-xmin, c[1]-ymin] for c in centers]
        ux = trim_to_equal_dim(u[:,:,0], even=True)
        uy = trim_to_equal_dim(u[:,:,1], even=True)

        uxraw, uyraw = ux.copy(), uy.copy()
        sigma = 2
        offset = 0
        ux = nan_gaussian_filter(ux, sigma) #ndimage.gaussian_filter(ux, sigma=sigma)  
        uy = nan_gaussian_filter(uy, sigma) #ndimage.gaussian_filter(uy, sigma=sigma)  
        f, ax = plt.subplots(3,8)  
        thetam, hetstrain, a, ss = hetstrain_from_adjacencies(centers, (adjacency_type > 0), ax[1,0], ax[2,0])  
        img = displacement_colorplot(None, ux, uy)
        plot_adjacency(img, centers, adjacency_type, ax=ax[0,0], colored=False)
        ax[0,0].axis('off')
        plot_rotated_gradients(ux, uy, offset, ax[0,1:], centers, adjacency_type, thetam, a, ss)
        plot_rotated_gradients(ux, uy, offset + 2/3*np.pi, ax[1,1:], centers, adjacency_type, thetam, a, ss)
        plot_rotated_gradients(ux, uy, offset + 4/3*np.pi, ax[2,1:], centers, adjacency_type, thetam, a, ss)
        plt.show()

        #f, ax = plt.subplots(2,3)
        #displacement_colorplot(ax[0,0], ux, uy)
        #displacement_colorplot(ax[1,0], uxraw, uyraw)
        #plot_grad(ux, uy, ax[0,1:], centers, adjacency_type, thetam, a, ss)
        #plot_grad(uxraw, uyraw, ax[1,1:], centers, adjacency_type, thetam, a, ss)
        #plt.show()

        sbool = boolquery("strain another saved dataset?")

       
        
        
