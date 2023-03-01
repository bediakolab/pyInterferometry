
from visualization import plot_voroni, plot_adjacency, displacement_colorplot, displacement_colorplot_lvbasis, make_legend, threshold_plot, plot_contour    
from interferometry_fitting import fit_u, fit_ABC
import matplotlib
import numpy as np
import os # for path functionality
import matplotlib.pyplot as plt
import pickle
from basis_utils import rotate_uvecs, cartesian_to_latticevec, latticevec_to_cartesian, lv_to_rzlv, flip_to_reference, rz_helper, cartesian_to_rzcartesian, cartesian_to_rz_WZ, adjust_zone, cart_to_zonebasis, zonebasis_to_cart     
from strain_utils import differentiate, differentiate_lv 
from strain import strain, plot_rotated_gradients  #unwrap
import scipy.ndimage as ndimage
from masking import get_aa_mask, get_sp_masks, get_sp_line_method2, get_region_centers
from utils import get_triangles, bin, tic, toc
from unwrap_utils import automatic_sp_rotation, geometric_unwrap, getAdjacencyMatrixAuto, refit_outliers, rzSignAlign, unwrapBFS, getAdjacencyMatrix, voronoi_finite_polygons_2d, normDistToNearestCenter, normNeighborDistance, median_fit, mip_fit, BFS_from_center_uwrap
from utils import merge_u, trim_to_equal_dim, read_excel, nan_gaussian_filter, getAdjacencyMatrixManual, get_AB_centers 
from new_utils import import_uvector


def NEW_TEST(theta_layer1, theta_layer2):

    delta = -5 / 100
    pr = 0.16 # graphene
    epsilon = 0 / 100
    theta_s = 0.0 * np.pi/180
    a0 = 1
    N = 100
    twist = theta_layer1 + theta_layer2 
    print('lattice mismatch is {} %'.format(100*delta))
    print('twist angle is {} deg'.format(twist*180/np.pi))
    expected_lambda = a0 * (1+delta) / np.sqrt(delta**2 + 2*(1+delta)*(1-np.cos(twist)))
    print('moire wavelength is {} times cell width'.format(expected_lambda/N))
    print('{} moire units per dimension '.format(N/expected_lambda))

    from moire_sandbox_2 import heterostrain, multiply_three_matrices, get_u_XY_grid, Rmat
    from diskset import DiskSet

    heterobilayer_mat = np.matrix([[1+delta, 0], [0, 1+delta]])
    HS_mat = heterostrain(epsilon, pr, theta_s)
    Transformation = multiply_three_matrices(Rmat(theta_layer2), heterobilayer_mat, HS_mat)
    print(Transformation)
    
    X, Y, Ux, Uy = get_u_XY_grid(Transformation, theta_layer1, N, plotflag=False)
    uvecs_cart = np.zeros((Ux.shape[0], Ux.shape[1],2))
    uvecs_cart[:,:,0], uvecs_cart[:,:,1] = Ux[:,:], Uy[:,:]
    uvecs_rz   = cartesian_to_rzcartesian(uvecs_cart, sign_wrap=False)
    uvecs_rzlv = cartesian_to_latticevec(uvecs_rz)
    uvecs_lv   = cartesian_to_latticevec(uvecs_cart)

    f, ax = plt.subplots(6,4)
    ax = ax.reshape((12,2))
    nx, ny = N, N
    ds = DiskSet(12, N, N)
    gvecs = [ [1,0], [0,1], [1,1], [1,-1], [-1,1], [-1,-1], [2,-1], [-2,1], [1,-2], [-1,2], [-1,0], [0,-1]]
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs = np.array(gvecs)
    ndisks = 12
    coefs = np.zeros((ndisks, 3)) #abc so 3
    coefs[:, 0] = np.ones(ndisks)
    coefs[:, 1] = 0.0 * np.ones(ndisks)
    coefs[:, 2] = 0.0 * np.ones(ndisks)
    I = np.zeros((ndisks, nx, ny))
    for disk in range(ndisks):
        for i in range(nx):
            for j in range(ny):
                gdotu = np.dot(gvecs[disk], np.array([uvecs_lv[i,j,0], uvecs_lv[i,j,1]]))
                I[disk, i, j] = coefs[disk,0] * np.cos(np.pi * gdotu)**2 
        ds.set_df(disk, I[disk,:,:])

    f, axes = plt.subplots(4, 3)
    axes = axes.flatten()
    for n in range(ndisks):
        img = I[n,:,:]
        axes[n].imshow(img, cmap='gray')
        axes[n].set_title("Disk {}{}".format(gvecs[n][0],gvecs[n][1]))
        axes[n].axis("off")
    plt.show()
    #exit()

    # unit test with raw intensities
    # testing the fit code to make sure it gives the right uvecs when NOT provided with perfect inital guesses
    fitcheck = False
    if fitcheck:
        ufit_lv, residuals = fit_u(I, coefs, nx, ny, gvecs, guess=None, parallel=True, nproc=12, norm_bool=True, multistart_bool=True, multistart_neighbor_bool=False)
        for n in range(1):
            coefs, resid = fit_ABC(I, ufit_lv, nx, ny, gvecs, coefs, nproc=12, parallel=False, norm_bool=True)
            ufit_lv, resid = fit_u(I, coefs, nx, ny, gvecs, guess=ufit_lv, nproc=12, parallel=True, norm_bool=True, multistart_bool=False)
        for n in range(1):
            coefs, resid = fit_ABC(I, ufit_lv, nx, ny, gvecs, coefs, nproc=12, parallel=False, norm_bool=True)
            ufit_lv, resid = fit_u(I, coefs, nx, ny, gvecs, nproc=12, guess=ufit_lv, parallel=True, norm_bool=True, multistart_neighbor_bool=True)
        ufit_lv = lv_to_rzlv(ufit_lv, sign_wrap=True)
        u = latticevec_to_cartesian(ufit_lv)
        f, ax = plt.subplots(1,2)
        displacement_colorplot(ax[0], u)
        displacement_colorplot(ax[1], uvecs_cart)
        plt.show()
        Ux, Uy = u[:,:,0], u[:,:,1]
    else:
        u = uvecs_cart
        f, ax = plt.subplots()
        displacement_colorplot(ax, u)
        Ux, Uy = u[:,:,0], u[:,:,1]
        plt.show()

    rotation_correction = - (theta_layer1 + theta_layer2/2)
    print('rotation_correction of ', rotation_correction)
    dux_dx, dux_dy, duy_dx, duy_dy = np.gradient(Ux * 0.5, axis=1), np.gradient(Ux * 0.5, axis=0), np.gradient(Uy * 0.5, axis=1), np.gradient(Uy * 0.5, axis=0)
    cor_dux_dx  = np.cos(rotation_correction) * dux_dx - np.sin(rotation_correction) * dux_dy 
    cor_dux_dy  = np.sin(rotation_correction) * dux_dx + np.cos(rotation_correction) * dux_dy 
    cor_duy_dx  = np.cos(rotation_correction) * duy_dx - np.sin(rotation_correction) * duy_dy 
    cor_duy_dy  = np.sin(rotation_correction) * duy_dx + np.cos(rotation_correction) * duy_dy 

    theta_tot = 180/np.pi * (cor_dux_dy-cor_duy_dx) 
    #print('expect {}'.format(0.5 * (1+delta) * (epsilon*(pr-1) - 2) * np.sin(twist)))
    expect_cor_dux_dy = (2+delta-pr*epsilon-pr*epsilon*delta)/2 * np.sin(theta_layer2/2)
    expect_cor_duy_dx = -(2+delta+epsilon+epsilon*delta)/2 * np.sin(theta_layer2/2)
    print('expect {}'.format(round(180/np.pi * (expect_cor_dux_dy-expect_cor_duy_dx),4)))
    approx_hs = ( 4-pr*epsilon+epsilon ) * theta_layer2/4
    approx_hbl = ( 2+delta ) * theta_layer2/2
    print('approx expect hs {}'.format(round(180/np.pi * (approx_hs),4)))
    print('approx expect hbl {}'.format(round(180/np.pi * (approx_hbl),4)))
    print('got {}'.format(round((theta_tot[N//2,N//2]),4)))
   
    dil = 100 * (cor_dux_dx+cor_duy_dy)
    #print('expect {}'.format(0.5 * (1+delta) * (2 - epsilon*(pr-1)) * np.cos(twist) - 1))
    expect_cor_dux_dx = (delta+epsilon+delta*epsilon)/2 * np.cos(theta_layer2/2)
    expect_cor_duy_dy = (delta-pr*epsilon-pr*delta*epsilon)/2 * np.cos(theta_layer2/2)
    print('expect {}'.format(round(100*(expect_cor_dux_dx+expect_cor_duy_dy),4)))
    approx_hs = (epsilon-pr*epsilon)/2
    approx_hbl = delta
    print('approx expect hs {}'.format(round(100*(approx_hs),4)))
    print('approx expect hbl {}'.format(round(100*(approx_hbl),4)))
    print('got {}'.format(round(100*(dil[N//2,N//2]/100),4)))

    f, ax = plt.subplots(1,2)
    ax[0].imshow(theta_tot)
    ax[1].imshow(dil)
    plt.show()

    savepath = '../data/fake_uvecs.pkl'
    print(u.shape)
    print(u[0,0,0])
    with open(savepath, 'wb') as f: 
        pickle.dump( [u], f )
    exit()

    exit()
  

###########################################################################
# utility assertion functions 
###########################################################################
def assert_vec_noteq(v1, v2, tol=1e-8): 
    d = np.max([np.abs(e1-e2) for e1, e2 in zip(v1, v2)])
    if (d < tol):
        print('ERROR: {} is {}'.format(v1, v2))
        exit()

def assertnoteq(v1, v2, tol=1e-8): 
    if (np.abs(v1-v2) < tol):
        print('ERROR: {} is {}'.format(v1, v2))
        exit()

def assert_vec_eq(v1, v2, tol=1e-8): 
    for e1, e2 in zip(v1, v2): asserteq(e1, e2, tol)

def asserteq(v1, v2, tol=1e-8): 
    if (np.abs(v1-v2) > tol):
        print('ERROR: {} isnt {}'.format(v1, v2))
        exit()

def rotation_test():
    xrange = np.arange(-1.5, 1.2, 0.05) 
    strainterms = [0.12, 0.11, 0.11, 0.12]
    boundary_val = 0.15 # AA find threshold
    delta_val = 0.1 # SP angular threshold
    combine_crit = 1.0 # combine if less than this dist away
    spdist = 5.0 # how close are AA centers to a SP line to say the line passes through the center 
    centerdist = 0.10 # how far from AA center need to be to unconstrain - stuff next to AA are unwrapped well and held fixed
    manual = False
    pad = 5   
    u, u_unwrap = make_fake_disps(xrange, strainterms) # make fake data   
    testangs = [0, 1/3*np.pi, -1/3*np.pi, 2/3*np.pi, -2/3*np.pi]
    f, ax = plt.subplots(2,len(testangs))
    for i in range(len(testangs)):
        u  = rotate_uvecs(u,  ang=testangs[i])   
        displacement_colorplot(ax[0,i], u)
        centers, adjacency_type = getAdjacencyMatrix(u, manual, boundary_val, delta_val, combine_crit, spdist)
        urot, ang, _ = automatic_sp_rotation(u,  centers, adjacency_type, transpose=True)
        ax[1,i].set_title('ang={}'.format(ang))
        displacement_colorplot(ax[1,i], urot)   
    plt.show()

###########################################################################
# tests
##########################################################################   
def tblg_unwrap_test(ds=4, ses=1):

    data_path = os.path.join('..','unittest-data','NK-tblg-uvecs')
    fx = os.path.join(data_path, 'DS{}_session{}_xdisplacements_beforeunwrapping.xlsx'.format(ds, ses))
    fy = os.path.join(data_path, 'DS{}_session{}_ydisplacements_beforeunwrapping.xlsx'.format(ds, ses))
    savepath = os.path.join(data_path, 'DS{}_session{}_unwrapped.pkl'.format(ds, ses))

    if False: #os.path.exists(savepath): 

        print('reading')
        with open(savepath, 'rb') as f: u_unwrap = pickle.load(f)

        ux, uy = read_excel(fx), read_excel(fy)
        ux, uy = trim_to_equal_dim(ux, even=True), trim_to_equal_dim(uy, even=True)
        u = np.zeros((ux.shape[0], ux.shape[1], 2))
        assert(u.shape[0] == u.shape[1])

        # NK code had displacement vectors in angstroms, this code expects displacement vectors in units of a_0
        # a_0 for graphene is 2.46 angs
        u[:,:,0], u[:,:,1] = ux, uy 
        u = u / 2.46
        u = rotate_uvecs(u, ang=2/3*np.pi) # rotate to my convention where want sp1(c) along y axis
        u = u[:160, :160, :] # only look at smaller portion to start

    else:
        ux, uy = read_excel(fx), read_excel(fy)
        ux, uy = trim_to_equal_dim(ux, even=True), trim_to_equal_dim(uy, even=True)
        u = np.zeros((ux.shape[0], ux.shape[1], 2))
        assert(u.shape[0] == u.shape[1])

        # NK code had displacement vectors in angstroms, this code expects displacement vectors in units of a_0
        # a_0 for graphene is 2.46 angs
        u[:,:,0], u[:,:,1] = ux, uy 
        u = u / 2.46
        u = rotate_uvecs(u, ang=2/3*np.pi) # rotate to my convention where want sp1(c) along y axis
        #u = u[:160, :160, :] # only look at smaller portion to start
        f, ax = plt.subplots(1,2); ax[0].quiver(u[:,:,0], u[:,:,1]); displacement_colorplot(ax[1], u, quiverbool=False); plt.show();

        # basis/plotting sanity checks
        if False:
            f, ax = plt.subplots(2,2)
            u_lv = cartesian_to_latticevec(u)
            rz_lv = lv_to_rzlv(u_lv)
            rz_cart = cartesian_to_rzcartesian(u)
            displacement_colorplot(ax[0,0], u, quiverbool=False)
            displacement_colorplot_lvbasis(ax[0,1], u_lv)
            displacement_colorplot(ax[1,0], rz_cart, quiverbool=False)
            displacement_colorplot_lvbasis(ax[1,1], rz_lv)
            print(np.max(np.abs(u.flatten())))
            print(np.max(np.abs(u_lv.flatten())))
            plt.show()

        # hyper parameters
        boundary_val = 0.45
        delta_val = 0.1
        combine_crit = 10.0
        spdist = 5.0
        centerdist = 0.1
        manual = False
        
        # unwrap and strain
        u = unwrap(u, manual=manual, ip=False, centerdist=centerdist, boundary_val=boundary_val, delta_val=delta_val, combine_crit=combine_crit, spdist=spdist)
        with open(savepath, 'wb') as f: pickle.dump(u, f) # save unwrapped

    pad = 25
    if False:
        f, ax = plt.subplots(1,3)
        d = normNeighborDistance(u, norm=False)
        ax[0].imshow(d, origin='lower') 
        d = normNeighborDistance(u_unwrap, norm=False)
        ax[1].imshow(d, origin='lower') 
        d = normNeighborDistance(u_unwrap[pad:-pad, pad:-pad, :], norm=False)
        ax[2].imshow(d, origin='lower') 
        plt.show(); 
    
    nk_filter, myfilter = False, True

    if nk_filter: 
        u = u_unwrap
        u = u[pad:-pad, pad:-pad, :]
        ux = bin(u[:,:,0], bin_w=2, size_retain=True, method=np.nanmedian)
        uy = bin(u[:,:,1], bin_w=2, size_retain=True, method=np.nanmedian)
        from skimage.restoration import denoise_tv_bregman
        ux, uy = denoise_tv_bregman(ux, weight=10), denoise_tv_bregman(uy, weight=10)
        u = merge_u(ux, uy)
        pad = 10
        u = u[pad:-pad, pad:-pad, :]

    elif myfilter:
        u = u_unwrap
        u = u[pad:-pad, pad:-pad, :]
        ux = bin(u[:,:,0], bin_w=2, size_retain=True, method=np.nanmedian)
        uy = bin(u[:,:,1], bin_w=2, size_retain=True, method=np.nanmedian)
        ux = ndimage.gaussian_filter(ux, sigma=2)  
        uy = ndimage.gaussian_filter(uy, sigma=2)  
        u = merge_u(ux, uy)
        pad = 10
        u = u[pad:-pad, pad:-pad, :]

    f, ax = plt.subplots(3,8)    
    plot_rotated_gradients(ux, uy, -0.05, ax[0,:])
    plot_rotated_gradients(ux, uy, -0.05 + 2/3*np.pi, ax[1,:])
    plot_rotated_gradients(ux, uy, -0.05 + 4/3*np.pi, ax[2,:])
    plt.show()

    exit()

def mip_fit_unit_test(plotting=True):

    n = Nx = Ny = 4
    fc1 = np.matrix([[ 0.41051403,  0.44712022,  0.48334049, -0.4810062 ],[ 0.46525198,  0.49814183,  0.46192155,  0.42626824],[ 0.47920058,  0.44259439,  0.40637412,  0.37072081],[ 0.42298027,  0.38637408,  0.3501538,   0.31450049]])
    fc2 = np.matrix([[-0.43853248, -0.40193991, -0.364293,   -0.32509712],[-0.45147681,  0.41488424,  0.37723733,  0.33804145],[ 0.46663268,  0.43004011,  0.3923932,   0.35319732],[ 0.4836269,   0.44703433,  0.40938742,  0.37019154]])
    variable_region =  np.matrix([[0., 0., 0., 0.],[0., 1., 1., 0.],[0., 1., 1., 0.],[0., 0., 0., 0.]])
    n =  np.matrix([[0., 0., 0., 1.],[0., 0., 0., 0.],[0., 0., 0., 0.],[0., 0., 0., 0.]])
    m =  np.matrix([[1., 1., 1., 1.],[1., 0., 0., 0.],[0., 0., 0., 0.],[0., 0., 0., 0.]])
    sign = np.ones((fc1.shape[0], fc1.shape[1])) 
    def neigbor_dist_penalty(Nx, Ny, n_mat, m_mat, s_mat, variable_region):
        def help_get_u(x, y, n_mat, m_mat, s_mat):
            vmat = np.matrix([[-1, 1/2],[0, np.sqrt(3)/2]])
            if variable_region[x,y]:
                v1 = (2 * s_mat[x][y] - 1)*fc1[x,y]+n_mat[x][y]
                v2 = (2 * s_mat[x][y] - 1)*fc2[x,y]+m_mat[x][y]
                return np.matmul(vmat, np.array([v1, v2]))
            else: 
                v1 = sign[x,y]*fc1[x,y]+n[x,y]
                v2 = sign[x,y]*fc2[x,y]+m[x,y]
                return np.matmul(vmat, np.array([v1, v2]))
        tot_pen = 0
        count = 0
        for x in range(Nx):
            for y in range(Ny):
                u_c = help_get_u(x, y, n_mat, m_mat, s_mat)
                if y < Ny-1: # add sq euclidean distance from bottom neighbor
                    u_b = help_get_u(x, y+1, n_mat, m_mat, s_mat)
                    tot_pen += np.abs(u_c[0,0] - u_b[0,0]) + np.abs(u_c[0,1] - u_b[0,1])
                    count += 1
                if x < Nx-1: # add sq euclidean distance from right neighbor
                    u_r = help_get_u(x+1, y, n_mat, m_mat, s_mat)
                    tot_pen += np.abs(u_c[0,0] - u_r[0,0]) + np.abs(u_c[0,1] - u_r[0,1])
                    count += 1
        return tot_pen
    n_mat = [[n[i,j] for j in range(Ny)] for i in range(Nx)]
    m_mat = [[m[i,j] for j in range(Ny)] for i in range(Nx)]
    s_mat = [[1 for j in range(Ny)] for i in range(Nx)]
    pen = neigbor_dist_penalty(Nx, Ny, n_mat, m_mat, s_mat, variable_region=np.ones((Nx,Ny)))
    pen2 = neigbor_dist_penalty(Nx, Ny, n_mat, m_mat, s_mat, variable_region=np.zeros((Nx,Ny)))
    asserteq(pen, pen2)
    u_cart  = zonebasis_to_cart(fc1, fc2, n, m, sign) # should capture all, allowed moves
    ufit, r, _, _ = mip_fit(u_cart, variable_region, plotting=True, verbose=True)

    # case that failed
    n = 3
    fc1 = np.matrix([[-0.08393319, -0.11885245, -0.15288864], [-0.13948063, -0.17439989, -0.20843608], [-0.19421858, -0.22913784, -0.26317403]]) 
    fc2 =  np.matrix([[-0.49748517,  0.46131348,  0.41769954], [-0.4823293,   0.47646935,  0.43285541], [-0.46938497,  0.48941368,  0.44579975]])  
    variable_region =  np.matrix([[0., 0., 0.], [1., 1., 0.], [1., 1., 1.]])
    n =  np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])                               
    m =  np.matrix([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])                              
    signs = np.ones((fc1.shape[0], fc1.shape[1])) 
    u_cart  = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
    u_cart[1,1] = np.nan, np.nan
    variable_region[1,1] = 0
    ufit, r, _, _ = mip_fit(u_cart, variable_region, plotting=True, verbose=True)
    exit()

    n = 4
    variable_region = np.zeros((n,n))
    variable_region[1,1] = variable_region[1,2] = variable_region[2,1] = variable_region[2,2] = 1
    xrange = np.arange(1.0, 1.0+0.05*(n+1), 0.05)
    x, y = np.meshgrid(xrange, xrange)
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            scale = 0 #0.05
            noisex, noisey = np.random.normal(0.0, 1.0) * scale, np.random.normal(0.0, 1.0) * scale
            u_cart[i,j,0]  =  y[i,j]  + 0.05 * np.cos(np.pi*x[i,j]) + 0.05 * np.sin(np.pi*y[i,j]) + noisex
            u_cart[i,j,1]  = -x[i,j]  + 0.10 * np.sin(np.pi*y[i,j]) + 0.10 * np.cos(np.pi*x[i,j]) + noisey
    u_cart_expect = cartesian_to_rzcartesian(u_cart, sign_wrap=False)
    fc1, fc2, n, m = cart_to_zonebasis(u_cart_expect)
    signs = np.ones((fc1.shape[0], fc1.shape[1])) 
    n[2,2], signs[2,2] = n[2,2]+1, -1
    m[1,2], signs[1,2] = m[1,2]-1, -1
    n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1]+1, -1 
    u_cart  = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
    ufit, r, _, _ = mip_fit(u_cart, variable_region, plotting=plotting)
    assert_vec_eq(u_cart_expect[2,2], ufit[2,2])
    assert_vec_eq(u_cart_expect[1,2], ufit[1,2])
    assert_vec_eq(u_cart_expect[1,1], ufit[1,1])

    n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1], 1
    u_cart  = zonebasis_to_cart(fc1, fc2, n, m, signs) # should miss [1,1] since n+1,s=1 not allowed
    ufit, r, _, _ = mip_fit(u_cart, variable_region, plotting=plotting)
    assert_vec_eq(u_cart_expect[2,2], ufit[2,2])
    assert_vec_eq(u_cart_expect[1,2], ufit[1,2])
    assert_vec_eq(u_cart_expect[1,1], ufit[1,1])

    n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1]-1, -1
    u_cart  = zonebasis_to_cart(fc1, fc2, n, m, signs) # should miss [1,1] since n+1,m-1 not allowed
    ufit, r, _, _ = mip_fit(u_cart, variable_region, plotting=plotting)
    assert_vec_eq(u_cart_expect[2,2], ufit[2,2])
    assert_vec_eq(u_cart_expect[1,2], ufit[1,2])
    assert_vec_eq(u_cart_expect[1,1], ufit[1,1])
    print('IP fit passes with N=4, needed to flip 3 cells, correctly handles sign constraints')

    n = 3
    variable_region = np.zeros((n,n))
    variable_region[1,1] = variable_region[1,2] = variable_region[2,1] = variable_region[2,2] = 1
    xrange = np.arange(1.0, 1.0+0.05*(n+1), 0.05)
    x, y = np.meshgrid(xrange, xrange)
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            scale = 0 #0.05
            noisex, noisey = np.random.normal(0.0, 1.0) * scale, np.random.normal(0.0, 1.0) * scale
            u_cart[i,j,0]  =  y[i,j]  + 0.05 * np.cos(np.pi*x[i,j]) + 0.05 * np.sin(np.pi*y[i,j]) + noisex
            u_cart[i,j,1]  = -x[i,j]  + 0.10 * np.sin(np.pi*y[i,j]) + 0.10 * np.cos(np.pi*x[i,j]) + noisey
    u_cart_expect = cartesian_to_rzcartesian(u_cart, sign_wrap=False)
    fc1, fc2, n, m = cart_to_zonebasis(u_cart_expect)
    signs = np.ones((fc1.shape[0], fc1.shape[1])) 
    n[2,2], signs[2,2] = n[2,2]+1, -1
    m[1,2], signs[1,2] = m[1,2]-1, -1
    n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1]+1, -1 
    u_cart  = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
    ufit, r, _, _ = mip_fit(u_cart, variable_region, plotting=plotting)
    assert_vec_eq(u_cart_expect[2,2], ufit[2,2])
    assert_vec_eq(u_cart_expect[1,2], ufit[1,2])
    assert_vec_eq(u_cart_expect[1,1], ufit[1,1])
    print('IP fit passes with N=3, needed to flip 3 cells')

    n = 3
    variable_region = np.zeros((n,n))
    variable_region[1:3,:] = 1
    xrange = np.arange(1.0, 1.0+0.05*(n+1), 0.05)
    x, y = np.meshgrid(xrange, xrange)
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            scale = 0 #0.05
            noisex, noisey = np.random.normal(0.0, 1.0) * scale, np.random.normal(0.0, 1.0) * scale
            u_cart[i,j,0]  =  y[i,j]  + 0.05 * np.cos(np.pi*x[i,j]) + 0.05 * np.sin(np.pi*y[i,j]) + noisex
            u_cart[i,j,1]  = -x[i,j]  + 0.10 * np.sin(np.pi*y[i,j]) + 0.10 * np.cos(np.pi*x[i,j]) + noisey
    u_cart_expect = cartesian_to_rzcartesian(u_cart, sign_wrap=False)
    fc1, fc2, n, m = cart_to_zonebasis(u_cart_expect)
    signs = np.ones((fc1.shape[0], fc1.shape[1])) 
    n[2,2], signs[2,2] = n[2,2]+1, -1
    m[1,2], signs[1,2] = m[1,2]-1, -1
    n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1]+1, -1 
    n[2,1], signs[2,1] = n[2,1]+1, -1
    m[1,0], signs[1,0] = m[1,0]-1, -1
    u_cart = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
    ufit, r, _, _ = mip_fit(u_cart, variable_region, plotting=plotting)
    for x,y in zip([2,1,1,2,1], [2,2,1,1,0]): assert_vec_eq(u_cart_expect[x,y], ufit[x,y]) 
    print('IP fit passes with N=2, needed to flip 5 cells')

    for N in [5,6,7,8]:
        variable_region = np.zeros((N,N))
        variable_region[1:4,1:4] = 1
        xrange = np.arange(1.0, 1.0+0.05*(N+1), 0.05)
        x, y = np.meshgrid(xrange, xrange)
        u_cart = np.zeros((N, N, 2))
        for i in range(N):
            for j in range(N):
                scale = 0 #0.05
                noisex, noisey = np.random.normal(0.0, 1.0) * scale, np.random.normal(0.0, 1.0) * scale
                u_cart[i,j,0]  =  y[i,j]  + 0.05 * np.cos(np.pi*x[i,j]) + 0.05 * np.sin(np.pi*y[i,j]) + noisex
                u_cart[i,j,1]  = -x[i,j]  + 0.10 * np.sin(np.pi*y[i,j]) + 0.10 * np.cos(np.pi*x[i,j]) + noisey
        u_cart_expect = cartesian_to_rzcartesian(u_cart, sign_wrap=False)
        fc1, fc2, n, m = cart_to_zonebasis(u_cart_expect)
        signs = np.ones((fc1.shape[0], fc1.shape[1])) 
        n[2,2], signs[2,2] = n[2,2]+1, -1
        m[1,2], signs[1,2] = m[1,2]-1, -1
        n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1]+1, -1 
        n[3,1], m[3,1], signs[3,1] = n[3,1]-1, m[3,1]-1, -1
        n[3,2], signs[3,2] = n[3,2]+1, -1
        m[1,3], signs[1,3] = m[1,3]-1, -1
        m[2,3], signs[2,3] = m[2,3]+1, -1
        n[3,3], m[3,3], signs[3,3] = n[3,3]-1, m[3,3]-1, -1
        u_cart = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
        ufit, r, _, _ = mip_fit(u_cart, variable_region, plotting=plotting)
        for x,y in zip([2,1,1,3,3,1,2,3], [2,2,1,1,2,3,3,3]): assert_vec_eq(u_cart_expect[x,y], ufit[x,y]) 
        print('IP fit passes with N={}, needed to flip 8 cells'.format(N))

def median_fit_unit_test():

    n = 4
    variable_region = np.zeros((n,n))
    variable_region[2,2] = 1
    xrange = np.arange(1.0, 1.0+0.05*(n+1), 0.05)
    x, y = np.meshgrid(xrange, xrange)
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            scale = 0 #0.05
            noisex, noisey = np.random.normal(0.0, 1.0) * scale, np.random.normal(0.0, 1.0) * scale
            u_cart[i,j,0]  =  y[i,j]  + 0.05 * np.cos(np.pi*x[i,j]) + 0.05 * np.sin(np.pi*y[i,j]) + noisex
            u_cart[i,j,1]  = -x[i,j]  + 0.10 * np.sin(np.pi*y[i,j]) + 0.10 * np.cos(np.pi*x[i,j]) + noisey
    u_cart_expect = cartesian_to_rzcartesian(u_cart, sign_wrap=False)
    fc1, fc2, n, m = cart_to_zonebasis(u_cart_expect)
    signs = np.ones((fc1.shape[0], fc1.shape[1])) 
    n[2,2], signs[2,2] = n[2,2]+1, -1
    u_cart  = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
    ufit, r = median_fit(u_cart, variable_region, plotting=True)
    plt.show()
    assert_vec_eq(u_cart_expect[2,2], ufit[2,2])
    print('median fit passes with N=4, needed to flip 1 cell')

    n = 3
    variable_region = np.zeros((n,n))
    variable_region[1,1] = variable_region[1,2] = variable_region[2,1] = variable_region[2,2] = 1
    xrange = np.arange(1.0, 1.0+0.05*(n+1), 0.05)
    x, y = np.meshgrid(xrange, xrange)
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            scale = 0 #0.05
            noisex, noisey = np.random.normal(0.0, 1.0) * scale, np.random.normal(0.0, 1.0) * scale
            u_cart[i,j,0]  =  y[i,j]  + 0.05 * np.cos(np.pi*x[i,j]) + 0.05 * np.sin(np.pi*y[i,j]) + noisex
            u_cart[i,j,1]  = -x[i,j]  + 0.10 * np.sin(np.pi*y[i,j]) + 0.10 * np.cos(np.pi*x[i,j]) + noisey
    u_cart_expect = cartesian_to_rzcartesian(u_cart, sign_wrap=False)
    fc1, fc2, n, m = cart_to_zonebasis(u_cart_expect)
    signs = np.ones((fc1.shape[0], fc1.shape[1])) 
    n[2,2], signs[2,2] = n[2,2]+1, -1
    m[1,2], signs[1,2] = m[1,2]-1, -1
    n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1]+1, -1 
    u_cart  = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
    ufit, r = median_fit(u_cart, variable_region, plotting=True)
    assert_vec_eq(u_cart_expect[2,2], ufit[2,2])
    assert_vec_eq(u_cart_expect[1,2], ufit[1,2])
    assert_vec_eq(u_cart_expect[1,1], ufit[1,1])
    print('median fit passes with N=3, needed to flip 3 cells')

    n = 3
    variable_region = np.zeros((n,n))
    variable_region[1:3,:] = 1
    xrange = np.arange(1.0, 1.0+0.05*(n+1), 0.05)
    x, y = np.meshgrid(xrange, xrange)
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            scale = 0 #0.05
            noisex, noisey = np.random.normal(0.0, 1.0) * scale, np.random.normal(0.0, 1.0) * scale
            u_cart[i,j,0]  =  y[i,j]  + 0.05 * np.cos(np.pi*x[i,j]) + 0.05 * np.sin(np.pi*y[i,j]) + noisex
            u_cart[i,j,1]  = -x[i,j]  + 0.10 * np.sin(np.pi*y[i,j]) + 0.10 * np.cos(np.pi*x[i,j]) + noisey
    u_cart_expect = cartesian_to_rzcartesian(u_cart, sign_wrap=False)
    fc1, fc2, n, m = cart_to_zonebasis(u_cart_expect)
    signs = np.ones((fc1.shape[0], fc1.shape[1])) 
    n[2,2], signs[2,2] = n[2,2]+1, -1
    m[1,2], signs[1,2] = m[1,2]-1, -1
    n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1]+1, -1 
    n[2,1], signs[2,1] = n[2,1]+1, -1
    m[1,0], signs[1,0] = m[1,0]-1, -1
    u_cart = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
    ufit, r = median_fit(u_cart, variable_region, plotting=False)
    for x,y in zip([2,1,1,2,1], [2,2,1,1,0]): assert_vec_eq(u_cart_expect[x,y], ufit[x,y]) 
    print('median fit passes with N=2, needed to flip 5 cells')

    for N in [5,6]:
        variable_region = np.zeros((N,N))
        variable_region[1:4,1:4] = 1
        xrange = np.arange(1.0, 1.0+0.05*(N+1), 0.05)
        x, y = np.meshgrid(xrange, xrange)
        u_cart = np.zeros((N, N, 2))
        for i in range(N):
            for j in range(N):
                scale = 0 #0.05
                noisex, noisey = np.random.normal(0.0, 1.0) * scale, np.random.normal(0.0, 1.0) * scale
                u_cart[i,j,0]  =  y[i,j]  + 0.05 * np.cos(np.pi*x[i,j]) + 0.05 * np.sin(np.pi*y[i,j]) + noisex
                u_cart[i,j,1]  = -x[i,j]  + 0.10 * np.sin(np.pi*y[i,j]) + 0.10 * np.cos(np.pi*x[i,j]) + noisey
        u_cart_expect = cartesian_to_rzcartesian(u_cart, sign_wrap=False)
        fc1, fc2, n, m = cart_to_zonebasis(u_cart_expect)
        signs = np.ones((fc1.shape[0], fc1.shape[1])) 
        n[2,2], signs[2,2] = n[2,2]+1, -1
        m[1,2], signs[1,2] = m[1,2]-1, -1
        n[1,1], m[1,1], signs[1,1] = n[1,1]+1, m[1,1]+1, -1 
        n[3,1], m[3,1], signs[3,1] = n[3,1]-1, m[3,1]-1, -1
        n[3,2], signs[3,2] = n[3,2]+1, -1
        m[1,3], signs[1,3] = m[1,3]-1, -1
        m[2,3], signs[2,3] = m[2,3]+1, -1
        n[3,3], m[3,3], signs[3,3] = n[3,3]-1, m[3,3]-1, -1
        u_cart = zonebasis_to_cart(fc1, fc2, n, m, signs) # should capture all, allowed moves
        ufit, r = median_fit(u_cart, variable_region, plotting=False)
        for x,y in zip([2,1,1,3,3,1,2,3], [2,2,1,1,2,3,3,3]): assert_vec_eq(u_cart_expect[x,y], ufit[x,y]) 
        print('median fit passes with N={}, needed to flip 8 cells'.format(N))

def make_fake_disps(xrange, strainterms):
    x, y = np.meshgrid(xrange, xrange)
    n = len(xrange)
    u_true = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            u_true[i,j,0] =   y[i,j]  + strainterms[0]  * np.cos(np.pi*x[i,j])  + strainterms[1] * np.sin(np.pi*y[i,j])
            u_true[i,j,1] =  -x[i,j]  + strainterms[2]  * np.sin(np.pi*y[i,j])  + strainterms[3] * np.cos(np.pi*x[i,j])
    u_wrap = cartesian_to_rzcartesian(u_true)
    for i in range(n):
        for j in range(n):
            if np.random.randint(0,2) < 1 : u_wrap[i,j,:] = -u_wrap[i,j,:] #flip sign, doesnt matter what basis
    return u_wrap, u_true

def fit_unwrap_unit_test():

    # set up fake u data for unit testing
    gvecs = [ [1,0], [0,1], [1,1], [1,-1], [-1,1], [-1,-1], [2,-1], [-2,1], [1,-2], [-1,2], [-1,0], [0,-1]]
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs = np.array(gvecs)
    ndisks = len(gvecs)
    nx, ny = 24,24 # small test want it fast
    strainterms = [0.12, 0.11, 0.11, 0.12]
    I = np.zeros((ndisks, nx, ny))
    xrange = np.arange(-0.2, 1.0, 0.05) 
    print(len(xrange))
    assert(len(xrange) == nx)
    make_fake_disps(xrange, strainterms)
    u_wrap, u_cart = make_fake_disps(xrange, strainterms)
    f, ax = plt.subplots()
    displacement_colorplot(ax, u_cart)
    plt.show()

    # set up fake intensity data for unit testing
    u_lv = cartesian_to_latticevec(u_cart)
    u_lvrz = lv_to_rzlv(u_lv, sign_wrap=True)
    coefs = np.zeros((ndisks, 3)) #abc so 3
    coefs[:, 0] = np.ones(ndisks)
    coefs[:, 1] = 0.0 * np.ones(ndisks)
    coefs[:, 2] = 0.0 * np.ones(ndisks)
    for disk in range(ndisks):
        for i in range(nx):
            for j in range(ny):
                gdotu = np.dot(gvecs[disk], np.array([u_lv[i,j,0], u_lv[i,j,1]]))
                I[disk, i, j] = 0.9 * np.cos(np.pi * gdotu)**2 + 0.1
    
    # testing the fit code to make sure it gives the right uvecs when NOT provided with perfect inital guesses
    ufit_lv, residuals = fit_u(I, coefs, nx, ny, gvecs, guess=None, parallel=True, nproc=12, norm_bool=True, multistart_bool=True, multistart_neighbor_bool=False)
    for n in range(2):
        coefs, resid = fit_ABC(I, ufit_lv, nx, ny, gvecs, coefs, nproc=12, parallel=False, norm_bool=True)
        ufit_lv, resid = fit_u(I, coefs, nx, ny, gvecs, guess=ufit_lv, nproc=12, parallel=True, norm_bool=True, multistart_bool=False)
    for n in range(5):
        coefs, resid = fit_ABC(I, ufit_lv, nx, ny, gvecs, coefs, nproc=12, parallel=False, norm_bool=True)
        ufit_lv, resid = fit_u(I, coefs, nx, ny, gvecs, nproc=12, guess=ufit_lv, parallel=True, norm_bool=True, multistart_neighbor_bool=True)
    ufit_lv = lv_to_rzlv(ufit_lv, sign_wrap=True)
    print(np.max(((u_lvrz[:,:,0]-ufit_lv[:,:,0])**2 + (u_lvrz[:,:,1]-ufit_lv[:,:,1])**2).flatten()))
    u = latticevec_to_cartesian(ufit_lv)

    f, ax = plt.subplots(1,2)
    displacement_colorplot(ax[0], u)
    displacement_colorplot(ax[1], u_cart)
    plt.show()

    # unwrap ufit_lv
    boundary_val = 0.25 # AA find threshold
    delta_val = 0.1 # SP angular threshold
    combine_crit = 1.0 # combine if less than this dist away
    spdist = 5.0 # how close are AA centers to a SP line to say the line passes through the center 
    centerdist = 0.10 # how far from AA center need to be to unconstrain - stuff next to AA are unwrapped well and held fixed
    manual = False
    pad = 5
    smooth_sigma = 3

    try: u = unwrap(u, manual, centerdist, boundary_val, delta_val, combine_crit, spdist)
    except: u = unwrap(u, manual=True)
    smoothfunc = lambda u: nan_gaussian_filter(u, sigma=smooth_sigma)
    exx2, exy2, eyx2, eyy2 = strain(u[pad:-pad, pad:-pad, :], smoothfunc)
    exx, exy, eyx, eyy = strain(u_cart[pad:-pad, pad:-pad, :], smoothfunc)

    f, ax = plt.subplots(3,5)
    displacement_colorplot(ax[0,0], u)
    ax[0,1].imshow(exx2, origin='lower')
    ax[0,2].imshow(exy2, origin='lower')
    ax[0,3].imshow(eyx2, origin='lower')
    ax[0,4].imshow(eyy2, origin='lower')
    displacement_colorplot(ax[1,0], u_cart)
    ax[1,1].imshow(exx, origin='lower')
    ax[1,2].imshow(exy, origin='lower')
    ax[1,3].imshow(eyx, origin='lower')
    ax[1,4].imshow(eyy, origin='lower')
    ax[2,0].imshow(((u_cart[pad:-pad,pad:-pad,0]-u[pad:-pad,pad:-pad,0])**2 + (u_cart[pad:-pad,pad:-pad,1]-u[pad:-pad,pad:-pad,1])**2), origin='lower')
    ax[2,1].imshow(exx-exx2, origin='lower')
    ax[2,2].imshow(exy-exy2, origin='lower')
    ax[2,3].imshow(eyx-eyx2, origin='lower')
    ax[2,4].imshow(eyy-eyy2, origin='lower')
    plt.show()
    print("interferometry unit test passing")

def unwrap_unit_test(xrange, strainterms, boundary_val, delta_val, combine_crit, spdist, centerdist, manual, pad, smooth_sigma): 
    
    n = len(xrange)
    assert(n%2 == 0) # default algo needs this to bin!
    u, u_cart = make_fake_disps(xrange, strainterms) # make fake data
         
    smoothfunc = lambda u: nan_gaussian_filter(u, sigma=smooth_sigma)  
    u = unwrap(u, manual=manual, centerdist=centerdist, boundary_val=boundary_val, delta_val=delta_val, combine_crit=combine_crit, spdist=spdist)
    exx2, exy2, eyx2, eyy2, _, _ = strain(u[pad:-pad, pad:-pad, :], smoothfunc)
    exx, exy, eyx, eyy, _, _ = strain(u_cart[pad:-pad, pad:-pad, :], smoothfunc)
  
    f, ax = plt.subplots(3,5)

    displacement_colorplot(ax[0,0], u)
    ax[0,1].imshow(exx2, origin='lower')
    ax[0,2].imshow(exy2, origin='lower')
    ax[0,3].imshow(eyx2, origin='lower')
    ax[0,4].imshow(eyy2, origin='lower')

    displacement_colorplot(ax[1,0], u_cart)
    ax[1,1].imshow(exx, origin='lower')
    ax[1,2].imshow(exy, origin='lower')
    ax[1,3].imshow(eyx, origin='lower')
    ax[1,4].imshow(eyy, origin='lower')
    
    ax[2,0].imshow(((u_cart[pad:-pad,pad:-pad,0]-u[pad:-pad,pad:-pad,0])**2 + (u_cart[pad:-pad,pad:-pad,1]-u[pad:-pad,pad:-pad,1])**2), origin='lower')
    ax[2,1].imshow(exx-exx2, origin='lower')
    ax[2,2].imshow(exy-exy2, origin='lower')
    ax[2,3].imshow(eyx-eyx2, origin='lower')
    ax[2,4].imshow(eyy-eyy2, origin='lower')

    plt.show()

def unwrap_unit_tests():   

    xrange = np.arange(-1.5, 1.2, 0.05) 
    strainterms = [0.12, 0.11, 0.11, 0.12]
    boundary_val = 0.15 # AA find threshold
    delta_val = 0.1 # SP angular threshold
    combine_crit = 1.0 # combine if less than this dist away
    spdist = 5.0 # how close are AA centers to a SP line to say the line passes through the center 
    centerdist = 0.10 # how far from AA center need to be to unconstrain - stuff next to AA are unwrapped well and held fixed
    manual = False
    pad = 5
    smooth_sigma = 3
    print('starting test 1')
    unwrap_unit_test(xrange, strainterms, boundary_val, delta_val, combine_crit, spdist, centerdist, manual, pad, smooth_sigma)

    xrange = np.arange(-1.5, 1.1, 0.1) 
    strainterms = [0.23, 0.01, 0.05, 0.03]
    boundary_val = 0.15 # AA find threshold
    delta_val = 0.1 # SP angular threshold
    combine_crit = 1.0 # combine if less than this dist away
    spdist = 5.0 # how close are AA centers to a SP line to say the line passes through the center 
    centerdist = 0.10 # how far from AA center need to be to unconstrain - stuff next to AA are unwrapped well and held fixed
    manual = True
    pad = 5
    smooth_sigma = 3
    unwrap_unit_test(xrange, strainterms, boundary_val, delta_val, combine_crit, spdist, centerdist, manual, pad, smooth_sigma)

    xrange = np.arange(-1.7, 1.8, 0.05) 
    strainterms = [0.12, 0.11, 0.11, 0.12]
    boundary_val = 0.15 # AA find threshold
    delta_val = 0.1 # SP angular threshold
    combine_crit = 1.0 # combine if less than this dist away
    spdist = 5.0 # how close are AA centers to a SP line to say the line passes through the center 
    centerdist = 0.03 # how far from AA center need to be to unconstrain - stuff next to AA are unwrapped well and held fixed
    manual = True
    pad = 5
    smooth_sigma = 3
    unwrap_unit_test(xrange, strainterms, boundary_val, delta_val, combine_crit, spdist, centerdist, manual, pad, smooth_sigma)
    
def basis_unit_tests():

    xrange = np.arange(-1.9, 1.9, 0.05) #0.05
    strainterms = [0.12, 0.11, 0.11, 0.12]
    x, y = np.meshgrid(xrange, xrange)
    n = len(xrange)
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            u_cart[i,j,0] = y[i,j]  
            u_cart[i,j,1] = -x[i,j]

    u, u_cart = make_fake_disps(xrange, strainterms) # make fake data
    u_zcart  = cartesian_to_rzcartesian(u_cart.copy(), sign_wrap=False) 
    u_zcart2 = cartesian_to_rz_WZ(u_cart.copy(), sign_wrap=False)
    zones = u_cart - u_zcart2
    zones_lv = cartesian_to_latticevec(zones)

    f, ax = plt.subplots(2,3)
    displacement_colorplot(ax[0,0], u_cart[:,:,0],   u_cart[:,:,1])
    displacement_colorplot(ax[0,1], u_zcart[:,:,0],  u_zcart[:,:,1])
    displacement_colorplot(ax[0,2], u_zcart2[:,:,0], u_zcart2[:,:,1])
    ax[1,0].quiver(zones[:,:,0],   zones[:,:,1])
    ax[1,1].imshow(zones_lv[:,:,0], origin='lower')
    ax[1,2].imshow(zones_lv[:,:,1], origin='lower')
    plt.show()

    uvecs, prefix, dsnum, _ = import_uvector()
    u_cart = latticevec_to_cartesian(uvecs)
    u_zcart2 = cartesian_to_rz_WZ(u_cart.copy(), sign_wrap=False)
    
    nx, ny = u_cart.shape[0], u_cart.shape[1]
    umag = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            umag[i,j] = (u_zcart2[i,j,0]**2 + u_zcart2[i,j,1]**2)**0.5
    f, ax = plt.subplots(1,6); ax[0].imshow(umag, origin='lower')

    
    from scipy import ndimage as ndi 
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    coords = peak_local_max(-umag)
    print(coords)
    mask = np.zeros(umag.shape, dtype=bool)
    mask[tuple(coords.T)] = True 
    ax[3].imshow(mask, origin='lower')
    markers, _ = ndi.label(mask)
    ax[4].imshow(markers, origin='lower')
    labels = watershed(umag, markers)
    ax[2].imshow(labels, origin='lower')

    # get voroni regions
    from scipy.spatial import Voronoi, voronoi_plot_2d
    mask_aa = get_aa_mask(u_zcart2, boundary=0.4)
    labeled, aa_regions = ndimage.label(mask_aa)
    centers = get_region_centers(labeled, mask_aa)
    dists = normDistToNearestCenter(u_zcart2.shape[0], u_zcart2.shape[1], centers)
    ax[1].imshow(dists, origin='lower')

    points = [ [c[1], c[0]] for c in centers ]
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    for i in range(len(regions)):
        region = regions[i]
        polygon = vertices[region]
        ax[1].scatter(points[i][0], points[i][1], color='k')
        ax[1].fill(*zip(*polygon), alpha=0.4)   

    plt.show()
    exit()

def strain_unit_tests():

    # set up fake u data for unit testing
    gradtol = 0.1

    xrange = np.arange(-1., 1., 0.01) #0.05
    x, y = np.meshgrid(xrange, xrange)
    n = len(xrange)
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            u_cart[i,j,0] = y[i,j]  + 0.005 * np.cos(np.pi*x[i,j]) + 0.005 * np.sin(2*np.pi*y[i,j])
            u_cart[i,j,1] = -x[i,j] + 0.004 * np.sin(np.pi*y[i,j]) + 0.004 * np.cos(2*np.pi*x[i,j])
    u_lv = cartesian_to_latticevec(u_cart)

    # testing first way of derivatives
    ss = xrange[1] - xrange[0]
    expect_gxx = np.gradient(u_cart[:,:,0], axis=0) * 1/ss
    expect_gxy = np.gradient(u_cart[:,:,0], axis=1) * 1/ss
    expect_gyx = np.gradient(u_cart[:,:,1], axis=0) * 1/ss
    expect_gyy = np.gradient(u_cart[:,:,1], axis=1) * 1/ss

    def teststrain(gxx, gyx, gxy, gyy, plotflag, absflag=False):
        if plotflag:
            f, axes = plt.subplots(4,2)
            from utils import bin
            if absflag: f = lambda x: np.abs(x)
            else: 
                def f(x):
                    #x = np.abs(x)
                    x = bin(x, bin_w=6, size_retain=True, method=np.nanmedian)
                    return x
            axes[0,0].imshow(f(gxx))#, vmin=np.min(f(expect_gxx)), vmax=np.max(f(expect_gxx)))
            axes[0,1].imshow(f(expect_gxx))
            axes[1,0].imshow(f(gxy))#, vmin=np.min(f(expect_gxy)), vmax=np.max(f(expect_gxy)))
            axes[1,1].imshow(f(expect_gxy))
            axes[2,0].imshow(f(gyx))#, vmin=np.min(f(expect_gyx)), vmax=np.max(f(expect_gyx)))
            axes[2,1].imshow(f(expect_gyx))
            axes[3,0].imshow(f(gyy))#, vmin=np.min(f(expect_gyy)), vmax=np.max(f(expect_gyy)))
            axes[3,1].imshow(f(expect_gyy))
            plt.show()
        assert(np.max(np.abs(gxx - expect_gxx)) < gradtol)
        assert(np.max(np.abs(gyy - expect_gyy)) < gradtol)
        assert(np.max(np.abs(gyx - expect_gyx)) < gradtol)
        assert(np.max(np.abs(gxy - expect_gxy)) < gradtol)

    u_rzlv = lv_to_rzlv(u_lv, sign_wrap=True)
    u_rzc = latticevec_to_cartesian(u_rzlv)
    gxx, gxy, gyx, gyy = differentiate(u_rzc, x, y, wrap=True)
    teststrain(gxx, gxy, gyx, gyy, True, absflag=False)
    print('strain test 5 passing')

    u_rzlv = lv_to_rzlv(u_lv, sign_wrap=True)
    gxx, gxy, gyx, gyy = differentiate_lv(u_rzlv, x, y, wrap=True)
    teststrain(gxx, gxy, gyx, gyy, True, absflag=False)
    print('strain test 7 passing')

    gxx, gyx, gxy, gyy = differentiate_lv(u_lv, x, y, wrap=False)
    teststrain(gxx, gyx, gxy, gyy, False)
    print('strain test 1 passing')

    gxx, gyx, gxy, gyy = differentiate(u_cart, x, y, wrap=False)
    teststrain(gxx, gyx, gxy, gyy, False)
    print('strain test 2 passing')

    gxx, gyx, gxy, gyy = differentiate_lv(u_lv, x, y, wrap=True)
    teststrain(gxx, gyx, gxy, gyy, False)
    print('strain test 3 passing')

    gxx, gyx, gxy, gyy = differentiate(u_cart, x, y, wrap=True)
    teststrain(gxx, gyx, gxy, gyy, False)
    print('strain test 4 passing')

    u_rzlv = lv_to_rzlv(u_lv, sign_wrap=True)
    gxx, gxy, gyx, gyy = differentiate_lv(u_rzlv, x, y, wrap=True)
    teststrain(gxx, gyx, gxy, gyy, True)
    print('strain test 6 passing')

def fit_unit_tests(test_serial=False):

    # set up fake u data for unit testing
    #           0      1       2       3       4        5       6       7       8       9       10     11
    gvecs = [ [0,1], [1,0], [1,-1], [-1,1], [-1,-1], [2,-1], [-2,1], [1,-2], [-1,2], [-1,0], [0,-1], [1,1] ]

    index_pairs = []
    for i in range(12):
        for j in range(i):
            if gvecs[i][0] != -gvecs[j][0] and gvecs[i][1] != -gvecs[j][1]:
                index_pairs.append([i,j])

    print(index_pairs)
    #g1 = np.array([ 0, 2/np.sqrt(3)])
    #g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs = np.array(gvecs)
    ndisks = len(gvecs)
    nx, ny = 100,100 # small test want it fast
    I = np.zeros((ndisks, nx, ny))
    u_cart = np.zeros((nx, ny, 2))
    X, Y = np.meshgrid(np.arange(-2.9, 2.7, 0.05), np.arange(-2.9, 2.7, 0.05)) # might not be realisitic resolution but not testing median fit 
    for i in range(nx):
        for j in range(ny):
            u_cart[i,j,:] = [-Y[i,j], X[i,j]]

    # set up fake intensity data for unit testing
    u_lv = cartesian_to_latticevec(u_cart)
    for disk in range(ndisks):
        for i in range(nx):
            for j in range(ny):
                gdotu = np.dot(gvecs[disk], np.array([u_lv[i,j,0], u_lv[i,j,1]]))
                noise = 0.01 * np.random.normal(0,1)
                I[disk, i, j] = np.cos(np.pi * gdotu)**2 + noise

    from utils import normalize, debugplot
    for disk in range(ndisks): I[disk,:,:] = normalize(I[disk,:,:]) # Ci=0, Ai=1 when Bi=0
    #for disk in range(ndisks): debugplot(I[disk,:,:])
    u_lv = lv_to_rzlv(u_lv, sign_wrap=True)


    if False:
        ### hi 
        x, y = 5,5
        Icurr = I[:,x,y]
        maxiter = 3

        from scipy.optimize import bisect 

        # use first gvector, [1,0] to get |u1| assuming no sincos term
        # find root of I[0] - cos2(pi*u1) with bisection, u1 in [1/2, 0]
        i = 1
        f = lambda x: Icurr[i] - np.cos(np.pi * x)**2
        if np.sign(f(0)) == np.sign(f(0.5)): u1 = 0
        else: u1 = bisect(f, 0, 0.5)

        # then do the same with I[1] for g = [0,1] and u2 
        i = 0
        f = lambda x: Icurr[i] - np.cos(np.pi * x)**2
        if np.sign(f(0)) == np.sign(f(0.5)): u2 = 0
        else: u2 = bisect(f, 0, 0.5)

        # now tell if [u1, u2] or [u1, -u2]. other two options degenerate without sincos
        i = 4
        option1 = Icurr[i] - np.cos(np.pi * u1 - np.pi * u2)**2
        option2 = Icurr[i] - np.cos(np.pi * u1 + np.pi * u2)**2
        if np.abs(option1) < np.abs(option2): guess = [u1, -u2]
        else: guess = [u1, u2]

        utrue = u_lv[x,y,:]
        print(utrue)
        print(guess)
        ufits = [0.0, 0.0]

        for indx_pair in index_pairs:
            i,j = indx_pair[:]

            ucurr = guess

            for nn in range(maxiter):
                ff = np.zeros((2,1))

                gdotu1 = np.dot(gvecs[i], ucurr)
                gdotu2 = np.dot(gvecs[j], ucurr)

                jacobian = np.zeros((2,2))
                jacobian[0,0] = gvecs[i][0]
                jacobian[0,1] = gvecs[i][1]
                jacobian[1,0] = gvecs[j][0]
                jacobian[1,1] = gvecs[j][1]
                jacobian[0,:] *= 2.0 * np.cos( np.pi * gdotu1 ) * np.sin( np.pi * gdotu1 ) * np.pi
                jacobian[1,:] *= 2.0 * np.cos( np.pi * gdotu2 ) * np.sin( np.pi * gdotu2 ) * np.pi

                ff[0,0] = Icurr[i] - np.cos( gdotu1*np.pi )**2
                ff[1,0] = Icurr[j] - np.cos( gdotu2*np.pi )**2

                aa = np.linalg.solve( jacobian, ff )
                ucurr[0] -= aa[0]
                ucurr[1] -= aa[1]

            ufits[0] += ucurr[0] / len(index_pairs)
            ufits[1] += ucurr[1] / len(index_pairs)

        
        print(ufits)
        from basis_utils import rz_helper_pair
        c1, c2 = rz_helper_pair(ufits[0], ufits[1], True)
        print(c1, c2)
        exit()

       












        ## end hi


    # coupled u, coef optimization 
    fitcoefs = np.zeros((ndisks, 3)) #abc so 3
    guess_c = np.zeros((ndisks, 3))
    guess_c[:,0] = 1
    guess_c[:,-1] = 0
    from interferometry_fitting import fit_u
    tic()
    u, residuals = fit_u(I, guess_c, nx, ny, gvecs, nproc=12)
    toc('fit u')
    f, ax = plt.subplots(2,3)
    displacement_colorplot_lvbasis(ax[0,0], u)
    ax[0,1].imshow(u[:,:,0], origin='lower')
    ax[0,2].imshow(u[:,:,1], origin='lower')
    displacement_colorplot_lvbasis(ax[1,0], u_lvrz)
    ax[1,1].imshow(u_lvrz[:,:,0], origin='lower')
    ax[1,2].imshow(u_lvrz[:,:,1], origin='lower')
    plt.show(); exit()


    # assertion tests to verify the lv <-> cartesian conversion
    asserteq(np.dot([0,1], np.array([ u_lv[0,0,0],   u_lv[0,0,1] ])), np.dot(g1, np.array([ u_cart[0,0,0], u_cart[0,0,1] ])))
    asserteq(np.dot([1,0], np.array([ u_lv[0,0,0],   u_lv[0,0,1] ])), np.dot(g2, np.array([ u_cart[0,0,0], u_cart[0,0,1] ])))
    u_cart2 = latticevec_to_cartesian(u_lv)
    for d in range(2): asserteq(u_cart[0,0,d],u_cart2[0,0,d])
    print("interferometry unit test 1 passing")

    # assertion tests to verify the displacement visualization being consistent with the conversion
    colors1 = displacement_colorplot_lvbasis(None, u_lv[:,:,0], u_lv[:,:,1])
    colors2 = displacement_colorplot(None, u_cart[:,:,0], u_cart[:,:,1])
    for i in range(colors1.shape[0]):
        for j in range(colors1.shape[1]):
            for c in range(colors1.shape[2]):
                assert(np.abs(colors1[i,j,c] - colors2[i,j,c]) < 1e-6)
    print("interferometry unit test 2 passing")

    # doing the same assertion tests after wrapping into the reduced zone
    colors1 = displacement_colorplot_lvbasis(None, u_lvrz[:,:,0], u_lvrz[:,:,1])
    for i in range(colors1.shape[0]):
        for j in range(colors1.shape[1]):
            for c in range(colors1.shape[2]):
                assert(np.abs(colors1[i,j,c] - colors2[i,j,c]) < 1e-6)
    print("interferometry unit test 3 passing")

    # testing the fit code to make sure it gives the right uvecs when provided with perfect inital guesses
    coefs = np.zeros((ndisks, 3)) #abc so 3
    coefs[:, 0] = 0.9 * np.ones(ndisks)
    coefs[:, 1] = 0.0 * np.ones(ndisks)
    coefs[:, 2] = 0.1 * np.ones(ndisks)
    ufit_lv, residuals = fit_u(I, coefs, nx, ny, gvecs, guess=u_lvrz, parallel=True, nproc=12, norm_bool=False, multistart_bool=False, multistart_neighbor_bool=False)
    if test_serial: ufit_lv2, residuals2 = fit_u(I, coefs, nx, ny, gvecs, guess=u_lvrz, parallel=False, nproc=12, norm_bool=False, multistart_bool=False, multistart_neighbor_bool=False)
    for i in range(residuals.shape[0]):
        for j in range(residuals.shape[1]):
            assert(np.abs(residuals[i,j]) < 1e-8)
            if test_serial: assert(np.abs(residuals2[i,j]) < 1e-8)
    print("interferometry unit test 4 passing")

    # checking fit coef code
    fitcoefs, residuals = fit_ABC(I, u_lvrz, nx, ny, g=gvecs, coef_guess=coefs, nproc=4, parallel=True, norm_bool=False)
    assert(np.max(np.abs(coefs - fitcoefs)) < 1e-8)
    print("interferometry unit test 5 passing")

    # checking fit coef code using subset of data
    fitcoefs, residuals = fit_ABC(I[:, :5,:5], u_lvrz[:5,:5,:], 5, 5, g=gvecs, coef_guess=coefs, nproc=4, parallel=True, norm_bool=False)
    assert(np.max(np.abs(coefs - fitcoefs)) < 1e-8)
    print("interferometry unit test 6 passing")
    
    # testing the fit code to make sure it gives the right uvecs when NOT provided with perfect inital guesses
    ufit_lv, residuals = fit_u(I, coefs, nx, ny, gvecs, guess=None, parallel=True, nproc=12, norm_bool=True, multistart_bool=True, multistart_neighbor_bool=False)
    for n in range(2):
        coefs, resid = fit_ABC(I, ufit_lv, nx, ny, gvecs, coefs, nproc=12, parallel=False, norm_bool=True)
        ufit_lv, resid = fit_u(I, coefs, nx, ny, gvecs, guess=ufit_lv, nproc=12, parallel=True, norm_bool=True, multistart_bool=False)
    for n in range(5):
        coefs, resid = fit_ABC(I, ufit_lv, nx, ny, gvecs, coefs, nproc=12, parallel=False, norm_bool=True)
        ufit_lv, resid = fit_u(I, coefs, nx, ny, gvecs, nproc=12, guess=ufit_lv, parallel=True, norm_bool=True, multistart_neighbor_bool=True)
    ufit_lv = lv_to_rzlv(ufit_lv, sign_wrap=True)
    print(np.max(((u_lvrz[:,:,0]-ufit_lv[:,:,0])**2 + (u_lvrz[:,:,1]-ufit_lv[:,:,1])**2).flatten()))
    assert(np.max(((u_lvrz[:,:,0]-ufit_lv[:,:,0])**2 + (u_lvrz[:,:,1]-ufit_lv[:,:,1])**2).flatten()) < 0.005)
    print("interferometry unit test 7 passing")

    # testing the fit code to make sure it gives the right uvecs when NOT provided with perfect inital guesses
    coefs = np.zeros((ndisks, 3)) #abc so 3
    coefs[:, 0] = 1.0 * np.ones(ndisks)
    coefs[:, 1] = 0.0 * np.ones(ndisks)
    coefs[:, 2] = 0.0 * np.ones(ndisks)
    ufit_lv, residuals = fit_u(I, coefs, nx, ny, gvecs, guess=None, parallel=True, nproc=12, norm_bool=True, multistart_bool=True, multistart_neighbor_bool=False)
    for n in range(2):
        coefs, resid = fit_ABC(I, ufit_lv, nx, ny, gvecs, coefs, nproc=12, parallel=False, norm_bool=True)
        ufit_lv, resid = fit_u(I, coefs, nx, ny, gvecs, guess=ufit_lv, nproc=12, parallel=True, norm_bool=True, multistart_bool=False)
    for n in range(5):
        coefs, resid = fit_ABC(I, ufit_lv, nx, ny, gvecs, coefs, nproc=12, parallel=False, norm_bool=True)
        ufit_lv, resid = fit_u(I, coefs, nx, ny, gvecs, nproc=12, guess=ufit_lv, parallel=True, norm_bool=True, multistart_neighbor_bool=True)
    ufit_lv = lv_to_rzlv(ufit_lv, sign_wrap=True)
    print(np.max(((u_lvrz[:,:,0]-ufit_lv[:,:,0])**2 + (u_lvrz[:,:,1]-ufit_lv[:,:,1])**2).flatten()))
    assert(np.max(((u_lvrz[:,:,0]-ufit_lv[:,:,0])**2 + (u_lvrz[:,:,1]-ufit_lv[:,:,1])**2).flatten()) < 0.005)
    print("interferometry unit test 8 passing")

if __name__ == "__main__":
    #fit_unit_tests()
    NEW_TEST(0, 0)
    #NEW_TEST(-1.5 * np.pi/180, 3.0 * np.pi/180) ---> -1.5 and 1.5
    #NEW_TEST(-6 * np.pi/180,   3.0 * np.pi/180) # --> -6 and -3 
    tests = [fit_unit_tests, basis_unit_tests, mip_fit_unit_test, strain_unit_tests, unwrap_unit_tests, rotation_test, tblg_unwrap_test, fit_unwrap_unit_test] #
    for test in tests: test()
