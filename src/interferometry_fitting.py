
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
do_par_fit_but_not_unwrap = False
if do_par_fit_but_not_unwrap:
    matplotlib.use('Agg') # need this backend when rendering within parallelized loops
# if you see the error 'main thread not in main loop', this is why
# GUI based backends for mpl need to be in main thread, agg doesnt. but
# then can't have gui for image manipulations
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares, lsq_linear
from diskset import *
from utils import *
from visualization import displacement_plt, displacement_colorplot, displacement_colorplot_lvbasis, displacement_plt_lvbasis
import pickle
from numpy.linalg import norm
import numpy.random as random
from pathos.multiprocessing import Pool # pathos.multiprocessing can serialize wrapper funcs, multiprocessing can't
from basis_utils import lv_to_rzlv, latticevec_to_cartesian, cartesian_to_latticevec, rz_helper_pair
import warnings

def verify_dfs(savepath, diskset, coefs, ulv):

    I = diskset.df_set()
    g = diskset.clean_normgset()
    for n in range(len(g)):
        I[n,:,:] = normalize(I[n,:,:])
    ndisks = I.shape[0]

    if isinstance(coefs, list):
        coefs_mat = np.zeros((ndisks, len(coefs)))
        for i in range(len(coefs)):
            coefs_mat[:,i] = coefs[i]
        coefs = coefs_mat

    f, ax = plt.subplots(ndisks,4)
    for disk in range(ndisks):
        vals  = np.zeros((ulv.shape[0],ulv.shape[1]))
        for i in range(ulv.shape[0]):
            for j in range(ulv.shape[1]):
                u  = [ulv[i,j,0], ulv[i,j,1]]
                if coefs.shape[1] == 2: # fit A, B only
                    vals[i,j] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u) ) ** 2 + coefs[disk,1]
                elif coefs.shape[1] == 3: # fit ABC
                    vals[i,j] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u) ) ** 2 + coefs[disk,2]
                    vals[i,j] += coefs[disk,1] * np.cos( np.pi * np.dot(g[disk],u) ) * np.sin( np.pi * np.dot(g[disk],u) )
        ax[disk,0].imshow(I[disk,:,:],      origin='lower')
        ax[disk,1].imshow(vals,   origin='lower')
        ax[disk,3].imshow(I[disk,:,:]-vals, origin='lower')
        ax[disk,0].set_title("raw disk {} g={}{}".format(disk, g[disk][0], g[disk][1]),fontsize=5)
        ax[disk,1].set_title("fit disk lv basis {} g={}{}".format(disk, g[disk][0], g[disk][1]),fontsize=5)
        ax[disk,3].set_title("resid disk {} g={}{}".format(disk, g[disk][0], g[disk][1]),fontsize=5)

    # Cart BASIS check
    g1  = np.array([ 0, 2/np.sqrt(3)])
    g2  = np.array([-1, 1/np.sqrt(3)])
    ucart = latticevec_to_cartesian(ulv.copy())
    for disk in range(ndisks):
        vals  = np.zeros((ucart.shape[0],ucart.shape[1]))
        for i in range(ulv.shape[0]):
            for j in range(ulv.shape[1]):
                u  = [ucart[i,j,0], ucart[i,j,1]]
                gvec = g[disk][0] * g2 + g[disk][1] * g1
                if coefs.shape[1] == 2: # fit A, B only
                    vals[i,j] =  coefs[disk,0] * np.cos( np.pi * np.dot(gvec,u) ) ** 2 + coefs[disk,1]
                elif coefs.shape[1] == 3: # fit ABC
                    vals[i,j] =  coefs[disk,0] * np.cos( np.pi * np.dot(gvec,u) ) ** 2 + coefs[disk,2]
                    vals[i,j] += coefs[disk,1] * np.cos( np.pi * np.dot(gvec,u) ) * np.sin( np.pi * np.dot(gvec,u) )
        ax[disk,2].imshow(vals,   origin='lower')
        ax[disk,2].set_title("fit cart basis disk {} g={}{}".format(disk, g[disk][0], g[disk][1]),fontsize=5)

    plt.savefig(savepath, dpi=300)
    plt.close('all')

def refit_full_hexagon_from_bin(diskset, coefs, uvecs, g=None):

    # get vector of g corresponding to diffraction pattern intensities 
    # so if I[:,:,i] is the intensity of disk i, which is at a gvector of g_1+g_2 (g_1 and g_2 are conventional hexagonal recip lattice vectors)
    # then g[i] = [1, 1] for g_1+g_2
    # note that I[x,y,i] = A[i] + B[i] * np.cos( np.pi * g[i] dot u[x,y] ) + ... 
    # where in our convention u[x,y] = u_1 * r_1 + u_2 * r_2 with r_1 and r_2 real space lattice vectors, g_i dot r_j is kroneckerdelta_ij 
    # so if we store the u vector in the lattice vector basis, u[x,y,:] = [u1, u2],
    # then assuming the g vectors are ordered as [g1, g2, g1+g2 ...]
    # I[x,y,0] = A[0] + B[0] * np.cos( np.pi * [1,0] dot [u1, u2] ) + ... 
    # I[x,y,1] = A[1] + B[1] * np.cos( np.pi * [0,1] dot [u1, u2] ) + ...
    # I[x,y,2] = A[2] + B[2] * np.cos( np.pi * [1,1] dot [u1, u2] ) + ...
    # note that can add a full lattice vector to u, so [u1, u2] -> [u1+1,u2] (can add any integer to both)
    # so when we fit we need for uniqueness to restrict to -1/2 <= u1 <= 1/2 and same with u2.
    # then finding U in the x,y basis involves the coordinate transformation,
    # for all i & j:
    #    u_xybasis[i,j,:] = u_latbasis[i,j,0] * r1 + u_latbasis[i,j,1] * r2

    if isinstance(diskset, DiskSet):
        I = diskset.df_set()
        g = diskset.clean_normgset()
    elif isinstance(diskset, np.ndarray):
        I = diskset 
        if not (isinstance(g, np.ndarray)):
            print('ERROR: gvectors must be provided to fit_full_hexagon if I isnt a DiskSet instance')
            exit()
    else: print('Unrecognized type for intensities given to fit_full_hexagon, needs to be an array or DiskSet instance')
    
    # if ncoefs=3 A,B,C for Acos^2 + Bsincos + C fit
    # if ncoefs=2 A,B   for Acos^2 + B fit
    # first multistart fit for u vectors with A and B fixed as 1 and 0 respectively
    ndisks, nx, ny = I.shape[0], I.shape[1], I.shape[2]
    uvecs_unbin = np.zeros((nx, ny, 2))
    uvecs_unbin[:,:,0] = unbin(uvecs[:,:,0], 2)
    uvecs_unbin[:,:,1] = unbin(uvecs[:,:,1], 2)
    uvecs_unbin, resid = fit_u(I, coefs, nx, ny, g, guess=uvecs_unbin, nproc=12, parallel=True, norm_bool=True, multistart_bool=False)

    print('starting iterative median refits')
    for n in range(1):
        tic()
        uvecs_unbin, residuals = fit_u(I, coefs, nx, ny, g, nproc=12, guess=uvecs_unbin, parallel=True, norm_bool=True, multistart_neighbor_bool=True)
        toc('u fit')
    return coefs, uvecs_unbin

def fit_full_hexagon(diskset, ncoefs=2, binw=1, g=None, plot=True, guess=None, A=None, useStagedOpt=True):

    # get vector of g corresponding to diffraction pattern intensities 
    # so if I[:,:,i] is the intensity of disk i, which is at a gvector of g_1+g_2 (g_1 and g_2 are conventional hexagonal recip lattice vectors)
    # then g[i] = [1, 1] for g_1+g_2
    # note that I[x,y,i] = A[i] + B[i] * np.cos( np.pi * g[i] dot u[x,y] ) + ... 
    # where in our convention u[x,y] = u_1 * r_1 + u_2 * r_2 with r_1 and r_2 real space lattice vectors, g_i dot r_j is kroneckerdelta_ij 
    # so if we store the u vector in the lattice vector basis, u[x,y,:] = [u1, u2],
    # then assuming the g vectors are ordered as [g1, g2, g1+g2 ...]
    # I[x,y,0] = A[0] + B[0] * np.cos( np.pi * [1,0] dot [u1, u2] ) + ... 
    # I[x,y,1] = A[1] + B[1] * np.cos( np.pi * [0,1] dot [u1, u2] ) + ...
    # I[x,y,2] = A[2] + B[2] * np.cos( np.pi * [1,1] dot [u1, u2] ) + ...
    # note that can add a full lattice vector to u, so [u1, u2] -> [u1+1,u2] (can add any integer to both)
    # so when we fit we need for uniqueness to restrict to -1/2 <= u1 <= 1/2 and same with u2.
    # then finding U in the x,y basis involves the coordinate transformation,
    # for all i & j:
    #    u_xybasis[i,j,:] = u_latbasis[i,j,0] * r1 + u_latbasis[i,j,1] * r2

    if isinstance(diskset, DiskSet):
        I = diskset.df_set()
        g = diskset.clean_normgset()
        #print(g); exit()
    elif isinstance(diskset, np.ndarray):
        I = diskset 
        if not (isinstance(g, np.ndarray)):
            print('ERROR: gvectors must be provided to fit_full_hexagon if I isnt a DiskSet instance')
            exit()
    else: print('Unrecognized type for intensities given to fit_full_hexagon, needs to be an array or DiskSet instance')

    Inorm = np.zeros((I.shape[0], I.shape[1], I.shape[2]))
    for disk in range(I.shape[0]): Inorm[disk,:,:] = normalize(I[disk,:,:])

    # if ncoefs=3 A,B,C for Acos^2 + Bsincos + C fit
    # if ncoefs=2 A,B   for Acos^2 + B fit
    # first multistart fit for u vectors with A and B fixed as 1 and 0 respectively
    ndisks, nx, ny = I.shape[0], I.shape[1], I.shape[2]
    coefs = np.zeros((ndisks, ncoefs))
    coefs[:,0] = 1
    tic()
    uvecs, residuals = fit_u(I, coefs, nx, ny, g=g, guess=guess, parallel=True, nproc=12, norm_bool=True, multistart_bool=True, multistart_neighbor_bool=False)
    
    ############## making a residual sanity plot #############
    resid_plot = False
    if resid_plot:
        counter = 0
        f, ax = plt.subplots(4,3)
        ax = ax.flatten()
        for i in range(residuals.shape[2]):
            im = ax[i].imshow(residuals[:,:,i])
            div = make_axes_locatable(ax[i])
            cax = div.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im, cax=cax, orientation='vertical')
        RMS = np.sqrt(np.mean([r**2 for r in residuals.flatten()]))
        ax[0].set_title(RMS)
        plt.savefig('{}.png'.format(counter))
        print('{} : RMS '.format(counter), RMS)
        counter += 1
    
    toc('ufit')

    if (not useStagedOpt): # serial together fit, is worse. see https://arxiv.org/abs/2406.04515
        guess = np.concatenate((coefs.flatten(), uvecs.flatten()), axis=0)
        coefs, uvecs, resid = fit_together_serial(Inorm, g, guess, ncoefs)
        f, ax = plt.subplots(4,3)
        ax = ax.flatten()
        for i in range(12):
            im = ax[i].imshow(resid[:,:,i])
            div = make_axes_locatable(ax[i])
            cax = div.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im, cax=cax, orientation='vertical')
        RMS = np.sqrt(np.mean([r**2 for r in resid.flatten()]))
        ax[0].set_title(RMS)
        plt.savefig('{}.png'.format(counter))
        print('{} : RMS '.format(counter), RMS)
        return coefs, uvecs

    # iterative fit
    print('starting iterative u, A, B, C fits')
    for n in range(15):
        tic()
        coefs, resid = fit_ABC(I, uvecs, nx, ny, g, coefs, nproc=12, parallel=False, norm_bool=True)
        
        # multistart fit using neighbors
        uvecs, resid = fit_u(I, coefs, nx, ny, g, nproc=12, guess=uvecs, parallel=True, norm_bool=True, multistart_neighbor_bool=True)
        
        # no multistart
        #uvecs, resid = fit_u(I, coefs, nx, ny, g, nproc=12, guess=uvecs, parallel=True, norm_bool=True, multistart_bool=False)
        
        # multistart fit using grid
        #uvecs, resid = fit_u(I, coefs, nx, ny, g, nproc=12, guess=uvecs, parallel=True, norm_bool=True, multistart_bool=True, multistart_neighbor_bool=False)
        if resid_plot:
            f, ax = plt.subplots(4,3)
            ax = ax.flatten()
            for i in range(12):
                im = ax[i].imshow(residuals[:,:,i])
                div = make_axes_locatable(ax[i])
                cax = div.append_axes('right', size='5%', pad=0.05)
                f.colorbar(im, cax=cax, orientation='vertical')
            RMS = np.sqrt(np.mean([r**2 for r in resid.flatten()]))
            ax[0].set_title(RMS)
            plt.savefig('{}.png'.format(counter))
            print('{} : RMS '.format(counter), RMS)
            counter += 1

        toc('u abc')

    return coefs, uvecs

####################################################################################################
# preform multistart nonlinear least squares given guesses and a cost function
####################################################################################################
def multistart(optfunc, costfunc, guesses):
    prev_max = np.inf
    had_success = False
    for guess in guesses:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                opt = optfunc(guess)
            opt_residuals_new = costfunc(opt.x)
            sucess = 1
            had_success = True
        except:
            sucess = 0
        if sucess and max(abs(opt_residuals_new)) < prev_max:
            opt_residuals = opt_residuals_new
            prev_max = max(abs(opt_residuals_new))
            opt_params = opt.x 
    if not had_success: 
        print("every multistart optimization failed for some reason check this out...")
        exit()
    return opt_params, opt_residuals

####################################################################################################
# fit u(x,y) such that I_j(x,y) = A_j * cos^2(pi g_j dot u(x,y)) + B_j
# using nonlinear least squares, done in serial
####################################################################################################
def fit_u_serial(ndisks, norm_df_set, coefs, nx, ny, g, guess=None, multistart_bool=False, multistart_neighbor_bool=False, delta=0.2):
    print('starting call to ufit serial')
    uvecs = np.zeros((nx, ny, 2))
    residuals = np.zeros((nx, ny,ndisks))
    # relies on the function fit_xy which fits an individual pixel, finds a single u from a single I_j = A_j * cos^2(pi g_j dot u) + B_j
    fit_xy_wrap = lambda x,y: fit_xy(ndisks, norm_df_set, coefs, x, y, nx, ny, g, guess, multistart_bool, multistart_neighbor_bool, delta)
    # loop over each pixel and use the fit_xy function on each
    for x in range(nx):
        print("{}% done...".format(100*x/nx))
        for y in range(ny):
            uvecs[x, y, :], residuals[x, y,:] = fit_xy_wrap(x, y)
    return uvecs, residuals

####################################################################################################
# fit u(x,y) such that I_j(x,y) = A_j * cos^2(pi g_j dot u(x,y)) + B_j
# using nonlinear least squares, done in parallel
####################################################################################################
def fit_u_parallel(ndisks, norm_df_set, coefs, nx, ny, g, nproc=4, guess=None, multistart_bool=False, multistart_neighbor_bool=False, delta=0.2):
    # relies on the function fit_xy which fits an individual pixel, finds a single u from a single I_j = A_j * cos^2(pi g_j dot u) + B_j
    # given a compound index xy, calles this function to calculate u
    print('starting call to ufit parallel')
    def fit_wrapper(xy):
        x = xy // ny
        y = xy % ny
        return fit_xy(ndisks, norm_df_set, coefs, x, y, nx, ny, g, guess, multistart_bool, multistart_neighbor_bool, delta)
    # loop over each pixel and use the fit_xy function on each, but now do so over multiple processors
    with Pool(processes=nproc) as pool:
        output = pool.map(fit_wrapper, range(nx*ny))
    uvec_slice =  [outputel[0] for outputel in output]
    resid_slice = [outputel[1] for outputel in output]
    uvecs = np.array(uvec_slice).reshape(nx,ny,2)
    residuals = np.array(resid_slice).reshape(nx,ny,ndisks)
    return uvecs, residuals

####################################################################################################
# fits the u vector for a single pixel of the dataset
# fit u s.t. I_j = A_j * cos^2(pi g_j dot u) + B_j
###################################################################################################
def find_g_index(g1, g2, g):
    for i in range(len(g)):
        if (g[i][0] == g1 and g[i][1] == g2): return i 
    print('failed to find {},{} gvector'.format(g1, g2))
    exit()

def fit_xynew(ndisks, I, coefs, x, y, g, guess=None, neighbor=False, asym=False):

    gvecs = g
    I = I[:,x,y].flatten() # get the intensity values, per disk normalized to [0,1]
    index_pairs = []
    for i in range(12):
        for j in range(i):
            if g[i][0] != -g[j][0] and g[i][1] != -g[j][1]:
                index_pairs.append([i,j])

    if guess is None:
        from scipy.optimize import bisect 

        # use first gvector, [1,0] to get |u1| assuming no sincos term
        # find root of I[0] - cos2(pi*u1) with bisection, u1 in [1/2, 0]
        i = find_g_index(1, 0, g)
        i2 = find_g_index(-1, 0, g)
        f = lambda x: (I[i] + I[i2])/2 - np.cos(np.pi * x)**2
        if np.sign(f(0)) == np.sign(f(0.5)): u1 = 0
        else: u1 = bisect(f, 0, 0.5)

        # then do the same with I[1] for g = [0,1] and u2 
        i = find_g_index(0, 1, g)
        i2 = find_g_index(0, -1, g)
        f = lambda x: (I[i]+I[i2])/2 - np.cos(np.pi * x)**2
        if np.sign(f(0)) == np.sign(f(0.5)): u2 = 0
        else: u2 = bisect(f, 0, 0.5)

        # now tell if [u1, u2] or [u1, -u2]. other two options degenerate without sincos
        i = find_g_index(1, 1, g)
        i2 = find_g_index(-1, -1, g)
        option1 = (I[i]+I[i2])/2 - np.cos(np.pi * u1 - np.pi * u2)**2
        option2 = (I[i]+I[i2])/2 - np.cos(np.pi * u1 + np.pi * u2)**2
        if np.abs(option1) < np.abs(option2): guess = [u1, u2]
        else: guess = [u1, -u2]

    else:
        u1, u2 = guess[x,y,:]
        guess = [u1, u2]


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

    return ufits, residuals
    """
    u1boundary, u2boundary = False, False
    tol = 1e-4
    if (np.abs(u1 - 0.5) < tol) or (np.abs(u1) < tol): u1boundary = True
    if (np.abs(u2 - 0.5) < tol) or (np.abs(u2) < tol): u2boundary = True

    # now using this guess we can optimize A, B, C for uvectors.. C might just allow overfitting since normalized
    def _fit_func(u):
        vals = np.zeros(ndisks)
        for disk in range(ndisks):
            if coefs.shape[1] == 2: # fit A, B only
                vals[disk] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u) ) ** 2 + coefs[disk,1]
            elif coefs.shape[1] == 3: # fit ABC
                vals[disk] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u) ) ** 2 + coefs[disk,2]
                vals[disk] += coefs[disk,1] * np.cos( np.pi * np.dot(g[disk],u) ) * np.sin( np.pi * np.dot(g[disk],u) )
        return vals

    def _costfunc(vec): return I - _fit_func(vec[0:2])
    lowerbounds = -0.5 * np.ones(2)
    upperbounds =  0.5 * np.ones(2)
    bounds = (lowerbounds, upperbounds)
    optfunc = lambda g: least_squares(_costfunc, g, bounds=bounds, verbose=0, ftol=None, xtol=1e-4, gtol=None)
    if not u1boundary and not u2boundary :
        if not asym:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                opt = optfunc(guess)
            opt_params = opt.x
            residuals = _costfunc(opt.x)
        else:
            u1, u2 = guess[:]
            guesses = [ [u1,u2], [-u1,-u2] ]
            opt_params, residuals = multistart(optfunc, _costfunc, guesses)
    elif u1boundary and u2boundary:
        u1, u2 = guess[:]
        if not asym: guesses = [ [0,0], [0.5,0], [0.5,0.5], [0,0.5], [0.5,-0.5]]
        else: guesses = [ [0,0], [0.5,0], [0.5,0.5], [0,0.5], [0.5,-0.5], [-0.5,-0.5], [-0.5,0.5]]
        opt_params, residuals = multistart(optfunc, _costfunc, guesses)
    elif u1boundary:
        u1, u2 = guess[:]
        if not asym: guesses = [ [0, u2], [0.5, u2], [-0.5, u2] ]
        elif asym: guesses = [ [0, u2], [0,-u2], [0.5, u2], [-0.5, u2], [0.5, -u2], [-0.5, -u2] ]
        opt_params, residuals = multistart(optfunc, _costfunc, guesses)
    elif u2boundary:
        u1, u2 = guess[:]
        if not asym: guesses = [ [u1, 0], [u1, 0.5], [u1, -0.5] ]
        else: guesses = [ [u1, 0], [u1, 0.5], [u1, -0.5], [-u1, 0.5], [-u1, -0.5] ]
        opt_params, residuals = multistart(optfunc, _costfunc, guesses)
    residuals = np.sum(np.abs(residuals))
    return opt_params, residuals
    """

def check_jacobian(coefs, g, _costfunc, guess_params, ndisks):
    opt = least_squares(_costfunc, guess_params, verbose=0, ftol=None, xtol=1e5, gtol=None)
    print("finite diff jacobian ", opt.jac)
    if (coefs.shape[1] == 2):
        def jacobian(u):
            ux, uy = u[0], u[1]
            jac = np.zeros((ndisks,2))
            for disk in range(ndisks):
                sincos = np.cos(np.pi * np.dot(g[disk],u))*np.sin(np.pi * np.dot(g[disk],u))
                for dim in range(2):
                    jac[disk,dim] = 2*g[disk][dim]*np.pi*coefs[disk,0]*sincos
            return jac
    elif (coefs.shape[1] == 3):
        def jacobian(u):
            ux, uy = u[0], u[1]
            # number of disks by number of u dimensions
            # jac[disk, dim] is partial deriv of A[disk] cos2 (pi u.g[disk]) + B[disk] wrt u[dim]
            jac = np.zeros((ndisks,2)) 
            for disk in range(ndisks):
                # d(Ai cos2 (pi u.gi))/dux = -2 Ai sin(pi u.gi) cos(pi u.gi) * (pi gi[x]) := 2 Ai sincos * (pi gi[x])
                sin = np.sin(np.pi * np.dot(g[disk],u))
                cos = np.cos(np.pi * np.dot(g[disk],u)) 
                sincos = sin*cos
                # d(Bi sincos (pi u.gi))/dux = Bi(cos2-sin2) * (pi gi[x])
                cos2minussin2 = cos*cos - sin*sin 
                for dim in range(2):
                    # dI[disk]/du[dim] from Ai term := -2 * g[disk][dim] * pi * A[disk] * sincos
                    jac[disk,dim]  = 2*g[disk][dim]*np.pi*coefs[disk,0]*sincos
                    # dI[disk]/du[dim] from Bi term := g[disk][dim] * pi * B[disk] * cos2minussin2
                    jac[disk,dim] += -1*g[disk][dim]*np.pi*coefs[disk,1]*cos2minussin2
            return jac
    print("analytic jacobian ", jacobian(opt.x))
    print("difference ", opt.jac - jacobian(opt.x))

def fit_xy(ndisks, norm_df_set, coefs, x, y, nx, ny, g, guess=None, multistart_bool=False, multistart_neighbor_bool=False, delta=0.2, quasiN=False):

    val_vec = norm_df_set[:,x,y].flatten() # get the intensity values
    # coefs[:,1] += 0.1 (used with check_jacobian... to verify on first iter when inital condition B[:]=0)
    
    # this is the function we want to fit to
    def _fit_func(u):
        vals = np.zeros(ndisks)
        for disk in range(ndisks):
            if coefs.shape[1] == 2: # fit A, B only
                vals[disk] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u) ) ** 2 + coefs[disk,1]
            elif coefs.shape[1] == 3: # fit ABC
                vals[disk] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u) ) ** 2 + coefs[disk,2]
                vals[disk] += coefs[disk,1] * np.cos( np.pi * np.dot(g[disk],u) ) * np.sin( np.pi * np.dot(g[disk],u) )
        return vals
    # returns the resiudals
    def _costfunc(vec): return val_vec - _fit_func(vec[0:2])

    # if the guess function is provided, start the optimization from this point, otherwise start from zero
    N = 2
    if guess is None: guess_params = np.zeros(N)
    else: guess_params = guess[x,y,:]
    # upper and lower bounds for u is [-1/2, 1/2] since displacement vectors greater or less than this value will be
    # equivalent to some in this range (displament field is invariant to translations by a full unit cell )
    lowerbounds = -0.50 * np.ones(N)
    upperbounds =  0.50 * np.ones(N)
    bounds = (lowerbounds, upperbounds)
    # check_jacobian(coefs, g, _costfunc, guess_params, ndisks)
        
    if quasiN: 
        optfunc = lambda guess_params: least_squares(_costfunc, guess_params, bounds=bounds, verbose=0, ftol=None, xtol=1e-3, gtol=None)
    
    else:
        # function returns Ii = Ai cos2 (pi u . gi) + Bi
        # parameters are [ux, uy]
        # jac[i,j] is partial derivative of func[i] with respect to parameter[j]
    
        if (coefs.shape[1] == 2): # Ii = Ai cos2 (pi u.gi) + Bi 
        
            def jacobian(u):
                ux, uy = u[0], u[1]
                # number of disks by number of u dimensions
                # jac[disk, dim] is partial deriv of A[disk] cos2 (pi u.g[disk]) + B[disk] wrt u[dim]
                jac = np.zeros((ndisks,2)) 
                for disk in range(ndisks):
                    # dAi/dux = -2 Ai sin(pi u.gi) cos(pi u.gi) * (pi gi[x]) := -2 Ai sincos * (pi gi[x])
                    sincos = np.cos(np.pi * np.dot(g[disk],u)) * np.sin(np.pi * np.dot(g[disk],u))
                    for dim in range(2):
                        # dA[disk]/du[dim] := -2 * g[disk][dim] * pi * A[disk] * sincos
                        jac[disk,dim] = 2*g[disk][dim]*np.pi*coefs[disk,0]*sincos
                return jac

            optfunc = lambda guess_params: least_squares(_costfunc, guess_params, jac=jacobian, bounds=bounds, verbose=0, ftol=None, xtol=1e-3, gtol=None)

        if (coefs.shape[1] == 3): # Ii = Ai cos2 (pi u.gi) + Bi sincos(pi u.gi) + Ci
        
            def jacobian(u):
                ux, uy = u[0], u[1]
                # number of disks by number of u dimensions
                # jac[disk, dim] is partial deriv of A[disk] cos2 (pi u.g[disk]) + B[disk] wrt u[dim]
                jac = np.zeros((ndisks,2)) 
                for disk in range(ndisks):
                    # d(Ai cos2 (pi u.gi))/dux = -2 Ai sin(pi u.gi) cos(pi u.gi) * (pi gi[x]) := 2 Ai sincos * (pi gi[x])
                    sin = np.sin(np.pi * np.dot(g[disk],u))
                    cos = np.cos(np.pi * np.dot(g[disk],u)) 
                    sincos = sin*cos
                    # d(Bi sincos (pi u.gi))/dux = Bi(cos2-sin2) * (pi gi[x])
                    cos2minussin2 = cos*cos - sin*sin 
                    for dim in range(2):
                        # dI[disk]/du[dim] from Ai term := -2 * g[disk][dim] * pi * A[disk] * sincos
                        jac[disk,dim]  = 2*g[disk][dim]*np.pi*coefs[disk,0]*sincos
                        # dI[disk]/du[dim] from Bi term := g[disk][dim] * pi * B[disk] * cos2minussin2
                        jac[disk,dim] += -1*g[disk][dim]*np.pi*coefs[disk,1]*cos2minussin2
                return jac

            optfunc = lambda guess_params: least_squares(_costfunc, guess_params, jac=jacobian, bounds=bounds, verbose=0, ftol=None, xtol=1e-3, gtol=None)

    if multistart_bool and not multistart_neighbor_bool:
        # get all the multi-start positions to try
        if guess is None:
            guesses = [ [0.25,0.25], [-0.25,-0.25], [0.25,-0.25], [-0.25,0.25], [0,0.25],     [0.25,0],     [-0.25,0],      [0,-0.25],
                        [0.37,0.12], [0.12,0.37],   [-0.37,0.12], [-0.12,0.37], [0.37,-0.12], [0.12,-0.37], [-0.37,-0.12],  [-0.12,-0.37], [0.0, 0.0]]
        else:
        # if provided a guess then do a series of displacements about this value of a magnitude set by delta
            guess_val = guess[x,y,:]
            guesses = [ guess_val + [0, delta], guess_val + [delta, 0], guess_val + [0, -delta], guess_val + [-delta,0],
                        guess_val + [delta, delta], guess_val + [delta, -delta], guess_val + [-delta, delta] ]
            for i in range(len(guesses)): 
                temp = rz_helper_pair(guesses[i][0], guesses[i][1], sign_wrap=False)
                guesses[i] = temp
        opt_params, opt_residuals = multistart(optfunc, _costfunc, guesses)
    elif multistart_neighbor_bool:
          if guess is None: print('for neighbor fit need guess vector pls')
          neighbor_u, neighbor_v = get_neighbors(x, y, guess, nx, ny)
          guesses = []
          for i in range(len(neighbor_u)): guesses.append([neighbor_u[i], neighbor_v[i]])
          opt_params, opt_residuals = multistart(optfunc, _costfunc, guesses)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            opt = least_squares(_costfunc, guess_params, bounds=bounds, verbose=0, ftol=None, xtol=1e-3, gtol=None)
        opt_params = opt.x
        opt_residuals = _costfunc(opt.x)
    uvecs = opt_params
    residuals = opt_residuals
    return uvecs, residuals

def fit_together_serial(df_set, g, guess = None, ncoefs= 2, quasiN=False):

    ndisks = df_set.shape[0]
    nx = df_set.shape[1]
    ny = df_set.shape[2]
    norm_df_set = np.zeros((ndisks, nx, ny))
    for disk in range(ndisks):
        norm_df_set[disk,:,:] = normalize(df_set[disk,:,:])
   
    # this is the function we want to fit to
    def _fit_func(coefs, u):
        vals = np.zeros((ndisks, nx, ny))
        for i in range(nx):
            for j in range(ny):
                for disk in range(ndisks):
                    if ncoefs == 3:
                        vals[disk, i,j] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u[i,j,:]) ) ** 2 + coefs[disk,2]
                        vals[disk, i,j] += coefs[disk,1] * np.cos( np.pi * np.dot(g[disk],u[i,j,:]) ) * np.sin( np.pi * np.dot(g[disk],u[i,j,:]) )
                    elif ncoefs == 2:
                        vals[disk, i,j] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u[i,j,:]) ) ** 2 + coefs[disk,1]
        return vals

    # returns the resiudals
    def _costfunc(vec): 
        coefs = vec[:ndisks*ncoefs].reshape(ndisks, ncoefs)
        u = vec[ndisks*ncoefs:].reshape(nx, ny, 2)
        return (norm_df_set - _fit_func(coefs, u)).flatten()

    lowerbounds = np.concatenate( (-1 * np.ones(ncoefs*ndisks), -0.50 * np.ones(2*nx*ny)), axis = 0 )
    upperbounds = np.concatenate( (1 * np.ones(ncoefs*ndisks),  0.50 * np.ones(2*nx*ny)), axis = 0 )
    
    #coefguess = np.zeros((ndisks, 3))
    #coefguess[:,0] = 1
    #guess_params = np.concatenate( (coefguess.flatten(), 0.0 * np.ones(2*nx*ny)), axis = 0 )

    guess_params = guess
    start_resid = _costfunc(guess_params)
    print('start RMS of ', np.sqrt(np.mean([r**2 for r in start_resid.flatten()])))

    bounds = (lowerbounds, upperbounds)

    test_jac = False
    if test_jac:
        
        opt = least_squares(_costfunc, guess_params, bounds=bounds, verbose=0, ftol=None, xtol=1e-3, gtol=None)
        #print(opt.jac)

        for i in range(10):
            print(' '.join([str(el) for el in opt.jac[i,:100]]), '...')
        print('...')

        def jacobian(inp_vec):

            # function values are intensity[disk, i,j].flatten()
            # function parameters are [ coef(12, ncoefs).flatten(), u(i,j,dim).flatten() ]
            
            n_vals = len(norm_df_set.flatten())
            n_params = len(inp_vec)
            u = inp_vec[ndisks*ncoefs:].reshape(nx, ny, 2)
            coefs = inp_vec[:ndisks*ncoefs].reshape(ndisks, ncoefs)

            jac = np.zeros((n_vals, n_params))

            # the coefficients

            for val in range(n_vals):

                vdisk, vi, vj = np.unravel_index(val, (ndisks, nx, ny))

                # coefficents
                for param in range(ndisks*ncoefs):

                    pdisk, pn = np.unravel_index(param, (ndisks, ncoefs))

                    if (vdisk == pdisk):
                        if pn == 1: # for coefs[disk,1], B term, constant 
                            jac[val,param] = - 1
                        elif pn == 0: # for coefs[disk,0], A term, cos2 coef
                            jac[val,param] = - np.cos(np.pi * np.dot(g[vdisk],u[vi, vj, :]))**2
                            
                # u values
                for param in range(ndisks*ncoefs, n_params):

                    pi,pj,pdim = np.unravel_index(param-ndisks*ncoefs, (nx, ny, 2))

                    if (pi == vi and pj == vj):
                        sincos = np.cos(np.pi * np.dot(g[vdisk],u[vi, vj, :]))*np.sin(np.pi * np.dot(g[vdisk],u[vi, vj, :]))
                        jac[val,param] = 2*g[vdisk][pdim]*np.pi*coefs[vdisk,0]*sincos

            return jac

        for _ in range(5): print('*************************************************')
        jtest = jacobian(opt.x)
        for i in range(10):
            print(' '.join([str(el) for el in jtest[i,:100]]), '...')
        print('...')

        print('checking for max difference')
        want0 = opt.jac - jtest
        print(np.max([abs(el) for el in want0.flatten()]))

        exit()

    if quasiN:

        opt = least_squares(_costfunc, guess_params, bounds=bounds, verbose=2, ftol=None, xtol=1e-3, gtol=None)

    else:

        # function returns Ii = Ai cos2 (pi u(x,y) . gi) + Bi, i from 0 to 12
        # parameters are [coefs ... ux, uy]
        # jac[i,j] is partial derivative of func[i] with respect to parameter[j]
        assert(ncoefs == 2)
        def jacobian(inp_vec):

            # function values are intensity[disk, i,j].flatten()
            # function parameters are [ coef(12, ncoefs).flatten(), u(i,j,dim).flatten() ]
            
            n_vals = len(norm_df_set.flatten())
            n_params = len(inp_vec)
            u = inp_vec[ndisks*ncoefs:].reshape(nx, ny, 2)
            coefs = inp_vec[:ndisks*ncoefs].reshape(ndisks, ncoefs)

            jac = np.zeros((n_vals, n_params))

            # the coefficients

            for val in range(n_vals):

                vdisk, vi, vj = np.unravel_index(val, (ndisks, nx, ny))

                # coefficents
                for param in range(ndisks*ncoefs):

                    pdisk, pn = np.unravel_index(param, (ndisks, ncoefs))

                    if (vdisk == pdisk):
                        if pn == 1: # for coefs[disk,1], B term, constant 
                            jac[val,param] = - 1
                        elif pn == 0: # for coefs[disk,0], A term, cos2 coef
                            jac[val,param] = - np.cos(np.pi * np.dot(g[vdisk],u[vi, vj, :]))**2
                            
                # u values
                for param in range(ndisks*ncoefs, n_params):

                    pi,pj,pdim = np.unravel_index(param-ndisks*ncoefs, (nx, ny, 2))

                    if (pi == vi and pj == vj):
                        sincos = np.cos(np.pi * np.dot(g[vdisk],u[vi, vj, :]))*np.sin(np.pi * np.dot(g[vdisk],u[vi, vj, :]))
                        jac[val,param] = 2*g[vdisk][pdim]*np.pi*coefs[vdisk,0]*sincos

            return jac
        opt = least_squares(_costfunc, guess_params, jac=jacobian, bounds=bounds, verbose=2, ftol=None, xtol=1e-3, gtol=None)

    opt_params = opt.x
    opt_residuals = _costfunc(opt.x)

    coefs = opt_params[:ndisks*ncoefs].reshape(ndisks, ncoefs)
    uvecs = opt_params[ndisks*ncoefs:].reshape(nx, ny, 2)
    residuals = opt_residuals.reshape(nx, ny, ndisks)
    return coefs, uvecs, residuals

####################################################################################################
# fit A_j and B_j s.t. I_j(x,y) = A_j * cos^2(pi g_j dot u(x,y)) + B_j
####################################################################################################
def fit_ABC_disk(norm_df, u, nx, ny, g, coef_guess):
    I = norm_df.flatten()
    u = u.reshape(nx*ny, 2)
    ncoefs = len(coef_guess)
    A = np.zeros((nx*ny, ncoefs))
    for i in range(nx*ny):
        A[i,0] = np.cos(np.pi * np.dot(g, u[i])) ** 2
        if ncoefs > 2 : A[i,1] = np.cos(np.pi * np.dot(g, u[i])) * np.sin(np.pi * np.dot(g, u[i]))
        A[i,-1] = 1 
    opt = lsq_linear(A, I)
    opt_params = opt.x 
    success = opt.success
    opt_cost = opt.cost
    return opt_params, opt_cost

def fit_ABC(df_set, u, nx, ny, g, coef_guess, nproc=1, parallel=False, norm_bool=False):
    ndisks = df_set.shape[0]
    norm_df_set = np.zeros((ndisks, nx, ny))
    uvecs = np.zeros((nx, ny, 2))
    if norm_bool:
        for disk in range(ndisks):
            norm_df_set[disk,:,:] = normalize(df_set[disk,:nx,:ny])
        df_set = norm_df_set   
    if parallel: coef, residuals = fit_ABC_parallel(ndisks, df_set, coef_guess, nx, ny, g, u, nproc)
    else: coef, residuals = fit_ABC_serial(ndisks, df_set, coef_guess, nx, ny, g, u)
    return coef, residuals

def fit_ABC_parallel(ndisks, norm_df_set, coef_guess, nx, ny, g, u, nproc=4):
    fit_wrap = lambda diskno: fit_ABC_disk(norm_df_set[diskno,:,:], u, nx, ny, g[diskno], coef_guess[diskno,:])
    with Pool(processes=nproc) as pool: output = pool.map(fit_wrap, range(ndisks))
    coefs = np.zeros(coef_guess.shape)
    residuals = np.zeros((ndisks, 1))
    for i in range(ndisks):     
        coefs[i, :] = output[i][0]
        residuals[i] = output[i][1]
    return coefs, residuals

def fit_ABC_serial(ndisks, norm_df_set, coef_guess, nx, ny, g, u):
    coefs = np.zeros((coef_guess.shape[0], coef_guess.shape[1]))
    residuals = np.zeros((ndisks,1))
    fit_wrap = lambda diskno: fit_ABC_disk(norm_df_set[diskno,:,:], u, nx, ny, g[diskno], coef_guess[diskno,:])
    for diskno in range(ndisks):
        print("{}% done...".format(100*diskno/ndisks))
        coefs[diskno,:], residuals[diskno] = fit_wrap(diskno)
    return coefs, residuals

####################################################################################################
# fit u(x,y) s.t. I_j(x,y) = A_j * cos^2(pi g_j dot u(x,y)) + B_j
####################################################################################################
def fit_u(df_set, coefs, nx, ny, g, bin_w=1, nproc=4, norm_bool=True, guess=None, multistart_bool=False, parallel=False, multistart_neighbor_bool=False, delta=0.2):
    ndisks = df_set.shape[0]
    nx = nx // bin_w
    ny = ny // bin_w
    norm_df_set = np.zeros((ndisks, nx, ny))
    if norm_bool:
        for disk in range(ndisks):
            temp = bin(normalize(df_set[disk,:,:]), bin_w)
            norm_df_set[disk,:nx,:ny] = temp[:nx, :ny]
    else:
        for disk in range(ndisks):
            temp = bin(df_set[disk,:,:], bin_w)
            norm_df_set[disk,:nx,:ny] = temp[:nx, :ny]
    if parallel: uvecs, residuals = fit_u_parallel(ndisks, norm_df_set, coefs, nx, ny, g, nproc, guess, multistart_bool, multistart_neighbor_bool, delta)
    else: uvecs, residuals = fit_u_serial(ndisks, norm_df_set, coefs, nx, ny, g, guess, multistart_bool, multistart_neighbor_bool, delta)
    return uvecs, residuals
