
import matplotlib
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
from basis_utils import lv_to_rzlv, latticevec_to_cartesian, cartesian_to_latticevec
import warnings

def verify_dfs(savepath, diskset, coefs, ulv):

    I = diskset.df_set()
    g = diskset.clean_normgset(sanity_plot = False)
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

def fit_full_hexagon(diskset, ncoefs=2, binw=1, g=None, plot=True, guess=None, A=None):

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
        g = diskset.clean_normgset(sanity_plot = False)
    elif isinstance(diskset, np.ndarray):
        I = diskset 
        if not (isinstance(g, np.ndarray)):
            print('ERROR: gvectors must be provided to fit_full_hexagon if I isnt a DiskSet instance')
            exit()
    else: print('Unrecognized type for intensities given to fit_full_hexagon, needs to be an array or DiskSet instance')


    Ibin = np.zeros((I.shape[0], I.shape[1]//binw, I.shape[2]//binw))
    for i in range(I.shape[0]):
        Ibin[i,:,:] = bin(I[i,:,:], bin_w=binw, method=np.sum)
    I = Ibin
    
    # if ncoefs=3 A,B,C for Acos^2 + Bsincos + C fit
    # if ncoefs=2 A,B   for Acos^2 + B fit
    # first multistart fit for u vectors with A and B fixed as 1 and 0 respectively
    ndisks, nx, ny = I.shape[0], I.shape[1], I.shape[2]
    coefs = np.zeros((ndisks, ncoefs))
    coefs[:,0] = 1
    tic()
    uvecs, residuals = fit_u(I, coefs, nx, ny, g=g, guess=guess, parallel=True, nproc=12, norm_bool=True, multistart_bool=True, multistart_neighbor_bool=False)
    toc('ufit')

    # iterative fit
    print('starting iterative u, A, B, C fits')
    for n in range(2):
        tic()
        coefs, resid = fit_ABC(I, uvecs, nx, ny, g, coefs, nproc=12, parallel=False, norm_bool=True)
        uvecs, resid = fit_u(I, coefs, nx, ny, g, guess=uvecs, nproc=12, parallel=True, norm_bool=True, multistart_bool=False)
        toc('u abc')

    print('starting iterative median u, A, B, C fits')
    for n in range(10):
        tic()
        coefs, resid = fit_ABC(I, uvecs, nx, ny, g, coefs, nproc=12, parallel=False, norm_bool=True)
        uvecs, residuals = fit_u(I, coefs, nx, ny, g, nproc=12, guess=uvecs, parallel=True, norm_bool=True, multistart_neighbor_bool=True)
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
    residuals = np.zeros((nx, ny))
    # relies on the function fit_xy which fits an individual pixel, finds a single u from a single I_j = A_j * cos^2(pi g_j dot u) + B_j
    fit_xy_wrap = lambda x,y: fit_xy(ndisks, norm_df_set, coefs, x, y, nx, ny, g, guess, multistart_bool, multistart_neighbor_bool, delta)
    # loop over each pixel and use the fit_xy function on each
    for x in range(nx):
        print("{}% done...".format(100*x/nx))
        for y in range(ny):
            uvecs[x, y, :], residuals[x, y] = fit_xy_wrap(x, y)
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
    residuals = np.array(resid_slice).reshape(nx,ny)
    return uvecs, residuals

####################################################################################################
# fits the u vector for a single pixel of the dataset
# fit u s.t. I_j = A_j * cos^2(pi g_j dot u) + B_j
####################################################################################################

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

    return ufits, 0
        
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


def fit_xy(ndisks, norm_df_set, coefs, x, y, nx, ny, g, guess=None, multistart_bool=False, multistart_neighbor_bool=False, delta=0.2):
    val_vec = norm_df_set[:,x,y].flatten() # get the intensity values
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
    # upper and lower bounds for u is [-1, 1] since displacement vectors greater or less than this value will be
    # equivalent to some in this range (displament field is invariant to translations by a full unit cell )
    lowerbounds = -0.50 * np.ones(N)
    upperbounds =  0.50 * np.ones(N)
    bounds = (lowerbounds, upperbounds)
    if multistart_bool and not multistart_neighbor_bool:
        optfunc = lambda guess_params: least_squares(_costfunc, guess_params, bounds=bounds, verbose=0, ftol=None, xtol=1e-3, gtol=None)
        # get all the multi-start positions to try
        if guess is None:
            guesses = [ [0.25,0.25], [-0.25,-0.25], [0.25,-0.25], [-0.25,0.25], [0,0.25],     [0.25,0],     [-0.25,0],      [0,-0.25],
                        [0.37,0.12], [0.12,0.37],   [-0.37,0.12], [-0.12,0.37], [0.37,-0.12], [0.12,-0.37], [-0.37,-0.12],  [-0.12,-0.37], [0.0, 0.0]]
        else:
        # if provided a guess then do a series of displacements about this value of a magnitude set by delta
            guess_val = guess[x,y,:]
            guesses = [ guess_val + [0, delta], guess_val + [delta, 0], guess_val + [0, -delta], guess_val + [-delta,0],
                        guess_val + [delta, delta], guess_val + [delta, -delta], guess_val + [-delta, delta] ]
            for i in range(len(guesses)): guesses[i,:] = rz_helper_pair(guesses[i,0], guesses[i,1], sign_wrap=False)
        opt_params, opt_residuals = multistart(optfunc, _costfunc, guesses)
    elif multistart_neighbor_bool:
          if guess is None: print('for neighbor fit need guess vector pls')
          optfunc = lambda guess_params: least_squares(_costfunc, guess_params, bounds=bounds, verbose=0, ftol=None, xtol=1e-3, gtol=None)
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
    residuals = np.sum(np.abs(opt_residuals))
    return uvecs, residuals

####################################################################################################
# fit A_j and B_j s.t. I_j(x,y) = A_j * cos^2(pi g_j dot u(x,y)) + B_j
####################################################################################################
def fit_ABC_disk_old(norm_df, u, nx, ny, g, coef_guess):
    val_vec = norm_df.flatten()
    def _fit_func(coefs):
        val = np.zeros((nx, ny))
        for x in range(nx):
            for y in range(ny):
                if len(coefs) == 2: 
                    val[x,y] =  coefs[0] * np.cos( np.pi * np.dot(g, u[x,y]) ) ** 2 + coefs[1]
                elif len(coefs) == 3: 
                    val[x,y] =  coefs[0] * np.cos( np.pi * np.dot(g, u[x,y]) ) ** 2 + coefs[2]
                    val[x,y] += coefs[1] * np.cos( np.pi * np.dot(g, u[x,y]) ) * np.sin( np.pi * np.dot(g, u[x,y]) )
        return val.flatten()
    def _costfunc(coef):
        pred_vec = _fit_func(coef)
        return val_vec - pred_vec.flatten()
    lowerbounds = [-2.0 for coef in coef_guess] # might change??
    upperbounds = [ 2.0 for coef in coef_guess]
    bounds = (lowerbounds, upperbounds)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opt = least_squares(_costfunc, coef_guess, bounds=bounds, verbose=0, ftol=1e-9, xtol=1e-8, gtol=1e-8)
        opt_params = opt.x # coefs!
        opt_residuals = np.sum(_costfunc(opt.x))
    return opt_params, opt_residuals

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
    fit_wrap = lambda diskno: fit_ABC_disk(norm_df_set[diskno,:,:], u, nx, ny, g[diskno,:], coef_guess[diskno,:])
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
    fit_wrap = lambda diskno: fit_ABC_disk(norm_df_set[diskno,:,:], u, nx, ny, g[diskno,:], coef_guess[diskno,:])
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
