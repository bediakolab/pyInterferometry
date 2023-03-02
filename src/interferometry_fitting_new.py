
import matplotlib
#matplotlib.use('Agg') # need this backend when rendering within parallelized loops
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

def refit_full_hexagon(diskset, inpcoefs, uvecs, prefix, dsnum=0, g=None, plot=True):

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
    
    ndisks, nx, ny = I.shape[0], I.shape[1], I.shape[2]

    ncoefs = len(inpcoefs)
    coefs = np.ones((ndisks, ncoefs))
    for n in range(ncoefs): coefs[:,n] = inpcoefs[n][:]
    displacement_plt_lvbasis(uvecs, nx, ny, 'before refit', prefix, dsnum, savebool=plot)

    print('starting iterative fits')
    tic()
    uvecs, residuals = fit_u(I, coefs, nx, ny, g, nproc=12)
    toc('fit 0')
    displacement_plt_lvbasis(uvecs, nx, ny, 'refit iter {}'.format(0), prefix, dsnum, savebool=plot)

    for n in range(5):
        tic()
        coefs, resid = fit_ABC(I, uvecs, nx, ny, g, coefs, nproc=4)
        uvecs, residuals = fit_u(I, coefs, nx, ny, g, nproc=12, asym=True)
        toc('fit {}'.format(n+1))
        displacement_plt_lvbasis(uvecs, nx, ny, 'refit iter {}'.format(n+1), prefix, dsnum, savebool=plot)

    for n in range(10):
        tic()
        coefs, resid = fit_ABC(I, uvecs, nx, ny, g, coefs, nproc=4)
        uvecs, residuals = fit_u(I, coefs, nx, ny, g, uvecs, nproc=12, asym=True, neighbor=True)
        toc('fit neighbor high tol {}'.format(n))
        displacement_plt_lvbasis(uvecs, nx, ny, 'neighbor refit iter {}'.format(n+1), prefix, dsnum, savebool=plot)

    return coefs, uvecs

def fit_full_hexagon(diskset, prefix, ncoefs=3, dsnum=0, g=None, plot=True):

    if isinstance(diskset, DiskSet):
        I = diskset.df_set()
        g = diskset.clean_normgset(sanity_plot = True, prefix = prefix, dsnum = dsnum)
    elif isinstance(diskset, np.ndarray):
        I = diskset 
        if not (isinstance(g, np.ndarray)):
            print('ERROR: gvectors must be provided to fit_full_hexagon if I isnt a DiskSet instance')
            exit()
    else: print('Unrecognized type for intensities given to fit_full_hexagon, needs to be an array or DiskSet instance')

    I = I[:, :50, :50]
    
    ndisks, nx, ny = I.shape[0], I.shape[1], I.shape[2]
    coefs = np.zeros((ndisks, ncoefs))
    coefs[:,1] = 1 # guess A are all 1, B are all 0
    uvecs, residuals = fit_u(I, coefs, nx, ny, g, nproc=12)
    displacement_plt_lvbasis(uvecs, nx, ny, 'after first fit', prefix, dsnum, savebool=plot)
    exit()

    # iterative fit
    print('starting iterative u, A, B, C fits')
    for n in range(1):
        coefs, resid = fit_ABC(I, uvecs, nx, ny, g, coefs, nproc=4)
        uvecs, resid = fit_u(I, coefs, nx, ny, g, guess=uvecs, nproc=12, asym=True)
        displacement_plt_lvbasis(uvecs, nx, ny, 'after refit {}'.format(n), prefix, dsnum, savebool=plot)

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
def fit_u_serial(ndisks, norm_df_set, coefs, nx, ny, g, guess=None, asym=False, neighbor=False):
    print('starting call to ufit serial')
    uvecs = np.zeros((nx, ny, 2))
    residuals = np.zeros((nx, ny))
    # relies on the function fit_xy which fits an individual pixel, finds a single u from a single I_j = A_j * cos^2(pi g_j dot u) + B_j
    if not neighbor: fit_xy_wrap = lambda x,y: fit_xy(ndisks, norm_df_set, coefs, x, y, g, guess, asym)
    else: fit_xy_wrap = lambda x,y: fit_xy_neighbor(ndisks, norm_df_set, coefs, guess, x, y, g)
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
def fit_u_parallel(ndisks, norm_df_set, coefs, nx, ny, g, guess=None, nproc=4, asym=False, neighbor=False):
    # relies on the function fit_xy which fits an individual pixel, finds a single u from a single I_j = A_j * cos^2(pi g_j dot u) + B_j
    # given a compound index xy, calles this function to calculate u
    print('starting call to ufit parallel')
    def fit_wrapper(xy):
        x = xy // ny
        y = xy % ny
        if not neighbor: return fit_xy(ndisks, norm_df_set, coefs, x, y, g, guess, asym)
        else: return fit_xy_neighbor(ndisks, norm_df_set, coefs, guess, x, y, g)
    # loop over each pixel and use the fit_xy function on each, but now do so over multiple processors
    with Pool(processes=nproc) as pool:
        output = pool.map(fit_wrapper, range(nx*ny))
    uvec_slice =  [outputel[0] for outputel in output]
    resid_slice = [outputel[1] for outputel in output]
    uvecs = np.array(uvec_slice).reshape(nx,ny,2)
    residuals = np.array(resid_slice).reshape(nx,ny)
    return uvecs, residuals

def find_g_index(g1, g2, g):
    for i in range(len(g)):
        if (g[i][0] == g1 and g[i][1] == g2): return i 
    print('failed to find {},{} gvector'.format(g1, g2))
    exit()

def fit_xy(ndisks, I, coefs, x, y, g, guess=None, neighbor=False, asym=False):

    I = I[:,x,y].flatten() # get the intensity values, per disk normalized to [0,1]

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

def fit_xy_neighbor(ndisks, I, coefs, uvecs, x, y, g, asym=True, hightol=False):
    
    nx, ny = I.shape[1], I.shape[2]
    I = I[:,x,y].flatten() # get the intensity values, per disk normalized to [0,1]
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
    if not hightol: optfunc = lambda g: least_squares(_costfunc, g, bounds=bounds, verbose=0, ftol=None, xtol=1e-4, gtol=None)
    elif hightol:   optfunc = lambda g: least_squares(_costfunc, g, bounds=bounds, verbose=0, ftol=None, xtol=1e-6, gtol=None)
    guesses = [ uvecs[x,y] ]
    if y > 0:  guesses.append( uvecs[x,y-1] )
    if x > 0:  guesses.append( uvecs[x-1,y] )
    if y < ny-1: guesses.append( uvecs[x,y+1] )
    if x < nx-1: guesses.append( uvecs[x+1,y] )
    if asym: 
        asym_guesses = []
        for guess in guesses:
            neg_guess = [-guess[0], -guess[1]]
            asym_guesses.append(guess)
            asym_guesses.append(neg_guess)
        guesses = asym_guesses
    opt_params, residuals = multistart(optfunc, _costfunc, guesses)
    residuals = np.sum(np.abs(residuals))
    return opt_params, residuals

def fit_u_coef(I, guessu, guessc, g):
    ndisks, nx, ny = I.shape[0], I.shape[1], I.shape[2]
    ncoefs = guessc.shape[1]
    val_vec = I.flatten() # get the intensity values ndisk x nx x ny variables
    def _fit_func(params):
        vals = np.zeros((ndisks, nx, ny))
        u = params[ndisks*ncoefs:].reshape(nx, ny, 2)
        coefs = params[:ndisks*ncoefs].reshape(ndisks, ncoefs)
        for disk in range(ndisks):
            for x in range(nx):
                for y in range(ny):
                    vals[disk,x,y] =  coefs[disk,0] * np.cos( np.pi * np.dot(g[disk],u[x,y]) ) ** 2 
                    if ncoefs > 1 : vals[disk,x,y] += coefs[disk,-1]
                    if ncoefs > 2 : vals[disk,x,y] += coefs[disk,1] * np.cos( np.pi * np.dot(g[disk],u[x,y]) ) * np.sin( np.pi * np.dot(g[disk],u[x,y]) )
        return vals.flatten()
    def _costfunc(vec): return val_vec - _fit_func(vec)
    lowerbounds = [-0.5 if i >= ndisks*ncoefs else -2 for i in range(nx*ny*2+ndisks*ncoefs)]
    upperbounds = [ 0.5 if i >= ndisks*ncoefs else  2 for i in range(nx*ny*2+ndisks*ncoefs)]
    guess = np.zeros(nx*ny*2+ndisks*ncoefs)
    guess[ndisks*ncoefs:] = guessu.flatten()
    guess[:ndisks*ncoefs] = guessc.flatten()
    bounds = (lowerbounds, upperbounds)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opt = least_squares(_costfunc, guess, bounds=bounds, verbose=2, ftol=1e-5, xtol=1e-5, gtol=None)
    opt_params = opt.x
    opt_residuals = _costfunc(opt.x)
    uvecs = opt_params[ndisks*ncoefs:].reshape(nx, ny, 2)
    coefs = opt_params[:ndisks*ncoefs].reshape(ndisks, ncoefs)
    residuals = np.sum(np.abs(opt_residuals))
    return uvecs, coefs, residuals

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

def fit_ABC(df_set, u, nx, ny, g, coef_guess, nproc=1):
    ndisks = df_set.shape[0]
    norm_df_set = np.zeros((ndisks, nx, ny))
    uvecs = np.zeros((nx, ny, 2))
    for disk in range(ndisks):
        norm_df_set[disk,:,:] = normalize(df_set[disk,:nx,:ny])
    df_set = norm_df_set   
    if nproc > 1: coef, residuals = fit_ABC_parallel(ndisks, df_set, coef_guess, nx, ny, g, u, nproc)
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
def fit_u(df_set, coefs, nx, ny, g, guess=None, bin_w=1, nproc=4, asym=False, neighbor=False):
    ndisks = df_set.shape[0]
    nx = nx // bin_w
    ny = ny // bin_w
    norm_df_set = np.zeros((ndisks, nx, ny))
    for disk in range(ndisks):
        temp = bin(normalize(df_set[disk,:,:]), bin_w)
        norm_df_set[disk,:nx,:ny] = temp[:nx, :ny]
    if nproc > 1: uvecs, residuals = fit_u_parallel(ndisks, norm_df_set, coefs, nx, ny, g, guess, nproc, asym, neighbor)
    else: uvecs, residuals = fit_u_serial(ndisks, norm_df_set, coefs, nx, ny, g, guess, asym, neighbor)
    return uvecs, residuals
