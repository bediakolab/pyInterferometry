
from utils import bomb_out
import numpy as np  
from visualization	import displacement_colorplot	
from basis_utils import cartesian_to_latticevec	
from scipy.optimize import least_squares
import warnings
from pathos.multiprocessing import Pool 
import matplotlib.pyplot as plt
from fitting_functions import cos2_sincos_bl_fitting


def optim_compare():

    # set up fake u data for testing
    gvecs = [ [1,0], [0,1], [1,1], [1,-1], [-1,1], [-1,-1], [2,-1], [-2,1], [1,-2], [-1,2], [-1,0], [0,-1]]
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs = np.array(gvecs)
    ncoefs = 3
    ndisks = len(gvecs)
    nx, ny = 10,10 # small test want it fast
    I = np.zeros((ndisks, nx, ny))
    u_cart = np.zeros((nx, ny, 2))
    X, Y = np.meshgrid(np.arange(-1, 0.7, 0.1), np.arange(-1, 0.7, 0.1)) # might not be realisitic resolution but not testing median fit 
    for i in range(nx):
        for j in range(ny):
            u_cart[i,j,:] = [-Y[i,j], X[i,j]]
    f, ax = plt.subplots()
    displacement_colorplot(ax, u_cart[:,:,0], u_cart[:,:,1])
    plt.show()

    # set up fake intensity data for testing
    u_lv = cartesian_to_latticevec(u_cart)
    for disk in range(ndisks):
        for i in range(nx):
            for j in range(ny):
                gdotu = np.dot(gvecs[disk], np.array([u_lv[i,j,0], u_lv[i,j,1]]))
                I[disk, i, j] = 0.9 * np.cos(np.pi * gdotu)**2 + 0.1

    # testing the fit code to make sure it gives the right uvecs when NOT provided with perfect inital guesses
    guess_u = np.zeros((nx,ny,2))
    guess_coef = np.zeros((ndisks, ncoefs))
    guess_coef[:,0] = 1
    guess_coef[:,1] = 0.1
    guess_coef[:,2] = 0.05
    guess = np.zeros()
    guess[:ndisks*ncoefs] = guess_coef.flatten()
    guess[ndisks*ncoefs:] = guess_u.flatten()
    fit_coef_and_u(I, gvecs, guess, nceof, nx, ny)
    print("interferometry unit test 8 passing")


def fit_coef_and_u(I, gvecs, guess, ncoefs, nx, ny):

    ndisks      = len(gvecs)
    nparams     = ndisks*ncoefs + nx*ny*2
    expected    = I.flatten()
    lowerbounds = [ -2.0 if i < ndisks*ncoefs else -0.5 for i in range(nparams) ]
    upperbounds = [  2.0 if i < ndisks*ncoefs else  0.5 for i in range(nparams) ]
    print(lowerbounds)
    fit_func = lambda params: cos2_sincos_bl_fitting(params[ndisks*ncoefs:].reshape(nx,ny,2), params[:ndisks*ncoefs].reshape(ndisks, ncoefs), gvecs)

    u, c = guess[ndisks*ncoefs:].reshape(nx,ny,2), guess[:ndisks*ncoefs].reshape(ndisks, ncoefs)
    print(u)
    print(c)
    print(fit_func(guess))
    exit()
        
    return fit_least_squares(fit_func, expected, guess, lowerbounds, upperbounds)


def fit_least_squares(fit_func, expected, guess, lowerbounds, upperbounds):
    def sqL2_norm(params): 
        diffs = expected - fit_func(params).flatten()
        return np.sum([diff**2 for diff in diffs]) 
    bounds = (lowerbounds, upperbounds)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opt = least_squares(sqL2_norm, guess, bounds=bounds, verbose=0, ftol=1e-9, xtol=1e-8, gtol=1e-8)
        opt_params = opt.x 
        opt_residuals = np.sum(sqL2_norm(opt.x))
    return opt_params, opt_residuals

# preform multistart 
def multistart(optfunc, costfunc, guesses):
    prev_max, had_success = np.inf, False
    for guess in guesses:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                opt = optfunc(guess)
            opt_residuals_new = costfunc(opt.x)
            sucess = 1
            had_success = True
        except: sucess = 0
        if sucess and max(abs(opt_residuals_new)) < prev_max:
            opt_residuals = opt_residuals_new
            prev_max = max(abs(opt_residuals_new))
            opt_params = opt.x 
    if not had_success: bomb_out("OPT ERROR: every multistart optimization failed")
    return opt_params, opt_residuals

# n : number of fit stages 
# fitfunc : wrapper to fitting function that just takes integer < n (stage #) as 
# argument and returns params (shape of guess) and residuals
def staged_fit_wrapper(n, guess, fitfunc, nproc=1):
    if nproc > 1: return fit_parallel(n, nproc, guess, fitfunc)
    else: return fit_serial(n, guess, fitfunc)

def staged_fit_parallel(n, nproc, guess, fitfunc):
    with Pool(processes=nproc) as pool: output = pool.map(fitfunc, range(n))
    params = np.zeros(guess.shape)
    residuals = np.zeros((n, 1))
    for i in range(n):     
        params[i, :] = output[i][0]
        residuals[i] = output[i][1]
    return params, residuals

def staged_fit_serial(n, guess, fitfunc):
    coefs = np.zeros(guess.shape)
    residuals = np.zeros((n,1))
    for i in range(n):
        if i%10 == 0: print("{}% done...".format(100*i/n))
        params[i,:], residuals[i] = fitfunc(i)
    return params, residuals


optim_compare()