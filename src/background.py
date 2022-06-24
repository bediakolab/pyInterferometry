
import os
import pickle
import py4DSTEM
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from diskset import *
from masking import *
from utils import *
from new_utils import parse_filepath
from visualization import overlay_vdf

def background_main(path):

    probe_kernel, probe_kernel_FT, beamcenter = get_probe(os.path.join('..','data','probe_08132020.dm3'))
    prefix, dsnum, scan_shape = parse_filepath(path)
    os.makedirs(os.path.join('..', 'plots',   prefix,     'ds_{}'.format(dsnum)), exist_ok=True)
    savepath     = os.path.join('..',  'results', prefix,     'dat_ds{}.pkl'.format(dsnum))
    datacube     = py4DSTEM.io.read(os.path.join(path, "dp.h5"), data_id="datacube_0")
    datacube.set_scan_shape(scan_shape[0],scan_shape[1])

    max_dp       = np.max(datacube.data, axis=(0,1)).astype(float)
    probe_kernel = bin(probe_kernel, int(probe_kernel.shape[0]/max_dp.shape[0]))
    peaks        = find_Bragg_disks_single_DP(max_dp, probe_kernel, corrPower=1.0, sigma=2, minPeakSpacing=0.01,
                   edgeBoundary=0, relativeToPeak=3, minRelativeIntensity=0.0005, maxNumPeaks=400)
    diskset      = get_diskset(max_dp, peaks, scan_shape, beamcenter, dsnum, prefix)
    _, diskset   = select_disks(diskset)

    print('masking')
    sum_dp    = np.sum(datacube.data, axis=(0,1))
    mask_bs   = get_beamstop_mask(max_dp, dsnum)
    mask_peak = get_peak_mask(max_dp.shape[0], max_dp.shape[1], peaks, diskset, max_dp, dsnum, radius_factor=2.0)
    masked_dp = mask_off(max_dp, [mask_bs, mask_peak], dsnum)
    anom_mask = get_anomoly_mask(masked_dp, beamcenter, dsnum, bin_w=27)
    masked_dp = mask_off(sum_dp, [mask_bs, mask_peak, anom_mask], dsnum)

    print('background subtraction')
    beamcenter = diskset.set_com_central()
    background_fit = fit_background_lorenzian(masked_dp.as_type(float), beamcenter, dsnum, pad=260, bin_w=4)
    #datacube_sub = subtract_background(datacube, background_fit)
    #diskset.set_background(background_fit)
    diskset = integrate_disks(datacube, diskset, dsnum, prefix, sub=True)

    counter = 0
    radius_factor=1.0
    f, axes = plt.subplots(3, diskset.size_in_use)
    for n in range(diskset.size):
        if diskset.in_use(n):
            x = diskset.x(n)
            y = diskset.y(n)
            r = diskset.r(n)
            img = diskset.df(n)
            img_bknd = integrate_bknd_circ(background_fit,x,y,radius_factor*r)
            axes[0, counter].imshow(img, cmap='gray')
            axes[0, counter].set_title("Disk {}".format(n))
            axes[1, counter].imshow(img_bknd, cmap='gray')
            axes[1, counter].set_title("Disk {} Background".format(n))
            axes[2, counter].imshow(img-img_bknd, cmap='gray')
            axes[2, counter].set_title("Disk {} - Background".format(n))
            counter += 1
    plt.show()
    overlay_vdf(diskset, dsnum, prefix_sub, sub=True)

def subtract_background(datacube, background):
    nx, ny, nqx, nqy = datacube.data.shape
    datacube.data = datacube.data.astype(float)
    for x in range(nx):
        for y in range(ny):
            temp = np.copy(datacube.data[x,y,:,:])
            newdata = temp - background
            for i in range(nqx):
                for j in range(nqy):
                    datacube.data[x,y,i,j] = newdata[i,j].astype(float)
                    if datacube.data[x,y,i,j] != newdata[i,j]:
                        print('wtf')
                        print(temp[x,y,i,j])
                        print(datacube.data[x,y,i,j])
                        print(newdata[i,j])
                        exit()
    return datacube

def fit_background_ringed_guassian(masked_dp, beamcenter_orig, dsnum, pad=0, bin_w=1, plotflag=True):

    masked_dp = bin(masked_dp, bin_w) # bin the data if requested
    beamcenter = (beamcenter_orig[0]/bin_w, beamcenter_orig[1]/bin_w) # adjust beamcenter after binning
    nx, ny = masked_dp.shape
    flat_dp = masked_dp.flatten() # flatten matrix into an array

    # define the ringed gaussian function to fit to
    def ringed_gaussian(x0, y0, nx, ny, I0, Ipeak, Iring, rRing, s0, s1, s2):
        I_predicted = np.zeros((nx, ny))
        for x in range(nx):
            for y in range(ny):
                r = np.sqrt((x-x0)**2 + (y-y0)**2)
                f = I0 + Ipeak * np.exp(-0.5*(r/s0)**2)
                f += (r<rRing) * Iring * np.exp(-0.5*((r-rRing)/s1)**2)
                f += (r>=rRing) * Iring * np.exp(-0.5*((r-rRing)/s2)**2)
                I_predicted[x,y] = f
        return I_predicted.flatten()

    # wrapped version of the above function which uses the beamcenter for the ringed gaussian center
    def _ringed_gaussian(I0, Ipeak, Iring, rRing, s0, s1, s2):
        return ringed_gaussian(beamcenter[0], beamcenter[1], nx, ny, I0, Ipeak, Iring, rRing, s0, s1, s2)

    # cost function we aim to optimize, wrapper function which removes nan values from the data and then
    # returns the residuals of the _ringed_gaussian function
    def _costfunc(params):
        bool_array = ~np.isnan(flat_dp)
        flat_dp_trimmed = flat_dp[bool_array]
        fit = _ringed_gaussian(*params)
        fit_trimmed = fit[bool_array]
        return flat_dp_trimmed - fit_trimmed

    s0 = 10/(2*np.sqrt(2*np.log(2))) # guesses optimizable parameters - first ring standard devation
    s1 = 5/(2*np.sqrt(2*np.log(2)))  # guesses optimizable parameters - second ring standard devation
    s2 = 5/(2*np.sqrt(2*np.log(2)))  # guesses optimizable parameters - third ring standard devation
    guess_params = [0.01, 1, 0.2, 60/bin_w, s0, s1, s2] # guesses optimizable parameters
    lowerbounds = [-0.1, 0, 0, 0, 0, 0, 0] # lower bounds for the optimizable parameters
    upperbounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] # upper bounds for the optimizable parameters
    bounds = (lowerbounds, upperbounds)
    guess_fvals = _ringed_gaussian(*guess_params) # get the function values for these parameters
    guess_residuals = flat_dp - guess_fvals # get the residuals for these parameters

    # sanity check, plot the residuals and corresponding lorenzian for the these parameters
    if plotflag:
        f, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(guess_fvals.reshape(nx, ny))#, cmap='gray')
        ax2.imshow(guess_residuals.reshape(nx, ny),vmin=-0.5, vmax=0.5)#, cmap='gray')
        ax1.set_title("Ringed Guassian Guess")
        ax2.set_title("Guess Residuals")
        ax1.scatter(beamcenter[1], beamcenter[0])
        ax2.scatter(beamcenter[1], beamcenter[0])
        plt.subplots_adjust(wspace=0.4)
        plt.show()

    # preform the nonlinear least squares, ftol xtol gtol control accuracy of fit
    opt = least_squares(_costfunc, guess_params, bounds=bounds, verbose=0, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    opt_params = opt.x # extract the parameters x0, y0, A, B for the best fit
    opt_fvals = _ringed_gaussian(*opt_params) # get the function values for these parameters
    opt_residuals = flat_dp - opt_fvals # get the residuals for these parameters

    # sanity check, plot the residuals and corresponding lorenzian for the these parameters
    if plotflag:
        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(opt_fvals.reshape(nx, ny))#, cmap='gray')
        ax2.imshow(opt_residuals.reshape(nx, ny),vmin=-0.5, vmax=0.5)#, cmap='gray')
        ax1.set_title("Ringed Guassian Fit")
        ax2.set_title("Residuals")
        ax1.scatter(beamcenter[1], beamcenter[0])
        ax2.scatter(beamcenter[1], beamcenter[0])
        plt.subplots_adjust(wspace=0.4)
        plt.show()

    # function to return the ringed_gaussian function values with a shape correspinding to that of the
    # original dataset, accounts for  possible requested binning of the data
    def _ringed_gaussian_unbinned(I0, Ipeak, Iring, rRing, s0, s1, s2):
        return ringed_gaussian(beamcenter_orig[0], beamcenter_orig[1],
                nx*bin_w, ny*bin_w, I0, Ipeak, Iring, rRing, s0, s1, s2)

    # reshapes the unbinned fit data to a matrix
    return _ringed_gaussian_unbinned(*opt_params).reshape(nx*bin_w, ny*bin_w)

def fit_background_lorenzian(masked_dp, beamcenter_orig, dsnum, pad=200, bin_w=1, plotflag=True):

    nx_orig, ny_orig = masked_dp.shape
    masked_dp = masked_dp[int(beamcenter_orig[0])-pad:int(beamcenter_orig[0])+pad,
                          int(beamcenter_orig[1])-pad:int(beamcenter_orig[1])+pad]
    masked_dp = bin(masked_dp, bin_w) # bin the data if requested
    beamcenter = (pad/bin_w, pad/bin_w) # adjust beamcenter after binning
    nx, ny = masked_dp.shape
    flat_dp = masked_dp.flatten() # flatten matrix into an array

    def lorenzian(nx, ny, x0, y0, A, B): # define the lorenzian function
        I_predicted = np.zeros((nx, ny))
        for x in range(nx):
            for y in range(ny):
                r = np.sqrt((x-x0)**2 + (y-y0)**2)
                f = A*(1.0/np.pi)*(0.5*B)/(r**2 + (0.5*B)**2)
                I_predicted[x,y] = f
        return I_predicted.flatten()

    # wrapped version of the above function which uses the beamcenter for the ringed gaussian center
    def _lorenzian(A, B):
        return lorenzian(nx, ny, beamcenter[0], beamcenter[1], A, B)

    # cost function we aim to optimize, wrapper function which removes nan values from the data and then
    # returns the residuals of the _ringed_gaussian function
    def _costfunc(params):
        bool_array = ~np.isnan(flat_dp)
        flat_dp_trimmed = flat_dp[bool_array]
        fit = _lorenzian(*params)
        fit_trimmed = fit[bool_array]
        return flat_dp_trimmed - fit_trimmed

    maxv = np.nanmax(flat_dp)
    # B is 2A/pi*max
    guessB = 50
    guessA = 1/2 * guessB * np.pi * maxv
    guess_params = [guessA, guessB] #beamcenter[0], beamcenter[1], 3, 50
    lowerbounds = [0,0] # lower bounds for the optimizable parameters
    upperbounds = [np.inf, np.inf] # upper bounds for the optimizable parameters
    bounds = (lowerbounds, upperbounds)
    guess_fvals = _lorenzian(*guess_params) # get the function values for these parameters
    guess_residuals = flat_dp - guess_fvals # get the residuals for these parameters

    # sanity check, plot the residuals and corresponding lorenzian for the these parameters
    if plotflag:
        f, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(guess_fvals.reshape(nx, ny))#, cmap='gray')
        ax2.imshow(guess_residuals.reshape(nx, ny))#, cmap='gray')
        ax3.imshow(flat_dp.reshape(nx, ny))#, cmap='gray')
        ax1.set_title("Lorenzian Guess")
        ax1.scatter(beamcenter[1], beamcenter[0])
        ax2.scatter(beamcenter[1], beamcenter[0])
        ax3.scatter(beamcenter[1], beamcenter[0])
        ax2.set_title("Guess Residuals")
        ax3.set_title("Masked DP to fit")
        plt.subplots_adjust(wspace=0.4)
        plt.show()

    # preform the nonlinear least squares, ftol xtol gtol control accuracy of fit
    opt = least_squares(_costfunc, guess_params, bounds=bounds, verbose=0, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    opt_params = opt.x # extract the parameters x0, y0, A, B for the best fit
    opt_fvals = _lorenzian(*opt_params) # get the function values for these parameters
    opt_residuals = flat_dp - opt_fvals # get the residuals for these parameters

    # sanity check, plot the residuals and corresponding lorenzian for the these parameters
    if plotflag:
        f, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(opt_fvals.reshape(nx, ny))#, cmap='gray')
        ax2.imshow(opt_residuals.reshape(nx, ny))#, cmap='gray')
        ax3.imshow(flat_dp.reshape(nx, ny))#, cmap='gray')
        ax1.set_title("Lorenzian Fit")
        ax2.set_title("Residuals")
        ax3.set_title("DP to Fit")
        for ax in [ax1, ax2, ax3]:
            ax.scatter(beamcenter[1], beamcenter[0])
            for r in [10, 15, 20, 25]:
                circle = plt.Circle((beamcenter[1], beamcenter[0]), r, color='r', fill=False)
                ax.add_patch(circle)
        plt.subplots_adjust(wspace=0.4)
        plt.show()

    # function to return the ringed_gaussian function values with a shape correspinding to that of the
    # original dataset, accounts for  possible requested binning of the data
    def _lorenzian_unbinned(A, B):
        return lorenzian(nx*bin_w, ny*bin_w, beamcenter[0]*bin_w, beamcenter[1]*bin_w, A, B)

    fit_region = _lorenzian_unbinned(*opt_params).reshape(nx*bin_w, ny*bin_w)
    fit_overall = np.zeros((nx_orig, ny_orig))
    fit_overall[int(beamcenter_orig[0])-pad:int(beamcenter_orig[0])+pad,
                int(beamcenter_orig[1])-pad:int(beamcenter_orig[1])+pad] = fit_region

    return fit_overall


if __name__ == "__main__":
    background_main(sys.argv[1])
