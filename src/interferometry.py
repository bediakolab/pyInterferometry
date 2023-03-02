
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pickle
from utils import boolquery
from new_utils import import_diskset, import_uvector, import_disket_uvector, get_diskset_index_options
from interferometry_fitting import fit_full_hexagon, refit_full_hexagon_from_bin

def main(ds):

    # fit binned by 2
    coefs, ufit = fit_full_hexagon(diskset, 3, binw=2)
    displacement_plt_lvbasis(ufit)
    with open(savepath, 'wb') as f: pickle.dump([diskset, coefs[:,0], coefs[:,1], coefs[:,2], ufit], f)

    # refit from bin
    coefs, ufit = refit_full_hexagon_from_bin(diskset, coefs, uvecs)
    displacement_plt_lvbasis(ufit)
    with open(savepath, 'wb') as f: pickle.dump([diskset, coefs[:,0], coefs[:,1], ufit], f)


def fit_all_datasets():
    # automatically run through all disksets
    for indx in get_diskset_index_options():
        diskset, prefix, dsnum = import_diskset(indx)
        if os.path.exists(os.path.join('..','results', prefix, 'dat_ds{}.pkl_fit_bin2_asym'.format(dsnum))): 
            print('dataset has already been fit (asym fit), skipping...')
            continue
        elif os.path.exists(os.path.join('..','results', prefix, 'dat_ds{}.pkl_bin2_fit'.format(dsnum))): 
            print('dataset has already been fit (sym fit), rerunning with asym fit...')
        interferometry_main(diskset, prefix, dsnum, binw=2, asym=True)

def refit_all_from_bin_datasets():
    for indx in get_diskset_index_options():
        diskset, prefix, dsnum = import_diskset(indx)
        binfitpath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_fit_bin2_asym'.format(dsnum))
        if not os.path.exists(binfitpath): 
            print('dataset has no asym bin2 fit skipping...')
        elif os.path.exists(os.path.join('..','results', prefix, 'dat_ds{}.pkl_asym_refit'.format(dsnum))): 
            print('dataset has already been refit skipping...')
        else:
            print('read bin fit from {}'.format(binfitpath))
            with open(binfitpath, 'rb') as f:
                data = pickle.load(f)
                uvecs_binned = data[4]
                A = data[1]
                B = data[2]
                C = data[3]
                coefs = np.zeros((A.shape[0], 3))
                coefs[:,0], coefs[:,1], coefs[:,2] = A, B, C
            refit_from_bin_main(uvecs_binned, prefix, dsnum, coefs, diskset)

def interferometry_main(diskset, prefix, dsnum, binw=1, asym=None):
    if asym is None: asym = boolquery('asymmetric fit w/sincos term? (helpful for AP data, wont affect results for P data)')
    os.makedirs(os.path.join('..','plots', prefix, 'ds_{}'.format(dsnum)), exist_ok=True)
    if asym:
        coefs, ufit = fit_full_hexagon(diskset, prefix, 3, dsnum, binw=2)
        savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_fit_bin2_asym'.format(dsnum))
        with open(savepath, 'wb') as f: pickle.dump([diskset, coefs[:,0], coefs[:,1], coefs[:,2], ufit], f)
    else:
        coefs, ufit = fit_full_hexagon(diskset, prefix, 2, dsnum, binw=2)
        savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_bin2_fit'.format(dsnum))
        with open(savepath, 'wb') as f: pickle.dump([diskset, coefs[:,0], coefs[:,1], ufit], f)
    return ufit

def refit_from_bin_main(uvecs, prefix, dsnum, coefs, diskset):
    coefs, ufit = refit_full_hexagon_from_bin(diskset, coefs, uvecs, prefix, dsnum)
    savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_asym_refit'.format(dsnum))
    with open(savepath, 'wb') as f: pickle.dump([diskset, coefs[:,0], coefs[:,1], ufit], f)
    return ufit

def refinefit_main(uvecs, prefix, dsnum, coefs, diskset):
    ncoef = len(coefs)
    if ncoef == 2:
        print('using A+Bcos^2')
        coefs, ufit = refit_full_hexagon(diskset, coefs, uvecs, prefix, dsnum)
        savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_refit'.format(dsnum))
        with open(savepath, 'wb') as f: pickle.dump([diskset, coefs[:,0], coefs[:,1], ufit], f)
    elif ncoef == 3:
        print('using A+Bcos^2+Csincos')
        coefs, ufit = refit_full_hexagon(diskset, coefs, uvecs, prefix, dsnum)
        savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_refit_asym'.format(dsnum))
        with open(savepath, 'wb') as f: pickle.dump([diskset, coefs[:,0], coefs[:,1], coefs[:,2], ufit], f)
    return ufit

if __name__ == "__main__":

    fitbool = boolquery("fit everything I can find? (all .pkl files in ../results/*/*.pkl) ... will use asymmetric fit for all")
    if fitbool: fit_all_datasets(); exit();

    fitbool = boolquery("refit from all bin2 I can find? (all .pkl_fit_bin2_asym files used) ... will use asymmetric fit for all")
    if fitbool: refit_all_from_bin_datasets(); exit();

    fitbool = boolquery("fit a set of extracted disk intensities (from virtual_df.py)?")
    while fitbool:
        diskset, prefix, dsnum = import_diskset()
        interferometry_main(diskset, prefix, dsnum)
        fitbool = boolquery("fit another set of disk intensities?")

    fitbool = boolquery("refine an exisiting displacement field?")
    while fitbool:
        uvecs, prefix, dsnum, coefs, diskset = import_disket_uvector()
        refinefit_main(uvecs, prefix, dsnum, coefs, diskset)
        fitbool = boolquery("refine another exisiting displacement field?")
