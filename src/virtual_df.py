
import os
import pickle
import py4DSTEM
from py4DSTEM.process.diskdetection import *
from py4DSTEM.process.virtualimage import *
import numpy as np
from diskset import *
from masking import *
from utils import *
from new_utils import parse_filepath
from visualization import overlay_vdf
from background import *

def virtualdf_main(ds, background_sub=False):    

    if not ds.check_has_raw():
        print('ERROR: dataset has no raw h5 or dm4 file to use')
        exit(1)

    probe_kernel, probe_kernel_FT, beamcenter = ds.extract_probe()
    datacube, scan_shape = ds.extract_raw()
    #print(datacube.data.shape)
    #print(scan_shape)
    max_dp       = np.max(datacube.data, axis=(0,1)).astype(float) #nqx, nqy
    probe_kernel = bin(probe_kernel, int(probe_kernel.shape[0]/max_dp.shape[0]))
    #f,ax = plt.subplots(); ax.imshow(max_dp, origin='lower'); plt.show()
    if not (max_dp.shape[0] == probe_kernel.shape[0] and max_dp.shape[1] == probe_kernel.shape[1]):
        print('resizing kernel')
        qx, qy = max_dp.shape[:]
        qxkernel, qykernel = probe_kernel.shape[:]
        removex, removey = qxkernel-qx, qykernel-qy
        probe_kernel = probe_kernel[ removex//2:- removex//2, removey//2:-removey//2]
        assert(max_dp.shape[0] == probe_kernel.shape[0] and max_dp.shape[1] == probe_kernel.shape[1])

    peaks        = find_Bragg_disks_single_DP(max_dp, probe_kernel, corrPower=1.0, sigma=2, minPeakSpacing=0.01, edgeBoundary=0, relativeToPeak=3, minRelativeIntensity=0.0005, maxNumPeaks=400)    
    diskset      = get_diskset(max_dp, peaks, scan_shape, beamcenter)
    _, diskset   = select_disks(diskset, max_dp)

    print('finding raw dark fields')
    diskset = integrate_disks(datacube, diskset)
    ds.update_diskset(diskset)
    ds.make_vdf_plots(diskset)

    if background_sub:
        print('masking')
        sum_dp    = np.sum(datacube.data, axis=(0,1)).astype(float)
        mask_peak = get_peak_mask(max_dp.shape[0], max_dp.shape[1], peaks, diskset, max_dp, dsnum, radius_factor=2.0)
        if boolquery("mask off beamstop? (done automatically, tends to correctly identify if there is no bs but can fail)"):
            mask_bs   = get_beamstop_mask(max_dp, dsnum)
            masked_dp = mask_off(max_dp, [mask_bs, mask_peak], dsnum)
        else:
            masked_dp = mask_off(max_dp, [mask_peak], dsnum)
        anom_mask = get_anomoly_mask(masked_dp, beamcenter, dsnum, bin_w=27)
        masked_dp = mask_off(masked_dp, [anom_mask], dsnum)

        print('background subtraction')
        beamcenter = diskset.set_com_central()
        background_fit = fit_background_lorenzian(masked_dp, beamcenter, dsnum)
        diskset = integrate_disks(datacube, diskset, background_fit=background_fit)
        ds.update_diskset(diskset)
        ds.make_vdf_plots(diskset)
        ds.update_parameter("BackgroundSubtraction", "Lorenzian", "virtualdf_main")
        
    return datacube, diskset

def virtualdf_main2(path, probe_kernel, beamcenter):

    prefix, dsnum, scan_shape = parse_filepath(path)
    prefix_sub = '{}_backgroundsub'.format(prefix)
    os.makedirs(os.path.join('..', 'results', prefix),     exist_ok=True)
    os.makedirs(os.path.join('..', 'plots',   prefix,     'ds_{}'.format(dsnum)), exist_ok=True)
    os.makedirs(os.path.join('..', 'results', prefix_sub), exist_ok=True)
    os.makedirs(os.path.join('..', 'plots',   prefix_sub, 'ds_{}'.format(dsnum)), exist_ok=True)
    savepath     = os.path.join('..',  'results', prefix,     'dat_ds{}.pkl'.format(dsnum))
    savepath_sub = os.path.join('..',  'results', prefix_sub, 'dat_ds{}.pkl'.format(dsnum))
    print('reading h5')
    datacube     = py4DSTEM.io.read(os.path.join(path, "dp.h5"), data_id="datacube_0")
    datacube.set_scan_shape(scan_shape[0],scan_shape[1])

    max_dp       = np.max(datacube.data, axis=(0,1)).astype(float) #nqx, nqy
    probe_kernel = bin(probe_kernel, int(probe_kernel.shape[0]/max_dp.shape[0]))
    print('finding bragg disks')

    if not (max_dp.shape[0] == probe_kernel.shape[0] and max_dp.shape[1] == probe_kernel.shape[1]):
        print('resizing kernel')
        qx, qy = max_dp.shape[:]
        qxkernel, qykernel = probe_kernel.shape[:]
        removex, removey = qxkernel-qx, qykernel-qy
        probe_kernel = probe_kernel[ removex//2:- removex//2, removey//2:-removey//2]
        assert(max_dp.shape[0] == probe_kernel.shape[0] and max_dp.shape[1] == probe_kernel.shape[1])

    peaks        = find_Bragg_disks_single_DP(max_dp, probe_kernel, corrPower=1.0, sigma=2, minPeakSpacing=0.01,
                   edgeBoundary=0, relativeToPeak=3, minRelativeIntensity=0.0005, maxNumPeaks=400)
    diskset      = get_diskset(max_dp, peaks, scan_shape, beamcenter, dsnum, prefix)
    _, diskset   = select_disks(diskset, max_dp)

    print('finding raw dark fields')
    diskset   = integrate_disks(datacube, diskset, dsnum, prefix, sub=False)
    overlay_vdf(diskset, dsnum, prefix, sub=False)
    with open(savepath, 'wb') as f: pickle.dump( diskset, f )

    print('masking')
    #f, ax = plt.subplots(1,2)
    sum_dp    = np.sum(datacube.data, axis=(0,1)).astype(float)
    mask_peak = get_peak_mask(max_dp.shape[0], max_dp.shape[1], peaks, diskset, max_dp, dsnum, radius_factor=2.0)
    if boolquery("mask off beamstop? (done automatically, tends to correctly identify if there is no bs but can fail)"):
        mask_bs   = get_beamstop_mask(max_dp, dsnum)
        masked_dp = mask_off(max_dp, [mask_bs, mask_peak], dsnum)
    else:
        masked_dp = mask_off(max_dp, [mask_peak], dsnum)
    anom_mask = get_anomoly_mask(masked_dp, beamcenter, dsnum, bin_w=27)
    masked_dp = mask_off(masked_dp, [anom_mask], dsnum)
    #ax[0].imshow(sum_dp)
    #ax[1].imshow(masked_dp)
    #plt.show()

    print('background subtraction')
    beamcenter = diskset.set_com_central()
    print(beamcenter)
    background_fit = fit_background_lorenzian(masked_dp, beamcenter, dsnum)
    diskset = integrate_disks(datacube, diskset, dsnum, prefix_sub, sub=True, background_fit=background_fit)
    overlay_vdf(diskset, dsnum, prefix_sub, sub=True)
    with open(savepath_sub, 'wb') as f: pickle.dump( diskset, f )

    return datacube, diskset

if __name__ == "__main__":
    probe_kernel, probe_kernel_FT, beamcenter = get_probe(os.path.join('..','data','probe_08132020.dm3')) # read in probe
    virtualdf_main(sys.argv[1], probe_kernel, beamcenter)
