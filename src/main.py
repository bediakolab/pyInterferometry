

import gc
import os
import py4DSTEM
if py4DSTEM.__version__ != '0.11.5':
    print('WARNING: you are using py4DSTEM version {}'.format(py4DSTEM.__version__))
    print('please use py4DSTEM version 0.11.5')
    print("type 'pip install py4DSTEM==0.11.5' in the virtual environment you're using")
import glob
from datetime import date
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from globals import default_parameters, default_parameter_filler, known_materials, data_quality_flags, fit_quality_flags, partition_quality_flags
from heterostrain import extract_twist_hetstrain, plot_twist_hetstrain, plotTris
from unwrap_utils import getAdjacencyMatrixManual, rotate_uvecs
from visualization import disp_categorize_plot, displacement_colorplot, plot_adjacency, colored_quiver
from utils import nan_gaussian_filter, boolquery, normalize, get_triangles
from basis_utils import latticevec_to_cartesian, cartesian_to_rz_WZ, cartesian_to_latticevec
from strain import strain
from masking import get_aa_mask, get_sp_masks, make_contour_mask
from io_utilities import DataSetContainer, compile_spreadsheet
from virtual_df import virtualdf_main
from interferometry_fitting import fit_full_hexagon, refit_full_hexagon_from_bin
from io_utilities import unwrap_main

def load_existing_dataset():
    dspaths = glob.glob(os.path.join('..', 'data', '*', 'ds*'))
    print('KEY:\t Name\t\t\t Has Raw?\t Has VDFs?\t Has Disps?\t Has Unwrap?')
    for i in range(len(dspaths)):
        dspath = dspaths[i] 
        ds = DataSetContainer(dspath)
        has_disp   = ds.check_has_displacement()
        has_disks  = ds.check_has_diskset()
        has_unwrap = ds.check_has_unwrapping()
        has_raw    = ds.check_has_raw()
        name       = ds.name
        print('{}:\t{}\t{}\t{}\t{}\t{}'.format(i, name, has_raw, has_disks, has_disp, has_unwrap))
    indx = int(input("Which Dataset to Use? ---> ").lower().strip())
    return DataSetContainer(dspaths[indx])

def load_all_datasets():
    dspaths = glob.glob(os.path.join('..', 'data', '*', 'ds*'))
    dsets = []
    for i in range(len(dspaths)):
        dspath = dspaths[i] 
        dsets.append(DataSetContainer(dspath))
    return dsets  

def main():
    
    useall = boolquery("extract strain and replot everything? (y) or just one dataset? (n)")    
    if useall: dsets = load_all_datasets() 
    else: dsets = [load_existing_dataset()] 
    counter = 0
    gc.collect()

    for ds in dsets:
    
        print(counter, " ", ds.name) 

        ########################################################
        #### disk intensity extraction
        ########################################################
        if False and (not ds.check_has_diskset()) and ds.check_has_raw() and boolquery("extract disks?"):
            # do the vdf/disk extraction
            datacube, diskset = virtualdf_main(ds)
            datacube, scan_shape = ds.extract_raw()
            dp = np.max(datacube.data, axis=(0,1)).astype(float) 
            dp = np.transpose(dp)
            diskset = ds.diskset
            diskset.adjust_qspace_aspect_ratio(dp)
            ds.update_diskset(diskset)

        ########################################################
        #### post vdf/disk extraction plots/analysis
        ########################################################
        if ds.check_has_diskset():
            ds.set_sample_rotation()
            ds.make_vdf_plots()

        ########################################################
        #### displacement fitting
        ########################################################
        if False and (not ds.check_has_displacement()) and ds.check_has_diskset() and boolquery("fit displacement (no bin)?"):
            # do the fitting
            coefs, ufit = fit_full_hexagon(ds.diskset, 3)
            ds.update_parameter("FittingFunction", "A+Bcos^2+Csincos", "main")
            ds.update_parameter("RefitFromBinned", "False", "main")
            ds.update_parameter("DisplacementBasis", "Cartesian", "main")
            ds.update_displacement_fit(coefs, latticevec_to_cartesian(ufit.copy()))
        elif False and (not ds.check_has_displacement()) and ds.check_has_diskset() and boolquery("fit displacement (bin then refit)?"):
            # do the fitting, first fit binned by 2
            coefs, ufitbin2 = fit_full_hexagon(ds.diskset, 3, binw=2)
            ds.update_parameter("FittingFunction", "A+Bcos^2+Csincos", "main")
            ds.update_parameter("RefitFromBinned", "True,Bin2", "main")
            ufitbin2 = latticevec_to_cartesian(ufitbin2.copy())
            ds.update_bindisplacement_fit(coefs, ufitbin2)
            ds.make_bindisplacement_plot()

            # then refit from binned fit
            coefs, ufit = refit_full_hexagon_from_bin(ds.diskset, coefs, ufitbin2)
            ufit = latticevec_to_cartesian(ufit.copy())
            ds.update_parameter("DisplacementBasis", "Cartesian", "main")
            ds.update_displacement_fit(coefs, ufit)

        ########################################################
        #### post displacement fitting plots/analysis 
        ########################################################
        if ds.check_has_displacement():
            
            coefs, u = ds.extract_coef_fit(), ds.extract_displacement_fit()
            ds.make_displacement_plot()
            ds.make_categorize_plot()
            ds.make_sanity_residuals()

        ########################################################
        #### unwrapping
        ########################################################
        if False and (not ds.check_has_unwrapping()) and ds.check_has_displacement() and boolquery("unwrap?"): 
            # do the unwrapping
            ufit, centers, adjacency_type = unwrap_main(ds)
            ds.update_unwraping(ufit, centers, adjacency_type)

        ########################################################
        #### post unwrapping plots/analysis 
        ########################################################
        if ds.check_has_unwrapping():

            ds.make_adjacency_plot()
            rigid_dil, rigid_twist, rigid_gamma = ds.make_twist_plot()
            ds.make_strainplots_localsubtraction(rigid_twist, rigid_dil, rigid_gamma)
            ds.make_piezo_plots()
            #ds.get_strain_stats()

        counter += 1

if __name__ == '__main__':

    #ds = load_existing_dataset()
    #theta, hs, rigid_dil, rigid_twist = ds.make_twist_plot()
    #ds.make_strainplots_localsubtraction(rigid_twist, rigid_dil)
    main()
    compile_spreadsheet()
   