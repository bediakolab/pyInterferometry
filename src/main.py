
import sys
import os
import gc
usep4dstem = True
if usep4dstem:
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
import shutil

from globals import default_parameters, default_parameter_filler, known_materials, data_quality_flags, fit_quality_flags, partition_quality_flags
from heterostrain import extract_twist_hetstrain, plot_twist_hetstrain, plotTris
from unwrap_utils import getAdjacencyMatrixManual, rotate_uvecs
from visualization import disp_categorize_plot, displacement_colorplot, plot_adjacency, colored_quiver
from utils import nan_gaussian_filter, boolquery, normalize, get_triangles, convert_all_h5, get_probe, boolquery, get_probe_dm4
from basis_utils import latticevec_to_cartesian, cartesian_to_rz_WZ, cartesian_to_latticevec
from strain import strain
from masking import get_aa_mask, get_sp_masks, make_contour_mask
from io_utilities import DataSetContainer, compile_spreadsheet, unwrap_main,  DataSetContainer, load_existing_dataset, load_all_datasets
from virtual_df import virtualdf_main
from interferometry_fitting import fit_full_hexagon, refit_full_hexagon_from_bin
from diskset import get_region_average_dp
from new_utils import chunk_split, parse_filename, import_probe
from utils import convert_dm4

def set_up_files(useh5=True):
    datapath = os.path.join("..", "input")
    print('looking for folders of the format Name/1_100x100... formated like the example in the ../input to read data from')
    print("expect that these folders each contain a dm4 file with the 4dstem. Please put your data in this folder. NOTE, module")
    print("used to read doesnt support symbolic links(?) so please just copy directly. Please call the dm4 file Diffraction_SI.dm4. ")
    
    chunk_split_bool = boolquery("Would you like to spit the data into N by N chunks?")
    if chunk_split_bool:
        Nchunk = int(input("chunk dimension? (100 will split into <= 100x100 nm)").lower().strip())
        
    foundsomething = False
    for name in [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, el))]:
        newdatapath = os.path.join(datapath, name)
        if chunk_split_bool: chunk_split(newdatapath, Nchunk)
        for scandir in [el for el in os.listdir(newdatapath) if os.path.isdir(os.path.join(newdatapath, el))]:
            if scandir == 'before_chunksplit': continue
            foundsomething = True
            m_dir = os.path.join(newdatapath, scandir)
            datasetnum, scan_shape = parse_filename(scandir)
            savepath = os.path.join(os.path.join("..", "data"), os.path.join(name, "ds{}".format(datasetnum)))
            if not os.path.exists(savepath): os.makedirs(savepath)          
            if useh5 and not os.path.exists(os.path.join(savepath, 'dp.h5')): 
                if not os.path.exists(os.path.join(m_dir, 'dp.h5')): 
                    print('converting a dm4 to a h5 file')
                    success = convert_dm4(scan_shape, m_dir)
                if success : shutil.copyfile(os.path.join(m_dir, 'dp.h5'), os.path.join(savepath, "dp.h5"))
            elif (not useh5) and not os.path.exists(os.path.join(savepath, 'dp.dm4')):
                shutil.copyfile(os.path.join(m_dir, 'Diffraction_SI.dm4'), os.path.join(savepath, "dp.dm4"))
            print("working on dataset {} of {} ".format(datasetnum, name))
            #avgdp_bool = boolquery("would you like to visualize the average dp in a real space region?")
            #while avgdp_bool:
            #    get_region_average_dp(datacube, diskset, plotbool=True)
            #    avgdp_bool = boolquery("visualize the avg dp in another region?")
            gc.collect()
            with open(os.path.join(savepath, "tmp.txt"), 'w') as f: f.write(m_dir)
    if not foundsomething: print('unable to find data, please reformat')

def main():

    set_up_files()
    useall = boolquery("process everything? (y) or just one dataset? (n)")    
    if useall: dsets = load_all_datasets() 
    else: dsets = [load_existing_dataset()] 
    counter = 0
    gc.collect()
    for ds in dsets:

        print("working on: ", counter, " of ", ds.name) 
        ds.autoset_parameters()

        ########################################################
        #### disk intensity extraction
        ########################################################
        if  (not ds.check_has_diskset()) and ds.check_has_raw() and boolquery("extract disk intensities?"):
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
            ds.make_stack_colormap() # NEW makes the colormap plot
            ds.make_vdf_plots()
        else:
            print('couldnt find any data to work with...')

        ########################################################
        #### displacement fitting
        ########################################################
        if (not ds.check_has_displacement()) and ds.check_has_diskset() and boolquery("fit displacement?"):
            # do the fitting
            coefs, ufit = fit_full_hexagon(ds.diskset, 3)
            ds.update_parameter("FittingFunction", "A+Bcos^2+Csincos", "main")
            ds.update_parameter("RefitFromBinned", "False", "main")
            ds.update_parameter("DisplacementBasis", "Cartesian", "main")
            ds.update_displacement_fit(coefs, latticevec_to_cartesian(ufit.copy()))
        elif False and (not ds.check_has_displacement()) and ds.check_has_diskset() and boolquery("fit displacement (bin then refit)?"): # dont use
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
        if (not ds.check_has_unwrapping()) and ds.check_has_displacement() and boolquery("unwrap?"): 
            # do the unwrapping
            ufit, centers, adjacency_type = unwrap_main(ds)
            ds.update_unwraping(ufit, centers, adjacency_type)

        ########################################################
        #### post unwrapping plots/analysis, strain, etc.
        ########################################################
        if ds.check_has_unwrapping():
            ds.make_adjacency_plot()
            rigid_dil, rigid_twist, rigid_gamma = ds.make_twist_plot()
            ds.make_strainplots_localsubtraction(rigid_twist, rigid_dil, rigid_gamma)
            #ds.calculate_uncertainties()
            ds.make_sanity_residuals()

        ds.update_data_flags()
        counter += 1

if __name__ == '__main__':
    main()
    #compile_spreadsheet()
   