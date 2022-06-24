
import sys
import gc
import os
import py4DSTEM
if py4DSTEM.__version__ != '0.11.5':
    print('WARNING: you are using py4DSTEM version {}'.format(py4DSTEM.__version__))
    print('please use py4DSTEM version 0.11.5')
    print("type 'pip install py4DSTEM==0.11.5' in the virtual environment you're using")
from utils import convert_all_h5, get_probe, boolquery, get_probe_dm4
from diskset import get_region_average_dp
from new_utils import chunk_split, parse_filepath, import_probe
from virtual_df import virtualdf_main

from io_utilities import DataSetContainer, load_existing_dataset

def main():
    #if boolquery("would you like to spit the data into chunks?"):
    #    chunk_split(datapath, int(input("chunk dimension? (100 will split into <= 100x100 nm)").lower().strip()))
    ds = load_existing_dataset()
    gc.collect()
    datacube, diskset = virtualdf_main(ds)
    avgdp_bool = boolquery("would you like to visualize the average dp in a real space region?")
    while avgdp_bool:
        get_region_average_dp(datacube, diskset, plotbool=True)
        avgdp_bool = boolquery("visualize the avg dp in another region?")
    gc.collect()

def main2(datapath):
    if boolquery("would you like to spit the data into chunks?"):
        chunk_split(datapath, int(input("chunk dimension? (100 will split into <= 100x100 nm)").lower().strip()))
    convert_all_h5(datapath)
    gc.collect()
    probe_path = import_probe()
    probe_kernel, probe_kernel_FT, beamcenter = get_probe(probe_path) # read in probe
    foundsomething = False
    print('looking for folders of the format 1_100x100 in the path to read data from...')
    for d in [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, el))]:
        if d == 'before_chunksplit': continue
        foundsomething = True
        m_dir = os.path.join(datapath,d)
        prefix, datasetnum, scan_shape = parse_filepath(m_dir)
        if os.path.exists(os.path.join('..','plots', prefix, 'ds_{}'.format(datasetnum))):
            print('../plots/{}/ds_{} exists'.format(prefix, datasetnum))
            if boolquery('skip (y) or re-run (n)?'): continue
        print("working on dataset {}".format(datasetnum))
        datacube, diskset = virtualdf_main(m_dir, probe_kernel, beamcenter)
        avgdp_bool = boolquery("would you like to visualize the average dp in a real space region?")
        while avgdp_bool:
            get_region_average_dp(datacube, diskset, plotbool=True)
            avgdp_bool = boolquery("visualize the avg dp in another region?")
        gc.collect()
    if not foundsomething: print('unable to find data, please reformat')

if __name__ == "__main__": #run using python3.py path
    main()#sys.argv[1])
