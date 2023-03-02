import sys
import gc
import os
from visualization import plot_disks, overlay_vdf
from diskset import get_region_average_dp
from new_utils import import_uvector, import_diskset, parse_filepath
from utils import manual_extract_angle

def dp_analyze_main():

    # importing stuff
    diskset, prefix, dsnum = import_diskset()
    plot_disks(diskset)
    f, ax = plt.subplots()
    overlay_vdf(diskset, ax=ax)
    plt.show()
    path = input('wheres the h5 file for this dataset? give a full path? -->')
    prefix, dsnum, scan_shape = parse_filepath(path)
    datacube = py4DSTEM.io.read(os.path.join(path, "dp.h5"), data_id="datacube_0")
    datacube.set_scan_shape(scan_shape[0],scan_shape[1])

    # get the average diffraction pattern in the region of interest, user defined region given the vdf.
    avg_dp = get_region_average_dp(datacube, diskset)

    # analyze it. get twist angle.
    # click to define three points for the angle. theta in radians
    theta = manual_extract_angle(avg_dp) 

    # click to define two points for the length. need to know camera length?

if __name__ == "__main__":
    dp_analyze_main()


    
