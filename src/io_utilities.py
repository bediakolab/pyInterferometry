
import gc
import os
usep4dstem = False
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from globals import default_parameters, default_parameter_filler, known_materials, data_quality_flags, fit_quality_flags, partition_quality_flags
from heterostrain import extract_twist_hetstrain, plot_twist_hetstrain, plotTris, matrixFromTriangleQuantity
from unwrap_utils import getAdjacencyMatrixManual, rotate_uvecs
from visualization import disp_categorize_plot, displacement_colorplot, plot_adjacency, colored_quiver, make_coloredvdf
from utils import nan_gaussian_filter, boolquery, normalize, get_triangles
from basis_utils import latticevec_to_cartesian, cartesian_to_rz_WZ, cartesian_to_rzcartesian, rotate_uvecs
from strain import strain, unscreened_piezocharge, strain_induced_polarization, strain_method_3
from masking import get_aa_mask, get_sp_masks, make_contour_mask
from new_utils import normNeighborDistance, crop_displacement, parse_filepath
from scipy import ndimage
from unwrap_utils import normDistToNearestCenter, getAdjacencyMatrix, automatic_sp_rotation, neighborDistFilter, geometric_unwrap, geometric_unwrap_tri

def bombout(message):
    print(message)
    exit(1)

def load_existing_dataset():
    dspaths = glob.glob(os.path.join('..', 'data', '*', 'ds*'))
    print('KEY:\t\t Name\t\t\t\t Has Raw?\t Has VDFs?\t Has Disps?\t Has Unwrap?')
    for i in range(len(dspaths)):
        dspath = dspaths[i] 
        ds = DataSetContainer(dspath)
        has_disp   = ds.check_has_displacement()
        has_disks  = ds.check_has_diskset()
        has_unwrap = ds.check_has_unwrapping()
        has_raw    = ds.check_has_raw()
        name       = ds.name
        print('{}:\t{}\t\t\t{}\t\t{}\t\t{}\t\t{}'.format(i, name, has_raw, has_disks, has_disp, has_unwrap))
    indx = int(input("Which Dataset to Use? ---> ").lower().strip())
    return DataSetContainer(dspaths[indx])

def load_all_datasets():
    dspaths = glob.glob(os.path.join('..', 'data', '*', 'ds*'))
    dsets = []
    for i in range(len(dspaths)):
        dspath = dspaths[i] 
        dsets.append(DataSetContainer(dspath))
    return dsets  

def compile_spreadsheet():
    dspaths = glob.glob(os.path.join('..', 'data', '*', 'ds*'))
    dicts = []
    for i in range(len(dspaths)):
        dspath = dspaths[i] 
        ds = DataSetContainer(dspath)
        has_disp   = ds.check_has_displacement()
        has_disks  = ds.check_has_diskset()
        has_unwrap = ds.check_has_unwrapping()
        has_raw    = ds.check_has_raw()
        param_dict, _ = parse_parameter_file(os.path.join(dspath, "info.txt"), use_nan=True)
        parts = dspath.replace(':', '-').split('/')[2:]
        param_dict["DataSetPath"] = " ".join(parts) #  MoS2_P/ds2 or something like this
        param_dict["DataSetPrefix"] = dspath.replace(':', '-').split('/')[2] # folder names, MoS2_P or something like this
        param_dict["FoundDiskset"]      =  has_disks
        param_dict["FoundDisplacement"] =  has_disp
        param_dict["FoundUnwrap"]       =  has_unwrap
        param_dict["FoundRawData"]      =  has_raw
        param_dict["SPPercent"] = param_dict["SP1Percent"] + param_dict["SP2Percent"] + param_dict["SP3Percent"]
        if i == 0:
            df = pd.DataFrame(data=param_dict, index=[0])
        else: 
            df2 = pd.DataFrame(data=param_dict, index=[0])
            df = pd.concat([df, df2])
    df.to_excel(os.path.join('..', 'data', 'summary.xlsx')) 
    return df   
    
def manual_define_good_triangles(img, centers, adjacency_type):
    # get region centers from the adjacencies 
    tris = get_triangles((adjacency_type > 0), n=3)
    tri_centers = []
    for tri in tris:
        center_y = np.mean([centers[tri[0]][0], centers[tri[1]][0], centers[tri[2]][0]])
        center_x = np.mean([centers[tri[0]][1], centers[tri[1]][1], centers[tri[2]][1]])
        tri_centers.append([center_y, center_x])
    if False: #boolquery('manually remove regions?'):
        mask, use_tris = manual_mark_tri(img, tri_centers, adjacency_type, centers, tris)   
    else:
        use_tris = [True for tri in tris]
    use_mask = np.zeros((img.shape[0], img.shape[1]))
    for i in range(len(tris)):
        if use_tris[i]:
            tri = tris[i]
            vertices = [centers[tri[0]], centers[tri[1]], centers[tri[2]]]
            mask = make_contour_mask(img.shape[0], img.shape[1], vertices, transpose=True)
            use_mask += mask 
    use_mask = (use_mask > 0).astype(int)
    return (1-use_mask), use_tris, tris, tri_centers

# unwraps and gets strain from a simple differentiation
def unwrap_main(ds, flip=False, transp=True, pad=1):

    centerdist, boundary_val, delta_val, combine_crit, spdist = 0.01, 0.3, 0.3, 0.0, 2.0

    if False: # sanity checking orrientation
        f, axes = plt.subplots(5,2)
        N = 50
        axes[0,0].quiver(ds.u_wrapped[:N,:N,0],ds.u_wrapped[:N,:N,1])
        displacement_colorplot(axes[0,1], ds.u_wrapped[:N,:N,:], sample_angle=0, quiverbool=True)
        u_wrapped_t = np.zeros((ds.u_wrapped.shape[0], ds.u_wrapped.shape[1], ds.u_wrapped.shape[2]))
        for i in range(ds.u_wrapped.shape[0]):
            for j in range(ds.u_wrapped.shape[1]):
                for d in range(ds.u_wrapped.shape[2]):
                    u_wrapped_t[i,j,d] = ds.u_wrapped[j,i,d]
        axes[1,0].quiver(u_wrapped_t[:N,:N,0],u_wrapped_t[:N,:N,1])
        displacement_colorplot(axes[1,1], u_wrapped_t[:N,:N,:], sample_angle=0, quiverbool=True)
        u_wrapped_t_rot = rotate_uvecs(u_wrapped_t, ang=(-np.pi/3))
        axes[2,0].quiver(u_wrapped_t_rot[:N,:N,0],u_wrapped_t_rot[:N,:N,1])
        displacement_colorplot(axes[2,1], u_wrapped_t_rot[:N,:N,:], sample_angle=0, quiverbool=True)
        u_wrapped_t_rot = rotate_uvecs(u_wrapped_t, ang=(np.pi/3))
        axes[3,0].quiver(u_wrapped_t_rot[:N,:N,0],u_wrapped_t_rot[:N,:N,1])
        displacement_colorplot(axes[3,1], u_wrapped_t_rot[:N,:N,:], sample_angle=0, quiverbool=True)
        axes[4,0].quiver(u[:N,:N,0],u[:N,:N,1])
        displacement_colorplot(axes[4,1], u[:N,:N,:], sample_angle=0, quiverbool=True)
        axes[0,0].title.set_text('untranspose, r=0')
        axes[1,0].title.set_text('transpose, r=0')
        axes[2,0].title.set_text('transpose, r=-pi/3')
        axes[3,0].title.set_text('transpose, r=pi/3')
        axes[4,0].title.set_text('orrientation used')
        plt.show()
        exit()

    while True:
        methodid = input("Method? \n1: voronoi AA \n2: watershed AA \n3: watershed AB \n").lower().strip()[0] 
        if int(methodid) == 1: 
            voronibool, tribool = True, False
            ds.update_parameter("UnwrapMethod", "Voronoi AA", "unwrap_main")
            break
        elif int(methodid) == 2: 
            voronibool, tribool = False, False
            print('using watershed')
            ds.update_parameter("UnwrapMethod", "Watershed AA", "unwrap_main")
            break
        elif int(methodid) == 3: 
            voronibool, tribool = False, True
            print('using watershed AB')
            ds.update_parameter("UnwrapMethod", "Watershed AB", "unwrap_main")
            break
        else:
            print('unrecognized/unimplemented method please try again'.format(methodid))

    """
    if ds.extract_parameter("DisplacementBasis", param_type=str) != "Cartesian":
        print('transforming LV to Cartesian')
        u = latticevec_to_cartesian(ds.u_wrapped.copy())
    else:
        u = ds.u_wrapped.copy()

    while True:
        methodid = input("Method? \n1: voronoi (good for large twist data, P or AP) \n2: watershed (good for most AP data unless very large twist)\n").lower().strip()[0] 
        if int(methodid) == 1: 
            voronibool, tribool = True, False
            ds.update_parameter("UnwrapMethod", "Voronoi", "unwrap_main")
            break
        elif int(methodid) == 2: 
            voronibool, tribool = False, False
            print('using watershed')
            ds.update_parameter("UnwrapMethod", "Watershed", "unwrap_main")
            break
        else:
            print('unrecognized/unimplemented method please try again'.format(methodid))

    img = displacement_colorplot(None, u)
    crop_displacement(img, u)

    u[:,:,0], u[:,:,1] = -u[:,:,0], u[:,:,1]
    u = cartesian_to_rz_WZ(u, sign_wrap=False)
    centers, adjacency_type = getAdjacencyMatrix(u, boundary_val, delta_val, combine_crit, spdist)
    points = [ [c[1], c[0]] for c in centers ]
    u, ang, adjacency_type = automatic_sp_rotation(u, centers, adjacency_type, transpose=True) # rotate so sp closest to vertical is sp1, gvector choice degenerate under 2pi/3 rotations so arbitrary sp1/sp2/sp3
    if not tribool: u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, vertices = geometric_unwrap(centers, adjacency_type, u, voronibool, plotting=True) 
    else: u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, vertices = geometric_unwrap_tri(centers, adjacency_type, u) 
    dists = normDistToNearestCenter(u.shape[0], u.shape[1], centers)
    variable_region = (dists > centerdist).astype(int)
    u = strain_method_3(u_unwrapped, points, variable_region)
    if nan_filter: u = neighborDistFilter(u, thresh=nan_thresh)
    """
    #4dstempc
    nan_filter = False
    flip = True    
    transp = False 

    #main
    #nan_filter = False
    #flip = False    
    #transp = True 

    if ds.u_wrapped is None: ds.extract_displacement_fit()
    u = ds.u_wrapped.copy()
    img = displacement_colorplot(None, u)
    crop_displacement(img, u)

    u = u[pad:-pad, pad:-pad, :]
    nx, ny = u.shape[0], u.shape[1]
    n = np.min([nx, ny])
    if n % 2 != 0: 
        u = u[:n-1, :n-1]
        nx, ny = u.shape[0], u.shape[1]
    else: u = u[:n, :n]
    assert(u.shape[0] % 2 == 0)
    
    if flip: u[:,:,0], u[:,:,1] = -u[:,:,0], u[:,:,1]
    u = cartesian_to_rz_WZ(u, sign_wrap=False)
    if transp:
        uorig = u.copy()
        for i in range(uorig.shape[0]):
            for j in range(uorig.shape[1]):
                for d in range(uorig.shape[2]):
                    u[i,j,d] = uorig[j,i,d]

    centers, adjacency_type = getAdjacencyMatrix(u, boundary_val, delta_val, combine_crit, spdist, refine=(not tribool))
    points = [ [c[1], c[0]] for c in centers ]
    u, ang, adjacency_type = automatic_sp_rotation(u, centers, adjacency_type, transpose=True) 

    if not tribool: 
        print('using aa methods')
        u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, vertices = geometric_unwrap(centers, adjacency_type, u, voronibool, plotting=True) 
    else: 
        print('using ab methods')
        u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, vertices = geometric_unwrap_tri(centers, adjacency_type, u, voronibool=voronibool) 

    dists = normDistToNearestCenter(u.shape[0], u.shape[1], centers)
    variable_region = (dists > centerdist).astype(int)
    u = strain_method_3(u_unwrapped, points, variable_region)
    #if nan_filter: u = neighborDistFilter(u, thresh=nan_thresh)
    return u, centers, adjacency_type   

def stder(v): return np.std(v, ddof=1) / np.sqrt(np.size(v))

def smooth(u, sigma):
    #d = normNeighborDistance(u, norm=False)
    nx, ny = u.shape[0], u.shape[1]
    smooth_u = np.zeros((nx, ny, 2))
    smooth_u[0, :,:] = np.nan 
    smooth_u[-1,:,:] = np.nan 
    smooth_u[:, 0,:] = np.nan 
    smooth_u[:,-1,:] = np.nan
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            smooth_u[x,y,:] = u[x,y,:]
            #if (d[x,y] > 0.05): smooth_u[x,y,:] = np.nan, np.nan
    smooth_u[:,:,0] = nan_gaussian_filter(smooth_u[:,:,0], sigma) 
    smooth_u[:,:,1] = nan_gaussian_filter(smooth_u[:,:,1], sigma) 
    return smooth_u

def unimplemented_error():
    print("I havent done this yet!!")
    exit()

def writedict(filepath, data, comments):
    with open(filepath, 'w') as f:
        if isinstance(data,dict):
            for k, v in data.items():
                if k in comments.keys(): 
                    comment = comments[k]
                    f.write('{} : {} #{}\n'.format(k, v, comment))
                else: 
                    f.write('{} : {}\n'.format(k, v))

def parse_parameter_file(path, use_nan=False):
    param_dictionary = dict()
    parameter_dict_comments = dict()
    lines = []
    with open(path) as f:
        line = f.readline()
        lines.append(line)
        while line:
            line = f.readline()
            lines.append(line)
    lines = [l for l in lines if len(l.strip()) > 0]
    for i in range(len(lines)):
        line = lines[i]
        comment_split = line.split('#')
        line = comment_split[0] # comment free line
        parsed = [el.strip() for el in line.split(':') if len(el.strip()) > 0]
        if len(parsed) == 2:
            k, v = parsed[0], parsed[1]
            try: param_dictionary[k] = float(v)
            except: 
                if v == default_parameter_filler and use_nan:
                    param_dictionary[k] = np.nan
                else: param_dictionary[k] = v
                if len(comment_split) > 1: 
                    comment = comment_split[1] # comment free line
                    comment = comment.strip('\n')
                    parameter_dict_comments[k] = comment
    return param_dictionary, parameter_dict_comments

class DataSetContainer:

    # folderprefix of something like MoS2_parallel
    def __init__(self, folderprefix, dataset_number = None):

        self.theta_colormap = 'RdBu_r'
        self.dilation_colormap = 'PuOr_r'
        self.gamma_colormap = 'plasma' #'gamma'
        self.piezo_colormap = 'RdBu_r'
        self.counter_colormap = 'viridis'
        self.localsubtractcheck_colormap = 'RdBu_r'

        if dataset_number is None: # one argument given, so parse path 
            folderpath = folderprefix
            head_tail = os.path.split(folderpath)
            head, tail = head_tail[0], head_tail[1]
            dataset_number = int(tail.split('ds')[1])
            head_tail = os.path.split(head)
            head, tail = head_tail[0], head_tail[1]
            folderprefix = tail

        # paths to important files, edit here to change directory organization
        self.name       = "{} : Dataset #{}".format(folderprefix, dataset_number)
        self.folderpath = os.path.join('..', 'data', folderprefix, "ds{}".format(dataset_number)) 
        self.rawpathh5  = os.path.join(self.folderpath, "dp.h5")
        self.rawpathdm  = os.path.join(self.folderpath, "dp.dm4")
        self.parampath  = os.path.join(self.folderpath, "info.txt")
        self.fitpath    = os.path.join(self.folderpath, "fit.pkl")
        self.fitbinpath = os.path.join(self.folderpath, "fitbin2.pkl")
        self.unwrappath = os.path.join(self.folderpath, "unwrap.pkl")
        self.diskpath   = os.path.join(self.folderpath, "diskset.pkl")
        self.maskpath   = os.path.join(self.folderpath, "mask.pkl")
        self.plotpath   = os.path.join(self.folderpath, "plots")
        self.folderprefix = folderprefix

        # plots to save, edit here to change directory organization
        self.stackcolormapath = os.path.join(self.plotpath, "stackcolor.png")
        self.twistplotpath    = os.path.join(self.plotpath, "twist.png")
        self.catplotpath      = os.path.join(self.plotpath, "categorize.png")
        self.dispplotpath     = os.path.join(self.plotpath, "displacement.png")
        self.dispbin2plotpath = os.path.join(self.plotpath, "displacementbin2.png")
        self.shearplotpath    = os.path.join(self.plotpath, "shear.png")
        self.dilplotpath      = os.path.join(self.plotpath, "dilation.png")
        self.adjplotpath      = os.path.join(self.plotpath, "adjacencies.png")
        self.rotplotpath      = os.path.join(self.plotpath, "rotation.png")
        self.vdfplotpath      = os.path.join(self.plotpath, "vdf.png")
        self.diskplotpath     = os.path.join(self.plotpath, "disks.png")
        self.hexplotpath      = os.path.join(self.plotpath, "hex_plots.png")
        self.subhexplotpath   = os.path.join(self.plotpath, "hex_plots_sub.png")
        self.croprotpath      = os.path.join(self.plotpath, "rotation_cropped.png")
        self.cropdilpath      = os.path.join(self.plotpath, "dilation_cropped.png")
        self.cropdisppath     = os.path.join(self.plotpath, "displacement_cropped.png")
        self.cropgammapath    = os.path.join(self.plotpath, "shear_cropped.png")
        self.quivplotpath     = os.path.join(self.plotpath, "quiver.png")
        self.piezoplotpath    = os.path.join(self.plotpath, "piezo_charge.png")
        self.piezoquiverpath  = os.path.join(self.plotpath, "piezo_polarization.png")
        self.piezohexplotpath = os.path.join(self.plotpath, "hex_piezocharge.png")
        self.subcroprotpath   = os.path.join(self.plotpath, "rotation_cropped_sub.svg")
        self.subcropdilpath   = os.path.join(self.plotpath, "dilation_cropped_sub.svg")
        self.subcropgammapath = os.path.join(self.plotpath, "shear_cropped_sub.png")

        self.dillinecutpath   = os.path.join(self.plotpath, "dil_linecut.png")
        self.rotlinecutpath   = os.path.join(self.plotpath, "rot_linecut.png")
        self.countlinecutpath = os.path.join(self.plotpath, "count_linecut.png")
        self.gamlinecutpath   = os.path.join(self.plotpath, "gamma_linecut.png")

        self.dillinecuttxtpath   = os.path.join(self.plotpath, "dil_linecut.txt")
        self.rotlinecuttxtpath   = os.path.join(self.plotpath, "rot_linecut.txt")
        self.countlinecuttxtpath = os.path.join(self.plotpath, "count_linecut.txt")
        self.gamlinecuttxtpath   = os.path.join(self.plotpath, "gamma_linecut.txt")

        self.rotsanitypath    = os.path.join(self.plotpath, "sanity_gvector_rotation_check.png")
        self.localsubplot     = os.path.join(self.plotpath, "sanity_local_substraction.png")
        self.sanity_intfit    = os.path.join(self.plotpath, "sanity_vdf_check2.png")
        self.sanity_vdf       = os.path.join(self.plotpath, "sanity_vdf_check.svg")
        self.sanity_axes      = os.path.join(self.plotpath, "sanity_axes.png")
        self.disp_orrientation_sanity =    os.path.join(self.plotpath, "sanity_disp_orrientation.png")
        self.unwrap_orrientation_sanity =  os.path.join(self.plotpath, "sanity_unwrap_orrientation.png")

        self.masked_diskset_path = os.path.join(self.folderpath, "masked_diskset.pkl")
        self.maskeddiskplotpath = os.path.join(self.plotpath, "masked_disks.png")
        self.maskedvdfplotpath = os.path.join(self.plotpath, "masked_vdf.png")

        # fields to hold the data
        self.u_unwrap       = None    # set by extract_unwraping, update_unwrapping
        self.u_wrapped_bin  = None  
        self.u_wrapped      = None    # set by extract_displacement, update_displacement
        self.centers        = None    # set by extract_unwraping, update_adjacency
        self.adjacency_type = None    # set by extract_unwraping, update_adjacency
        self.diskset        = None    # set by extract_diskset, update_diskset
        self.masked_diskset = None
        self.raw_data       = None

        self.parameter_dict = default_parameters # defined in globals.py  
        self.parameter_dict_comments = dict() 
        self.write_directory()

    def autoset_parameters(self):   
        if os.path.exists(os.path.join(self.folderpath, "tmp.txt")):
            print('attempting to automatically set aquisition and material parameters')
            print('parsing from folder prefix of {}'.format(self.folderprefix)) 
            with open(os.path.join(self.folderpath, "tmp.txt"), 'r') as f: 
                orig_dir = f.readline()
            _, dsnum, scan_shape, ss = parse_filepath(orig_dir, ss=True)
            self.update_parameter("ScanShapeX", scan_shape[0], "parsed from given directory")
            self.update_parameter("ScanShapeY", scan_shape[1], "parsed from given directory")
            self.update_parameter("PixelSize",  ss, "parsed from given directory")
            self.update_parameter("Original Data Location", orig_dir, "parsed from given directory")  
        else:
            if not self.check_parameter_is_set("ScanShapeX"):
                self.update_parameter("ScanShapeX")
            if not self.check_parameter_is_set("ScanShapeY"):
                self.update_parameter("ScanShapeY")
            if not self.check_parameter_is_set("PixelSize"):
                self.update_parameter("PixelSize")  

        self.update_material_parameters(self.folderprefix)
        #self.update_parameter("SmoothingSigma", 2.0, "asssumed, default")
        #self.update_parameter("PixelSize", 0.5, "asssumed, default")
        self.update_parameter("FittingFunction", "A+Bcos^2+Csincos", "asssumed, default")
        self.update_parameter("BackgroundSubtraction", "Lorenzian", "asssumed, default")

        if not self.check_parameter_is_set("HeteroBilayer"):
            print('please set HeteroBilayer to yes, t, y, or true if it is a HeteroBilayer.')
            self.update_parameter("HeteroBilayer")
        self.set_sample_rotation()

    def set_sample_rotation(self):
        if False:#self.check_parameter_is_set("SampleRotation"): 
            print("not resetting sample rotation since theres a value in info.txt...")
            return
        if not self.check_has_diskset(): 
            print("data has no diskset or specified sample rotation")
            return
        else:
            diskset = self.extract_diskset()
            mean_ang = diskset.get_rotatation(self.rotsanitypath)
            self.update_diskset(diskset) # has info about rotation now
            rotation_correction = -11.3
            mean_ang = mean_ang + rotation_correction
            print('Adding rotatational correction of {} degrees to the measured sample rotation'.format(rotation_correction))
            self.update_parameter("SampleRotation", mean_ang )
            print('total SampleRotation (convenience + instrument) is : ', mean_ang)
            self.update_parameter("K3toHAADFRotation_Used", rotation_correction )

    def update_data_flags(self):
        dataflag = self.update_flags(data_quality_flags, "DataQualityFlag", checkfunc=self.check_has_displacement, plotfunc=self.make_displacement_plot)
        fitflag = self.update_flags(fit_quality_flags, "FitQualityFlag", checkfunc=self.check_has_displacement, plotfunc=None)
        if fitflag != "bad":
            catflag = self.update_flags(partition_quality_flags, "PartitionQualityFlag",  checkfunc=self.check_has_displacement, plotfunc=self.make_categorize_plot)
        else: catflag = self.update_parameter("PartitionQualityFlag", "bad")

    def update_flags(self, flagdict, flagid, checkfunc=None, plotfunc=None):
        flag_descriptions = [i for i in flagdict.values()]
        flags = [ i for i in flagdict.keys()]
        existing_flag = self.extract_parameter(flagid, param_type=str)
        if existing_flag in flags: return existing_flag
        if not checkfunc(): return None
        if plotfunc is not None: plotfunc(showflag=True)
        msg = "\n".join(["{}: {} ".format(i, flag_descriptions[i]) for i in range(len(flags))])
        msg = "Flags? \n{}\n ---> ".format(msg)
        while True:
            try:
                indx = int(input(msg).lower().strip()[0])
                break 
            except: print('unexpected query entry please try again')
        flag = flags[indx]
        self.update_parameter(flagid, flags[indx])
        return flag

    def update_material_parameters(self, folderprefix):
        # asks for manual input of these parameters
        folderprefix = folderprefix.replace('/', '-').replace(':', '-').replace('_', '-')
        split_prefix = folderprefix.split("-")
        if len(split_prefix) > 1: orrientation = split_prefix[-1].lower()
        materials = split_prefix[0:-1]
        materials = [el.lower() for el in materials]
        if len(materials) > 1: 
            self.is_hbl = True
            self.update_parameter("Material", '-'.join(materials))
        elif len(materials) == 1:
            self.update_parameter("Material", materials[0])
            tag = self.extract_parameter("HeteroBilayer", param_type=str)
            if tag is not None and tag.strip().lower() in ['t','y','yes','true']:
                self.is_hbl = True
            else:
                self.is_hbl = False
        else:
            if not self.check_parameter_is_set("Material"):
                self.update_parameter("Material")
            materials = [self.extract_parameter("Material", param_type=str)]
            if not self.check_parameter_is_set("HeteroBilayer"):
                self.update_parameter("HeteroBilayer")
            tag = self.extract_parameter("HeteroBilayer", param_type=str)
            if tag is not None and tag.strip().lower() in ['t','y','yes','true']:
                self.is_hbl = True
            else:
                self.is_hbl = False

        if len(split_prefix) > 1: 
            self.update_parameter("Orientation", orrientation)
        else:
            print('Did not provide P or AP in folder name, will assume P and/or no distinction')
            orrientation = "p" # treat as P by default
        for material in materials:
            if material not in known_materials.keys():
                print("material {} doesnt have tabulated lattice constant data will ask for manual definition".format(material))
                return
            if known_materials[material][3] is not None:    
                self.update_parameter("PiezoChargeConstant", known_materials[material][3], "update_material_parameters")
        if self.is_hbl:
            a1 = known_materials[materials[0]][0]
            a2 = known_materials[materials[1]][0]
            aL, aS = np.max([a1,a2]), np.min([a1,a2])
            delta = ((aS/aL) - 1)  
            a = aL
            self.update_parameter("LatticeConstant", a, "update_material_parameters")
            self.update_parameter("LatticeMismatch", delta, "update_material_parameters")
        else:
            material = materials[0]
            is_parallel = (orrientation.lower() in ["p", "par", "parallel"])
            if is_parallel: pr = known_materials[material][1]
            else: pr = known_materials[material][2]
            self.update_parameter("LatticeConstant", known_materials[material][0], "update_material_parameters")
            self.update_parameter("LatticeMismatch", 0.0, "update_material_parameters")
            self.update_parameter("PoissonRatio", pr, "update_material_parameters")

    def extract_probe(self, indx=None):
        from utils import get_probe
        if not os.path.exists(self.parameter_dict["ProbeUsed"]):    
            probes = glob.glob(os.path.join('..', 'probes', '*'))
            if indx == None:
                for i in range(len(probes)): print('{}:    {}'.format(i, probes[i]))
                indx = int(input("Which Probe File to Use? ---> ").lower().strip())
            self.update_parameter("ProbeUsed", probes[indx], "extract_probe")
            return get_probe(probes[indx])
        return get_probe(self.parameter_dict["ProbeUsed"])

    def extract_masked_diskset(self):
        with open(self.masked_diskset_path, 'rb') as f:
            self.masked_diskset = pickle.load(f)
        return self.masked_diskset

    # makes the bivariate colormap
    def make_stack_colormap(self, showflag=False):
        if self.diskset is None: self.extract_diskset()
        diskset = self.diskset
        f, ax = plt.subplots(1, 1)
        gvecs = diskset.clean_normgset(self.rotsanitypath)
        ringnos = diskset.determine_rings()
        counter = 0
        ring1 = np.zeros((diskset.nx, diskset.ny))
        ring2 = np.zeros((diskset.nx, diskset.ny))
        for n in range(diskset.size):
            if diskset.in_use(n): 
                img = diskset.df(n)
                ringno = ringnos[counter]
                if ringno == 1: ring1 += img
                elif ringno == 2: ring2 += img
                counter += 1
        rgb = make_coloredvdf(ring1, ring2, gaussian_sigma=None)
        ax.imshow(rgb)
        ax.axis("off")
        if not showflag: plt.savefig(self.stackcolormapath, dpi=300)
        if not showflag: plt.close()
        if showflag: plt.show()

    def select_vdf_masked_diskset(self):
        self.extract_masked_diskset()
        diskset = self.masked_diskset
        nx, ny = int(np.ceil(diskset.size ** 0.5)), int(np.ceil(diskset.size ** 0.5))
        f, axes = plt.subplots(nx, ny)
        axes = axes.flatten()
        gvecs = diskset.clean_normgset(self.rotsanitypath)
        for n in range(diskset.size):
            img = diskset.df(n)
            axes[n].imshow(img, cmap='gray')
            axes[n].set_title("Disk {}".format(n))
        for n in range(diskset.size, len(axes)): axes[n].axis("off")
        plt.subplots_adjust(hspace=0.55, wspace=0.3)
        plt.show()
        while True:
            try:
                use_indx = [int(el.strip()) for el in input("which vdfs to use? enter indices separated by commas").split(',')]
                break
            except:
                print('uh try again parse error')
        diskset.reset_useflags()
        for n in use_indx: diskset.set_useflag(n, True)
        print(use_indx)
        self.update_masked_diskset(diskset)
    
    def make_vdf_plots(self, showflag=False, masked=False):

        if (not masked):
            self.extract_diskset()
            diskset = self.diskset
            pltpath1 = self.diskplotpath
            pltpath2 = self.vdfplotpath
            Ndiskparam = "NumberDisksUsed"
        if (masked):
            self.extract_masked_diskset()
            diskset = self.masked_diskset
            pltpath1 = self.maskeddiskplotpath
            pltpath2 = self.maskedvdfplotpath
            Ndiskparam = "NumberDisksUsed_Masked"
            
        counter = 0
        tot_img = np.zeros((diskset.nx, diskset.ny))
        self.update_parameter(Ndiskparam, diskset.size_in_use, "make_vdf_plots")
        nx, ny = int(np.ceil(diskset.size_in_use ** 0.5)), int(np.ceil(diskset.size_in_use ** 0.5))
        f, axes = plt.subplots(nx, ny)
        axes = axes.flatten()
        gvecs = diskset.clean_normgset(self.rotsanitypath)
        for n in range(diskset.size):
            if diskset.in_use(n): 
                img = diskset.df(n)
                tot_img = tot_img + img
                axes[counter].imshow(img, cmap='gray')
                #axes[counter].set_title("Disk {}".format(counter))
                axes[counter].set_title("Disk {}:{}{}".format(counter, gvecs[counter][0],gvecs[counter][1]))
                counter += 1
        for n in range(counter, len(axes)): axes[n].axis("off")
        tot_img = normalize(tot_img)
        plt.subplots_adjust(hspace=0.55, wspace=0.3)
        if not showflag: 
            plt.savefig(pltpath1, dpi=300)
            plt.close()
        if showflag: plt.show()

        # saves sum of all disk vdfs
        f, ax = plt.subplots()
        cf = ax.imshow(tot_img, cmap='gray')
        ax.set_title("Sum of Selected Disk Virtual DFs")
        cb = plt.colorbar(cf, ax=ax, orientation='vertical')
        ax.set_xticks(np.arange(0, np.round(diskset.nx/50)*50+1, 50))
        ax.set_yticks(np.arange(0, np.round(diskset.ny/50)*50+1, 50))
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
        if not showflag: 
            plt.savefig(pltpath2, dpi=300)
            print("saving vdf plot to {}".format(pltpath2))
            plt.close()
        if showflag: plt.show()
        if False:
            f, ax = plt.subplots(4,3)
            counter = 0
            I = diskset.df_set()
            g1  = np.array([ 0, 2/np.sqrt(3)])
            g2  = np.array([-1, 1/np.sqrt(3)])
            g   = diskset.clean_normgset(self.rotsanitypath)
            otherdisk = -1
            for n in range(12):
                df = I[n,:,:]
                gvec = g[n][0] * g1 + g[n][1] * g2
                print(n, " -- ", gvec)
                if gvec[0] == 0:
                    if otherdisk == -1: 
                        otherdisk = n
                        print('set to ', otherdisk)
                    else:
                        print('plotting diff')
                        ax[counter,2].imshow(df - I[otherdisk,:,:])
                        ax[counter,2].set_title("disk {} - {}".format(n, otherdisk))
                        otherdisk = -1
                        print('set to ', otherdisk)
                    ax[counter,0].plot(np.sum(df, axis=1), 'k')
                    ax[counter,0].set_title("disk {} y".format(n))
                    ax[counter,1].imshow(df)
                    ax[counter,1].set_title("disk {}".format(n))
                    counter += 1
                elif gvec[1] == 0:
                    if otherdisk == -1: 
                        otherdisk = n
                        print('set to ', otherdisk)
                    else:
                        print('plotting diff')
                        ax[counter,2].imshow(df - I[otherdisk,:,:])
                        ax[counter,2].set_title("disk {} - {}".format(n, otherdisk))
                        otherdisk = -1
                        print('set to ', otherdisk)
                    ax[counter,0].plot(np.sum(df, axis=0), 'k')
                    ax[counter,0].set_title("disk {} x".format(n))
                    ax[counter,1].imshow(df)
                    ax[counter,1].set_title("disk {}".format(n))
                    counter += 1
            plt.show()
        return tot_img    

    def make_twist_plot(self):
        #if os.path.exists(self.twistplotpath): return
        if self.adjacency_type is None: self.extract_unwraping()
        if self.u_wrapped is None: self.extract_displacement_fit()
        rho = self.extract_parameter("PoissonRatio", update_if_unset=True, param_type=float)
        f, ax = plt.subplots(3,2) 
        N = self.u_wrapped.shape[0]
        tri_centers, thetas, het_strains, deltas, het_strain_proxies = extract_twist_hetstrain(self)
        
        if self.is_hbl:
            print("DiffractionPatternTwist is the average twist angle we got by looking at the average diffraction pattern")
            twist_dp = self.extract_parameter("DiffractionPatternTwist", update_if_unset=True, param_type=float)
            print('Using Diffraction pattern twist of {} degress, {} radians'.format(twist_dp, twist_dp * np.pi/180))
            self.update_parameter("AvgMoireMismatch", np.nanmean(deltas), "make_twist_plot")
            self.update_parameter("AvgHeteroStrainHBLProxy", np.nanmean(het_strain_proxies), "make_twist_plot")
            self.update_parameter("ErrMoireMismatch", stder(deltas), "make_twist_plot")
            self.update_parameter("ErrHeteroStrainHBLProxy", stder(het_strain_proxies), "make_twist_plot")
            plot_twist_hetstrain(self, ax[0,0], ax[0,1], deltas, het_strain_proxies, tri_centers, N)
        else:
            self.update_parameter("AvgMoireTwist", np.nanmean(thetas), "make_twist_plot")
            self.update_parameter("AvgHeteroStrain", np.nanmean(het_strains), "make_twist_plot")
            self.update_parameter("ErrMoireTwist", stder(thetas), "make_twist_plot")
            self.update_parameter("ErrHeteroStrain", stder(het_strains), "make_twist_plot")
            plot_twist_hetstrain(self, ax[0,0], ax[0,1], thetas, het_strains, tri_centers, N)
        
        triangles  = get_triangles((self.adjacency_type > 0).astype(int))
        if not self.is_hbl: 
            theta_mat  = matrixFromTriangleQuantity(triangles, self.centers, thetas, N)
            strain_mat = matrixFromTriangleQuantity(triangles, self.centers, het_strains, N)
            for i in range(theta_mat.shape[0]):
                for j in range(theta_mat.shape[1]): 
                    if theta_mat[i,j] == 0:
                        theta_mat[i,j] = np.nan
                        strain_mat[i,j] = np.nan 
            nx, ny = theta_mat.shape[0], theta_mat.shape[1]
        else:
            delta_mat = matrixFromTriangleQuantity(triangles, self.centers, deltas, N)
            for i in range(delta_mat.shape[0]):
                for j in range(delta_mat.shape[1]): 
                    if delta_mat[i,j] == 0:
                        delta_mat[i,j] = np.nan   
            nx, ny =  delta_mat.shape[0], delta_mat.shape[1]

        rigid_dilation = np.zeros((nx, ny))
        rigid_local_twist = np.zeros((nx, ny))
        rigid_gamma = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):  
                if self.is_hbl: 
                    delta = delta_mat[i,j] * 1/100
                    ang = twist_dp * np.pi/180
                    rigid_dilation[i,j] = - delta 
                    rigid_local_twist[i,j] = ang * (1 - delta/2)
                    rigid_gamma[i,j] = 0
                else:
                    ang = theta_mat[i,j] * np.pi/180
                    eps = strain_mat[i,j] * 1/100
                    delta_effective = 0.5 * (eps - eps*rho)
                    rigid_dilation[i,j] = delta_effective 
                    rigid_local_twist[i,j] = ang * (1 + delta_effective/2)
                    rigid_gamma[i,j] = 0.5 * (eps + eps*rho)
        
        ax[1,0].imshow(rigid_local_twist * 180/np.pi, origin='lower')
        ax[1,1].imshow(rigid_dilation * 100, origin='lower') 
        ax[2,0].imshow(rigid_gamma * 100, origin='lower') 
        ax[1,0].set_title("$theta^{rigid} (^o)$")
        ax[1,1].set_title("$dilation^{rigid} (\%)$")
        ax[2,0].set_title("$gamma^{rigid} (\%)$")
        ax[1,0].axis('off')
        ax[1,0].set_aspect('equal')
        ax[1,1].axis('off')
        ax[1,1].set_aspect('equal')
        ax[2,0].axis('off')
        ax[2,0].set_aspect('equal')
        for n in range(len(triangles)):
            if self.is_hbl and not np.isnan(deltas[n]):
                ang = twist_dp
                tc = tri_centers[n]
                dil = deltas[n] * 1/100
                twist = ang * (1 + dil/2)
                g = 0
                ax[1,0].text(tc[0], tc[1], "{:.2f}".format(twist*180/np.pi), color='grey', fontsize='xx-small', horizontalalignment='center')
                ax[1,1].text(tc[0], tc[1], "{:.2f}".format(dil*100), color='grey', fontsize='xx-small', horizontalalignment='center')
                ax[2,0].text(tc[0], tc[1], "{:.2f}".format(g*100), color='grey', fontsize='xx-small', horizontalalignment='center')
            elif not np.isnan(thetas[n]):
                ang = thetas[n] * np.pi/180 
                tc = tri_centers[n]
                eps = het_strains[n] * 1/100
                delta_effective = 0.5 * (eps - eps*rho)
                dil = delta_effective
                twist = ang * (1 + delta_effective/2)
                g = 0.5 * (eps + eps*rho)
                ax[1,0].text(tc[0], tc[1], "{:.2f}".format(twist*180/np.pi), color='grey', fontsize='xx-small', horizontalalignment='center')
                ax[1,1].text(tc[0], tc[1], "{:.2f}".format(dil*100), color='grey', fontsize='xx-small', horizontalalignment='center')
                ax[2,0].text(tc[0], tc[1], "{:.2f}".format(g*100), color='grey', fontsize='xx-small', horizontalalignment='center')
        #plt.show(); exit()
        print("saving twist plot to {}".format(self.twistplotpath))
        plt.savefig(self.twistplotpath, dpi=300)
        plt.close('all')
        return rigid_dilation, rigid_local_twist, rigid_gamma

    def make_adjacency_plot(self):

        #if os.path.exists(self.adjplotpath): return
        if self.u_unwrap is None: self.extract_unwraping()
        sigma = self.extract_parameter("SmoothingSigma", update_if_unset=True, param_type=float)

        #u = np.zeros((self.u_unwrap.shape[0], self.u_unwrap.shape[1], 2))
        #u[:,:,0] = self.u_unwrap[:,:,1]
        #u[:,:,1] = self.u_unwrap[:,:,0]

        smoothed_u = smooth(self.u_unwrap, sigma)
        f, ax = plt.subplots()
        img = displacement_colorplot(None, smoothed_u)
        plot_adjacency(img, self.centers, self.adjacency_type, ax, colored=True) 
        print("saving adjacency plot to {}".format(self.adjplotpath)) 
        ax.set_title("$u_{smooth}(x,y)$") 
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$") 
        plt.savefig(self.adjplotpath, dpi=300)
        plt.close('all')

        f, ax = plt.subplots()
        colored_quiver(ax, smoothed_u[:,:,0], smoothed_u[:,:,1])
        plt.savefig(self.quivplotpath, dpi=300)
        plt.close('all')

        """
        f, ax = plt.subplots()
        colored_quiver(ax, self.u_unwrap[:,:,0], self.u_unwrap[:,:,1])
        plt.savefig("TEST", dpi=300)
        plt.close('all')

        f, ax = plt.subplots()
        if self.u_wrapped is None: self.extract_displacement_fit()
        u = self.u_wrapped.copy()
        u = cartesian_to_rz_WZ(u, False)
        colored_quiver(ax, u[:,:,0], u[:,:,1])
        plt.savefig("TEST2", dpi=300)
        plt.close('all')
        exit()
        """

    def make_categorize_plot(self, showflag=False):
        #if os.path.exists(self.catplotpath): return
        if self.u_wrapped is None: self.extract_displacement_fit()
        f, ax = plt.subplots()
        pAA, pSP1, pSP2, pSP3, pXX, pMM, rAA, eAA, wSP, eSP = disp_categorize_plot(self.u_wrapped.copy(), ax)
        if self.extract_parameter("Orientation", update_if_unset=True, param_type=str).strip().lower() == 'ap':
            if pXX > pMM:
                print('WARNING: flipping XX and MM since think that the hexagon was rotated')
                pXX, pMM = pMM, pXX
            self.update_parameter("XMMXPercent", pAA, "make_categorize_plot")
            self.update_parameter("XXPercent", pXX, "make_categorize_plot")
            self.update_parameter("MMPercent", pMM, "make_categorize_plot")
            self.update_parameter("SP1Percent", pSP1, "make_categorize_plot")
            self.update_parameter("SP2Percent", pSP2, "make_categorize_plot")
            self.update_parameter("SP3Percent", pSP3, "make_categorize_plot")
            self.update_parameter("AvgXMMXradius", rAA, "make_categorize_plot")
            self.update_parameter("ErrXMMXradius", eAA, "make_categorize_plot")
            self.update_parameter("AvgSPwidth", wSP, "make_categorize_plot")
            self.update_parameter("ErrSPwidth", eSP, "make_categorize_plot")
            t_1 = "{:.2f} % \\XMMX {:.2f} % \\XX {:.2f} % MM {:.2f} % SP".format(pAA, pXX, pMM, pSP1+pSP2+pSP3)
            t_2 = "{:.2f}+/-{:.2f} XMMXr (pix)  {:.2f}+/-{:.2f} SPw (pix)".format(rAA, eAA, wSP, eSP)
            ax.set_title("{}\n{}".format(t_1, t_2))
        elif self.extract_parameter("Orientation", update_if_unset=True, param_type=str).strip().lower() == 'p':
            self.update_parameter("MMXXPercent", pAA, "make_categorize_plot")
            self.update_parameter("MXorXMPercent", pXX + pMM, "make_categorize_plot")
            self.update_parameter("SP1Percent", pSP1, "make_categorize_plot")
            self.update_parameter("SP2Percent", pSP2, "make_categorize_plot")
            self.update_parameter("SP3Percent", pSP3, "make_categorize_plot")
            self.update_parameter("AvgMMXXradius", rAA, "make_categorize_plot")
            self.update_parameter("ErrMMXXradius", eAA, "make_categorize_plot")
            self.update_parameter("AvgSPwidth", wSP, "make_categorize_plot")
            self.update_parameter("ErrSPwidth", eSP, "make_categorize_plot")
            t_1 = "{:.2f} % MMXX {:.2f} % MX or XM {:.2f} % SP".format(pAA, pXX + pMM, pSP1+pSP2+pSP3)
            t_2 = "{:.2f}+/-{:.2f} MMXXr (pix)  {:.2f}+/-{:.2f} SPw (pix)".format(rAA, eAA, wSP, eSP)
            ax.set_title("{}\n{}".format(t_1, t_2))
        else:
            print('Couldnt determine if material is p or ap, assuming no distinction and graphene like')
            self.update_parameter("AAPercent", pAA, "make_categorize_plot")
            self.update_parameter("BAorABPercent", pXX + pMM, "make_categorize_plot")
            self.update_parameter("SP1Percent", pSP1, "make_categorize_plot")
            self.update_parameter("SP2Percent", pSP2, "make_categorize_plot")
            self.update_parameter("SP3Percent", pSP3, "make_categorize_plot")
            self.update_parameter("AvgAAradius", rAA, "make_categorize_plot")
            self.update_parameter("ErrAAradius", eAA, "make_categorize_plot")
            self.update_parameter("AvgSPwidth", wSP, "make_categorize_plot")
            self.update_parameter("ErrSPwidth", eSP, "make_categorize_plot")
            t_1 = "{:.2f} % AA {:.2f} % AB or BA {:.2f} % SP".format(pAA, pXX + pMM, pSP1+pSP2+pSP3)
            t_2 = "{:.2f}+/-{:.2f} AAr (pix)  {:.2f}+/-{:.2f} SPw (pix)".format(rAA, eAA, wSP, eSP)
            ax.set_title("{}\n{}".format(t_1, t_2))
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        if showflag:
            plt.show()
        else:
            plt.savefig(self.catplotpath, dpi=300)
            plt.close('all')

    def make_sanity_residuals(self, smooth_unwrap=False, transp=False):

        coefs = self.extract_coef_fit()
        if len(coefs) == 3:
            A = coefs[0]
            B = coefs[1]
            C = coefs[2]
        elif len(coefs) == 2:
            A = coefs[0]
            B = np.zeros(len(A))
            C = coefs[1]
        else:
            print('got an unexpected number of coefs...')
            exit()

        diskset = self.diskset
        I = diskset.df_set()
        #I = I[:, :100, :100]
        if self.u_wrapped is None: self.extract_displacement_fit()

        g1  = np.array([ 0, 2/np.sqrt(3)])
        g2  = np.array([-1, 1/np.sqrt(3)])
        dfs = []
        dfsA = []
        dfsB = []
        g   = diskset.clean_normgset(self.rotsanitypath)
        ringnos = diskset.determine_rings()

        for n in range(len(g)): I[n,:,:] = normalize(I[n,:,:])

        I_fit = np.zeros((I.shape[0], I.shape[1], I.shape[2]))

        for n in range(I.shape[0]):
            for i in range(I.shape[1]):
                for j in range(I.shape[2]):
                    gvec = g[n][1] * g1 + g[n][0] * g2
                    u    = [self.u_wrapped_raw[i,j,0], self.u_wrapped_raw[i,j,1]]
                    I_fit[n,i,j]   = A[n] * (np.cos(np.pi * np.dot(gvec, u)))**2
                    I_fit[n,i,j]  += B[n] * (np.cos(np.pi * np.dot(gvec, u)))*(np.sin(np.pi * np.dot(gvec, u)))
                    I_fit[n,i,j]  += C[n]     

        resid = I - I_fit    
        
        f, axes = plt.subplots(2,2)
        ax = axes.flatten()

        sample_angle = self.extract_parameter("SampleRotation", update_if_unset=True, param_type=float)
        img = displacement_colorplot(ax[0], self.u_wrapped, sample_angle=sample_angle, quiverbool=False)

        ring1 = np.zeros((I.shape[1], I.shape[2]))
        ring2 = np.zeros((I.shape[1], I.shape[2]))
        ring1f = np.zeros((I.shape[1], I.shape[2]))
        ring2f = np.zeros((I.shape[1], I.shape[2]))
        rms = np.zeros((I.shape[1], I.shape[2]))
        for n in range(I.shape[0]):
            img = I[n,:,:]
            imgf = I_fit[n,:,:]
            ringno = ringnos[n]
            if ringno == 1: 
                ring1 += img
                ring1f += imgf;
            elif ringno == 2: 
                ring2 += img;
                ring2f += imgf;

        for i in range(I.shape[1]):
            for j in range(I.shape[2]):
                rms[i,j] = np.sqrt(np.mean([el**2 for el in resid[:,i,j]]))

        img = ax[1].imshow(rms, cmap='plasma')
        div = make_axes_locatable(ax[1])
        cax = div.append_axes('right', size='5%',pad=0.05)
        f.colorbar(img, cax=cax, orientation='vertical')
        ax[2].imshow(make_coloredvdf(ring1, ring2, gaussian_sigma=None))
        ax[2].axis("off")
        ax[3].imshow(make_coloredvdf(ring1f, ring2f, gaussian_sigma=None))
        ax[3].axis("off")
            
        plt.savefig(self.sanity_vdf, dpi=300)
        plt.close('all')


    def make_displacement_plot(self, showflag=False, rewrite=False):

        if False:
            showflag=True
            f, ax = plt.subplots(1,3)
            if self.u_wrapped is None: self.extract_displacement_fit()
            ax[0].quiver(self.u_wrapped[:,:,0], self.u_wrapped[:,:,1])      
            ax[1].imshow(self.u_wrapped[:,:,0]) 
            ax[2].imshow(self.u_wrapped[:,:,1])  
            plt.show()
            self.u_unwrap  = self.u_wrapped[:, :, :]
            self.u_wrapped = self.u_wrapped[:, :, :]
            _, _, gamma, theta, dil = self.extract_strain(smoothbool=False)# plot reconstruction rotation
            cmap = self.theta_colormap 
            title = '$\\theta_r(^o)$'
            f, ax = plt.subplots()
            lim = np.max(np.abs(theta.flatten())) # want colormap symmetric about zero
            im = ax.imshow(theta, origin='lower', cmap=cmap)#, vmin=-lim, vmax=lim) 
            plt.colorbar(im, ax=ax, orientation='vertical')
            ax.set_title(title)
            ax.set_xlabel("$x(pixels)$")  
            ax.set_ylabel("$y(pixels)$")
            plt.show()

            # plot shear strain
            cmap = self.gamma_colormap 
            title = '$\\gamma(\\%)$'
            f, ax = plt.subplots()
            im = ax.imshow(gamma, origin='lower', cmap=cmap) 
            plt.colorbar(im, ax=ax, orientation='vertical')
            ax.set_title(title)
            ax.set_xlabel("$x(pixels)$")  
            ax.set_ylabel("$y(pixels)$")
            plt.show()

            # plot dilation strain
            cmap = self.dilation_colormap 
            title = '$dil(\\%)$'
            f, ax = plt.subplots()
            lim = np.max(np.abs(dil.flatten())) # want colormap symmetric about zero
            im = ax.imshow(dil, origin='lower', cmap=cmap)#, vmin=-lim, vmax=lim) 
            plt.colorbar(im, ax=ax, orientation='vertical')
            ax.set_title(title)
            ax.set_xlabel("$x(pixels)$")  
            ax.set_ylabel("$y(pixels)$")
            plt.show()
            #exit()

        #if os.path.exists(self.dispplotpath) and not rewrite: 
        #    print('{} exists so skipping'.format(self.dispplotpath))
        #    return
        if self.u_wrapped is None: self.extract_displacement_fit()
        f, ax = plt.subplots(1,3)
        sample_angle = self.extract_parameter("SampleRotation", update_if_unset=True, param_type=float)
        img = displacement_colorplot(ax[0], self.u_wrapped, sample_angle=sample_angle, quiverbool=False)
        img = displacement_colorplot(ax[1], self.u_wrapped, sample_angle=sample_angle, quiverbool=True)
        sample_angle = self.extract_parameter("SampleRotation", update_if_unset=True, param_type=float)
        ax[2].imshow(ndimage.rotate(img, -sample_angle, reshape=False), origin='lower') #, extent=[0,100,0,100]
        print("saving displacement plot to {}".format(self.dispplotpath)) 
        ax[0].set_title("$u_{raw}(x,y)$")    
        ax[0].set_xlabel("$x(pixels)$")  
        ax[0].set_ylabel("$y(pixels)$")
        ax[1].set_title("$u_{raw}(x,y)$")    
        ax[1].set_xlabel("$x(pixels)$") 
        ax[1].set_ylabel("$y(pixels)$")
        ax[2].set_title("$u(x||a_1,y)$")   
        ax[2].set_xlabel("$x||a_1$") 
        ax[2].set_ylabel("$y$")
        if showflag: 
            plt.show() 
        else:
            plt.savefig(self.dispplotpath, dpi=300)
            plt.close('all')

    def make_bindisplacement_plot(self):
        #if os.path.exists(self.dispbin2plotpath): return
        if self.u_wrapped_bin is None: self.extract_bindisp_fit()
        f, ax = plt.subplots()
        sample_angle = self.extract_parameter("SampleRotation", update_if_unset=True, param_type=float)
        img = displacement_colorplot(ax, self.u_wrapped_bin, sample_angle=sample_angle)
        print("saving binned displacement plot to {}".format(self.dispbin2plotpath))  
        ax.set_title("$u_{bin}(x,y)$")  
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        plt.savefig(self.dispbin2plotpath, dpi=300)
        plt.close('all')

    def extract_strain(self, smoothbool=True):
        if self.u_unwrap is None: self.extract_unwraping()          
        sigma = self.extract_parameter("SmoothingSigma", update_if_unset=True, param_type=float)
        ss = self.extract_parameter("PixelSize", force_set=True, param_type=float)
        a = self.extract_parameter("LatticeConstant", force_set=True, param_type=float)
        sample_angle = self.extract_parameter("SampleRotation", update_if_unset=True, param_type=float)
        if smoothbool: smoothed_u = smooth(self.u_unwrap, sigma)
        else: smoothed_u = self.u_unwrap
        smooth_scale_u = smoothed_u * a/ss
        #f, ax = plt.subplots(1,3)
        #img = displacement_colorplot(ax[0], self.u_unwrap) 
        #img = displacement_colorplot(ax[1], smooth(self.u_unwrap, sigma)) 
        #img = displacement_colorplot(ax[2], smooth(self.u_unwrap, sigma) * a/ss) 
        #plt.show()
        #exit()
        _, _, _, _, gamma, thetap, theta, dil = strain(smooth_scale_u, sample_angle)
        return smoothed_u, smooth_scale_u, 100*gamma, theta, 100*dil

    def make_strain_plot(self, values, centered, colormap, title, savepath, tris, use_tris):
        # plot values with triangle area mask
        f, ax = plt.subplots()
        lim = np.nanmax(np.abs(values.flatten()))
        if centered: im = ax.imshow(values, origin='lower', cmap=colormap, vmax=lim, vmin=-lim)
        else: im = ax.imshow(values, origin='lower', cmap=colormap, vmax=lim, vmin=0)
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(title)
        plotTris(tris, ax, self.centers, manual=False, use_tris=use_tris)
        plt.savefig(savepath, dpi=300)
        plt.close('all')       

    def linecut_avghex(self, u, mask, values, savepath, textsavepath, title):
        uvecs_cart = cartesian_to_rz_WZ(u.copy(), sign_wrap=False) #
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if values is not None: 
                    if mask[i,j]: 
                        values[i,j] = uvecs_cart[i,j,0] = uvecs_cart[i,j,1] = np.nan   
        start, spacing = 0.6, 15
        print('using a bin width of a0/{} for the linecuts!'.format(spacing))
        xrang = np.arange(-start,start+(1/spacing),1/spacing)
        N = len(xrang) 
        avg_vals, counter = np.zeros((N,N)), np.zeros((N,N))
        umag = np.zeros((N,N))
        std_vals = np.zeros((N,N))
        eps = 1e-7
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not np.isnan(uvecs_cart[i,j,0]) and not np.isnan(uvecs_cart[i,j,1]):
                    ux_index = int(np.round((uvecs_cart[i,j,0]+start)*spacing))
                    uy_index = int(np.round((uvecs_cart[i,j,1]+start)*spacing))
                    if values is not None: 
                        avg_vals[ux_index, uy_index] += values[i,j]
                        std_vals[ux_index, uy_index] += (values[i,j]*values[i,j])
                    counter[ux_index, uy_index]  += 1
                    umag[ux_index, uy_index] += uvecs_cart[i,j,1]
        for i in range(N):
            for j in range(N):    
                if counter[i,j] > 0: 
                    if values is not None: 
                        x2 = std_vals[i,j]
                        x = avg_vals[i, j] 
                        n = counter[i,j]
                        m = x/n
                        std_vals[i,j] = np.sqrt(x2/n - m*m)/np.sqrt(n)
                        avg_vals[i, j] = m
                        umag[i, j] /= n
                else: 
                    avg_vals[i, j] = counter[i,j] = np.nan
        
        if values is None: 
            avg_vals = counter

        f, ax = plt.subplots(1,3)
        ax[0].imshow(avg_vals)
        n0 = 2
        ax[0].plot([i for i in range(spacing)], [int(N/2) for i in range(spacing)], color="red", linewidth=2)
        ax[0].plot([i for i in range(spacing)], [n0 for i in range(spacing)], color="red", linewidth=2)
        yaxis = [ v for v in avg_vals[int(N/2),:] if not np.isnan(v) ] 
        stdevs = [ v for v,u in zip(std_vals[int(N/2),:],avg_vals[int(N/2),:]) if not np.isnan(u) ] 
        umagv = [ v for v,u in zip(umag[int(N/2),:],avg_vals[int(N/2),:]) if not np.isnan(u) ]
        yaxis2 = [ v for v in avg_vals[n0,:] if not np.isnan(v) ]  
        stdevs2 = [ v for v,u in zip(std_vals[n0,:],avg_vals[n0,:]) if not np.isnan(u) ] 
        umagv2 = [ v+1/np.sqrt(3) for v,u in zip(umag[n0,:],avg_vals[n0,:]) if not np.isnan(u) ] 
        yaxis_cat = yaxis#np.concatenate((yaxis, yaxis2), axis=None)
        stdevs_cat = stdevs#np.concatenate((stdevs, stdevs2), axis=None)
        umagv = umagv#np.concatenate((umagv, umagv2), axis=None)
        ax[1].plot(umagv, yaxis, c='k')   
        ax[1].fill_between(umagv, [mean-st for mean,st in zip(yaxis,stdevs)], [mean+st for mean,st in zip(yaxis,stdevs)],color='gray')    
        ax[1].set_xticks([-1/np.sqrt(3),0,1/np.sqrt(3)])
        ax[1].set_xticklabels(['MM','XMMX','XX'])
        umagv2 = umagv2 - np.min(umagv2)
        umagv2 = [u+1/np.sqrt(3) for u in umagv2]
        ax[2].plot(umagv2, yaxis2, c='k')   
        ax[2].fill_between(umagv2, [mean-st for mean,st in zip(yaxis2,stdevs2)], [mean+st for mean,st in zip(yaxis2,stdevs2)],color='gray') 

        with open(textsavepath, "w") as fid:
            fid.write("mean value \t\t\t standard error \t\t\t stacking param (-1/root3=MM, 0=XMMX, 1/root3=XX, 2/root3=XX \n") 
            for i in range(len(umagv)): 
                fid.write("{} \t\t\t {} \t\t\t {} \t\t\t \n".format(yaxis[i], stdevs[i], umagv[i]))
            for i in range(1,len(umagv2)):
                fid.write("{} \t\t\t {} \t\t\t {} \t\t\t \n".format(yaxis2[i], stdevs2[i], umagv2[i]))

        ax[2].set_xticks([1/np.sqrt(3),2/np.sqrt(3)])
        ax[2].set_xticklabels(['XX','MM'])
        ax[0].set_title(title)  
        #plt.show()
        #exit()
        plt.savefig(savepath, dpi=300)
        plt.close('all')
        return 

    def make_hex_plot(self, axis, counts_axis, u, mask, values, centered, colormap, title, partition_axis=None):

        #u = self.u_unwrap       
        uvecs_cart = cartesian_to_rz_WZ(u.copy(), sign_wrap=False)
        #f, ax = plt.subplots()
        #disp_categorize_plot(uvecs_cart, ax)
        #plt.show()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]: values[i,j] = uvecs_cart[i,j,0] = uvecs_cart[i,j,1] = np.nan  
        umag = np.zeros((uvecs_cart.shape[0],uvecs_cart.shape[1]))
        for i in range(uvecs_cart.shape[0]):
            for j in range(uvecs_cart.shape[1]):
                umag[i,j] = (uvecs_cart[i,j,0]**2 + uvecs_cart[i,j,1]**2)**0.5     
        boundary=(0.5 * np.nanmax(umag.flatten()) )    
        #f, ax = plt.subplots()
        #disp_categorize_plot(uvecs_cart, ax)
        #plt.show()
        #aa_mask = get_aa_mask(uvecs_cart, boundary=(0.5 * np.nanmax(umag.flatten()) ), smooth=None) 
        #sp1mask, sp2mask, sp3mask, mm_mask, xx_mask = get_sp_masks(uvecs_cart, aa_mask, plotbool=False, exclude_aa=True, include_aa=False, window_filter_bool=False)
        start, spacing = 0.6, 25
        xrang = np.arange(-start,start+(1/spacing),1/spacing)
        #print(xrang)
        N = len(xrang)
        avg_vals, counter, colors = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N,3))
        avg_ux, avg_uy, avg_ang = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
        eps = 1e-7
        cAA, cSP1, cSP2, cSP3, cXX, cMM, cTot = eps,eps,eps,eps,eps,eps,eps 
        AA_avg, SP1_avg, SP2_avg, SP3_avg, MM_avg, XX_avg = [],[],[],[],[],[]   

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not np.isnan(uvecs_cart[i,j,0]) and not np.isnan(uvecs_cart[i,j,1]):
                    ux_index = int(np.round((uvecs_cart[i,j,0]+start)*spacing))
                    uy_index = int(np.round((uvecs_cart[i,j,1]+start)*spacing))
                    #print(uvecs_cart[i,j,:], xrang[ux_index], xrang[uy_index])
                    avg_ux[ux_index, uy_index] += uvecs_cart[i,j,0]
                    avg_uy[ux_index, uy_index] += uvecs_cart[i,j,1]
                    avg_vals[ux_index, uy_index] += values[i,j]
                    counter[ux_index, uy_index] += 1
                    umax = (xrang[ux_index]**2 + xrang[uy_index]**2)**0.5
                    uang = np.arctan(xrang[uy_index]/(1e-7 + xrang[ux_index])) # cartesian!
                    uang = uang * 12/np.pi
                    uang += 6
                    if xrang[uy_index] < 0 : uang *= -1
                    delta = 1
                    avg_ang[ux_index, uy_index] = uang
                    if umax < boundary: 
                        AA_avg.append(values[i,j])
                        colors[ux_index, uy_index, :] = [0, 0, 0] #k
                        cAA += 1
                    elif ( np.abs(uang - 6 ) < delta ) | ( np.abs(uang + 6) < delta ):
                        SP1_avg.append(values[i,j])
                        colors[ux_index, uy_index, :] = [0, 225, 225] # SP1 c
                        cSP1 += 1
                    elif ( np.abs(uang - 2 ) < delta ) | ( np.abs(uang + 2) < delta ) :
                        SP2_avg.append(values[i,j])
                        colors[ux_index, uy_index, :] = [225, 0, 225] # SP2 m
                        cSP2 += 1  
                    elif ( np.abs(uang - 10 ) < delta ) | ( np.abs(uang + 10) < delta ):
                        SP3_avg.append(values[i,j])
                        colors[ux_index, uy_index, :] = [225, 225, 0] #SP3 y
                        cSP3 += 1
                    elif ( np.abs(uang - 4) < delta ) | ( np.abs(uang - 8) < delta ) | ( np.abs(uang + 12) < delta ):
                        XX_avg.append(values[i,j])
                        colors[ux_index, uy_index, :] = [225, 0, 0] # XX r
                        cXX += 1 
                    elif ( np.abs(uang + 4)  < delta ) | ( np.abs(uang + 8) < delta ) | ( np.abs(uang - 12) < delta ):
                        MM_avg.append(values[i,j])
                        colors[ux_index, uy_index, :] = [0, 0, 255] # MM b
                        cMM += 1 
                    elif ( np.abs(uang)  < delta ):
                        if xrang[ux_index] < 0: 
                            MM_avg.append(values[i,j])
                            colors[ux_index, uy_index, :] = [0, 0, 255] # MM b
                            cMM += 1
                        else:
                            XX_avg.append(values[i,j])
                            colors[ux_index, uy_index, :] = [225, 0, 0] # XX r
                            cXX += 1 
                    cTot += 1

        pAA,pSP1,pSP2,pSP3,pMM,pXX = cAA/cTot,cSP1/cTot,cSP2/cTot,cSP3/cTot,cMM/cTot,cXX/cTot
        #AA_avg, SP1_avg, SP2_avg, SP3_avg, MM_avg, XX_avg = AA_avg/cAA, SP1_avg/cSP1, SP2_avg/cSP2, SP3_avg/cSP3, MM_avg/cMM, XX_avg/cXX
        percents = [cAA/cTot,cSP1/cTot,cSP2/cTot,cSP3/cTot,cMM/cTot,cXX/cTot]
        averages = [AA_avg, SP1_avg, SP2_avg, SP3_avg, MM_avg, XX_avg]
        for i in range(N):
            for j in range(N):    
                if counter[i,j] > 0: 
                    avg_vals[i, j] /= counter[i,j]
                    avg_ux[i, j] /= counter[i,j]
                    avg_uy[i, j] /= counter[i,j]
                else: avg_vals[i, j] = counter[i,j] = np.nan
        if False: #averaging sanity
            f, axes = plt.subplots(1,2)
            axes[0].imshow(avg_ux)
            axes[1].imshow(avg_uy)
            plt.show()
            exit()
        lim = np.nanmax(np.abs(avg_vals.flatten())) 
        if centered:
            im = axis.imshow(avg_vals, origin='lower', cmap=colormap, vmax=lim, vmin=-lim)
        else:
            im = axis.imshow(avg_vals, origin='lower', cmap=colormap, vmax=lim, vmin=0)    
        plt.colorbar(im, ax=axis, orientation='vertical')
        axis.set_title(title)  
        if counts_axis is not None: 
            im = counts_axis.imshow(counter, origin='lower', cmap=self.counter_colormap)
            plt.colorbar(im, ax=counts_axis, orientation='vertical')
            counts_axis.set_title('counts') 
            counts_axis.axis('off')
            counts_axis.set_aspect('equal')
        if partition_axis is not None: 
            im = partition_axis.imshow(colors, origin='lower')
            partition_axis.set_title('partition') 
            partition_axis.axis('off')
            partition_axis.set_aspect('equal')    
        axis.axis('off')
        axis.set_aspect('equal')
        return percents, averages   

    def make_strainplots_localsubtraction(self, rigid_local_twist, rigid_dilation, rigid_gamma):

        rig_dil = rigid_dilation * 100
        rig_gamma = rigid_gamma * 100
        rig_theta = rigid_local_twist * 180/np.pi
        u, scaled_u, gamma, theta, dil = self.extract_strain(smoothbool=True)
        if np.nanmean(theta.flatten()) < 0: theta *= -1 #convention positive dilation overall, diverging, will substract from rigid
        if np.nanmean(dil.flatten()) < 0: dil *= -1 #convention positive dilation overall, diverging, will substract from rigid

        print('mean twist ', np.nanmean(theta.flatten()))
        print('mean dilation ',np.nanmean(dil.flatten()))
        print('mean gamma ',np.nanmean(gamma.flatten()))
        print('rigid twist ',np.nanmean(rig_theta.flatten()))
        print('rigid dilation ',np.nanmean(rig_dil.flatten()))
        print('rigid gamma ',np.nanmean(rig_gamma.flatten()))        
        print('doing local subtractions')
        subt = theta - rig_theta[:theta.shape[0], :theta.shape[1]]
        subd = dil - rig_dil[:dil.shape[0], :dil.shape[1]]
        subg =  gamma - rig_gamma[:gamma.shape[0], :gamma.shape[1]]
        print('sub twist ',np.nanmean(subt.flatten()))
        print('sub dilation ',np.nanmean(subd.flatten()))
        print('sub gamma ',np.nanmean(subg.flatten())) 

        assert(np.nanmean(dil.flatten()) > 0)
        assert(np.nanmean(theta.flatten()) > 0)

        def make_subplot(axis, title, matrix, center_colormap=True):
            if center_colormap: 
                lim = np.nanmax(np.abs(matrix.flatten())) 
                im = axis.imshow(matrix, origin='lower', cmap=self.localsubtractcheck_colormap, vmin=-lim, vmax=lim)
            else: 
                im = axis.imshow(matrix, origin='lower', vmin=0) 
            plot_adjacency(None, self.centers, self.adjacency_type, ax=axis, colored=False) # overlay triangles
            plt.colorbar(im, ax=axis, orientation='vertical')
            axis.set_title(title)
            axis.axis('off')
            axis.set_aspect('equal')

        f, ax = plt.subplots(3, 3)
        make_subplot(ax[0,0], '$\\theta_{raw}(^o)$', theta, center_colormap=True)
        make_subplot(ax[0,1], '$dilation_{raw}(\%)$', dil, center_colormap=True)
        make_subplot(ax[0,2], '$\\gamma_{raw}(\%)$', gamma, center_colormap=True)
        make_subplot(ax[1,0], '$\\theta_{sub}(^o)$', theta - rig_theta[:theta.shape[0], :theta.shape[1]])
        make_subplot(ax[1,1], '$dilation_{sub}(\%)$', dil - rig_dil[:dil.shape[0], :dil.shape[1]])
        make_subplot(ax[1,2], '$\\gamma_{sub}(\%)$', gamma - rig_gamma[:gamma.shape[0], :gamma.shape[1]])
        make_subplot(ax[2,0], "$\\theta^{rigid} (^o)$", rig_theta, center_colormap=True)
        make_subplot(ax[2,1], "$dilation^{rigid} (\%)$", rig_dil, center_colormap=True)
        make_subplot(ax[2,2], "$\\gamma^{rigid} (\%)$", rig_gamma, center_colormap=True)
        plt.savefig(self.localsubplot, dpi=300)
        plt.close('all')

        mask, use_tris, tris, tri_centers = manual_define_good_triangles(theta, [[c[1], c[0]] for c in self.centers], self.adjacency_type)
        img = displacement_colorplot(None, u)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]: 
                    gamma[i,j] = theta[i,j] = dil[i,j] = np.nan
                    img[i,j] = [1.0, 1.0, 1.0]

        # plot displacement with triangle area mask
        f, ax = plt.subplots()
        ax.imshow(img, origin='lower')
        ax.set_title('$u$') 
        plotTris(tris, ax, self.centers, manual=False, use_tris=use_tris)
        plt.savefig(self.cropdisppath, dpi=300)
        plt.close('all')

        # plot before subtraction
        self.make_strain_plot(theta, True, self.theta_colormap, '$\\theta_r(^o)$', self.croprotpath, tris, use_tris)
        self.make_strain_plot(dil, True, self.dilation_colormap, '$dil(\\%)$', self.cropdilpath, tris, use_tris)  
        self.make_strain_plot(gamma, True, self.gamma_colormap, '$\\gamma(\\%)$', self.cropgammapath, tris, use_tris)
        f, axes = plt.subplots(2,3)
        axes = axes.flatten()    
        percents, averages = self.make_hex_plot(axes[3], None, u, mask, theta, True, self.theta_colormap, '$<\\theta_r(^o)>%$')
        self.update_strain_stats(averages, "ReconRot")  
        percents, averages = self.make_hex_plot(axes[2], None, u, mask, dil, True , self.dilation_colormap, '$<dil>$', partition_axis=axes[4])
        self.update_strain_stats(averages, "Dil")  
        percents, averages = self.make_hex_plot(axes[1], axes[0], u, mask, gamma, True , self.gamma_colormap, '$<\\gamma>$')
        self.update_strain_stats(averages, "Gamma")  
        self.update_strain_stats(percents, "Percent")
        print("saving hex plots to {}".format(self.hexplotpath))
        plt.savefig(self.hexplotpath, dpi=300)
        plt.close('all')

    
        theta = theta - rig_theta[:theta.shape[0], :theta.shape[1]]
        dil = dil - rig_dil[:dil.shape[0], :dil.shape[1]]
        gamma = gamma - rig_gamma[:gamma.shape[0], :gamma.shape[1]]
        # plot after subtraction
        self.make_strain_plot(theta, True, self.theta_colormap, '$\\theta_r(^o)$', self.subcroprotpath, tris, use_tris)
        self.make_strain_plot(dil, True, self.dilation_colormap, '$dil(\\%)$',     self.subcropdilpath, tris, use_tris)
        self.make_strain_plot(gamma, True,  self.gamma_colormap, '$\\gamma(\\%)$', self.subcropgammapath, tris, use_tris)

        ########################################
        # area for the line cut implementation #
        ########################################
        # self.dillinecutpath, self.rotlinecutpath, self.countlinecutpath, self.gamlinecutpath
        f, axes = plt.subplots(2,2)
        axes = axes.flatten()    
        percents, averages = self.make_hex_plot(axes[3], None, u, mask, theta, True, self.theta_colormap, '$<\\theta_r(^o)>%$')
        pAA,pSP1,pSP2,pSP3,pMM,pXX = percents[:]
        self.update_strain_stats(averages, "SubReconRot")
        percents, averages = self.make_hex_plot(axes[2], None, u, mask, dil, True , self.dilation_colormap, '$<dil>$')  
        self.update_strain_stats(averages, "SubDil")      
        percents, averages = self.make_hex_plot(axes[1], axes[0], u, mask, gamma, True ,  self.gamma_colormap, '$<\\gamma>$')
        self.update_strain_stats(averages, "SubGamma")  
        print("saving hex plots to {}".format(self.subhexplotpath))
        plt.savefig(self.subhexplotpath, dpi=300)
        plt.close('all')

        print('saving line cut plots and textfiles with data to {} and {}'.format(self.dillinecutpath, self.dillinecuttxtpath))
        self.linecut_avghex(u, mask, dil, self.dillinecutpath, self.dillinecuttxtpath, '$<dil>$')  
        self.linecut_avghex(u, mask, theta, self.rotlinecutpath, self.rotlinecuttxtpath, '$<\\theta_r(^o)>%$')
        self.linecut_avghex(u, mask, gamma, self.gamlinecutpath, self.gamlinecuttxtpath, '$<\\gamma>$')
        #self.linecut_avghex(u, mask, None, self.countlinecutpath, self.countlinecuttxtpath, '$counts$')
           
    def update_strain_stats(self, averages, straintype):
        aAA,aSP1,aSP2,aSP3,aMM,aXX = averages[:]
        if straintype == "Percent":
            self.update_parameter("SP1Percent", aSP1*100, "make_categorize_plot")
            self.update_parameter("SP2Percent", aSP2*100, "make_categorize_plot")
            self.update_parameter("SP3Percent", aSP3*100, "make_categorize_plot")
            if self.extract_parameter("Orientation", update_if_unset=True, param_type=str).strip().lower() == 'ap':
                self.update_parameter("XMMXPercent", aAA*100, "make_categorize_plot")
                self.update_parameter("XXPercent", aXX*100, "make_categorize_plot")
                self.update_parameter("MMPercent", aMM*100, "make_categorize_plot")
            if self.extract_parameter("Orientation", update_if_unset=True, param_type=str).strip().lower() == 'p':
                self.update_parameter("MMXXPercent", aAA*100, "make_categorize_plot")
                self.update_parameter("MXorXMPercent", (aXX+aMM)*100, "make_categorize_plot")
        else:
            self.update_parameter("AvgSP1{}".format(straintype),   np.nanmean(aSP1), "get_strain_stats")
            self.update_parameter("ErrSP1{}".format(straintype),   stder(aSP1), "get_strain_stats")
            self.update_parameter("AvgSP2{}".format(straintype),   np.nanmean(aSP2), "get_strain_stats")
            self.update_parameter("ErrSP2{}".format(straintype),   stder(aSP2), "get_strain_stats")
            self.update_parameter("AvgSP3{}".format(straintype),   np.nanmean(aSP3), "get_strain_stats")
            self.update_parameter("ErrSP3{}".format(straintype),   stder(aSP3), "get_strain_stats")
            if self.extract_parameter("Orientation", update_if_unset=True, param_type=str).strip().lower() == 'ap':
                self.update_parameter("AvgXMMX{}".format(straintype), np.nanmean(aAA), "get_strain_stats")
                self.update_parameter("ErrXMMX{}".format(straintype), stder(aAA), "get_strain_stats")
                self.update_parameter("AvgMM{}".format(straintype),   np.nanmean(aMM), "get_strain_stats")
                self.update_parameter("ErrMM{}".format(straintype),   stder(aMM), "get_strain_stats")
                self.update_parameter("AvgXX{}".format(straintype),   np.nanmean(aXX), "get_strain_stats")
                self.update_parameter("ErrXX{}".format(straintype),   stder(aXX), "get_strain_stats")
            if self.extract_parameter("Orientation", update_if_unset=True, param_type=str).strip().lower() == 'p':
                self.update_parameter("AvgMMXX{}".format(straintype),   np.nanmean(aAA), "get_strain_stats")
                self.update_parameter("ErrMMXX{}".format(straintype),   stder(aAA), "get_strain_stats")
                self.update_parameter("AvgMX{}".format(straintype), np.nanmean(aMM), "get_strain_stats")
                self.update_parameter("ErrMX{}".format(straintype), stder(aMM), "get_strain_stats")
                self.update_parameter("AvgXM{}".format(straintype), np.nanmean(aXX), "get_strain_stats")
                self.update_parameter("ErrXM{}".format(straintype), stder(aXX), "get_strain_stats")

    def update_diskset(self, diskset):
        if os.path.exists(self.diskpath):
            print('WARNING: overwriting diskset for {}'.format(self.name))
        with open(self.diskpath, 'wb') as f: 
            pickle.dump( diskset, f )

    def update_masked_diskset(self, diskset):
        if os.path.exists(self.masked_diskset_path):
            print('WARNING: overwriting diskset for {}'.format(self.name))
        with open(self.masked_diskset_path, 'wb') as f: 
            pickle.dump( diskset, f )

    def update_displacement_fit(self, coefs, fit):
        print('updating displacements for {}'.format(self.name))
        self.u_wrapped = fit
        nc = (coefs.shape[1])
        if nc == 3:
            with open(self.fitpath, 'wb') as f: 
                pickle.dump([self.diskset, coefs[:,0], coefs[:,1], coefs[:,2], fit], f)
        elif nc == 2:
            with open(self.fitpath, 'wb') as f: 
                pickle.dump([self.diskset, coefs[:,0], coefs[:,1], fit], f)

    def update_bindisplacement_fit(self, coefs, fit):
        print('updating binned displacements for {}'.format(self.name))
        self.u_wrapped_bin = fit
        with open(self.fitbinpath, 'wb') as f: 
            pickle.dump([self.diskset, coefs[:,0], coefs[:,1], coefs[:,2], fit], f)

    def manual_update_adjacency_matrix(self):
        if self.adjacency_type is None: self.extract_unwraping()
        centers, self.adjacency_type = getAdjacencyMatrixManual(img, [[c[1], c[0]] for c in self.centers], self.adjacency_type)
        self.centers = [[c[1], c[0]] for c in centers]
        print('updating adjacencies for {}'.format(self.name))
        with open(self.unwrappath, 'wb') as f: 
            pickle.dump( [self.u_unwrap, self.centers, self.adjacency_type, None], f )

    def update_unwraping(self, fit, centers, adjmat):
        print('updating unwrapping, adjacencies, centers for {}'.format(self.name))
        self.u_unwrap, self.centers, self.adjacency_type = fit, centers, adjmat
        f, ax = plt.subplots(); ax.quiver(fit[:,:,0], fit[:,:,1]); plt.show()
        with open(self.unwrappath, 'wb') as f: 
            pickle.dump( [self.u_unwrap, self.centers, self.adjacency_type, None], f )

    def write_directory(self):
        os.makedirs(self.folderpath, exist_ok=True) # make folder for dataset
        if not os.path.exists(self.parampath): # write parameters to info.txt file
            writedict(self.parampath, self.parameter_dict, self.parameter_dict_comments) 
        else: # read parameters from info.txt file if it already exists
            self.parameter_dict, self.parameter_dict_comments = parse_parameter_file(self.parampath)
        os.makedirs(self.plotpath, exist_ok=True) # make subdir for plots
           
    def update_parameter(self, field, newvalue=None, comment=""):
        self.param_dictionary, self.parameter_dict_comments = parse_parameter_file(self.parampath)
        if field not in self.param_dictionary.keys():
            self.param_dictionary[field] = "Unset"
        if newvalue is None:
            newvalue = input("Enter value for {} for {} --> ".format(field, self.name)).lower().strip()
            try: newvalue = float(newvalue)
            except: newvalue = newvalue
            today = date.today()
            today_str = today.strftime("%m/%d/%y")
            self.parameter_dict_comments[field] = " - set manually on {}".format(today_str)
        else:
            today = date.today()
            today_str = today.strftime("%m/%d/%y")
            self.parameter_dict_comments[field] = " {}- set automatically on {}".format(comment, today_str)
        oldvalue = self.param_dictionary[field]
        #print('Writing parameter {}: changing from {} to {}'.format(field, oldvalue, newvalue))
        self.param_dictionary[field] = newvalue
        writedict(self.parampath, self.param_dictionary, self.parameter_dict_comments)
        return newvalue

    def extract_raw(self, use_hyperspy=False):
        scan_shape0 = int(self.extract_parameter("ScanShapeX", force_set=True, param_type=float))
        scan_shape1 = int(self.extract_parameter("ScanShapeY", force_set=True, param_type=float))
        if not self.check_has_raw():
            print("ERROR: {} has no raw dataset file.".format(self.name))
            exit(1)
        if use_hyperspy:
            if not os.path.exists(self.rawpathdm):
                print("need dm4 for the hyperspy loading...")
                exit(1)
            import hyperspy.api as hs  
            data = hs.load(self.rawpathdm, lazy=True)
        else:
            if not (os.path.exists(self.rawpathh5) or os.path.islink(self.rawpathh5)):
                data = py4DSTEM.io.read(self.rawpathdm)
                data.set_scan_shape(scan_shape0, scan_shape1)
                py4DSTEM.io.save(self.rawpathh5, data, overwrite=True)
            else:
                data = py4DSTEM.io.read(self.rawpathh5, data_id="datacube_0")
                data.set_scan_shape(scan_shape0, scan_shape1)
        self.raw_data = data
        return data, [scan_shape0, scan_shape1]

    def extract_parameter(self, field, force_set=False, update_if_unset=False, param_type=str, default_value=None):
        self.param_dictionary, self.parameter_dict_comments = parse_parameter_file(self.parampath)
        additional_keys = ['HeteroBilayer']
        if field not in self.param_dictionary.keys():
            if field not in additional_keys: print('WARNING parameter {} is unrecognized'.format(field))
            value = default_parameter_filler
        else: 
            value = self.param_dictionary[field]
        if value == default_parameter_filler: 
            if default_value is not None:
                    value = default_value
                    self.update_parameter(field, newvalue=value, comment="default")
            elif update_if_unset: 
                value = self.update_parameter(field) # set the parameter if unset
            elif force_set:
                print('ERROR: required parameter {} is unset'.format(field))
                exit(1)
            else:
                value = None
        if value is not None and not isinstance(value, param_type):
            print('WARNING: parameter {} of value {} is not of type {}, instead {}'.format(field, value, param_type, type(value)))
        return value

    def check_parameter_is_set(self, field):
        param_dictionary, self.parameter_dict_comments = parse_parameter_file(self.parampath)
        if field not in param_dictionary.keys():
            return False
        value = param_dictionary[field]
        return (value != default_parameter_filler)    

    def check_has_displacement(self): return os.path.exists(self.fitpath)

    def check_has_diskset(self): return os.path.exists(self.diskpath)

    def check_has_masked_diskset(self): return os.path.exists(self.masked_diskset_path)

    def check_has_mask(self): return os.path.exists(self.maskpath)

    def check_has_raw(self): 
        hasdm4 = os.path.exists(self.rawpathdm) or os.path.islink(self.rawpathdm)
        hash5 = os.path.exists(self.rawpathh5) or os.path.islink(self.rawpathh5)
        return hasdm4 or hash5

    def check_has_unwrapping(self): return os.path.exists(self.unwrappath)

    def extract_mask(self):
        with open(self.maskpath, 'rb') as f: mask = pickle.load(f)
        return mask

    def extract_diskset(self):
        if not self.check_has_diskset():
            print("ERROR: {} has no diskset file.".format(self.name))
            exit(1)
        else:
            filepath = self.diskpath
            print('reading from {}'.format(filepath))
            with open(filepath, 'rb') as f: self.diskset = pickle.load(f)
            return self.diskset

    def extract_unwraping(self, hbl_correct=True, spcorrection=True):
        if not self.check_has_unwrapping():
            print("ERROR: {} has no unwrapping file.".format(self.name))
            exit(1)
        else:
            filepath = self.unwrappath
            print('reading from {}'.format(filepath))
            with open(filepath, 'rb') as f: d = pickle.load(f)
            u, centers, adjacency_type = d[0], d[1], d[2]
            self.u_unwrap, self.centers, self.adjacency_type = u, centers, adjacency_type
            #self.u_unwrap, _, self.adjacency_type = automatic_sp_rotation(u, centers, adjacency_type)
            if hbl_correct and self.is_hbl:
                #u_wrapped_t = np.zeros((self.u_unwrap.shape[0], self.u_unwrap.shape[1], self.u_unwrap.shape[2]))
                #for i in range(self.u_wrapped.shape[0]):
                #    for j in range(self.u_wrapped.shape[1]):
                #        for d in range(self.u_wrapped.shape[2]):
                #            u_wrapped_t[i,j,d] = self.u_wrapped[j,i,d]
                u = -1 * rotate_uvecs(u, ang=-1/3*np.pi) # -1 so diverging by default, rotate so sp1 horizontal
                for i in range(adjacency_type.shape[0]):
                    for j in range(adjacency_type.shape[0]):
                        if adjacency_type[i,j] > 0: adjacency_type[i,j] = ((adjacency_type[i,j]+1)%3)+1 #123->312
                self.u_unwrap, self.centers, self.adjacency_type = u, centers, adjacency_type

            if spcorrection:
                ang = self.extract_parameter("UnwrapSPCorrectionRotation", default_value=0, param_type=float)
                assert(np.round(ang,2) in [np.round(v,2) for v in [-np.pi/3, np.pi/3, -2*np.pi/3, 2*np.pi/3, 0]])
                self.u_unwrap = rotate_uvecs(self.u_unwrap, ang=ang)    
                if np.round(ang,2) in [np.round(-1/3*np.pi), np.round(2/3*np.pi)]: 
                    for i in range(adjacency_type.shape[0]):
                        for j in range(adjacency_type.shape[0]):
                            if adjacency_type[i,j] > 0: adjacency_type[i,j] = ((adjacency_type[i,j])%3)+1 #123->231
                elif np.round(ang,2) in [np.round(1/3*np.pi), np.round(-2/3*np.pi)]: 
                    for i in range(adjacency_type.shape[0]):
                        for j in range(adjacency_type.shape[0]):
                            if adjacency_type[i,j] > 0: adjacency_type[i,j] = ((adjacency_type[i,j]+1)%3)+1 #123->312
            
            f, ax = plt.subplots(1,2); 
            ax[1].set_title('unwrap'); 
            img = displacement_colorplot(ax[0], self.u_unwrap, quiverbool=True); 
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if np.isnan(img[i,j,0]): img[i,j,:] = 0 
            sample_angle = self.extract_parameter("SampleRotation", force_set=True, param_type=float)
            ax[1].imshow(ndimage.rotate(img, -sample_angle, reshape=False), origin='lower') 
            plt.savefig(self.unwrap_orrientation_sanity, dpi=300)
            print('generating plot ', self.unwrap_orrientation_sanity)
            plt.close('all')
            return self.u_unwrap, self.centers, self.adjacency_type

    def extract_displacement_fit(self, transp=False):
        if not self.check_has_displacement():
            print("ERROR: {} has no displacement file.".format(self.name))
            exit(1)
        else:
            filepath = self.fitpath
            print('reading from {}'.format(filepath))                
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.u_wrapped = data[-1] # stored as diskset, coefs, ufit
            if self.extract_parameter("DisplacementBasis", param_type=str) != "Cartesian":
                print('DISPLCEMENT BASIS CHANGE LV->CART')
                self.u_wrapped = latticevec_to_cartesian(self.u_wrapped)
            self.u_wrapped_raw = self.u_wrapped.copy()
            self.u_wrapped = cartesian_to_rz_WZ(self.u_wrapped, sign_wrap=False)
            from basis_utils import rotate_uvecs
            ang = 0 # self.extract_parameter("DispSPCorrectionRotation", default_value=np.pi/3, param_type=float)
            assert(np.round(ang,2) in [np.round(v,2) for v in [-np.pi/3, np.pi/3, -2*np.pi/3, 2*np.pi/3, 0]])
            if transp:
                u_wrapped_t = np.zeros((self.u_wrapped.shape[0], self.u_wrapped.shape[1], self.u_wrapped.shape[2]))
                for i in range(self.u_wrapped.shape[0]):
                    for j in range(self.u_wrapped.shape[1]):
                        for d in range(self.u_wrapped.shape[2]):
                            u_wrapped_t[i,j,d] = self.u_wrapped[j,i,d]
                self.u_wrapped = rotate_uvecs(u_wrapped_t, ang=ang)
            else:
                self.u_wrapped = rotate_uvecs(self.u_wrapped, ang=ang)
            f, ax = plt.subplots(1,2); 
            img = displacement_colorplot(ax[0], self.u_wrapped); 
            ax[0].set_title('fit'); 
            sample_angle = self.extract_parameter("SampleRotation", update_if_unset=True, param_type=float)
            ax[1].imshow(ndimage.rotate(img, -sample_angle, reshape=False), origin='lower') 
            plt.savefig(self.disp_orrientation_sanity, dpi=300)
            plt.close('all')
            return self.u_wrapped

    def extract_coef_fit(self):
        if not self.check_has_displacement():
            print("ERROR: {} has no displacement file.".format(self.name))
            exit(1)
        else:
            filepath = self.fitpath
            print('reading from {}'.format(filepath))                
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.coefs = data[1:-1] # stored as diskset, coefs, ufit
            return self.coefs        

    def extract_bindisp_fit(self):
        if not self.check_has_bindisp():
            print("ERROR: {} has no binned displacement file.".format(self.name))
            exit(1)
        else:
            filepath = self.fitbinpath
            print('reading from {}'.format(filepath))                
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.u_wrapped_bin = data[-1] # stored as diskset, coefs, ufit
            return self.u_wrapped_bin

