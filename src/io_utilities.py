
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
from heterostrain import extract_twist_hetstrain, plot_twist_hetstrain, plotTris, matrixFromTriangleQuantity
from unwrap_utils import getAdjacencyMatrixManual, rotate_uvecs
from visualization import disp_categorize_plot, displacement_colorplot, plot_adjacency, colored_quiver
from utils import nan_gaussian_filter, boolquery, normalize, get_triangles
from basis_utils import latticevec_to_cartesian, cartesian_to_rz_WZ
from strain import strain, unscreened_piezocharge, strain_induced_polarization, strain_method_3
from masking import get_aa_mask, get_sp_masks, make_contour_mask
from new_utils import normNeighborDistance, crop_displacement
from scipy import ndimage
from unwrap_utils import normDistToNearestCenter, getAdjacencyMatrix, automatic_sp_rotation, neighborDistFilter, geometric_unwrap, geometric_unwrap_tri

def bombout(message):
    print(message)
    exit(1)

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

    if ds.u_wrapped is None: ds.extract_displacement_fit()
    u = ds.u_wrapped.copy()

    if False:
        from basis_utils import rotate_uvecs
        #f, axes = plt.subplots(3,2)
        #axes = axes.flatten()
        #N = 50
        #axes[0].quiver(ds.u_wrapped[:N,:N,0],ds.u_wrapped[:N,:N,1])
        #displacement_colorplot(axes[1], ds.u_wrapped[:N,:N,:], sample_angle=0, quiverbool=True)
        u_wrapped_t = np.zeros((ds.u_wrapped.shape[0], ds.u_wrapped.shape[1], ds.u_wrapped.shape[2]))
        for i in range(ds.u_wrapped.shape[0]):
            for j in range(ds.u_wrapped.shape[1]):
                for d in range(ds.u_wrapped.shape[2]):
                    u_wrapped_t[i,j,d] = ds.u_wrapped[j,i,d]
        #axes[2].quiver(u_wrapped_t[:N,:N,0],u_wrapped_t[:N,:N,1])
        #displacement_colorplot(axes[3], u_wrapped_t[:N,:N,:], sample_angle=0, quiverbool=True)
        u_wrapped_t = rotate_uvecs(u_wrapped_t, ang=(-np.pi/3))
        #axes[4].quiver(u_wrapped_t[:N,:N,0],u_wrapped_t[:N,:N,1])
        #displacement_colorplot(axes[5], u_wrapped_t[:N,:N,:], sample_angle=0, quiverbool=True)
        #plt.show()
        if transp: u = u_wrapped_t
        else: u = ds.u_wrapped.copy()

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
    u = u[pad:-pad, pad:-pad, :]
    nx, ny = u.shape[0], u.shape[1]
    n = np.min([nx, ny])
    if n % 2 != 0: 
        u = u[:n-1, :n-1]
        nx, ny = u.shape[0], u.shape[1]
    else:
        u = u[:n, :n]
    assert(u.shape[0] % 2 == 0)

    if flip: u[:,:,0], u[:,:,1] = -u[:,:,0], u[:,:,1]
    u = cartesian_to_rz_WZ(u, sign_wrap=False)
    print(u.shape)
    centers, adjacency_type = getAdjacencyMatrix(u, boundary_val, delta_val, combine_crit, spdist)
    points = [ [c[1], c[0]] for c in centers ]
    #u, ang, adjacency_type = automatic_sp_rotation(u, centers, adjacency_type, transpose=False) # rotate so sp closest to vertical is sp1, gvector choice degenerate under 2pi/3 rotations so arbitrary sp1/sp2/sp3
    
    f, axes = plt.subplots(1,2)
    axes[0].quiver(u[:50,:50,0],u[:50,:50,1])
    displacement_colorplot(axes[1], u[:50,:50,:], sample_angle=0, quiverbool=True)
    plt.show()

    #print('WARNING NOT UNWRAPPING BOMB OUT TO SAVE ADJ MAT ONLY')
    #ds.update_unwraping(u, centers, adjacency_type)
    #f, ax = plt.subplots()
    #ax.quiver(u[:,:,0], u[:,:,1])
    #plt.show()
    #return u, centers, adjacency_type  

    if not tribool: u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, vertices = geometric_unwrap(centers, adjacency_type, u, voronibool, plotting=True) 
    else: u_signalign, u_unwrapped, u_adjusts, nmcenters, regions, vertices = geometric_unwrap_tri(centers, adjacency_type, u) 
    dists = normDistToNearestCenter(u.shape[0], u.shape[1], centers)
    variable_region = (dists > centerdist).astype(int)
    u = strain_method_3(u_unwrapped, points, variable_region)
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
        self.gamma_colormap = 'PiYG_r' #'gamma'
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
        self.plotpath   = os.path.join(self.folderpath, "plots")

        # plots to save, edit here to change directory organization
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
        self.subcroprotpath   = os.path.join(self.plotpath, "rotation_cropped_sub.png")
        self.subcropdilpath   = os.path.join(self.plotpath, "dilation_cropped_sub.png")
        self.subcropgammapath = os.path.join(self.plotpath, "shear_cropped_sub.png")

        self.rotsanitypath    = os.path.join(self.plotpath, "sanity_gvector_rotation_check.png")
        self.localsubplot     = os.path.join(self.plotpath, "sanity_local_substraction.png")
        self.sanity_intfit    = os.path.join(self.plotpath, "sanity_vdf_check2.png")
        self.sanity_vdf       = os.path.join(self.plotpath, "sanity_vdf_check.png")
        self.sanity_axes      = os.path.join(self.plotpath, "sanity_axes.png")
        self.disp_orrientation_sanity =    os.path.join(self.plotpath, "sanity_disp_orrientation.png")
        self.unwrap_orrientation_sanity =  os.path.join(self.plotpath, "sanity_unwrap_orrientation.png")

        # fields to hold the data
        self.u_unwrap       = None     # set by extract_unwraping, update_unwrapping
        self.u_wrapped_bin  = None  
        self.u_wrapped      = None    # set by extract_displacement, update_displacement
        self.centers        = None    # set by extract_unwraping, update_adjacency
        self.adjacency_type = None  # set by extract_unwraping, update_adjacency
        self.diskset        = None  # set by extract_diskset, update_diskset
        self.raw_data       = None

        self.parameter_dict = default_parameters # defined in globals.py  
        self.parameter_dict_comments = dict() 
        self.write_directory()
        self.update_material_parameters(folderprefix)
        #self.update_parameter("SmoothingSigma", 2.0, "asssumed, default")
        #self.update_parameter("PixelSize", 0.5, "asssumed, default")
        self.update_parameter("FittingFunction", "A+Bcos^2+Csincos", "asssumed, default")
        self.update_parameter("BackgroundSubtraction", "Lorenzian", "asssumed, default")
        #self.set_sample_rotation()
        self.update_data_flags()

    def set_sample_rotation(self):
        #if self.check_parameter_is_set("SampleRotation"): 
        #    return
        if not self.check_has_diskset(): 
            print("data has no diskset or specified sample rotation")
            return
        else:
            diskset = self.extract_diskset()
            mean_ang, stderr_ang = diskset.get_rotatation(True, self.rotsanitypath)
            rotation_correction = 11.3
            mean_ang = mean_ang + rotation_correction
            print('Adding rotatational correctio of {} degrees to the measured sample rotation'.format(rotation_correction))
            self.update_parameter("SampleRotation", mean_ang )
            self.update_parameter("K3toHAADFRotation_Used", rotation_correction )
            self.update_parameter("SampleRotationStdErr", stderr_ang )

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
        orrientation = split_prefix[-1].lower()
        materials = split_prefix[0:-1]
        materials = [el.lower() for el in materials]
        if len(materials) > 1: 
            self.is_hbl = True
            self.update_parameter("Material", '-'.join(materials))
        else:
            self.update_parameter("Material", materials[0])
            tag = self.extract_parameter("HeteroBilayer", param_type=str)
            if tag is not None and tag.strip().lower() in ['t','y','yes','true']:
                self.is_hbl = True
            else:
                self.is_hbl = False
        self.update_parameter("Orientation", orrientation)
        for material in materials:
            if material not in known_materials.keys():
                #print("material {} doesnt have tabulated lattice constant data will ask for manual definition".format(material))
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
            probes = glob.glob(os.path.join('..', 'data', 'probe*'))
            if indx == None:
                for i in range(len(probes)): print('{}:    {}'.format(i, probes[i]))
                indx = int(input("Which Probe File to Use? ---> ").lower().strip())
            self.update_parameter("ProbeUsed", probes[indx], "extract_probe")
            print(probes[indx])
            print(self.parameter_dict["ProbeUsed"])
        return get_probe(self.parameter_dict["ProbeUsed"])

    def make_vdf_plots(self, showflag=False):

        if self.diskset is None: self.extract_diskset()
        #if os.path.exists(self.vdfplotpath): return
        diskset = self.diskset
        # saves all disk vdfs
        counter = 0
        tot_img = np.zeros((diskset.nx, diskset.ny))
        print(diskset._in_use)
        self.update_parameter("NumberDisksUsed", diskset.size_in_use, "make_vdf_plots")
        nx, ny = int(np.ceil(diskset.size_in_use ** 0.5)), int(np.ceil(diskset.size_in_use ** 0.5))
        f, axes = plt.subplots(nx, ny)
        axes = axes.flatten()
        gvecs = diskset.clean_normgset()
        for n in range(diskset.size):
            if diskset.in_use(n): 
                img = diskset.df(n)
                tot_img = tot_img + img
                axes[counter].imshow(img, cmap='gray')
                axes[counter].set_title("Disk {}{}".format(gvecs[counter][0],gvecs[counter][1]))
                counter += 1
        for n in range(counter, len(axes)):
            axes[n].axis("off")
        tot_img = normalize(tot_img)
        plt.subplots_adjust(hspace=0.55, wspace=0.3)
        if not showflag: plt.savefig(self.diskplotpath, dpi=300)
        if not showflag: plt.close()
        if showflag: plt.show()

        # saves sum of all disk vdfs
        f, ax = plt.subplots()
        cf = ax.imshow(tot_img, cmap='gray')
        ax.set_title("Sum of Selected Disk Virtual DFs")
        cb = plt.colorbar(cf, ax=ax, orientation='vertical')
        ax.set_xticks(np.arange(0, np.round(diskset.nx/50)*50+1, 50))
        ax.set_yticks(np.arange(0, np.round(diskset.ny/50)*50+1, 50))
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
        print("saving vdf plot to {}".format(self.vdfplotpath))
        if not showflag: plt.savefig(self.vdfplotpath, dpi=300)
        if not showflag: plt.close()
        if showflag: plt.show()
        if False:
            f, ax = plt.subplots(4,3)
            counter = 0
            I = diskset.df_set()
            g1  = np.array([ 0, 2/np.sqrt(3)])
            g2  = np.array([-1, 1/np.sqrt(3)])
            g   = diskset.clean_normgset(sanity_plot = False)
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
        pAA, pSP1, pSP2, pSP3, pAB, rAA, eAA, wSP, eSP = disp_categorize_plot(self.u_wrapped.copy(), ax)
        self.update_parameter("AAPercent", pAA, "make_categorize_plot")
        self.update_parameter("ABPercent", pAB, "make_categorize_plot")
        self.update_parameter("SP1Percent", pSP1, "make_categorize_plot")
        self.update_parameter("SP2Percent", pSP2, "make_categorize_plot")
        self.update_parameter("SP3Percent", pSP3, "make_categorize_plot")
        self.update_parameter("AvgAAradius", rAA, "make_categorize_plot")
        self.update_parameter("ErrAAradius", eAA, "make_categorize_plot")
        self.update_parameter("AvgSPwidth", wSP, "make_categorize_plot")
        self.update_parameter("ErrSPwidth", eSP, "make_categorize_plot")
        t_1 = "{:.2f} % AA {:.2f} % AB {:.2f} % SP".format(pAA, pAB, pSP1+pSP2+pSP3)
        t_2 = "{:.2f}+/-{:.2f} AAr (pix)  {:.2f}+/-{:.2f} SPw (pix)".format(rAA, eAA, wSP, eSP)
        ax.set_title("{}\n{}".format(t_1, t_2))
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        if showflag:
            plt.show()
        else:
            plt.savefig(self.catplotpath, dpi=300)
            plt.close('all')

    def make_sanity_residuals(self, transp=False):

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
        if self.u_wrapped is None: self.extract_displacement_fit()

        g1  = np.array([ 0, 2/np.sqrt(3)])
        g2  = np.array([-1, 1/np.sqrt(3)])
        dfs = []
        dfsA = []
        dfsB = []
        g   = diskset.clean_normgset(sanity_plot = False)

        for n in range(len(g)): I[n,:,:] = normalize(I[n,:,:])

        AB_ratio = []
        for n in range(len(g)):
            df_tot   = np.zeros((self.u_wrapped.shape[0], self.u_wrapped.shape[1]))
            df_Aonly = np.zeros((self.u_wrapped.shape[0], self.u_wrapped.shape[1]))
            df_Bonly = np.zeros((self.u_wrapped.shape[0], self.u_wrapped.shape[1]))
            for i in range(self.u_wrapped.shape[0]):
                for j in range(self.u_wrapped.shape[1]):
                    gvec = g[n][1] * g1 + g[n][0] * g2
                    u    = [self.u_wrapped_raw[i,j,0], self.u_wrapped_raw[i,j,1]]
                    df_tot[i,j]   = A[n] * (np.cos(np.pi * np.dot(gvec, u)))**2
                    df_tot[i,j]  += B[n] * (np.cos(np.pi * np.dot(gvec, u)))*(np.sin(np.pi * np.dot(gvec, u)))
                    df_Aonly[i,j] = A[n] * (np.cos(np.pi * np.dot(gvec, u)))**2
                    df_Bonly[i,j] = B[n] * (np.cos(np.pi * np.dot(gvec, u)))*(np.sin(np.pi * np.dot(gvec, u)))
                    df_tot[i,j]  += C[n] 
            dfs.append(df_tot)
            dfsA.append(df_Aonly)
            dfsB.append(df_Bonly)
            AB_ratio.append(np.abs(B[n]/A[n]) * 100)
        
        print("mean unsigned B/A is ", np.mean(AB_ratio), " percent")
        f, axes = plt.subplots(6,10)

        mean_resid_all_disks = []

        for n in range(6):
            axes[n,0].imshow(dfs[n], cmap='gray', vmin=0., vmax=1.)
            axes[n,1].imshow(I[n,:,:], cmap='gray', vmin=0., vmax=1.)
            axes[n,2].imshow(dfsA[n], vmin=-1., vmax=1.)
            axes[n,3].imshow(dfsB[n], vmin=-1., vmax=1.)
            axes[n,4].imshow(I[n,:,:] - dfs[n], cmap='gray')
            mean_resid_all_disks.append( 100*np.mean(np.abs((I[n,:,:] - dfs[n])).flatten()) )
            axes[n,0].set_title('fit {}-{}{}'.format(n, g[n][1], g[n][0])); 
            axes[n,1].set_title('raw {}-{}{}'.format(n, g[n][1], g[n][0])); 
            axes[n,2].set_title('cos2'); 
            axes[n,3].set_title('sincos'); 
            axes[n,4].set_title('resid'); 
        for n in range(6):
            try:
                axes[n,5+0].set_title('fit {}-{}{}'.format(6+n, g[6+n][1], g[6+n][0])); 
                axes[n,5+1].set_title('raw {}-{}{}'.format(6+n, g[6+n][1], g[6+n][0])); 
                axes[n,5+2].set_title('cos2'); 
                axes[n,5+3].set_title('sincos'); 
                axes[n,5+4].set_title('resid'); 
                axes[n,5+0].imshow(dfs[6+n], cmap='gray', vmin=0., vmax=1.)
                axes[n,5+1].imshow(I[6+n,:,:], cmap='gray', vmin=0., vmax=1.)
                axes[n,5+2].imshow(dfsA[6+n], vmin=-1., vmax=1.)
                axes[n,5+3].imshow(dfsB[6+n], vmin=-1., vmax=1.)
                axes[n,5+4].imshow(I[6+n,:,:] - dfs[6+n], cmap='gray')
                mean_resid_all_disks.append( 100*np.mean(np.abs((I[n+6,:,:] - dfs[n+6])).flatten()) )
            except:
                continue

        print("mean unsigned error is ", np.mean(mean_resid_all_disks), " percent") 
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
            _, _, gamma, theta, dil = self.extract_strain(subtract=False, smoothbool=False)# plot reconstruction rotation
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

    def make_piezo_plots(self, rewrite=True):

        if self.is_hbl: 
            print('skipping HBL piezocharge calculations')
            return 

        if os.path.exists(self.piezoplotpath) and not rewrite: 
            print('{} exists so skipping'.format(self.piezoplotpath))
            return

        if self.u_unwrap is None: self.extract_unwraping()  
        sigma = self.extract_parameter("SmoothingSigma",      update_if_unset=True, param_type=float)
        ss    = self.extract_parameter("PixelSize",           force_set=True,       param_type=float)
        a     = self.extract_parameter("LatticeConstant",     force_set=True,       param_type=float)
        pc    = self.extract_parameter("PiezoChargeConstant", update_if_unset=True, param_type=float)
        sample_angle = self.extract_parameter("SampleRotation", update_if_unset=True, param_type=float)
        smoothed_u = smooth(self.u_unwrap, sigma)
        smooth_scale_u = smoothed_u * a/ss
        piezo_top = unscreened_piezocharge(smooth_scale_u, sample_angle=sample_angle, ss=ss, coef=pc)
        P_top = strain_induced_polarization(smooth_scale_u, sample_angle=sample_angle, ss=ss, coef=pc)

        x, y, px, py = [], [], [], []
        P_mag = np.zeros((P_top.shape[1], P_top.shape[2]))
        for i in range(P_top.shape[1]):
            for j in range(P_top.shape[2]):
                x.append(j)
                y.append(i)
                px.append(P_top[0,i,j])
                py.append(P_top[1,i,j])
                P_mag[i,j] = ( P_top[0,i,j] ** 2 + P_top[1,i,j] ** 2 ) ** 0.5

        f, ax = plt.subplots()
        lim = np.nanmax(np.abs(piezo_top.flatten())) # want colormap symmetric about zero
        im = ax.imshow(piezo_top, origin='lower', cmap=self.piezo_colormap, vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title("$rho_{unscreened piezo}^{top}(e*nm^{-2})$")   
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$") 
        print("saving piezocharge plot to {}".format(self.piezoplotpath)) 
        plt.savefig(self.piezoplotpath, dpi=300)      
        
        f, ax = plt.subplots()
        lim = np.nanmax(np.abs(P_mag.flatten())) # want colormap symmetric about zero
        im = ax.imshow(P_mag, origin='lower', cmap=self.piezo_colormap, vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=ax, orientation='vertical')       
        ax.quiver(x, y, px, py)
        ax.set_title("$P_{unscreened piezo}^{top}(e*nm^{-1})$")    
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        plt.savefig(self.piezoquiverpath, dpi=300)     

        # get the averaged strains
        u, scaled_u, _, _, _ = self.extract_strain()
        tri_centers, thetas, het_strains, deltas, het_strain_proxies = extract_twist_hetstrain(self)
        mask, use_tris, tris, _ = manual_define_good_triangles(piezo_top, [[c[1], c[0]] for c in self.centers], self.adjacency_type)
        uvecs_cart = cartesian_to_rz_WZ(u.copy(), sign_wrap=False)

        # apply mask
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]: piezo_top[i,j] = uvecs_cart[i,j,0] = uvecs_cart[i,j,1] = np.nan
        
        start, spacing = 0.6, 25
        xrang = np.arange(-start,start+(1/spacing),1/spacing)
        N = len(xrang)
        avg_piezo, counter = np.zeros((N,N)), np.zeros((N,N))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not np.isnan(uvecs_cart[i,j,0]) and not np.isnan(uvecs_cart[i,j,1]):
                    ux_index = int(np.round((uvecs_cart[i,j,0]+start)*spacing))
                    uy_index = int(np.round((uvecs_cart[i,j,1]+start)*spacing))
                    avg_piezo[ux_index, uy_index] += piezo_top[i,j]
                    counter[ux_index, uy_index] += 1
        for i in range(N):
            for j in range(N):    
                if counter[i,j] > 0: avg_piezo[i, j] /= counter[i,j]
                else: avg_piezo[i, j] = np.nan

        f, axes = plt.subplots(1,2)
        axes = axes.flatten()
        lim = np.nanmax(np.abs(avg_piezo.flatten())) # want colormap symmetric about zero
        im = axes[0].imshow(avg_piezo, origin='lower', cmap=self.piezo_colormap, vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=axes[0], orientation='vertical')
        axes[0].set_title('$<\\rho_{piezo}^{top}>%$')  
        im = axes[1].imshow(counter, origin='lower', cmap=self.counter_colormap)
        plt.colorbar(im, ax=axes[1], orientation='vertical')
        axes[1].set_title('counts')
        for ax in axes: ax.axis('off')
        for ax in axes: ax.set_aspect('equal')
        print("saving piezo hex plot to {}".format(self.piezohexplotpath))
        plt.savefig(self.piezohexplotpath, dpi=300)
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

    def extract_strain(self, smoothbool=True, subtract=True):
        if self.u_unwrap is None: self.extract_unwraping()          
        sigma = self.extract_parameter("SmoothingSigma", update_if_unset=True, param_type=float)
        thetam = self.extract_parameter("AvgMoireTwist", force_set=True, param_type=float)
        ss = self.extract_parameter("PixelSize", force_set=True, param_type=float)
        a = self.extract_parameter("LatticeConstant", force_set=True, param_type=float)
        sample_angle = self.extract_parameter("SampleRotation", update_if_unset=True, param_type=float)
        if self.is_hbl: 
            twist = self.extract_parameter("DiffractionPatternTwist", update_if_unset=False, param_type=float)
            thetam, deltam = twist, np.abs(thetam)
            print("Will subtract off given twist of {} deg and calculated average lattice mismatch of {} percent ".format(thetam, deltam))
        else:
            deltam = 0.0
        if smoothbool: smoothed_u = smooth(self.u_unwrap, sigma)
        else: smoothed_u = self.u_unwrap
        smooth_scale_u = smoothed_u * a/ss
        _, _, _, _, gamma, thetap, theta, dil = strain(smooth_scale_u, sample_angle)
        if subtract:
            print('Subtract of global delta_rigid, theta_rigid of {} and {}'.format(thetam, deltam))
            return smoothed_u, smooth_scale_u, 100*gamma, theta-thetam, 100*dil-deltam
        else:
            return smoothed_u, smooth_scale_u, 100*gamma, theta, 100*dil

    def make_strainplots_uncropped_dep(self):

        showflag=True
        self.extract_unwraping()  
        #self.u_unwrap = self.u_unwrap[20:30, 20:30, :]
        Ux = self.u_unwrap[:,:,0]
        Uy = self.u_unwrap[:,:,1]
        #print(Ux)
        #print(Uy)
        #if os.path.exists(self.rotplotpath): return
        _, _, gamma, theta, dil = self.extract_strain(subtract=False, smoothbool=False)

        # plot reconstruction rotation
        cmap = self.theta_colormap
        title = '$\\theta_r(^o)$'
        f, ax = plt.subplots()
        lim = np.max(np.abs(theta.flatten())) # want colormap symmetric about zero
        im = ax.imshow(theta, origin='lower', cmap=cmap)#, vmin=-lim, vmax=lim) 
        plot_adjacency(None, self.centers, self.adjacency_type, ax=ax, colored=False) # overlay triangles
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(title)
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        print("saving reconstruction rotation plot to {}".format(self.rotplotpath))  
        if showflag:
            plt.show()
        else:
            plt.savefig(self.rotplotpath, dpi=300)
            plt.close('all')

        # plot shear strain
        cmap = self.gamma_colormap
        title = '$\\gamma(\\%)$'
        f, ax = plt.subplots()
        im = ax.imshow(gamma, origin='lower', cmap=cmap) 
        plot_adjacency(None, self.centers, self.adjacency_type, ax=ax, colored=False) # overlay triangles
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(title)
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        print("saving shear strain plot to {}".format(self.shearplotpath))  
        if showflag:
            plt.show()
        else:
            plt.savefig(self.shearplotpath, dpi=300)
            plt.close('all')

        # plot dilation strain
        cmap = self.dilation_colormap 
        title = '$dil(\\%)$'
        f, ax = plt.subplots()
        lim = np.max(np.abs(dil.flatten())) # want colormap symmetric about zero
        im = ax.imshow(dil, origin='lower', cmap=cmap)#, vmin=-lim, vmax=lim) 
        plot_adjacency(None, self.centers, self.adjacency_type, ax=ax, colored=False) # overlay triangles
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(title)
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        print("saving dilaton plot to {}".format(self.dilplotpath)) 
        if showflag:
            plt.show()
        else:
            plt.savefig(self.dilplotpath, dpi=300)
            plt.close('all')  

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

    def make_hex_plot(self, axis, counts_axis, u, mask, values, centered, colormap, title):
        uvecs_cart = cartesian_to_rz_WZ(u.copy(), sign_wrap=False)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]: values[i,j] = uvecs_cart[i,j,0] = uvecs_cart[i,j,1] = np.nan
        start, spacing = 0.6, 25
        xrang = np.arange(-start,start+(1/spacing),1/spacing)
        N = len(xrang)
        avg_vals, counter = np.zeros((N,N)), np.zeros((N,N))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not np.isnan(uvecs_cart[i,j,0]) and not np.isnan(uvecs_cart[i,j,1]):
                    ux_index = int(np.round((uvecs_cart[i,j,0]+start)*spacing))
                    uy_index = int(np.round((uvecs_cart[i,j,1]+start)*spacing))
                    avg_vals[ux_index, uy_index] += values[i,j]
                    counter[ux_index, uy_index] += 1
        for i in range(N):
            for j in range(N):    
                if counter[i,j] > 0: avg_vals[i, j] /= counter[i,j]
                else: avg_vals[i, j] = counter[i,j] = np.nan
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
        axis.axis('off')
        axis.set_aspect('equal')

    def make_strainplots_localsubtraction(self, rigid_local_twist, rigid_dilation, rigid_gamma):

        rig_dil = rigid_dilation * 100
        rig_gamma = rigid_gamma * 100
        rig_theta = rigid_local_twist * 180/np.pi
        u, scaled_u, gamma, theta, dil = self.extract_strain(smoothbool=True, subtract=False)
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
        f, axes = plt.subplots(2,2)
        axes = axes.flatten()    
        self.make_hex_plot(axes[3], None, u, mask, theta, True, self.theta_colormap, '$<\\theta_r(^o)>%$')
        self.make_hex_plot(axes[2], None, u, mask, dil, True , self.dilation_colormap, '$<dil>$')
        self.make_hex_plot(axes[1], axes[0], u, mask, gamma, True , self.gamma_colormap, '$<\\gamma>$')
        print("saving hex plots to {}".format(self.hexplotpath))
        plt.savefig(self.hexplotpath, dpi=300)
        plt.close('all')

        theta = theta - rig_theta[:theta.shape[0], :theta.shape[1]]
        dil = dil - rig_dil[:dil.shape[0], :dil.shape[1]]
        gamma = gamma - rig_gamma[:gamma.shape[0], :gamma.shape[1]]
        # plot after subtraction
        self.make_strain_plot(theta, True, self.theta_colormap, '$\\theta_r(^o)$', self.subcroprotpath, tris, use_tris)
        self.make_strain_plot(dil, True, self.dilation_colormap, '$dil(\\%)$', self.subcropdilpath, tris, use_tris)
        self.make_strain_plot(gamma, True,  self.gamma_colormap, '$\\gamma(\\%)$', self.subcropgammapath, tris, use_tris)
        f, axes = plt.subplots(2,2)
        axes = axes.flatten()    
        self.make_hex_plot(axes[3], None, u, mask, theta, True, self.theta_colormap, '$<\\theta_r(^o)>%$')
        self.make_hex_plot(axes[2], None, u, mask, dil, True , self.dilation_colormap, '$<dil>$')
        self.make_hex_plot(axes[1], axes[0], u, mask, gamma, True ,  self.gamma_colormap, '$<\\gamma>$')
        print("saving hex plots to {}".format(self.subhexplotpath))
        plt.savefig(self.subhexplotpath, dpi=300)
        plt.close('all')

    def get_strain_stats(self):

        scaleu, u, gamma, theta, dil = self.extract_strain()
        delta=(np.pi/12)
        umag = np.zeros((u.shape[0],u.shape[1]))
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                umag[i,j] = (u[i,j,0]**2 + u[i,j,1]**2)**0.5
        boundary = 0.5 * np.max(u.flatten()) 
        AAmask = get_aa_mask(u, boundary=boundary, smooth=None)
        sp1mask, sp2mask, sp3mask = get_sp_masks(u, AAmask, delta=delta, include_aa=False, window_filter_bool=False)
        SPmask = ( (sp1mask.astype(int) + sp2mask.astype(int) + sp3mask.astype(int)) > 0 ).astype(int)

        AARot, SPRot, ABRot = [], [], []
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                if AAmask[i,j]: AARot.append(theta[i,j])
                elif SPmask[i,j]: SPRot.append(theta[i,j])
                else: ABRot.append(theta[i,j])
  
        self.update_parameter("AvgAAReconRot", np.nanmean(AARot), "get_strain_stats")
        self.update_parameter("ErrAAReconRot", stder(AARot), "get_strain_stats")
        self.update_parameter("AvgABReconRot", np.nanmean(ABRot), "get_strain_stats")
        self.update_parameter("ErrABReconRot", stder(ABRot), "get_strain_stats")
        self.update_parameter("AvgSPReconRot", np.nanmean(SPRot), "get_strain_stats")
        self.update_parameter("ErrSPReconRot", stder(SPRot), "get_strain_stats")

        AADil, SPDil, ABDil = [], [], []
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                if AAmask[i,j]: AADil.append(dil[i,j])
                elif SPmask[i,j]: SPDil.append(dil[i,j])
                else: ABDil.append(dil[i,j])

        self.update_parameter("AvgSPDil", np.nanmean(SPDil), "get_strain_stats")
        self.update_parameter("ErrSPDil", stder(SPDil), "get_strain_stats")
        self.update_parameter("AvgAADil", np.nanmean(AADil), "get_strain_stats")
        self.update_parameter("ErrAADil", stder(AADil), "get_strain_stats")
        self.update_parameter("AvgABDil", np.nanmean(ABDil), "get_strain_stats")
        self.update_parameter("ErrABDil", stder(ABDil), "get_strain_stats")

        AAGamma, SPGamma, ABGamma = [], [], []
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                if AAmask[i,j]: AAGamma.append(gamma[i,j])
                elif SPmask[i,j]: SPGamma.append(gamma[i,j])
                else: ABGamma.append(gamma[i,j])

        self.update_parameter("AvgSPGamma", np.nanmean(SPGamma), "get_strain_stats")
        self.update_parameter("ErrSPGamma", stder(SPGamma), "get_strain_stats")
        self.update_parameter("AvgAAGamma", np.nanmean(AAGamma), "get_strain_stats")
        self.update_parameter("ErrAAGamma", stder(AAGamma), "get_strain_stats")
        self.update_parameter("AvgABGamma", np.nanmean(ABGamma), "get_strain_stats")
        self.update_parameter("ErrABGamma", stder(ABGamma), "get_strain_stats")
     
    def make_cropped_plots_dep(self):
        
        #if os.path.exists(self.cropdisppath): return
        # crop based on triangles
        u, scaled_u, gamma, theta, dil = self.extract_strain()
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

        # plot theta with triangle area mask
        f, ax = plt.subplots()
        lim = np.nanmax(np.abs(theta.flatten()))
        im = ax.imshow(theta, origin='lower', cmap=self.theta_colormap, vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title('$\\theta_r(^o)$')
        plotTris(tris, ax, self.centers, manual=False, use_tris=use_tris)
        plt.savefig(self.croprotpath, dpi=300)
        plt.close('all')

        # plot dil with triangle area mask
        f, ax = plt.subplots()
        lim = np.nanmax(np.abs(dil.flatten()))
        im = ax.imshow(dil, origin='lower', cmap=self.dilation_colormap, vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=ax, orientation='vertical')
        plotTris(tris, ax, self.centers, manual=False, use_tris=use_tris)
        ax.set_title('$dil(\\%)$')
        plt.savefig(self.cropdilpath, dpi=300)
        plt.close('all')
        
        # plot gamma with triangle area mask
        f, ax = plt.subplots()
        im = ax.imshow(gamma, origin='lower', cmap=self.gamma_colormap)
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title('$\\gamma(\\%)$')  
        plotTris(tris, ax, self.centers, manual=False, use_tris=use_tris)
        plt.savefig(self.cropgammapath, dpi=300)
        plt.close('all')

    def make_averaged_hexplots_dep(self):

        #if os.path.exists(self.hexplotpath): return
        # get the averaged strains
        u, scaled_u, gamma, theta, dil = self.extract_strain()

        tri_centers, thetas, het_strains, deltas, het_strain_proxies = extract_twist_hetstrain(self)
        mask, use_tris, tris, _ = manual_define_good_triangles(theta, [[c[1], c[0]] for c in self.centers], self.adjacency_type)
        
        uvecs_cart = cartesian_to_rz_WZ(u.copy(), sign_wrap=False)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]: gamma[i,j] = theta[i,j] = dil[i,j] = uvecs_cart[i,j,0] = uvecs_cart[i,j,1] = np.nan
        
        start, spacing = 0.6, 25
        xrang = np.arange(-start,start+(1/spacing),1/spacing)
        N = len(xrang)
        avg_gamma, avg_theta, avg_dil, counter = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not np.isnan(uvecs_cart[i,j,0]) and not np.isnan(uvecs_cart[i,j,1]):
                    ux_index = int(np.round((uvecs_cart[i,j,0]+start)*spacing))
                    uy_index = int(np.round((uvecs_cart[i,j,1]+start)*spacing))
                    avg_gamma[ux_index, uy_index] += gamma[i,j]
                    avg_theta[ux_index, uy_index] += theta[i,j]
                    avg_dil[ux_index, uy_index] += dil[i,j]
                    counter[ux_index, uy_index] += 1
        for i in range(N):
            for j in range(N):    
                if counter[i,j] > 0:
                    avg_gamma[i, j] /= counter[i,j]
                    avg_theta[i, j] /= counter[i,j]
                    avg_dil[i, j] /= counter[i,j]
                else:
                    avg_gamma[i, j] = avg_theta[i, j] = avg_dil[i, j] = counter[i,j] = np.nan

        f, axes = plt.subplots(2,2)
        axes = axes.flatten()
        lim = np.nanmax(np.abs(avg_theta.flatten())) # want colormap symmetric about zero
        im = axes[0].imshow(avg_theta, origin='lower', cmap=self.theta_colormap, vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=axes[0], orientation='vertical')
        axes[0].set_title('$<\\theta_r(^o)>%$')  

        im = axes[1].imshow(avg_gamma, origin='lower', cmap=self.gamma_colormap)#, vmax=5.0, vmin=0.0)
        plt.colorbar(im, ax=axes[1], orientation='vertical')
        axes[1].set_title('$<\\gamma>$')  
        lim = np.nanmax(np.abs(avg_dil.flatten()))
        im = axes[2].imshow(avg_dil, origin='lower', cmap=self.dilation_colormap, vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=axes[2], orientation='vertical')
        axes[2].set_title('$<dil>$')

        im = axes[3].imshow(counter, origin='lower', cmap=self.counter_colormap) 
        plt.colorbar(im, ax=axes[3], orientation='vertical')
        axes[3].set_title('counts')
         
        for ax in axes: ax.axis('off')
        for ax in axes: ax.set_aspect('equal')
        print("saving hex plots to {}".format(self.hexplotpath))
        plt.savefig(self.hexplotpath, dpi=300)
        plt.close('all')

    def update_diskset(self, diskset):
        if os.path.exists(self.diskpath):
            print('WARNING: overwriting diskset for {}'.format(self.name))
        with open(self.diskpath, 'wb') as f: 
            pickle.dump( diskset, f )

    def update_displacement_fit(self, coefs, fit):
        print('updating displacements for {}'.format(self.name))
        self.u_wrapped = fit
        with open(self.fitpath, 'wb') as f: 
            pickle.dump([self.diskset, coefs[:,0], coefs[:,1], coefs[:,2], fit], f)

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

    def extract_raw(self):
        scan_shape0 = int(self.extract_parameter("ScanShapeX", force_set=True, param_type=float))
        scan_shape1 = int(self.extract_parameter("ScanShapeY", force_set=True, param_type=float))
        if not self.check_has_raw():
            print("ERROR: {} has no raw dataset file.".format(self.name))
            exit(1)
        elif not os.path.exists(self.rawpathh5):
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
        return (value is not default_parameter_filler)    

    def check_has_displacement(self): return os.path.exists(self.fitpath)

    def check_has_diskset(self): return os.path.exists(self.diskpath)

    def check_has_raw(self): return os.path.exists(self.rawpathdm) or os.path.exists(self.rawpathh5)

    def check_has_unwrapping(self): return os.path.exists(self.unwrappath)

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


if __name__ == '__main__':
    main()
