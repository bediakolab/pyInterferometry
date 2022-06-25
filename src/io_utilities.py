
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
from basis_utils import latticevec_to_cartesian, cartesian_to_rz_WZ
from strain import strain
from masking import get_aa_mask, get_sp_masks, make_contour_mask
#from new_utils import normNeighborDistance

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
def unwrap_main(ds):

    centerdist, boundary_val, delta_val, combine_crit, spdist = 0.01, 0.3, 0.3, 0.0, 2.0

    if ds.extract_parameter("DisplacementBasis", param_type=str) != "Cartesian":
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
    return u, centers, adjacency_type   

def stder(v): 
    return np.std(v, ddof=1) / np.sqrt(np.size(v))

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
        self.croprotpath      = os.path.join(self.plotpath, "rotation_cropped.png")
        self.cropdilpath      = os.path.join(self.plotpath, "dilation_cropped.png")
        self.cropdisppath     = os.path.join(self.plotpath, "displacement_cropped.png")
        self.cropgammapath    = os.path.join(self.plotpath, "shear_cropped.png")
        self.quivplotpath     = os.path.join(self.plotpath, "quiver.png")

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
        self.update_data_flags()

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
            self.is_hbl = False
        self.update_parameter("Orientation", orrientation)
        for material in materials:
            if material not in known_materials.keys():
                print("material {} doesnt have tabulated lattice constant data will ask for manual definition".format(material))
                return
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

    def extract_probe(self):
        from utils import get_probe
        if not os.path.exists(self.parameter_dict["ProbeUsed"]):    
            probes = glob.glob(os.path.join('..', 'data', 'probe*'))
            if indx == None:
                for i in range(len(probes)): print('{}:    {}'.format(i, probes[i]))
                indx = int(input("Which Probe File to Use? ---> ").lower().strip())
            self.update_parameter("ProbeUsed", probes[indx], "extract_probe")
        return get_probe(self.parameter_dict["ProbeUsed"])

    def make_vdf_plots(self, showflag=False):
        if self.diskset is None: self.extract_diskset()
        if os.path.exists(self.vdfplotpath): return
        diskset = self.diskset
        # saves all disk vdfs
        counter = 0
        tot_img = np.zeros((diskset.nx, diskset.ny))
        self.update_parameter("NumberDisksUsed", diskset.size_in_use, "make_vdf_plots")
        nx, ny = int(np.ceil(diskset.size_in_use ** 0.5)), int(np.ceil(diskset.size_in_use ** 0.5))
        f, axes = plt.subplots(nx, ny)
        axes = axes.flatten()
        for n in range(diskset.size):
            if diskset.in_use(n): 
                img = diskset.df(n)
                tot_img = tot_img + img
                axes[counter].imshow(img, cmap='gray')
                axes[counter].set_title("Disk {}".format(n))
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
        return tot_img    

    def make_twist_plot(self):
        if os.path.exists(self.twistplotpath): return
        if self.adjacency_type is None: self.extract_unwraping()
        f, ax = plt.subplots(1,2) 
        tri_centers, thetas, het_strains = extract_twist_hetstrain(self)
        self.update_parameter("AvgMoireTwist", np.nanmean(thetas), "make_twist_plot")
        self.update_parameter("AvgHeteroStrain", np.nanmean(het_strains), "make_twist_plot")
        self.update_parameter("ErrMoireTwist", stder(thetas), "make_twist_plot")
        self.update_parameter("ErrHeteroStrain", stder(het_strains), "make_twist_plot")
        plot_twist_hetstrain(self, ax[0], ax[1], thetas, het_strains, tri_centers)
        print("saving twist plot to {}".format(self.twistplotpath))
        plt.savefig(self.twistplotpath, dpi=300)
        plt.close('all')

    def make_piezoplot(self):
        
        print("WARNING: assuming that displacement in top/bottom layers of same magnitude. Heterostrain can cause deviation from this. For HBL this might be far from truth.")

        

        return

    def make_adjacency_plot(self):

        if os.path.exists(self.adjplotpath): return
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
        if self.extract_parameter("DisplacementBasis", param_type=str) != "Cartesian":
            u = latticevec_to_cartesian(self.u_wrapped.copy())
        else:
            u = self.u_wrapped.copy()
        u = cartesian_to_rz_WZ(u, False)
        colored_quiver(ax, u[:,:,0], u[:,:,1])
        plt.savefig("TEST2", dpi=300)
        plt.close('all')
        exit()
        """

    def make_categorize_plot(self, showflag=False):
        if os.path.exists(self.catplotpath): return
        if self.u_wrapped is None: self.extract_displacement_fit()
        f, ax = plt.subplots()
        pAA, pSP1, pSP2, pSP3, pAB, rAA, eAA, wSP, eSP = disp_categorize_plot(self.u_wrapped, ax)
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
        
    def make_displacement_plot(self, showflag=False):
        if os.path.exists(self.dispplotpath): return
        if self.u_wrapped is None: self.extract_displacement_fit()
        f, ax = plt.subplots(1,2)
        if self.extract_parameter("DisplacementBasis", param_type=str) != "Cartesian":
            img = displacement_colorplot(ax[0], latticevec_to_cartesian(self.u_wrapped), quiverbool=False)
            img = displacement_colorplot(ax[1], latticevec_to_cartesian(self.u_wrapped), quiverbool=True)
        else:
            img = displacement_colorplot(ax[0], self.u_wrapped, quiverbool=False)
            img = displacement_colorplot(ax[1], self.u_wrapped, quiverbool=True)

        print("saving displacement plot to {}".format(self.dispplotpath)) 
        ax[0].set_title("$u_{raw}(x,y)$")    
        ax[0].set_xlabel("$x(pixels)$")  
        ax[0].set_ylabel("$y(pixels)$")
        ax[1].set_title("$u_{raw}(x,y)$")    
        ax[1].set_xlabel("$x(pixels)$") 
        if showflag: plt.show() 
        else:
            plt.savefig(self.dispplotpath, dpi=300)
            plt.close('all')

    def make_bindisplacement_plot(self):
        if os.path.exists(self.dispbin2plotpath): return
        if self.u_wrapped_bin is None: self.extract_bindisp_fit()
        f, ax = plt.subplots()
        img = displacement_colorplot(ax, self.u_wrapped_bin)
        print("saving binned displacement plot to {}".format(self.dispbin2plotpath))  
        ax.set_title("$u_{bin}(x,y)$")  
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        plt.savefig(self.dispbin2plotpath, dpi=300)
        plt.close('all')

    def extract_strain(self):
        if self.u_unwrap is None: self.extract_unwraping()  
        sigma = self.extract_parameter("SmoothingSigma", update_if_unset=True, param_type=float)
        thetam = self.extract_parameter("AvgMoireTwist", force_set=True, param_type=float)
        ss = self.extract_parameter("PixelSize", force_set=True, param_type=float)
        a = self.extract_parameter("LatticeConstant", force_set=True, param_type=float)
        if self.is_hbl: 
            twist = self.extract_parameter("DiffractionPatternTwist", update_if_unset=False, param_type=float)
            thetam, deltam = twist, np.abs(thetam)
            print("Will subtract off given twist of {} deg and calculated average lattice mismatch of {} percent ".format(thetam, deltam))
        else:
            deltam = 0.0
        smoothed_u = smooth(self.u_unwrap, sigma)
        smooth_scale_u = smoothed_u * a/ss
        _, _, _, _, gamma, thetap, theta, dil = strain(smooth_scale_u)
        return smoothed_u, smooth_scale_u, 100*gamma, theta-thetam, 100*dil-deltam

    def make_strainplots_uncropped(self):

        if os.path.exists(self.rotplotpath): return
        _, _, gamma, theta, dil = self.extract_strain()

        # plot reconstruction rotation
        cmap = 'RdBu_r'
        title = '$\\theta_r(^o)$'
        f, ax = plt.subplots()
        lim = np.max(np.abs(theta.flatten())) # want colormap symmetric about zero
        im = ax.imshow(theta, origin='lower', cmap=cmap, vmin=-lim, vmax=lim) 
        plot_adjacency(None, self.centers, self.adjacency_type, ax=ax, colored=False) # overlay triangles
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(title)
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        print("saving reconstruction rotation plot to {}".format(self.rotplotpath))  
        plt.savefig(self.rotplotpath, dpi=300)
        plt.close('all')

        # plot shear strain
        cmap = 'inferno'
        title = '$\\gamma(\\%)$'
        f, ax = plt.subplots()
        im = ax.imshow(gamma, origin='lower', cmap=cmap) 
        plot_adjacency(None, self.centers, self.adjacency_type, ax=ax, colored=False) # overlay triangles
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(title)
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        print("saving shear strain plot to {}".format(self.shearplotpath))  
        plt.savefig(self.shearplotpath, dpi=300)
        plt.close('all')

        # plot dilation strain
        cmap = 'RdBu_r'
        title = '$dil(\\%)$'
        f, ax = plt.subplots()
        lim = np.max(np.abs(dil.flatten())) # want colormap symmetric about zero
        im = ax.imshow(dil, origin='lower', cmap=cmap, vmin=-lim, vmax=lim) 
        plot_adjacency(None, self.centers, self.adjacency_type, ax=ax, colored=False) # overlay triangles
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title(title)
        ax.set_xlabel("$x(pixels)$")  
        ax.set_ylabel("$y(pixels)$")
        print("saving dilaton plot to {}".format(self.dilplotpath)) 
        plt.savefig(self.dilplotpath, dpi=300)
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
     
    def make_cropped_plots(self):
        
        if os.path.exists(self.cropdisppath): return
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
        im = ax.imshow(theta, origin='lower', cmap='RdBu_r', vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title('$\\theta_r(^o)$')
        plotTris(tris, ax, self.centers, manual=False, use_tris=use_tris)
        plt.savefig(self.croprotpath, dpi=300)
        plt.close('all')

        # plot dil with triangle area mask
        f, ax = plt.subplots()
        lim = np.nanmax(np.abs(dil.flatten()))
        im = ax.imshow(dil, origin='lower', cmap='RdBu_r', vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=ax, orientation='vertical')
        plotTris(tris, ax, self.centers, manual=False, use_tris=use_tris)
        ax.set_title('$dil(\\%)$')
        plt.savefig(self.cropdilpath, dpi=300)
        plt.close('all')
        
        # plot gamma with triangle area mask
        f, ax = plt.subplots()
        im = ax.imshow(gamma, origin='lower', cmap='inferno')
        plt.colorbar(im, ax=ax, orientation='vertical')
        ax.set_title('$\\gamma(\\%)$')  
        plotTris(tris, ax, self.centers, manual=False, use_tris=use_tris)
        plt.savefig(self.cropgammapath, dpi=300)
        plt.close('all')

    def make_averaged_hexplots(self):

        if os.path.exists(self.hexplotpath): return
        # get the averaged strains
        u, scaled_u, gamma, theta, dil = self.extract_strain()
        tri_centers, thetas, het_strains = extract_twist_hetstrain(self)
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
        im = axes[0].imshow(avg_theta, origin='lower', cmap='RdBu_r', vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=axes[0], orientation='vertical')
        axes[0].set_title('$<\\theta_r(^o)>%$')  

        im = axes[1].imshow(avg_gamma, origin='lower', cmap='inferno')#, vmax=5.0, vmin=0.0)
        plt.colorbar(im, ax=axes[1], orientation='vertical')
        axes[1].set_title('$<\\gamma>$')  
        lim = np.nanmax(np.abs(avg_dil.flatten()))
        im = axes[2].imshow(avg_dil, origin='lower', cmap='RdBu_r', vmax=lim, vmin=-lim)
        plt.colorbar(im, ax=axes[2], orientation='vertical')
        axes[2].set_title('$<dil>$')

        im = axes[3].imshow(counter, origin='lower', cmap='inferno')
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
            print('ERROR parameter {} is unrecognized for {}'.format(field, self.name))
            exit(1)
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
        scan_shape0 = self.extract_parameter("ScanShapeX", force_set=True, param_type=float)
        scan_shape1 = self.extract_parameter("ScanShapeY", force_set=True, param_type=float)
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
        return data

    def extract_parameter(self, field, force_set=False, update_if_unset=False, param_type=str):
        self.param_dictionary, self.parameter_dict_comments = parse_parameter_file(self.parampath)
        if field not in self.param_dictionary.keys():
            print('parameter {} is unrecognized'.format(field))
            exit(1)
        value = self.param_dictionary[field]
        if value == default_parameter_filler: 
            if update_if_unset: 
                value = self.update_parameter(field) # set the parameter if unset
            elif force_set:
                print('ERROR: required parameter {} is unset'.format(field))
                exit(1)
            else:
                value = None
        if value is not None and not isinstance(value, param_type):
            print('ERROR: parameter {} of value {} is not of type {}, instead {}'.format(field, value, param_type, type(value)))
            exit(1)
        return value

    def check_parameter_is_set(self, field):
        param_dictionary, self.parameter_dict_comments = parse_parameter_file(self.parampath)
        if field not in param_dictionary.keys():
            print('parameter {} is unrecognized'.format(field))
            exit(1)
        value = param_dictionary[field]
        return (value is not default_parameter_filler)    

    def check_has_displacement(self):
        return os.path.exists(self.fitpath)

    def check_has_diskset(self): 
        return os.path.exists(self.diskpath)

    def check_has_raw(self): 
        return os.path.exists(self.rawpathdm) or os.path.exists(self.rawpathh5)

    def check_has_unwrapping(self): 
        return os.path.exists(self.unwrappath)

    def extract_diskset(self):
        if not self.check_has_diskset():
            print("ERROR: {} has no diskset file.".format(self.name))
            exit(1)
        else:
            filepath = self.diskpath
            print('reading from {}'.format(filepath))
            with open(filepath, 'rb') as f: self.diskset = pickle.load(f)
            return self.diskset

    def extract_unwraping(self):
        if not self.check_has_unwrapping():
            print("ERROR: {} has no unwrapping file.".format(self.name))
            exit(1)
        else:
            filepath = self.unwrappath
            print('reading from {}'.format(filepath))
            with open(filepath, 'rb') as f: d = pickle.load(f)
            u, centers, adjacency_type = d[0], d[1], d[2]
            self.u_unwrap, self.centers, self.adjacency_type = u, centers, adjacency_type
            if self.is_hbl:
                u = -1 * rotate_uvecs(u, ang=-1/3*np.pi) # -1 so diverging by default, rotate so sp1 horizontal
                for i in range(adjacency_type.shape[0]):
                    for j in range(adjacency_type.shape[0]):
                        if adjacency_type[i,j] > 0: adjacency_type[i,j] = ((adjacency_type[i,j]+1)%3)+1 #123->312
                self.u_unwrap, self.centers, self.adjacency_type = u, centers, adjacency_type
            return self.u_unwrap, self.centers, self.adjacency_type

    def extract_displacement_fit(self):
        if not self.check_has_displacement():
            print("ERROR: {} has no displacement file.".format(self.name))
            exit(1)
        else:
            filepath = self.fitpath
            print('reading from {}'.format(filepath))                
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.u_wrapped = data[-1] # stored as diskset, coefs, ufit
            return self.u_wrapped

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
