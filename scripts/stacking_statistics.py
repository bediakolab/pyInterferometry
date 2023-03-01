
from utils import *
from masking import *
import matplotlib
import numpy as np
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import pickle
from visualization import *
from utils import *
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
import warnings
import numpy.random as random
from new_utils import crop_displacement, import_uvector, get_lengths, get_angles, get_area
from visualization import displacement_colorplot, make_legend


# to do show this on the visualization legend
def extract_stacking_stats(ufit, filenm, savename, radname, spname, boundary_val=None, delta_val=None):
    nx, ny = ufit.shape[0:2]
    img = displacement_colorplot(None, ufit[:,:,0].reshape(nx, ny), ufit[:,:,1].reshape(nx, ny))
    #img, ufit = crop_displacement(img, ufit)
    if boundary_val == None:
        boundary_val = float(input('What value boundary to use for AA regions? Value used will be illustrated on hexagon legend.  '))
    if delta_val == None:
        delta_val = float(input('What angle threshold (in radians) to use for SP regions? Will be illustrated on hexagon legend.  '))
    mask_aa = get_aa_mask(ufit, boundary=boundary_val)
    mask_sp1, mask_sp2, mask_sp3 = get_sp_masks(ufit, mask_aa, delta=delta_val, include_aa=False, window_filter_bool=False)
    nx, ny = mask_aa.shape
    percent_aa  = np.sum(mask_aa)/(nx*ny)
    percent_sp1 = np.sum(mask_sp1)/(nx*ny)
    percent_sp2 = np.sum(mask_sp2)/(nx*ny)
    percent_sp3 = np.sum(mask_sp3)/(nx*ny)
    f, ((ax1,ax2,ax3,ax4,ax5),(ax1_2,ax2_2,ax3_2,ax4_2,ax5_2)) = plt.subplots(2,5)
    ax1.imshow(mask_aa, cmap='plasma')
    aa_radii = extract_contour_radius(mask_aa, ax=ax1, contour_boundary=0.5)
    ax1_2.hist(aa_radii, color='k')
    for axis in ['top','bottom','left','right']: ax1_2.spines[axis].set_linewidth(2)
    writefile(radname, 'radii for {}'.format(filenm), aa_radii)
    ax5.set_title(' AA bound = {} \n SP bound = {}'.format(boundary_val, delta_val))
    if len(aa_radii) > 0: ax1_2.set_title('AA radii \n mean = {:2.3f} \n sd = {:2.3f}'.format(np.mean(aa_radii), np.std(aa_radii)))
    ax1.set_title('AA, {:2.3f}% area'.format(100*percent_aa))
    for axis in ['top','bottom','left','right']: ax1.spines[axis].set_linewidth(2)
    ax2.imshow(mask_sp1, cmap='plasma')
    ax2.set_title('SP1, {:2.3f}% area'.format(100*percent_sp1))
    for axis in ['top','bottom','left','right']: ax2.spines[axis].set_linewidth(2)
    ax3.imshow(mask_sp2, cmap='plasma')
    ax3.set_title('SP2, {:2.3f}% area'.format(100*percent_sp2))
    for axis in ['top','bottom','left','right']: ax3.spines[axis].set_linewidth(2)
    ax4.imshow(mask_sp3, cmap='plasma')
    ax4.set_title('SP3, {:2.3f}% area'.format(100*percent_sp3))
    for axis in ['top','bottom','left','right']: ax4.spines[axis].set_linewidth(2)
    make_legend(ax=ax5, boundary=boundary_val, delta=delta_val, plotflag=False)
    displacement_colorplot(ax5_2, ufit[:,:,0].reshape(nx, ny), ufit[:,:,1].reshape(nx, ny))
    widths = extract_widths(mask_sp1, ax2)
    ax2_2.hist(widths, color = 'c')
    for axis in ['top','bottom','left','right']: ax2_2.spines[axis].set_linewidth(2)
    tot_widths = []
    for el in widths: tot_widths.append(str(el))
    if len(widths) > 0: ax2_2.set_title('SP1 width \n mean = {:2.1f} \n max = {:2.1f} \n std = {:2.1f}'.format(np.mean(widths), np.max(widths), np.std(widths)))
    widths = extract_widths(mask_sp2, ax3)
    ax3_2.hist(widths, color = 'y')
    for axis in ['top','bottom','left','right']: ax3_2.spines[axis].set_linewidth(2)
    for el in widths: tot_widths.append(str(el))
    if len(widths) > 0: ax3_2.set_title('SP2 width \n mean = {:2.1f} \n max = {:2.1f} \n std = {:2.1f}'.format(np.mean(widths), np.max(widths), np.std(widths)))
    widths = extract_widths(mask_sp3, ax4)
    ax4_2.hist(widths, color = 'm')
    for axis in ['top','bottom','left','right']: ax4_2.spines[axis].set_linewidth(2)
    for el in widths: tot_widths.append(str(el))
    writefile(spname, 'SP widths for {}'.format(filenm), tot_widths)
    if len(widths) > 0: ax4_2.set_title('SP3 width \n mean = {:2.1f} \n max = {:2.1f} \n std = {:2.1f}'.format(np.mean(widths), np.max(widths), np.std(widths)))
    f.set_size_inches(20.0, 7.0)
    plt.subplots_adjust(hspace=0.582, wspace=0.26, left=0.068, right=0.965)
    plt.savefig(savename, dpi=300)
    print('saving to ', savename)#plt.show()
    return percent_aa, percent_sp1+percent_sp2+percent_sp3, aa_radii, [float(v) for v in tot_widths]

def extract_widths(mask, ax=None):
    contours = measure.find_contours(mask, 0.5)
    from utils import is_connected
    widths = []
    for contour in contours:
        contlen = len(contour[:, 1])
        if contlen > 20 and is_connected(contour):
            xcont = smooth(contour[:, 1], 15) #contour[:, 1]
            ycont = smooth(contour[:, 0], 15) #contour[:, 0]
            xcent = np.mean(xcont)
            ycent = np.mean(ycont)
            rads = [ np.sqrt((x-xcent)**2 + (y-ycent)**2) for x,y in zip(xcont, ycont) ]
            min_rad = np.min(rads)
            widths.append(min_rad*2)
            circle = plt.Circle((xcent, ycent), min_rad, color='r', fill=False)
            if ax is not None: ax.add_patch(circle)
    return widths

def stackingstats_main(uvecs, prefix, dsnum, boundary_val=None, delta_val=None):
    filenm = os.path.join(prefix,'ds_{}'.format(dsnum))
    filepath = os.path.join('..','results',filenm)
    nx, ny = uvecs.shape[0:2]
    savename = "{}_geomstat.png".format(filepath)
    radname = "{}_rads.txt".format(filepath)
    spname = "{}_spwidths.txt".format(filepath)
    percent_aa, percent_sp, aa_radii, widths = extract_stacking_stats(uvecs, filepath, savename, radname, spname, boundary_val, delta_val)
    return filepath, percent_aa, percent_sp, aa_radii, widths

def plot_all(aathresh=0.5, spthresh=0.25, amaterial=0.315): #default a = 3.15A = mos2
    aar = []
    arm = []
    aas = []
    spw = []
    spm = []
    sps = []
    angs = [0.52, 1.89, 1.81, 1.80, 1.71, 1.26, 1.67, 1.37, 1.98, 1.63, 1.77, 1.92, 1.2, 1.21, 1.84, 1.76, 1.36, 1.6, 0.8, 0.68, 0.37, 0.4]
    files = glob.glob(os.path.join('..', 'results', 'MV_2.10.21_4d_background*', '*rads.txt'))
    for filepath in files:
        v = []
        with open(filepath, 'rb') as f: 
            for line in f:
                try: v.append(float(line))
                except: continue
        aar.append(np.mean(v))
        aas.append(np.std(v))
        arm.append(np.max(v))
    plt.errorbar(angs, aar, yerr=aas, fmt='o', c='r')
    #plt.scatter(angs, arm)
    files = glob.glob(os.path.join('..', 'results', 'MV_2.10.21_4d_background*', '*spwidths.txt'))
    for filepath in files:
        v = []
        with open(filepath, 'rb') as f: 
            for line in f:
                try: v.append(float(line))
                except: continue
        spw.append(np.mean(v))
        sps.append(np.std(v))
        spm.append(np.max(v))
    plt.errorbar(angs, spw, yerr=sps, fmt='o', c='b')
    ang_dense = np.arange(0, 3, 0.1)
    wavels = [amaterial / (2 * np.sin(np.pi/180 * ang_el)) for ang_el in ang_dense]
    rigid_aa = [aathresh * w/2 for w in wavels]
    rigid_sp = [2 * w * np.tan(spthresh) for w in wavels]
    plt.plot(ang_dense, rigid_aa, c='r')
    #plt.plot(ang_dense, rigid_sp, c='b')
    #plt.scatter(angs, spm)
    plt.xticks(rotation='vertical')
    plt.show()

if __name__ == "__main__":
    # plot_all()
    # exit()
    # for i in range(95): #95
    #     uvecs, prefix, dsnum = import_uvector(i)
    #     if dsnum != None:
    #         stackingstats_main(uvecs, prefix, dsnum, 0.5, 0.25)
    # exit()
    stackbool = boolquery("would you extract stacking statistics from a saved dataset?")
    while stackbool:
        uvecs, prefix, dsnum, _ = import_uvector()
        stackingstats_main(uvecs, prefix, dsnum, 0.5, 0.25)
        stackbool = boolquery("would you extract stacking statistics from another saved dataset?")
