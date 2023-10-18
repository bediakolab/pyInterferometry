
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import warnings
from utils import *
from skimage import measure
from scipy.spatial import ConvexHull

def convex_mask(img, plotflag=False):
    contours = measure.find_contours(img, 0.5)
    #if len(contours) > 0: print('WARNING found more than one contour for region')
    contour = contours[0]
    if plotflag: f, ax = plt.subplots(); ax.imshow(img, origin='lower');
    if plotflag: ax.plot(contour[:,1], contour[:,0], 'r')
    hull = ConvexHull(contour)
    xpts, ypts = contour[hull.vertices,1], contour[hull.vertices,0]
    contour = np.zeros((len(xpts), 2))
    contour[:,1], contour[:,0] = xpts, ypts
    if plotflag: ax.plot(contour[:,1], contour[:,0], 'b')
    if plotflag: plt.show()
    nx, ny = img.shape
    return make_contour_mask(nx, ny, contour)

#USED!
def get_region_centers(labeled, mask, minz=-0.5):
    centers = []
    sizes = []
    for index in range(np.max(labeled) + 1):
        sizes.append(np.count_nonzero((labeled == index)))
    mean = np.nanmean(sizes)
    std = np.nanstd(sizes)
    for index in range(np.max(labeled) + 1):
        label_mask = (labeled == index)
        region_indeces = label_mask.nonzero()
        size = sizes[index]
        region_mask = (mask * label_mask).astype(float)
        region_mask[~label_mask] = np.nan
        if (size - mean)/std > minz and size > 2 and np.nanmean(region_mask) > 0.8:
            avg_i = np.mean(region_indeces[0])
            avg_j = np.mean(region_indeces[1])
            centers.append([avg_i, avg_j])
    return centers

#USED!
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    npad = int(np.ceil(box_pts*1.))
    y_smooth[0:npad] = y[0:npad]
    y_smooth[len(y)-npad:len(y)] = y[len(y)-npad:len(y)]
    return y_smooth

#USED!
def get_sp_line_method2(sp_mask, plotbool=False):
    sp_lines = []
    nx, ny = sp_mask.shape
    if plotbool: f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
    labeled, regions = ndimage.label(sp_mask)
    if plotbool: ax1.imshow(sp_mask, origin="lower")
    sizes = []
    lines1 = []
    lines2 = []
    uselines = []
    for index in range(np.max(labeled) + 1): sizes.append(np.count_nonzero((labeled == index)))
    mean = np.nanmean(sizes)
    std = np.nanstd(sizes)
    for index in range(np.max(labeled) + 1):
        label_mask = (labeled == index)
        if sizes[index] < 25:
            labeled[label_mask] = 0
    if plotbool: ax2.imshow(labeled, origin="lower")
    if plotbool: ax3.imshow(labeled, origin="lower")
    with warnings.catch_warnings(): # ignore mean of empty slice warning this is fine
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for index in range(1, np.max(labeled) + 1):
            label_mask = (labeled == index)
            x, y = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)))
            y[~label_mask] = np.nan
            x[~label_mask] = np.nan
            lines1.append([np.nanmean(x, axis=0), np.nanmean(y, axis=0)])
            lines2.append([np.nanmean(x, axis=1), np.nanmean(y, axis=1)])
            if plotbool: ax3.plot( np.nanmean(x, axis=0), np.nanmean(y, axis=0), color='r')
            if plotbool: ax3.plot( np.nanmean(x, axis=1), np.nanmean(y, axis=1), color='b')
            xrange = np.nanmax(x) - np.nanmin(x)
            yrange = np.nanmax(y) - np.nanmin(y)
            if xrange > yrange:
                uselines.append([np.nanmean(x, axis=0), np.nanmean(y, axis=0)])
                if plotbool: ax3.plot( np.nanmean(x, axis=0), np.nanmean(y, axis=0), color='c')
            else:
                uselines.append([np.nanmean(x, axis=1), np.nanmean(y, axis=1)])
                if plotbool: ax3.plot( np.nanmean(x, axis=1), np.nanmean(y, axis=1), color='c')
    total_line_mask = np.zeros((nx,ny))
    for line in uselines: total_line_mask += make_line_mask(nx, ny, line[0], line[1], pad_w=1)
    if plotbool: ax4.imshow(total_line_mask.T, origin="lower")
    if plotbool: plt.show()
    return total_line_mask.T, uselines

#USED!
def get_aa_mask(ufit, boundary=0.75, plotbool=False, smooth=None):
    nx, ny, dim = ufit.shape
    umag = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            umag[i,j] = (ufit[i,j,0]**2 + ufit[i,j,1]**2)**0.5
    if smooth is not None:       
         umag = ndimage.gaussian_filter(umag, smooth)
    mask = umag < boundary
    if plotbool:
        f, ax = plt.subplots()
        ax.imshow(ndimage.gaussian_filter(mask, 0.2))
        plt.show()
    return ndimage.gaussian_filter(mask, 0.2)

#USED!
def get_sp_masks(ufit, aa_mask, delta=np.pi/12, plotbool=False, include_aa=True, exclude_aa=False, window_filter_bool=True, eps=1e-6):
    # if delta = np.pi/6 complete partition
    # if delta = np.pi/12 half into sp half into AB-type
    nx, ny, dim = ufit.shape
    uang = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            uang[i,j] = np.arctan(ufit[i,j,1]/(eps + ufit[i,j,0])) # cartesian!
            if uang[i,j] < 0: uang[i,j] += 2*np.pi # uang now between 0 and 2pi
    mask_sp1 = ( np.abs(uang - 0        ) < delta ) | ( np.abs(uang - np.pi     ) < delta ) | ( np.abs(uang - 2*np.pi) < delta )
    mask_sp2 = ( np.abs(uang - np.pi/3  ) < delta ) | ( np.abs(uang - 4*np.pi/3 ) < delta ) 
    mask_sp3 = ( np.abs(uang - 2*np.pi/3) < delta ) | ( np.abs(uang - 5*np.pi/3 ) < delta ) 
    mask_xx = np.zeros((ufit.shape[0], ufit.shape[1]))
    mask_mm = np.zeros((ufit.shape[0], ufit.shape[1]))

    """
    for i in range(nx):
        for j in range(ny):
            uang[i,j] = (np.arctan(ufit[i,j,1]/(1e-7 + ufit[i,j,0])) * 12/np.pi) + 6
            if ufit[i,j,1] < 0 : uang[i,j] *= -1
    mask_sp1 = ( np.abs(uang - 0 ) < 1 ) | ( np.abs(uang - 12) < 1 ) | ( np.abs(uang + 12) < 1 )
    mask_sp2 = ( np.abs(uang - 4 ) < 1 ) | ( np.abs(uang + 4) < 1 ) 
    mask_sp3 = ( np.abs(uang - 8 ) < 1 ) | ( np.abs(uang + 8) < 1 )    
    """

    if include_aa:
        mask_sp1 = mask_sp1.astype(int) | aa_mask.astype(int)
        mask_sp2 = mask_sp2.astype(int) | aa_mask.astype(int)
        mask_sp3 = mask_sp3.astype(int) | aa_mask.astype(int)
        mask_xx = mask_xx.astype(int) | aa_mask.astype(int)
        mask_mm = mask_mm.astype(int) | aa_mask.astype(int)

    if exclude_aa:
        mask_sp1 = ((mask_sp1.astype(int) - aa_mask.astype(int)) > 0 ).astype(int)
        mask_sp2 = ((mask_sp2.astype(int) - aa_mask.astype(int)) > 0 ).astype(int)
        mask_sp3 = ((mask_sp3.astype(int) - aa_mask.astype(int)) > 0 ).astype(int)
        mask_xx  = ((mask_xx.astype(int) - aa_mask.astype(int)) > 0 ).astype(int)
        mask_mm  = ((mask_mm.astype(int) - aa_mask.astype(int)) > 0 ).astype(int)
            
    if window_filter_bool:
        from utils import window_filter
        mask_sp1 = window_filter(mask_sp1, 2, method=np.max)
        mask_sp2 = window_filter(mask_sp2, 2, method=np.max)
        mask_sp3 = window_filter(mask_sp3, 2, method=np.max)
        mask_xx = window_filter(mask_xx, 2, method=np.max)
        mask_mm = window_filter(mask_mm, 2, method=np.max)
    if plotbool:
        f, axes = plt.subplots(1,5)
        axes = axes.flatten()
        axes[0].imshow(mask_sp1, origin='lower')
        axes[1].imshow(mask_sp2, origin='lower')
        axes[2].imshow(mask_sp3, origin='lower')
        axes[3].imshow(mask_xx, origin='lower')
        axes[4].imshow(mask_mm, origin='lower')
        plt.show()
    return mask_sp1, mask_sp2, mask_sp3, mask_mm, mask_xx

def get_peak_mask(nx, ny, peaks, diskset, dp, dsnum, radius_factor=1.15):
    nspots = len(peaks.data['qy'])
    mask = np.zeros((nx,ny))
    radii = np.zeros(nspots)
    centers = np.zeros((nspots,2))
    for i in range(nspots): # for each bragg disk
      qx = diskset.x(i) # find x,y coordinates of center
      qy = diskset.y(i)
      radii[i] = radius_factor * diskset.r(i) # extract radius and multiply by radius_factor
      centers[i,0] = qx
      centers[i,1] = qy
    # create a mask around these peak of radius equal to the peak radius scaled by radius_factor
    mask = circular_mask( nx, ny, centers, radii )
    return mask

def get_anomoly_mask(masked_dp, beamcenter, dsnum, probe_rad_f=0.2, bin_w=32, zscore=3):
    # first make sure that we can bin the data by with bin width bin_w
    while masked_dp.shape[0] % bin_w != 0.0:
        bin_w = bin_w // 2
        print('bin width of anomoly mask changed to be divisible by scan dimension')
    masked_dp = bin(masked_dp, bin_w) # bin the data
    nx, ny = masked_dp.shape
    # mask off probe area
    mask = circular_mask( nx, ny, [ (int(beamcenter[1]/bin_w), int(beamcenter[0]/bin_w)) ], [ probe_rad_f*nx ] )
    dp = normalize(mask_off(masked_dp, [mask]))
    dpstd = np.nanstd(dp)
    dpm = np.nanmean(dp)
    # find all areas of high z score
    contours = measure.find_contours(dp, dpm+zscore*dpstd )
    nx, ny = dp.shape
    mask = np.zeros((nx,ny))
    # create mask identifying areas of high zscore
    for contour in contours: mask = mask + make_contour_mask(nx, ny, contour)
    return unbin((mask > 0), bin_w) # unbin the mask before returning

def circular_mask(nx, ny, centers, radii):
    X, Y = np.ogrid[:nx, :ny]
    mask = np.zeros((nx,ny))
    for i in range(len(radii)): # for each radius in the array radii
        try: center = centers[i,:]
        except: center = centers[i]
        radius = radii[i]
        # generate a circle centered at the value in centers and this radius
        dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
        # and add this circle to the mask
        mask = (dist_from_center <= radius) + mask
    return mask

def get_beamstop_mask(dp, dsnum, contour_boundary=0.20):
    # find the boundary between points in the dataset that are above/below a value of contour_boundary
    # in the normalized diffraction pattern
    contours = measure.find_contours(normalize(dp), contour_boundary)
    max_l = 0
    nx, ny = dp.shape
    for contour in contours: # finds the maximum length boundary in the dataset
        if len(contour) > max_l:
            max_l = len(contour)
            bs_contour = contour
    mask = make_contour_mask(nx, ny, bs_contour) #  turns boundary into a mask
    return mask

def make_contour_mask(nx, ny, contour, transpose=False):
    try: p = Path(contour) # generates a path between the points
    except: p = contour
    x, y = np.meshgrid(np.arange(nx), np.arange(ny)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    if transpose: points = np.vstack((y,x)).T
    else: points = np.vstack((x,y)).T
    # identifies if each coordinate is contained in the path of points, generating a mask
    grid = p.contains_points(points)
    return grid.reshape(nx,ny).T # reshape into a matrix

def make_line_mask(nx, ny, xcoords, ycoords, pad_w=1):
    mask = np.zeros((nx, ny)) # allocate mask
    # remove nan values from the data
    xcoords_new = [xcoords[i] for i in range(len(xcoords)) if not (np.isnan(xcoords[i]) or np.isnan(ycoords[i]))]
    ycoords_new = [ycoords[i] for i in range(len(xcoords)) if not (np.isnan(xcoords[i]) or np.isnan(ycoords[i]))]
    for i in range(len(xcoords_new)-1):
        startx = int(np.min([xcoords_new[i],xcoords_new[i+1]]))
        endx   = int(np.max([xcoords_new[i],xcoords_new[i+1]]))
        starty = int(np.min([ycoords_new[i],ycoords_new[i+1]]))
        endy   = int(np.max([ycoords_new[i],ycoords_new[i+1]]))
        # set a small range of values in the mask equal to true
        # corresponds to the rectangle drawn between adjacent points
        mask[startx:endx+1, starty:endy+1] = 1
    return mask

def mask_off(dp, masks, prefix=None, dsnum=None, plotflag=False):
    nx,ny = dp.shape
    # for each mask, set corresponding values in dp to nan
    for mask in masks: dp[mask > 0] = np.nan
    if plotflag:
        # plot and save
        f, ax = plt.subplots()
        im = ax.imshow(dp)
        ax.set_title('Automatically Masked DP')
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig("../plots/{}/ds_{}/masked_dp.png".format(prefix,dsnum), dpi=300)
        plt.close()
    return dp
