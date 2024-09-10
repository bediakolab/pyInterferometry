
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import *
from time import sleep
import os
import pickle
from skimage.measure import approximate_polygon
from scipy.optimize import least_squares
import numpy as np

# for first two rings, fit to the hexagons expected of triangular lattice on-zone
# assuming minimal impact of stigmation and heterostrain on effective g vectors
# needed since will be using more idealized g_i in the interferometry fitting and 
# need to know the rotational offset between actual g_i and the values used
def expected_bragg_locs(gvecs, com_x, com_y, rotation, scale=1): # no stigmation assume!
    center = np.array([com_x, com_y])
    a1rot = scale * rotate2d(np.array([0,1]), rotation*np.pi/180)
    a2rot = scale * rotate2d(np.array([-np.sqrt(3)/2, 0.5]), rotation*np.pi/180)
    pts = []
    locs = [ (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (2,-1), (-1,2), (-2,1), (1,-2), (1,1), (-1,-1) ]
    for nm in locs: pts.append( center + nm[0]*a1rot + nm[1]*a2rot )
    # getting an ordering of the expected disks aligning with provided g,
    # also will deal with scenario where only measure 11 disks from beam stop
    use_pts = []
    #print(pts)
    for g in gvecs:
        dists = [(g[0] - pts[i][0])**2 + (g[1] - pts[i][1])**2 for i in range(len(pts))]
        #print(dists)
        index_guess_pt = np.argmin(dists)
        use_pts.append(pts[index_guess_pt])
        pts.pop(index_guess_pt)
    return use_pts

def expected_bragg_dists(gvecs, com_x, com_y, rotation, scale=1): # no stigmation assume!
    pts = expected_bragg_locs(gvecs, com_x, com_y, rotation, scale)
    dists = []
    for pt in pts: dists.append(((pt[0]-com_x)**2 + (pt[1]-com_y)**2)**0.5)
    return dists 
def expected_hex_g(gvecs, com_x, com_y, rotation, scale=1): # no stigmation assume!
    center = np.array([com_x, com_y])
    a1rot = scale * rotate2d(np.array([0,1]), rotation*np.pi/180)
    a2rot = scale * rotate2d(np.array([-np.sqrt(3)/2, 0.5]), rotation*np.pi/180)
    pts = []
    locs = [ (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (2,-1), (-1,2), (-2,1), (1,-2), (1,1), (-1,-1) ]
    for nm in locs: pts.append( center + nm[0]*a1rot + nm[1]*a2rot )
    # getting an ordering of the expected disks aligning with provided g,
    # also will deal with scenario where only measure 11 disks from beam stop
    simplifygvecs, use_pts = [], []
    for g in gvecs:
        dists = [(g[0] - pts[i][0])**2 + (g[1] - pts[i][1])**2 for i in range(len(pts))]
        index_guess_pt = np.argmin(dists)
        use_pts.append(pts[index_guess_pt])
        pts.pop(index_guess_pt)
        simplifygvecs.append(locs[index_guess_pt])
        locs.pop(index_guess_pt)        
    return simplifygvecs, use_pts

def fit_disklocations_to_hexagonal(ax, diskset):

    distances = []
    diskset.set_com_central()
    com = diskset.centralbeam
    graw = diskset.d_set()
    #print('fitting with ', graw)
    ringnos = diskset.determine_rings()
    for i in range(len(graw)): 
        if ringnos[i] == 1: 
            distances.append((graw[i][0]**2 + graw[i][1]**2)**0.5)
        if ringnos[i] == 2: 
            distances.append(1/np.sqrt(3) * (graw[i][0]**2 + graw[i][1]**2)**0.5)
    guess_scale = np.mean(distances)

    def _cost_func(vars):
        comx, comy, rotation, scale = vars[:]
        pts = expected_bragg_locs(graw, comx, comy, rotation, scale)
        return [((y[0] - l[0])**2 + (y[1] - l[1])**2)**0.5 for y,l in zip(pts,graw)]

    guess_prms = [ com[0], com[1], 0, guess_scale ]
    print("starting with central=[{:.2f},{:.2f}], twist={:.2f} degrees, scale={:.2f}".format(com[0], com[1], 0, guess_scale))
    opt = least_squares(_cost_func, guess_prms, verbose=1, max_nfev=8000)
    comx, comy, rotation, scale = opt.x[:]
    # can add integer multiples of 60 degrees, want to pick smaller angle (-30 to 30)
    # probably better to use modulo math here, but really not bottleneck and this works fine
    orig_rot = rotation
    while np.abs(rotation) > 30:
        if rotation > 0: 
            rotation -= 60
        elif rotation < 0: 
            rotation += 60
    print('rotation {} --> {}'.format(orig_rot, rotation))
    print("ending with central=[{:.2f},{:.2f}], twist={:.2f} degrees, scale={:.2f}".format(comx, comy, rotation, scale))
    idealized_gvecs, pts = expected_hex_g(graw, comx, comy, rotation, scale)
    if ax != None:
        for i in range(len(graw)): 
            ax.scatter(pts[i][0], pts[i][1], c='r')
            ax.scatter(graw[i][0], graw[i][1], c='k')
            ax.text(pts[i][0], pts[i][1], "{:.1f}{:.1f}".format(idealized_gvecs[i][0],idealized_gvecs[i][1]), c='k')
        idealized_gvecs, pts = expected_hex_g(graw, comx, comy, 0, scale)
        for i in range(len(pts)): 
            ax.scatter(pts[i][0], pts[i][1], c='b')
            ax.text(pts[i][0], pts[i][1], "{:.1f}{:.1f}".format(idealized_gvecs[i][0],idealized_gvecs[i][1]), c='b')
        ax.axis('equal')
    dists = expected_bragg_dists(graw, comx, comy, rotation, scale)
    print('stigmation estimate: ', dists)
    ax.text(0,0, 'hex fit (rot={:.2f} deg)'.format(rotation), c='r')
    ax.text(0,0.25, 'input',c='k')
    ax.text(0,-0.25, 'g used',c='b')
    return idealized_gvecs, rotation

# click instead of asking for entries
def manual_select_disks(diskset, dp, use_log):
    plt.close('all')
    fig, ax = plt.subplots()
    def click_event(click):
        y,x = click.xdata, click.ydata
        disk_id = diskset.get_closest_disk(x,y)
        print('selecting disk ', disk_id)
        diskset.set_useflag(disk_id,True)
        print('now have {} disks selected'.format(diskset.size_in_use))
        x,y = diskset.x(disk_id), diskset.y(disk_id)
        ax.scatter(y,x,color='w')
        fig.canvas.draw()
    print("please click to define the desired peaks, close figure when done")
    diskset.plot(dp, use_log=use_log, ax=ax, f=fig)
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    return diskset.size_in_use, diskset

def manual_define_rectangle(diskset, vdf=None):
    plt.close('all')
    fig, ax = plt.subplots()
    if vdf is None: vdf = overlay_vdf(diskset, plotflag=False)
    vertices = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        vertices.append([x,y])
        print('vertex {} at ({},{})'.format(len(vertices), x, y))
        ax.scatter(x,y,color='k')
        if len(vertices) > 1:
            ax.plot([vertices[-1][0], vertices[-2][0]], [vertices[-1][1], vertices[-1][1]], color='k')
            ax.plot([vertices[-1][0], vertices[-1][0]], [vertices[-1][1], vertices[-2][1]], color='k')
            ax.plot([vertices[-1][0], vertices[-2][0]], [vertices[-2][1], vertices[-2][1]], color='k')
            ax.plot([vertices[-2][0], vertices[-2][0]], [vertices[-1][1], vertices[-2][1]], color='k')
        fig.canvas.draw()
        if len(vertices) == 2:
            sleep(1)
            fig.canvas.mpl_disconnect(cid)
            plt.close('all')
    print("please click twice to define rectangular region")
    ax.imshow(vdf, cmap='gray')
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    print('finished with manual region definition')
    return vertices

def manual_define_disks(diskset, r, dp, use_log):
    plt.close('all')
    fig, ax = plt.subplots()
    def click_event(click):
        y,x = click.xdata, click.ydata # this is a little slower since resizes after each add but not a bottleneck
        diskset.add_disk(x, y, False, r)
        ax.scatter(y,x,color='b')
        fig.canvas.draw()
    print("please click on the center of the disk in the plot that appears to define new peaks close when done.")
    diskset.plot(dp, use_log=use_log, ax=ax, f=fig)
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    print('finished with manual disk definition')

####################################################################################################
# returns virtual dark field image for given bragg disk centered at (x0,y0) of radius r
####################################################################################################
def integrate_disks(datacube, diskset, sub=False, background_fit=None, radius_factor=1.0):
    for n in range(diskset.size):
        if diskset.in_use(n): # for disks which have been tagged as important
            x = diskset.x(n) # extract x position of disk center from the provided diskset container class
            y = diskset.y(n) # extract y position of disk center from the provided diskset container class
            r = diskset.r(n) # extract radius of disk from the provided diskset container class
            # sum pixels in this circular region to generate dark field
            img = py4DSTEM.process.virtualimage.get_virtualimage_circ(datacube,x,y,radius_factor*r)
            if sub and background_fit is not None:
                img_bknd_v = integrate_bknd_circ(background_fit,x,y,radius_factor*r)
                img_bknd_v /= np.float(img.shape[0] * img.shape[1])
                img_bknd = np.ones(img.shape) * img_bknd_v
                sub_img = img-img_bknd
                diskset.set_df(n, sub_img)
            else:
                diskset.set_df(n, img)
    return diskset

def integrate_disks_from_mask(datacube, diskset, mask, isLazy=False):
    nregions = np.max(mask.flatten())+1 
    print('there are {} regions found in the mask'.format(nregions))
    for n in range(0, nregions):
        region_mask = (mask == n)
        qxvals, qyvals = np.where(region_mask == 1)
        qxmin, qxmax, qymin, qymax = np.min(qxvals), np.max(qxvals), np.min(qyvals), np.max(qyvals)
        region_mask_chunk = region_mask[qxmin:qxmax, qymin:qymax]
        if isLazy:
            # datacube is massive object, just load small portion that I need if hyperspy lazyload
            print('loading in revevant chunk from lazy hyperspy import of data')
            data_chunk = datacube.data[:,:,qxmin:qxmax, qymin:qymax].compute()
        else:
            data_chunk = datacube[:,:,qxmin:qxmax, qymin:qymax]
        vdf = np.sum(data_chunk*region_mask_chunk, axis=(2,3))/np.sum(region_mask_chunk) 
        diskset.set_df(n, vdf)
    return diskset

####################################################################################################
# finds the average diffraction pattern corresponding to a given real space area
# used to visualize and compare the average diffraction pattern in given regions of the dataset
####################################################################################################
def get_region_average_dp(datacube, diskset, plotbool=False):
    nx, ny, nqx, nqy = datacube.data.shape
    vdf = overlay_vdf(diskset, plotflag=False) # get the sum of virtual dark fields
    # calls a utility function that asks the user to click twice to define a rectangle
    vertices = manual_define_rectangle(diskset, vdf)
    vertices = np.array(vertices)
    xmin = int(np.min(vertices[:,0]))
    ymin = int(np.min(vertices[:,1]))
    xmax = int(np.max(vertices[:,0]))
    ymax = int(np.max(vertices[:,1]))
    # extract the average diffraction pattern in this rectangular region
    avg_dp = np.sum(datacube.data[ymin:ymax,xmin:xmax,:,:], axis=(0,1))
    if plotbool:
        # generate plot of the sum of virtual dark fields, overlay the user defined region
        f, (ax,ax2,ax3) = plt.subplots(1,3)
        ax.imshow(avg_dp)
        diskset.plot(avg_dp, selected_only=True, ax=ax2, f=f)
        ax3.imshow(vdf, cmap='gray')
        ax3.plot([vertices[-1][0], vertices[-2][0]], [vertices[-1][1], vertices[-1][1]], color='k')
        ax3.plot([vertices[-1][0], vertices[-1][0]], [vertices[-1][1], vertices[-2][1]], color='k')
        ax3.plot([vertices[-1][0], vertices[-2][0]], [vertices[-2][1], vertices[-2][1]], color='k')
        ax3.plot([vertices[-2][0], vertices[-2][0]], [vertices[-1][1], vertices[-2][1]], color='k')
        plt.show()
    return avg_dp

def bin(data, bin_w, size_retain=False, method=np.nanmean):
    s = len(data)
    s_bin = s//bin_w
    data_binned = np.zeros((s_bin, s_bin))
    for i in range(s_bin):
        for j in range(s_bin):
            d = data[ i*bin_w:(i+1)*bin_w , j*bin_w:(j+1)*bin_w ]
            if len(d) > 0:
                data_binned[i, j] = method(d)
            else:
                data_binned[i, j] = np.nan
    if not size_retain: return data_binned
    else: return unbin(data_binned, bin_w)

def rotate2d(v, ang):
    r = np.zeros((2,2))
    r[0,0] = np.cos(ang)
    r[0,1] = -np.sin(ang)
    r[1,0] = np.sin(ang)
    r[1,1] = np.cos(ang)
    return np.dot(r, v)

def normalize(d):
    d = d - min(d.flatten())
    d = d/max(d.flatten())
    return d

def get_peak_radius(dp, peaks, dsnum=None, prefix=None, contour_boundary=0.35, disk_to_use=3, w=25, plotflag=False):
    center = (peaks.data['qx'][disk_to_use], peaks.data['qy'][disk_to_use])
    dp_slice = np.array(dp[int(center[0]-w):int(center[0]+w),int(center[1]-w):int(center[1]+w)])
    dp_slice = normalize(dp_slice)
    contours = measure.find_contours(dp_slice, contour_boundary)
    max_r = 0
    mr_contour = None
    center = (0,0)
    if plotflag:
        f, ax = plt.subplots()
        im = ax.matshow(dp_slice)
    for contour in contours:
        xcent = np.mean(contour[:, 1])
        ycent = np.mean(contour[:, 0])
        rads = [ np.sqrt((x-xcent)**2 + (y-ycent)**2) for x,y in zip(contour[:, 1], contour[:, 0]) ]
        rad = np.mean(rads)
        if (rad > max_r):
            max_r = rad
            center = (xcent, ycent)
            mr_contour = contour
    if plotflag:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im, cax=cax, orientation='vertical')
        circle = plt.Circle(center, max_r, color='r',alpha=0.5)
        ax.add_patch(circle)
        ax.plot(mr_contour[:, 1], mr_contour[:, 0])
        ax.set_title('Automatic Disk Radii Detection')
        plt.axis('off')
        if prefix is not None: 
            fullpath = os.path.join(plots,prefix,'ds_{}'.format(dsnum),'disk_radius.png')
            plt.savefig(fullpath, dpi=300)
            plt.close()
        else: 
            plt.show()
    return max_r

############################################################################################################
# container class for a bragg disk set:
#      holds the bragg disk positions, their dark fields, if they are tagged as important from the user, etc
#      also has various utility functions for determining the reciprocal space lattice vectors and their
#      orrientation
# using a structure of array format (SoA)
############################################################################################################
class DiskSet:

    def __init__(self, size, nx, ny):
        self._conven_rotation = None
        self._idealized_gvecs = None
        self.size_in_use = 0                # number of bragg disks the user is interested in using
        self._xvec = np.zeros(size)         # the x positions of the bragg disk centers
        self._yvec = np.zeros(size)         # the y positions of the bragg disk centers
        self._rvec = np.zeros(size)         # the radii bragg disks
        self._dvec = np.zeros((size,2))     # the distance from the bragg disk centers to the central beam
        self.size = size                    # total number of disks
        self.nx = nx                        # number of x-axis pixels in the 4dstem dataset
        self.ny = ny                        # number of y-axis pixels in the 4dstem dataset
        self._in_use = np.zeros(size)       # boolean array containing ones if a given disk is deemed interesting by the user
        self._df = np.zeros((size, nx, ny)) # holds the dark fields for each disk
        self.centralbeam = None             # holds the coordinates of the central beam
        self.emptyslots = 0                 # internal variable for use in resizing the data containers
        self.n_manualdef = 0                # internal variable counting the number of additional manually defined disks
        self.qscale = [1,1]   # when diffraction pattern is distorted as seen in some simulated data 
        # ie the aspect ratio of (qx,qy) space is not one, then set qscale such that nqx*qscale[0] = nqy*qscale[1]
        # diffraction patterns will be distorted into a square shape before processing

    def adjust_qspace_aspect_ratio(self, dp):
        if hasattr(self, 'qscale') and (self.qscale[0] != 1 or self.qscale[1] != 1):
            # was already scaled!
            return 
        desired_q   = np.min([dp.shape[0], dp.shape[1]])
        qx_scale    = desired_q/dp.shape[1]
        qy_scale    = desired_q/dp.shape[0]  
        self.qscale = [qx_scale, qy_scale]
        for n in range(self.size):
            self._xvec[n] = qx_scale * self._xvec[n]
            self._yvec[n] = qy_scale * self._yvec[n]

    def reset_useflags(self):
        self._in_use *= 0 
        self.size_in_use = 0

    # sets a given bragg disk as active or inactive
    # usage: to 'activate' disk 3, diskset.set_useflag(3, True)
    def set_useflag(self, n, useflag):
        olduse = self._in_use[n]
        self._in_use[n] = useflag
        if useflag and (not olduse):
            self.size_in_use += 1
        if (not useflag) and (olduse):
            self.size_in_use -= 1

    # internal function to resizes the container class corresponding to a
    # change in the total number of bragg disks
    # usage: to change the size of the class to 50, diskset.resize(50)
    # the new size should be greater than or equal to the old size
    def resize(self, new_size):
        new_xvec   = np.zeros(new_size)
        new_yvec   = np.zeros(new_size)
        new_rvec   = np.zeros(new_size)
        new_dvec   = np.zeros((new_size,2))
        new_in_use = np.zeros(new_size)
        new_df     = np.zeros((new_size, self.nx, self.ny))
        new_xvec[:self.size]   = self._xvec
        new_yvec[:self.size]   = self._yvec
        new_rvec[:self.size]   = self._rvec
        new_dvec[:self.size]   = self._dvec
        new_in_use[:self.size] = self._in_use
        new_df[:self.size]     = self._df
        self._xvec = new_xvec
        self._yvec = new_yvec
        self._rvec = new_rvec
        self._dvec = new_dvec
        self._in_use = new_in_use
        self._df = new_df
        self.emptyslots = new_size - self.size
        self.size = new_size

    # append a new disk to the container
    def add_disk(self, x, y, use_bool, r):
        if self.emptyslots == 0:
            self.resize(self.size + 1) # resize if needed
        indx = self.size - self.emptyslots
        self._xvec[indx] = x
        self._yvec[indx] = y
        self._rvec[indx] = r
        self._in_use[indx] = use_bool
        self.emptyslots -= 1
        self.n_manualdef += 1

    def get_closest_disk(self, x, y):
        dists = [(x - self._xvec[i])**2 + (y - self._yvec[i])**2 for i in range(len(self._xvec))]
        return np.argmin(dists)

    # plots the identified disks over a provided diffraction pattern (dp)
    # only plots selected disks if selected_only is true
    def plot(self, dp, use_log=False, selected_only=False, ax=None, f=None, saveflag=False, prefix=None, dsnum=None, origin=None):
        if ax is None: f, ax = plt.subplots()
        if origin is None:
            if use_log:
                im = ax.matshow(dp, norm=LogNorm())
            else: im = ax.matshow(dp)
        else:
            if use_log:
                im = ax.imshow(dp, origin=origin, norm=LogNorm())
            else: im = ax.imshow(dp, origin=origin)
        for i in range(self.size):
            # only plot selected disks if selected_only is true
            if selected_only and not self.in_use(i): continue
            qy = self.x(i)
            qx = self.y(i)
            r = self.r(i)
            # plots manually defined disks in blue and others in red
            if (self.size-1)-i > self.n_manualdef: circle = plt.Circle((qx, qy), r, color='r', alpha=0.25)
            else: circle = plt.Circle((qx, qy), r, color='b', alpha=0.25)
            ax.add_patch(circle)
            ax.text(qx, qy, "[{}]".format(i), color="k", size=5, verticalalignment='center', horizontalalignment='center')
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im, cax=cax, orientation='vertical')
        if saveflag: # save the plot
            path = os.path.join('..', 'plots', prefix, 'ds_{}'.format(dsnum))
            os.makedirs(path, exist_ok=True)
            files = os.listdir(path)
            for file in files:
                max_i = 0
                if 'disks_detected' in file:
                    try:
                        tmp = file.split('_')[-1]
                        tmp = int(tmp.split('.')[0])
                    except: tmp = 0
                    max_i = np.max([max_i, tmp])
            plt.savefig(os.path.join(path,"disks_detected_{}.png".format(i)), dpi=300)
            plt.close(f)

    def set_x(self, n, xval):   self._xvec[n] = xval # set the disk center x coordinate for a given disk (index n)
    def set_y(self, n, yval):   self._yvec[n] = yval # set the disk center y coordinate for a given disk (index n)
    def set_r(self, n, rval):   self._rvec[n] = rval # set the disk radius for a given disk (index n)
    def set_df(self, n, df):    self._df[n, :df.shape[0], :df.shape[1]] = df # set the diffraction pattern for a given disk (index n)
    def set_central(self, bm):  self.centralbeam = bm # set the central beam coordinate (should be a 2 element interable)

    def x(self, n):           return self._xvec[n] # return disk center x coordinate of a given disk (index n)
    def y(self, n):           return self._yvec[n] # return disk center y coordinate of a given disk (index n)
    def r(self, n):           return self._rvec[n] # return disk radius of a given disk (index n)
    def df(self, n):          return self._df[n, :, :] # return disk virtual dark field of a given disk (index n)
    def xvec(self):           return self._xvec # return vector of disk center x coordinates
    def yvec(self):           return self._yvec # return vector of disk center y coordinates
    def rvec(self):           return self._rvec # return vector of disk radii
    def dvec(self):           return self._dvec # return vector of distances from disk centers to central beam
    def in_use(self, n):      return self._in_use[n] # return is a given disk (index n) is tagged as important

    # return the distance from teh central beam to this disk (index n) in diffraction space
    def d(self, n):
        if self.centralbeam is None:
            print("DiskSet error, must call set_com_central before computing distances using diskset.d(i)")
            exit(1)
        return [self._xvec[n]-self.centralbeam[0], self._yvec[n]-self.centralbeam[1]]

    # calculate the disk center as the center of mass of all the identified peaks in the diffraction pattern
    # not a good guess at all when have less than 12 disks due to beam stop, but good enough as an initial 
    # guess for the least squares gvector fit to what's expected of perfect triangular lattice (without stig or heterostrain)
    def set_com_central(self):
        self.centralbeam = [0,0]
        ndisk = 0
        for n in range(self.size):
            if self.in_use(n):
                self.centralbeam[0] += self._xvec[n]
                self.centralbeam[1] += self._yvec[n]
                ndisk += 1
        self.centralbeam[0] /= self.size_in_use
        self.centralbeam[1] /= self.size_in_use
        return self.centralbeam

    # returns set of all the calculated virtual dfs for this diskset
    def df_set(self):
        dfset = np.zeros((self.size_in_use, self.nx, self.ny))
        i = 0
        for n in range(self.size):
            if self.in_use(n):
                dfset[i, :, :] = self.df(n)
                i += 1
        return dfset

    # returns set of vectors from central beam for disks that have been integrated (tagged as important) only
    # note for a moire, these will be centers of overlap regions so gvectors will be halfway between a given set of bilayers
    def d_set(self):
        self.set_com_central()
        #assert( self.size_in_use == len([n for n in range(self.size) if self.in_use(n)]) )
        self.size_in_use = len([n for n in range(self.size) if self.in_use(n)]) 
        dset = np.zeros((self.size_in_use,2))
        i = 0
        for n in range(self.size):
            if self.in_use(n):
                dset[i,:] = self.d(n)
                i += 1
        distances = [(el[0]**2 + el[1]**2)**0.5 for el in dset]
        dset = dset/min(distances)
        dset[:,0] = -dset[:,0]
        return dset

    # finds friedel pairs
    def get_pairs(self):
        g = self.clean_normgset()
        pairs = []
        unpaired_disks = [i for i in range(g.shape[0])]
        while len(unpaired_disks) > 0:
            disk1 = unpaired_disks.pop()
            disk2 = -1
            for i in unpaired_disks:
                if g[i][0] == -g[disk1][0] and g[i][1] == -g[disk1][1]:
                    disk2 = i
                    unpaired_disks.remove(disk2)
                    break
            pairs.append([disk1, disk2])
        return pairs      

    def get_rotatation(self, plotpath=None):
        if (not hasattr(self, '_conven_rotation')) or self._conven_rotation == None or boolquery("redo sample rotation? (fit dp)"):
            if plotpath != None: f, ax = plt.subplots()
            idealized_gvecs, rotation = fit_disklocations_to_hexagonal(ax, self)
            self._conven_rotation = rotation
            self._idealized_gvecs = idealized_gvecs
            if plotpath != None: plt.savefig(plotpath, dpi=300)
        return self._conven_rotation

    def clean_normgset(self, plotpath=None):
        if (not hasattr(self, '_conven_rotation')) or self._conven_rotation == None:
            if plotpath != None: f, ax = plt.subplots()
            idealized_gvecs, rotation = fit_disklocations_to_hexagonal(ax, self)
            self._conven_rotation = rotation
            self._idealized_gvecs = idealized_gvecs
            if plotpath != None: plt.savefig(plotpath, dpi=300)
        return self._idealized_gvecs

    def x_set(self): # returns set of distances from central beam for disks that have been integrated only
        xset = np.zeros(self.size_in_use)
        i = 0
        for n in range(self.size):
            if self.in_use(n):
                xset[i] = self.x(n)
                i += 1
        return xset

    def y_set(self): # returns set of distances from central beam for disks that have been integrated only
        yset = np.zeros(self.size_in_use)
        i = 0
        for n in range(self.size):
            if self.in_use(n):
                yset[i] = self.y(n)
                i += 1
        return yset

    # for disks that have been integrated, determine their effective "rings"
    # ie, first ring is [100], [010], [1-10], [-110], [-100], [0-10]
    def determine_rings(self, tol=0.5):
        ring_no = np.zeros(self.size_in_use)
        dset = self.d_set()
        dset = [ np.sqrt(dx**2 + dy**2) for dx,dy in zip(dset[:,0], dset[:,1]) ]
        min_d = np.min(dset)
        max_d = np.max(dset)
        ring2_d = np.sqrt(3) * min_d
        for n in range(self.size_in_use):
            d = dset[n]
            if np.abs(d - min_d) < tol:
                ring_no[n] = 1
            elif np.abs(d - ring2_d) < tol:
                ring_no[n] = 2
            elif np.abs(d - 2*min_d) < tol:
                ring_no[n] = 3
            else:
                ring_no[n] = -1
            #print("disk {} at d={} is tagged as being in ring {}".format(n, d, ring_no[n]))
        return ring_no

def get_diskset(dp, peaks, scan_shape, centralbeam, dsnum=None, prefix=None, radius_factor=0.65, plotflag=True):
    try: 
        disk_radius = get_peak_radius(dp, peaks, dsnum, prefix)
    except: 
        print('failed to get a good disk radius, using default radius for disks of 5')
        disk_radius = 5
    diskset = DiskSet(len(peaks.data['qy']), scan_shape[0], scan_shape[1])
    nx, ny = dp.shape
    r = disk_radius*radius_factor
    for i in range(diskset.size):
      qy = peaks.data['qy'][i]
      qx = peaks.data['qx'][i]
      diskset.set_x(i,qx)
      diskset.set_y(i,qy)
      diskset.set_r(i,r)
    if prefix is not None: diskset.plot(dp, saveflag=True, prefix=prefix, dsnum=dsnum)
    else: diskset.plot(dp, saveflag=False)
    use_log = boolquery('use a log plot?')
    manual_define_disks(diskset, r, dp, use_log)
    if prefix is not None: diskset.plot(dp, saveflag=True, prefix=prefix, dsnum=dsnum)
    else: diskset.plot(dp, saveflag=False)
    return diskset

def get_masked_diskset(scan_shape, mask):
    nregions = np.max(mask.flatten())+1
    nx, ny = scan_shape[0], scan_shape[1]
    masked_ds = DiskSet(nregions, nx, ny)
    for i in range(nregions):
        region_mask = (mask == i)
        qxvals, qyvals = np.where(region_mask == 1)
        meanqx, meanqy = np.mean(qxvals), np.mean(qyvals)
        # qx and qy rough locations of each masked disk used to figure out corresponding gvecs
        # need not be exact since algorithm knows expected gvecs - just to ascribe each vdf to one
        masked_ds.set_x(i,meanqx)
        masked_ds.set_y(i,meanqy)
        masked_ds.set_useflag(i,True) # all disk objects active for the masked approach
    return masked_ds

def select_disks(diskset, dp, disks_to_use=None):
    use_log = boolquery('use a log plot?')
    if disks_to_use == None:
        ndisks_used, diskset = manual_select_disks(diskset, dp, use_log)
    else:
        ndisks_used = len(disks_to_use)
        for n in disks_to_use:
            diskset.set_useflag(n,True)
    if ndisks_used == 0:
        print('ERROR: you didnt select any disks!')
        exit()
    return ndisks_used, diskset
