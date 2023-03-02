
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import *
from time import sleep
import os
import pickle

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

    # sets a given bragg disk as active or inactive
    # usage: to 'activate' disk 3, diskset.set_useflag(3, True)
    def set_useflag(self, n, useflag):
        self._in_use[n] = useflag
        self.size_in_use += 1

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
    def set_com_central(self):
        self.centralbeam = [0,0]
        for n in range(self.size):
            if self.in_use(n):
                self.centralbeam[0] += self._xvec[n]
                self.centralbeam[1] += self._yvec[n]
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

    # returns set of distances from central beam for disks that have been integrated (tagged as important) only
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

    def norm_dset(self):
        dset = self.d_set()
        for i in range(len(dset)):
            dset[i,:] /= (dset[i,0]**2 + dset[i,1]**2)**0.5
        return dset

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

    def rotated_normgset(self, sanity_plot=False):
        g = self.norm_dset()
        ringnos = self.determine_rings()
        indx_1100 = -1
        val_1100 = -np.inf
        for i in range(len(g)):
            if ringnos[i] == 1:
                if g[i,1] > val_1100:
                    val_1100 = g[i,1]
                    indx_1100 = i
        # get angle from g[i,:] to [0,1]
        theta = np.arccos(np.dot([0,1],g[indx_1100,:]))
        if sanity_plot:
            f, ax = plt.subplots()
            for i in range(len(g)): 
                if ringnos[i] == 1:
                    ax.scatter(g[i][0], g[i][1], c='k')
                if ringnos[i] == 2:
                    ax.scatter(2*g[i][0], 2*g[i][1], c='k')    
        for i in range(len(g)):
            g[i,:] = rotate2d(g[i,:], -theta)
        if sanity_plot:
            for i in range(len(g)): 
                if ringnos[i] == 1:
                    ax.scatter(g[i][0], g[i][1], c='r')
                if ringnos[i] == 2:
                    ax.scatter(2*g[i][0], 2*g[i][1], c='r')    
            plt.show()    
        return g

    def get_rotatation(self, sanity_plot=True, savepath=None, printing=False):
        g1 = np.array([ 0, 2/np.sqrt(3)])
        g2 = np.array([-1, 1/np.sqrt(3)])
        graw = self.norm_dset()   
        g = self.clean_normgset()   
        angs = []
        ringnos = self.determine_rings()
        for i in range(len(g)):
            if g[i][0] == 1 and g[i][1] == 0:
                indx = i 
                break
        ptC = g[indx][0] * g1 + g[indx][1] * g2
        ptA = graw[indx][:]
        # want rotation from A to C, so if x coordinate of A larger, rotate back
        if ptA[0] > ptC[0]: 
            rotation_sign = -1
            if printing: print('raw had larger x')
        else: 
            rotation_sign = 1
            if printing: print('raw had smaller x')
        if printing: print('index of top was ', indx)
        if printing: print('raw x coord here was ', ptA[0])
        if printing: print('target x coord here was ', ptC[0])
        if printing: print('so rotate by a angle with sign ', rotation_sign)
        # quick helper - get angle from v1 through v2 to v3 in degrees
        def get_angle(pts):
            v1, v2, v3 = pts[0], pts[1], pts[2] 
            l1 = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
            l2 = ((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)**0.5
            l3 = ((v3[0] - v2[0])**2 + (v3[1] - v2[1])**2)**0.5
            return np.arccos((l1**2 + l3**2 - l2**2)/(2*l1*l3)) 

        use_single_ring = True #( len(g) != 12 )

        if use_single_ring:

            print('WARNING: original sample rotation might be incorrect')
            n_inner = len([i for i in range(len(g)) if ringnos[i] == 1])
            n_outter = len([i for i in range(len(g)) if ringnos[i] == 2])
            if n_inner == 6:
                use_out = False
                print('using only ring1 for sample rotation')
            elif n_outter == 6: 
                use_out = True
                print('using only ring2 for sample rotation')
            else:
                print('AAAA')
                exit()
        
        for i in range(len(g)):
            if use_single_ring:
                if use_out and ringnos[i] == 1: 
                    continue
                elif (not use_out) and ringnos[i] == 2: 
                    continue
            ptA = graw[i][:]
            ptB = [0,0]
            ptC = g[i][0] * g1 + g[i][1] * g2
            angs.append(get_angle([ptA, ptB, ptC]))
        mean_ang = rotation_sign * np.mean(angs) 
        stderr_ang = np.std(angs, ddof=1)/np.sqrt(np.size(angs))
        if printing: print("{} +/- {} rad".format(mean_ang, stderr_ang))
        if sanity_plot:
            f, ax = plt.subplots()
            for i in range(len(g)):
                if use_single_ring:
                    if use_out and ringnos[i] == 1: continue
                    elif (not use_out) and ringnos[i] == 2: continue
                gvec = g[i][0] * g1 + g[i][1] * g2
                scale = 1/((graw[i][0] ** 2 + graw[i][1] ** 2) ** 0.5)
                ax.scatter(scale*graw[i][0], scale*graw[i][1], c='b')
                #ax.text(scale*gvec[0], scale*gvec[1], '{}{}'.format(g[i][0],g[i][1]))
                ax.text(scale*graw[i][0], scale*graw[i][1], i)
                grot = rotate2d(scale*graw[i,:], -mean_ang)
                if ringnos[i] == 1:
                    ax.scatter(0.75*grot[0], 0.75*grot[1], c='c')
                else:
                    ax.scatter(grot[0], grot[1], c='c')
                #ax.text(grot[0], grot[1], i)
                scale = 1/((gvec[0] ** 2 + gvec[1] ** 2) ** 0.5) * 0.75 #0.75 factor so offset enough to see
                ax.scatter(scale*gvec[0], scale*gvec[1], c='r')
                ax.text(scale*gvec[0], scale*gvec[1], i)
                ax.text(scale*0.75*gvec[0], scale*0.75*gvec[1], '{}{}'.format(g[i][0],g[i][1]))
            ax.set_title("{} +/- {} rad".format(mean_ang, stderr_ang))       
            if savepath is not None:
                plt.savefig(savepath, dpi=300)
                plt.close('all')
            else:
                plt.show()
        return mean_ang * 180/np.pi, stderr_ang * 180/np.pi

    ####################################################################################################
    # I know this is gross it works for now will clean up later
    ####################################################################################################
    def clean_normgset(self, sanity_plot=False, prefix=None, dsnum=None):
        graw = self.norm_dset()
        g = self.rotated_normgset()
        ringnos = self.determine_rings()
        if sanity_plot:
            #savepath = os.path.join('..','plots', prefix, 'ds_{}'.format(dsnum), 'gvectors.png')
            f, ax = plt.subplots()
            ax.scatter(0, 0, c='g')
            for i in range(len(g)):
                if ringnos[i] == 1:
                    #ax.scatter(g[i][0], g[i][1], c='k')
                    #ax.text(g[i][0], g[i][1], i)
                    length = (graw[i][0] ** 2 + graw[i][1] ** 2) ** 0.5
                    scale = 1/length
                    ax.scatter(scale*graw[i][0], scale*graw[i][1], c='b')
                    ax.text(scale*graw[i][0], scale*graw[i][1], i)
                if ringnos[i] == 2:
                    #ax.scatter(2*g[i][0], 2*g[i][1], c='k')
                    #ax.text(2*g[i][0], 2*g[i][1], i)
                    length = (graw[i][0] ** 2 + graw[i][1] ** 2) ** 0.5
                    scale = 1/length
                    ax.scatter(scale*graw[i][0], scale*graw[i][1], c='b')
                    ax.text(scale*graw[i][0], scale*graw[i][1], i)

        ind0n1 = np.argmin([g[i,1] if ringnos[i] == 1 else np.inf for i in range(len(g))]) # find ind of disk with smallest y coord in ring 1
        g[ind0n1,:] = [-1,0] # -g1
        ind01 = np.argmax([g[i,1] if ringnos[i] == 1 else -np.inf for i in range(len(g))]) # find ind of disk with largest y coord in ring 1
        g[ind01,:] = [1,0]  # g1
        ind1n2 = np.argmax([g[i,0] if ringnos[i] == 2 else -np.inf for i in range(len(g))]) # find ind of disk with largest x coord in ring 2
        g[ind1n2,:] = [1,-2] #-2*g2+g1
        indn12 = np.argmin([g[i,0] if ringnos[i] == 2 else np.inf for i in range(len(g))]) # find ind of disk with smallest x coord in ring 2
        g[indn12,:] = [-1,2]#2*g2-g1
        for i in range(len(g)):
            if ringnos[i] == 1 and (i != ind0n1 and i != ind01):
                if   g[i,1] > 0 and g[i,0] > 0: g[i,:] = [1,-1]  #g1-g2
                elif g[i,1] > 0 and g[i,0] < 0: g[i,:] = [0,1]   #g2
                elif g[i,1] < 0 and g[i,0] > 0: g[i,:] = [0,-1]  #-g2
                elif g[i,1] < 0 and g[i,0] < 0: g[i,:] = [-1,1]  #g2-g1
        for i in range(len(g)):
            if ringnos[i] == 2 and (i != ind1n2 and i != indn12):
                if   g[i,1] > 0 and g[i,0] > 0: g[i,:] = [2,-1]  #2*g1-g2
                elif g[i,1] > 0 and g[i,0] < 0: g[i,:] = [1,1]   #g1+g2
                elif g[i,1] < 0 and g[i,0] > 0: g[i,:] = [-1,-1] #-g1-g2
                elif g[i,1] < 0 and g[i,0] < 0: g[i,:] = [-2,1]  #-2*g1+g2
        if sanity_plot:
            g1 = np.array([ 0, 2/np.sqrt(3)])
            g2 = np.array([-1, 1/np.sqrt(3)])
            for i in range(len(g)):
                gvec = g[i][0] * g1 + g[i][1] * g2
                length = (gvec[0] ** 2 + gvec[1] ** 2) ** 0.5
                scale = 1/length
                ax.scatter(scale*gvec[0], scale*gvec[1], c='r')
                ax.text(scale*gvec[0], scale*gvec[1], i)# "{}-{}/{}".format(i,g[i][0], g[i][1]))
            #plt.savefig(savepath, dpi=300)
            plt.show()    
        return g

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

