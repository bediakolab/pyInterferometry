##############################################################################
## calculate the heterostrain and twist angle for a provived STM dataset.   ##
## this is accomplished by fitting the tunneling intensities to bivariate   ##
## gaussians, then preforming Deluanay triangulation on the resultant       ##
## gasssian centers.                                                        ##
## WARNING: this is highly dependent on the hyper-parameters chosen, which  ##
## are printed to an output file upon execution for book keeping. Some      ##
## example datasets and their corresponding parameters are provided in the  ##
## folder 'examples' for reference.                                         ##
##                                                                          ##
## usage : python3 STMtriangulate.py folder/dataset.txt                     ##
## hyperparameters are set in the dictionary at the bottom of the file.     ##
## must be called in a directory containing a folder holding the STM data   ##
## STM data must be formatted in accordance with the provided example data. ##
##############################################################################

import numpy as np
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from skimage import measure
from scipy.spatial import Delaunay
import re
import os
import sys
import seaborn as sns
from time import sleep
from matplotlib.colors import Normalize
from matplotlib.path import Path

def get_angles_from_vertical(v1, v2, v3):

    v12 = np.array([v2[0] - v1[0], v2[1] - v1[1]])
    v13 = np.array([v3[0] - v1[0], v3[1] - v1[1]])
    v23 = np.array([v3[0] - v2[0], v3[1] - v2[1]])
    l12 = np.linalg.norm(v12)
    l13 = np.linalg.norm(v13)
    l23 = np.linalg.norm(v23)
    a12 = np.arccos( np.dot(np.array([1,0]), v12) / l12 )
    a13 = np.arccos( np.dot(np.array([1,0]), v13) / l13 )
    a23 = np.arccos( np.dot(np.array([1,0]), v23) / l23 )
    angs = np.array([a12, a13, a23]) * 180/np.pi
    for i in range(len(angs)):
        if angs[i] > 90:
            angs[i] = 180 - angs[i]

    angs = np.sort(angs)
    # if equilateral expect phi, 60-phi, 60+phi
    return [angs[0], 60-angs[1], angs[2]-60]

def main_vdf(filepath, vdf, FOV_len):

    vdf = 1 - vdf

    params = {
        # change me to cut data up in the more/less chunks
        "num_slices"        : 1,               # slice data into chunks, will be faster but omits some data

        # change me if there are too few or too many peaks
        "contour_boundary"  : 0.8,            # initial boundaries will be at this percent intentsity
        'guess_radius_criterion' : 1.0,       # decrease me if there are a lot of erroneous mall peaks
        'combine_criterion' : 1.0,             # increase me if plot_avg breaks peaks into two nearby circles
        'times_to_combine' : 2,                # times to run through the nearby point combination proecedure

        # change me if there are too few or too many triangles that are accepted and used to get theta
        'ml_criterion' : 1.5,                  # remove points that have mean delaunay lengths to nearby points greater than a given criterion
                                               # due to common issue with delaunay algorithm - will include erroneous connections
        "angle_criterion"   : 0.35, #0.35,      # will ignore regions where moire triangle angles are greater than angle_criterion rad from pi/3

        # change me for special purposes
        "xtol"              : 1e-1,            # tolerance for gaussian fit,
                                               # increase for better guassian fits (slower, bad for noisy data, good for clean data)
        "removed_before_keys" : [],            # manually removed points (will ask you to enter these while running program so can leave blank)
                                               # or can enter them here if you've already run this program and know what to remove
        'truncation' : False,                  # to trucate field of view

        # these have very little effect on the results
        "guess_theta_t"     : 0.2,             # degrees of guess angle for twist (results don't depend on this I've found)
        "guess_theta_s"     : 25,              # guess angle of heterostrain application (results don't depend on this I've found)
        "guess_hs"          : 0.05,            # guessed percent heterostrain (results don't depend on this I've found)
    }

    # get inputs
    params['filename'] = filepath
    filedir = os.path.split(params['filename'])
    filedir = filedir[0]

    # parse
    params['FOV_length'] = FOV_len
    params['a_graphene'] = 0.246/params['FOV_length'] # 0.246 nm, normalized in units of FOV
    
    (nx, ny) = vdf.shape
    thetas, het_strains, area_percents = mesh_process(vdf, filedir, 0, 0, params)

    # plot histograms
    f, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.hist(thetas, bins='auto')
    ax1.set_title('local twist angle')
    ax2.hist(het_strains, bins='auto')
    ax2.set_title('local heterostrain')
    ax3.hist(area_percents, bins='auto')
    ax3.set_title('local AA(?) area percent')
    plt.savefig("{}/histo.png".format(filedir))

    # write data
    write("{}/heterostrains.txt".format(filedir), "percent heterostrain obtained", het_strains)
    write("{}/angles.txt".format(filedir), "twist angles (degrees)", thetas)
    write("{}/areapercent.txt".format(filedir), "area percents of AA per moire triangle", thetas)
    write("{}/output.txt".format(filedir), "parameters used:", params)

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

def manual_define_point(img, spots):
    plt.close('all')
    fig, ax = plt.subplots()
    points = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        points.append([x, y])
        ax.scatter(x,y,color='b')
        fig.canvas.draw()
        #if len(points) == 1:
        #    fig.canvas.mpl_disconnect(cid)
        #    sleep(1)
        #    plt.close('all')
    ax.imshow(img, origin='lower')
    ax.set_title('press to define new center for this point, close plot when done')
    for i in range(len(spots)):
        circle = plt.Circle((spots[i][0],spots[i][1]), color='r', radius=spots[i][2], linewidth=2.5, fill=False)
        ax.text(spots[i][0],spots[i][1], i)
        ax.add_patch(circle)
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    return points

# 2d gaussian
def gaussian(x, y, x0, y0, alpha, A):
    return A * np.exp( -((x-x0)/alpha)**2 -((y-y0)/alpha)**2)

# unwrapped lc of gaussians for lsq fit
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//4):
       arr += gaussian(x, y, *args[i*4:i*4+4])
    return arr

# given three triangle lengths return angles
def get_angles(l1, l2, l3):
    a12 = np.arccos((l1**2 + l2**2 - l3**2)/(2*l1*l2))
    a23 = np.arccos((l2**2 + l3**2 - l1**2)/(2*l3*l2))
    a31 = np.arccos((l1**2 + l3**2 - l2**2)/(2*l1*l3))
    return a12, a23, a31

# given three vertices give lengths of triangle
def get_lengths(v1, v2, v3):
    l1 = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
    l2 = ((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)**0.5
    l3 = ((v3[0] - v2[0])**2 + (v3[1] - v2[1])**2)**0.5
    return l1, l2, l3

# distance between two vertices
def get_dist(v1, v2):
    l = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
    return l

# determine heterostrain and twist angle given lengths of moire triangle
def fit_heterostrain(l1, l2, l3, params):
    delta = 0.16 # graphene Poisson ratio

    # returns residuals
    def cost_func(L, theta_t, theta_s, eps):
        k = 4*np.pi/(np.sqrt(3)*params['a_graphene'])
        K = np.array( [[k, 0], [k*0.5, k*0.86602540378], [-k*0.5, k*0.86602540378]] )
        R_t = np.array([[np.cos(theta_t), -np.sin(theta_t)], [np.sin(theta_t), np.cos(theta_t)]])
        R_s = np.array([[np.cos(theta_s), -np.sin(theta_s)], [np.sin(theta_s), np.cos(theta_s)]])
        R_ns = np.array([[np.cos(-theta_s), -np.sin(-theta_s)], [np.sin(-theta_s), np.cos(-theta_s)]])
        E = np.array([[1/(1+eps), 0],[0, 1/(1-delta*eps)]])
        M = R_t - np.matmul(R_ns, np.matmul(E, R_s))
        Y = [0,0,0]
        for i in range(3):
            v = np.matmul(M, K[i,:])
            l = (np.dot(v,v))**0.5
            Y[i] = 4*np.pi/(np.sqrt(3)*l)
        return [y - l for y,l in zip(Y,L)]

    # wrapped cost function
    def _cost_func(vars):
        L = np.array([l1,l2,l3])
        return cost_func(L, vars[0], vars[1], vars[2])

    guess_prms = [params['guess_theta_t'] * np.pi/180, params['guess_theta_s'] * np.pi/180, params['guess_hs']/100]
    opt = least_squares(_cost_func, guess_prms)
    return opt.x # theta_t, theta_s, eps

# normalizes dataset, filtering out low intensity outliers if requested
def normalize(dat, truncation):
    # normalize
    (nx, ny) = dat.shape
    if ( truncation ):
        d_avg = np.mean(dat[0:nx-50,0:ny])
        d_std = np.std(dat[0:nx-50,0:ny])
        dat[nx-50:nx,0:ny] = d_avg - 0.5*d_std
    d_max = np.max(dat)
    d_min = np.min(dat)
    d_range = d_max - d_min
    dat = (dat - d_min)/d_range
    return dat

# filter points based on a z-score criterion of valuevec (sigmas, intensities, etc)
def filter(valuevec, points, criterion, filter_name):
    m = np.nanmean(valuevec)
    sd = np.nanstd(valuevec)
    for i in range(len(valuevec)):
        if np.abs(m - valuevec[i])/sd > criterion:
            print("removing {},{} from {} filter".format(points[i, 0], points[i, 1], filter_name))
            points[i, :] = np.nan
            valuevec[i] = np.nan

# points within a distance criterion are combined
def combine_nearby_spots(spots, combine_criterion):
    n = len(spots)
    distance_table = 100 * np.ones((n,n))
    bool_arr = np.ones((n,1))
    for i in range(n):
        for j in range(i+1, n):
            distance_table[i,j] = get_dist([spots[i][0], spots[i][1]], [spots[j][0], spots[j][1]])
    for i in range(n):
        d = np.min(distance_table[i,:])
        if d < combine_criterion:
            j = np.argmin(distance_table[i,:])
            spot_i = spots[i]
            spot_j = spots[j]
            #print('combining points at {:.2f},{:.2f} and {:.2f},{:.2f} at d={:.2f}'.format(spot_i[0], spot_i[1], spot_j[0], spot_j[1], d))
            spots[i] = [ (spot_i[0]*spot_i[2]+spot_j[0]*spot_j[2])*1/(spot_i[2]+spot_j[2]),
                         (spot_i[1]*spot_i[2]+spot_j[1]*spot_j[2])*1/(spot_i[2]+spot_j[2]),
                          spot_i[2]+spot_j[2], (spot_i[3]+spot_j[3])*0.5 ]
            spots[j] = spots[i]
            bool_arr[i] = 0 # remove point i
    new_spots = []
    for i in range(len(spots)):
        if bool_arr[i]:
            new_spots.append(spots[i])
    spots = new_spots
    return spots

# obtain heterostrain and twist angle through fitting data to a series of guassians,
# triangulating, and extracting these from the resultant mesh
#   plots to the provided mpl axis
#   returns vectors of heterostrain and twist angle
#   will plot data, gassian fit, Delaunay mesh, and fit heterostrain/twist to provided ax
def mesh_process(dat, filedir, chunkno, nchunks, params):

    # get domain and normalize
    (nx, ny) = dat.shape
    x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    dat = normalize(dat, params['truncation'])

    # determine countours
    print('finding average spots')
    contours = measure.find_contours(dat, params['contour_boundary'])
    spots = []
    rads = []

    # plot contours and averages
    for n, contour in enumerate(contours):
        xcent = np.mean(contour[:, 1])
        ycent = np.mean(contour[:, 0])
        avg_y_rad = np.mean([np.abs(v - xcent) for v in contour[:, 1]])
        avg_x_rad = np.mean([np.abs(v - ycent) for v in contour[:, 0]])
        rad = (avg_x_rad + avg_y_rad) * 0.5
        rads.append(rad)
        I = dat[int(ycent),int(xcent)]
        spots.append([xcent, ycent, rad, I])

    # combine spots close to eachother
    print('filtering average spots')
    for i in range(params['times_to_combine']):
        spots = combine_nearby_spots(spots, params['combine_criterion'])

    # remove spots with small radii
    avg_rad = np.mean(rads)
    sd_rad = np.std(rads)
    new_spots = []
    for i in range(len(spots)):
        if (avg_rad - spots[i][2])/sd_rad < params['guess_radius_criterion'] :
            new_spots.append(spots[i])
    spots = new_spots

    bool_arr = np.ones((n,1))
    for i in params["removed_before_keys"]:
        bool_arr[i] = 0
    new_spots = []
    for i in range(len(spots)):
        if bool_arr[i]: new_spots.append(spots[i])
    spots = new_spots

    # plot averaged spots
    f, ax = plt.subplots()
    ax.imshow(dat, origin='lower')
    ax.axis('image')
    AA_area = 0
    for i in range(len(spots)):
        rad = spots[i][2]
        circle = plt.Circle((spots[i][0],spots[i][1]), color='r', radius=spots[i][2], linewidth=2.5, fill=False)
        ax.text(spots[i][0],spots[i][1], i) 
        AA_area += np.pi * rad * rad
        ax.add_patch(circle)
    ax.set_xticks([])
    ax.set_yticks([])
    print('total AA area in the displayed circles is {} (this will depend on threshold value used)'.format(AA_area))
    print('total area in scan is {} '.format(nx*ny))
    print('fraction AA is around {} based on AA circle guess (this will depend on threshold value used)'.format(AA_area/(nx*ny)))
    ax.set_title("showing first guess for AA centers \n save this plot for manual adjustment \n close plot to continue")
    plt.show()

    # to do : change to clicking?
    target_i = input("enter number of point to adjust, or enter -1 once done: \n")
    bool_arr = np.ones((n,1))
    while (target_i.strip().lower() != '-1'):
        try:
            i = int(target_i.strip())
            params["removed_before_keys"].append(i)
            remove = input('remove this point? (type y or n): ').lower() == 'y'
            if remove: bool_arr[i] = 0
            else:
                adjust = input('adjust this point? (type y or n): ').lower() == 'y'
                if adjust:
                    print('click on new center for this point (you picked point {})'.format(target_i))
                    pt = manual_define_point(dat, spots)
                    print(pt)
                    spots[i][0], spots[i][1] = pt[0][0], pt[0][1]
            target_i = input("enter number of point to adjust, or enter -1 once done: \n")
        except:
            print('something went wrong, please try again')
            target_i = input("enter number of point to adjust, or enter -1 once done: \n")
    new_spots = []
    for i in range(len(spots)):
        if bool_arr[i]: new_spots.append(spots[i])
    spots = new_spots

    f, ax = plt.subplots()
    ax.imshow(dat, origin='lower')
    ax.axis('image')
    AA_area = 0
    for i in range(len(spots)):
        rad = spots[i][2]
        circle = plt.Circle((spots[i][0],spots[i][1]), color='r', radius=rad, linewidth=2.5, fill=False)
        AA_area += np.pi * rad * rad
        ax.text(spots[i][0],spots[i][1], i) #horizontalalignment='center'
        ax.add_patch(circle)

    #print('total AA area in the displayed circles is {} (this will depend on threshold value used)'.format(AA_area))
    #print('total area in scan is {} '.format(nx*ny))
    #print('fraction AA is around {} based on filtered AA circles (this will depend on threshold value used)'.format(AA_area/(nx*ny)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("showing AA centers after manual adjustment, close to continue")
    plt.savefig("{}/centers_chunk{}of{}.png".format(filedir, chunkno+1, nchunks), dpi=600)
    plt.show()

    # use averaged data as input to guassian fit, guess_prms holds initial guesses
    guess_prms = []
    for spot in spots:
        t = [spot[0]/nx, spot[1]/ny, spot[2]/nx, spot[3]]
        guess_prms.extend(t)

    # plot as 2D image and the fit as overlaid contours
    guess = np.zeros(dat.shape)
    for i in range(len(spots)):
        g = gaussian(X, Y, *guess_prms[i*4:i*4+4])
        guess += g

    # least squares opt
    print('fitting guassians')
    xdata = np.vstack((X.ravel(), Y.ravel()))
    popt, pcov = curve_fit(_gaussian, xdata, dat.ravel(), guess_prms, xtol=params['xtol'])

    # extract fit function and toss out erroneous points
    points = np.zeros((len(spots), 2))
    sigs   = np.zeros((len(spots), 1))
    intens = np.zeros((len(spots), 1))
    fit    = np.zeros(dat.shape)
    AA_area = 0
    for i in range(len(spots)):
        g = gaussian(X, Y, *popt[i*4:i*4+4])
        if (popt[i*4] > 1.5 or popt[i*4+1] > 1.5):
            print("removing {},{} from FOV filter".format(popt[i*4], popt[i*4+1]))
            points[i, :] = np.nan
            sigs[i] = np.nan
            intens[i] = np.nan
        else:
            points[i, :] = popt[i*4:i*4+2]
            sigs[i] = popt[i*4+2]
            radius = sigs[i] * 0.8326182348
            AA_area += np.pi * radius * radius
            try:
                intens[i] = dat[int(points[i, 1]*ny),int(points[i, 0]*nx)]
            except:
                intens[i] = np.nan
        fit += g
    #print('fraction AA is around {} based on guassians (area fraction in the green circles shown) \n--> (using FWHM as diameter, I suggest increasing xtol from default to rely on this value)'.format(AA_area))
    #print('--> WARNING: this is an overestimate as it right now just calculates the area in all green circles and doesnt account for them bleeding out of the FOV')

    # fit mesh
    print('meshing')
    points = points[~np.isnan(points)]
    points = np.reshape(points, (len(points)//2,2))
    try:
        tri = Delaunay(points)
    except:
        print("Failed to identify enough points for triangularization after filtering")
        print("Please increase num_slices or change filering criteria")
        exit(0)

    # plot fit guassians and Delaunay mesh
    f, (ax, ax2, ax3) = plt.subplots(1,3)
    nx, ny = dat.shape[0], dat.shape[1]
    AA_mask = np.zeros((nx, ny))
    ax.imshow(dat, origin='lower', extent=(0, 1, 0, 1))
    ax.axis('image')
    ax2.imshow(dat, origin='lower', extent=(0, 1, 0, 1))
    ax2.contour(X, Y, fit, colors='w')
    ax2.axis('image')
    for i in range(len(sigs)):
        try:
            radius = sigs[i] * 0.8326182348
            xcenter = points[i, 0]
            ycenter = points[i, 1]
            circle = plt.Circle((xcenter, ycenter), color='w', radius=radius, linewidth=2.5, fill=False)
            mask = np.fromfunction(lambda x,y: ((x-xcenter*nx)**2 + (y-ycenter*ny)**2) < (radius*nx)**2, (nx,ny)).astype(int) 
            AA_mask += mask
            ax.add_patch(circle)
        except:
            continue
    AA_mask = (AA_mask > 0).astype(int)    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(points[:,0], points[:,1], 'ro')
    #ax.triplot(points[:,0], points[:,1], tri.simplices, color='b')
    ax3.imshow(AA_mask.transpose(), origin='lower')

    # obtain heterostrains and angles from mesh
    print('calculating heterostrain and twist')
    het_strains = []
    thetas = []
    lens = []
    triangles = []
    area_percents = []
    tri_centers = []
    orrient = []
    wls = []

    # get threshold for AA area percent calculation

    # get traingles within angle criterion for further filtering by length
    for i in range(len(tri.simplices)):
        v1, v2, v3 = points[tri.simplices[i,:]]
        l1, l2, l3 = get_lengths(v1, v2, v3)
        m_l = np.mean([l1, l2, l3])
        a12, a23, a31 = get_angles(l1,l2,l3) # want roughly pi/3 for hexagonal
        if (np.abs(a12 - np.pi/3) < params['angle_criterion'] and
            np.abs(a23 - np.pi/3) < params['angle_criterion'] and
            np.abs(a31 - np.pi/3) < params['angle_criterion'] ):
            lens.append(m_l)
    for i in range(len(tri.simplices)):
        v1, v2, v3 = points[tri.simplices[i,:]]
        l1, l2, l3 = get_lengths(v1, v2, v3)
        m_l = np.mean([l1, l2, l3])
        a12, a23, a31 = get_angles(l1,l2,l3) # want roughly pi/3 for hexagonal
        if (np.abs(a12 - np.pi/3) < params['angle_criterion'] and
            np.abs(a23 - np.pi/3) < params['angle_criterion'] and
            np.abs(a31 - np.pi/3) < params['angle_criterion'] and
            ( m_l - np.mean(lens))/np.std(lens) < params['ml_criterion'] ):
            ax.plot([v2[0], v3[0]], [v2[1], v3[1]], color="r")
            ax.plot([v2[0], v1[0]], [v2[1], v1[1]], color="r")
            ax.plot([v3[0], v1[0]], [v3[1], v1[1]], color="r")


            wls.append(m_l * params['FOV_length'])

            angs = get_angles_from_vertical(v1, v2, v3)
            orrient.append(np.mean(angs))
            print(angs, '-------', np.mean(angs))

            center_x = np.mean([v1[0], v2[0], v3[0]])
            center_y = np.mean([v1[1], v2[1], v3[1]])
            theta_t, theta_s, eps = fit_heterostrain(l1, l2, l3, params)

            scaled_v1 = [nx*v1[0], ny*v1[1]]
            scaled_v2 = [nx*v2[0], ny*v2[1]]
            scaled_v3 = [nx*v3[0], ny*v3[1]]
            in_tri_mask = make_contour_mask(nx, ny, [scaled_v1, scaled_v2, scaled_v3], transpose=True)
            AA_and_in_tri_mask = (in_tri_mask * AA_mask) 
            AA_f = np.sum(AA_and_in_tri_mask.flatten()) / np.sum(in_tri_mask.flatten())
            thetas.append(np.abs(theta_t) * 180/np.pi)
            het_strains.append(np.abs(eps*100))
            triangles.append([v1, v2, v3])
            area_percents.append(100 * AA_f)
            tri_centers.append([center_x, center_y])

    print('AA mask determined from gaussian fit using FWHM as diameter - I suggest increasing xtol from default to rely on this value')
    ax.set_title('angle = {:.2f} +/- {:.2f} (std) deg'.format(np.mean(thetas), np.std(thetas)))
    
    print('wl = {:.4f} +/- {:.4f} (std) nm'.format(np.mean(wls), np.std(wls)))
    #print(wls)
    print('wl from avg angle = {:.4f} nm'.format( 0.246/(2 * np.sin(np.pi/180 * 0.5 * np.mean(thetas))) ))

    ax2.set_title('het strain = {:.2f} +/- {:.2f} (std) %'.format(np.mean(het_strains), np.std(het_strains)))
    ax3.set_title('overall AA = {:.2f}%'.format(100 * np.sum(AA_mask.flatten())/(nx*ny)))
    plt.savefig("{}/mesh_chunk{}of{}.png".format(filedir, chunkno+1, nchunks), dpi=600)
    plt.show()
    plt.close()
    f, (ax2, ax3, ax4) = plt.subplots(1,3)

    # get colormaps - all plasma
    colormap_func = matplotlib.cm.get_cmap('plasma')
    
    theta_colors = [t if not np.isnan(t) else 0 for t in thetas]
    norm = Normalize()
    norm.autoscale(theta_colors)
    theta_colors = colormap_func(norm(theta_colors))
    
    het_strain_colors = [t if not np.isnan(t) else 0 for t in het_strains]
    norm = Normalize()
    norm.autoscale(het_strain_colors)
    het_strain_colors = colormap_func(norm(het_strain_colors))     
    
    areapercent_colors = [t if not np.isnan(t) else 0 for t in area_percents]
    norm = Normalize()
    norm.autoscale(areapercent_colors)
    areapercent_colors = colormap_func(norm(areapercent_colors))    

    orrient_colors =  [t if not np.isnan(t) else 0 for t in orrient]
    norm = Normalize()
    norm.autoscale(orrient_colors)
    orrient_colors = colormap_func(norm(orrient_colors))  

    for i in range(len(triangles)):

        v1, v2, v3 = triangles[i][:]
        tri_center = tri_centers[i]
        
        for axis in [ax2, ax3, ax4]:
            axis.plot([v2[0], v3[0]], [v2[1], v3[1]], color="grey", linewidth=0.5)
            axis.plot([v2[0], v1[0]], [v2[1], v1[1]], color="grey", linewidth=0.5)
            axis.plot([v3[0], v1[0]], [v3[1], v1[1]], color="grey", linewidth=0.5)
                
        if not np.isnan(thetas[i]):
            trix = [v1[0], v2[0], v3[0]]
            triy = [v1[1], v2[1], v3[1]]
            ax2.fill(trix, triy, color=theta_colors[i])
            ax3.fill(trix, triy, color=het_strain_colors[i])
            ax2.text(tri_center[0], tri_center[1], "{:.2f}".format(thetas[i]), color='grey', fontsize='xx-small', horizontalalignment='center')
            ax3.text(tri_center[0], tri_center[1], "{:.2f}".format(het_strains[i]), color='grey', fontsize='xx-small', horizontalalignment='center')

            #ax4.fill(trix, triy, color=areapercent_colors[i])
            #ax4.text(tri_center[0], tri_center[1], "{:.2f}".format(area_percents[i]), color='grey', fontsize='xx-small', horizontalalignment='center')
            ax4.fill(trix, triy, color=orrient_colors[i])
            ax4.text(tri_center[0], tri_center[1], "{:.2f}".format(orrient[i]), color='grey', fontsize='xx-small', horizontalalignment='center')

    # after have triangles now can calculate percent AA area in each 
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax2.set_title('$<\\theta_m> = {:.2f} +/- {:.2f}^o$'.format(np.nanmean(thetas), np.nanstd(thetas)))
    ax3.set_title('$<\epsilon> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(het_strains), np.nanstd(het_strains)))
    #ax4.set_title('$<AA_f> = {:.2f}\%$'.format(np.nanmean(area_percents)))
    ax4.set_title('$<\\varphi> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(orrient), np.nanstd(orrient)))
    plt.show()
    return thetas, het_strains, area_percents
 

# reads in excel file and returns partitioned data
def read_excel(filenm, num_slices):
    dat_pd = pd.read_excel(filenm, index_col=0)
    dat = dat_pd.to_numpy()
    (nx, ny) = dat.shape
    increment = nx//num_slices
    if (nx%num_slices > 0):
        print("Cannot cleanly slice into {}x{} chunks".format(num_slices,num_slices))
        exit(0)
    chunks = []
    for i in range(num_slices):
        for j in range(num_slices):
            chunks.append( dat[i*increment:i*increment+increment, j*increment:j*increment+increment] )
    return chunks, np.nan

# reads the text file of formatted STM data
def read_txt(filenm, num_slices):

    # read the POSCAR file
    lines = []
    with open(filenm) as f:
       line = f.readline()
       lines.append(line)
       while line:
           line = f.readline()
           lines.append(line)

    # parse
    for line in lines:
        if re.match("Data Size:*", line):
            l = line.split(":")
            domain_size = [ int(e.strip()) for e in l[1].split("x") ]
            if (domain_size[0] != domain_size[1]):
                print("expect domain sizes equal, cannot have {}x{} atm".format(domain_size[0], domain_size[1]))
                exit(0)
            dat = np.zeros((domain_size[0], domain_size[1]))
            i = 0
        elif re.match("Surface Size:*", line):
            l = line.split(":")
            domain_len = [ float(e.strip()) for e in l[1].split("x") ]
            if (domain_len[0] != domain_len[1]):
                print("expect domain sizes equal, cannot have {}x{} atm".format(domain_len[0], domain_len[1]))
                exit(0)
        elif re.match("X Unit:*", line) or re.match("Y Unit:*", line):
            l = line.split(":")
            unit = l[1].strip()
            if (unit.lower() != "nanometer"):
                print("unexpected unit in STM data, expect domain to be in nm")
                exit(0)
        elif (line.strip() != ""):
            l = line.split("\t")
            if re.match("-?\d+.\d+", l[0]):
                dat[i, :] = [float(e) for e in l[0:-1]]
                i += 1

    nx = domain_size[0]
    increment = nx//num_slices
    if (nx%num_slices > 0):
        print("Cannot cleanly slice into {}x{} chunks".format(num_slices,num_slices))
        exit(0)
    chunks = []
    for i in range(num_slices):
        for j in range(num_slices):
            chunks.append( dat[i*increment:i*increment+increment, j*increment:j*increment+increment] )
    return chunks, domain_len[0]/num_slices

# write a file
def write(filenm, data_name, data):
    with open(filenm, 'w') as f:
        f.write('{}\n'.format(data_name))
        if isinstance(data,dict):
            for k, v in data.items():
                f.write('{} : {}\n'.format(k, v))
        else:
            for el in data:
                f.write('{}\n'.format(el))

# read the data, wrapper for all file types
def read(filenm, num_slices):
    f = filenm.split(".")
    if (f[1] == "txt"):
        chunks, FOV_len = read_txt(filenm, num_slices)
    elif (f[1] == "xlsx"):
        chunks, FOV_len = read_excel(filenm, num_slices)
    else:
        print("STM data has unrecongnized file format")
        exit(0)
    return chunks, FOV_len

if __name__ == '__main__':

    params = {
        # change me to cut data up in the more/less chunks
        "num_slices"        : 1,               # slice data into chunks, will be faster but omits some data

        # change me if there are too few or too many peaks
        "contour_boundary"  : 0.55,            # initial boundaries will be at this percent intentsity
        'guess_radius_criterion' : -0.1,       # decrease me if there are a lot of erroneous mall peaks
        'combine_criterion' : 4.0,             # increase me if plot_avg breaks peaks into two nearby circles
        'times_to_combine' : 3,                # times to run through the nearby point combination proecedure

        # change me if there are too few or too many triangles that are accepted and used to get theta
        'ml_criterion' : 1.5,                  # remove points that have mean delaunay lengths to nearby points greater than a given criterion
                                               # due to common issue with delaunay algorithm - will include erroneous connections
        "angle_criterion"   : 0.35,            # will ignore regions where moire triangle angles are greater than angle_criterion rad from pi/3

        # change me for special purposes
        "xtol"              : 1e-1,            # tolerance for gaussian fit,
                                               # increase for better guassian fits (slower, bad for noisy data, good for clean data)
        "removed_before_keys" : [],            # manually removed points (will ask you to enter these while running program so can leave blank)
                                               # or can enter them here if you've already run this program and know what to remove
        'truncation' : False,                  # to trucate field of view

        # these have very little effect on the results
        "guess_theta_t"     : 0.2,             # degrees of guess angle for twist (results don't depend on this I've found)
        "guess_theta_s"     : 25,              # guess angle of heterostrain application (results don't depend on this I've found)
        "guess_hs"          : 0.05,            # guessed percent heterostrain (results don't depend on this I've found)
    }

    # get inputs
    params['filename'] = sys.argv[1]
    filedir = os.path.split(params['filename'])
    filedir = filedir[0]

    # parse
    chunks, FOV_len = read(params['filename'], params['num_slices'])
    params['FOV_length'] = FOV_len
    params['a_graphene'] = 0.246/params['FOV_length'] # 0.246 nm, normalized in units of FOV
    nchunks = params['num_slices']**2

    # process chunks
    for i in range(nchunks):
        # process chunk i
        print("processing chunk {} of {} ".format(i+1,nchunks))
        (nx, ny) = chunks[i].shape
        thetas, het_strains, area_percents = mesh_process(chunks[i], filedir, i, nchunks, params)

        # plot histograms
        f, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.hist(thetas, bins='auto')
        ax1.set_title('local twist angle')
        ax2.hist(het_strains, bins='auto')
        ax2.set_title('local heterostrain')
        ax3.hist(area_percents, bins='auto')
        ax3.set_title('local AA area percent')
        plt.savefig("{}/histo_chunk{}of{}.png".format(filedir, i+1,nchunks))

        # write data
        write("{}/heterostrains_chunk{}of{}.txt".format(filedir, i+1,nchunks), "percent heterostrain obtained", het_strains)
        write("{}/angles_chunk{}of{}.txt".format(filedir, i+1,nchunks), "twist angles (degrees)", thetas)
        write("{}/areapercent_chunk{}of{}.txt".format(filedir, i+1,nchunks), "area percents of AA per moire triangle", thetas)
        write("{}/output_chunk{}of{}.txt".format(filedir, i+1,nchunks), "parameters used:", params)
