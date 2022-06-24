
import py4DSTEM
import numba
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # for excel io utilites
from py4DSTEM.io import *
from py4DSTEM.process.diskdetection import *
from py4DSTEM.process.virtualimage import *
from ncempy.io import dm
from skimage import measure
from matplotlib.path import Path
from scipy.optimize import curve_fit, least_squares
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.restoration import denoise_tv_bregman
from time import sleep, time
import numpy.random as random
from numpy.linalg import norm
import scipy.ndimage as ndimage

def bomb_out(str):
    print(str)
    exit()

def debugplot(m):
    f, ax = plt.subplots()
    ax.imshow(m, origin='lower')
    plt.show()

def merge_u(ux, uy):
    u = np.zeros((ux.shape[0], ux.shape[1], 2))
    u[:,:,0], u[:,:,1] = ux, uy
    return u

def trim_to_equal_dim(mat, even=False):
    nx, ny = mat.shape[0], mat.shape[1]
    if even and nx%2 == 1: nx -= 1
    if even and ny%2 == 1: ny -= 1
    if nx < ny: return mat[:nx,:nx]
    else: return mat[:ny, :ny]

def read_excel(fpath):
    data = pd.read_excel(fpath, engine='openpyxl') # just use pandas function, openpyxl needed for xlsx files
    return np.matrix(data)

def tic():
    global globaltime
    globaltime = time()

def toc(msg=''):
    print('{}: {} seconds'.format(msg, time() - globaltime))

def get_AB_centers(adjacency_type, centers):
    triangles = get_triangles(adjacency_type > 0)
    AB_centers = []
    for tri in triangles:
        sp_types = np.unique([adjacency_type[tri[0], tri[1]], adjacency_type[tri[2], tri[1]], adjacency_type[tri[0], tri[2]]])
        if len(sp_types) > 2:
            x = np.mean([centers[i][1] for i in tri])
            y = np.mean([centers[i][0] for i in tri])
            AB_centers.append([x,y])
    return AB_centers

# see stack exchange for 'guassian filtering a image with nan in python'
def nan_gaussian_filter(array, sigma):
    u = array.copy()
    v = array.copy()
    v[np.isnan(u)] = 0
    vv = ndimage.gaussian_filter(v, sigma=sigma)
    w = 0*u.copy()+1
    w[np.isnan(u)] = 0
    ww = ndimage.gaussian_filter(w, sigma=sigma)
    return vv/ww

def integrate_bknd_circ(mat, x0, y0, R):
    nqx, nqy = mat.shape
    xmin,xmax = max(0,int(np.floor(x0-R))),min(nqx,int(np.ceil(x0+R)))
    ymin,ymax = max(0,int(np.round(y0-R))),min(nqy,int(np.ceil(y0+R)))
    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < R**2, (xsize,ysize))
    return np.sum(mat[xmin:xmax,ymin:ymax]*mask, axis=(0,1))

def integrate_ds_circ(mat, x0, y0, R):
    nx, ny, nqx, nqy = mat.shape
    xmin,xmax = max(0,int(np.floor(x0-R))),min(nqx,int(np.ceil(x0+R)))
    ymin,ymax = max(0,int(np.round(y0-R))),min(nqy,int(np.ceil(y0+R)))
    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < R**2, (xsize,ysize))
    return np.sum(mat[:,:,xmin:xmax,ymin:ymax]*mask, axis=(2,3))

def boolquery(msg):
    return input("{} (y/n) ".format(msg)).lower().strip()[0] == 'y'

def extract_contour_radius(mask, ax=None, contour_boundary=0.35):
    radii = []
    contours = measure.find_contours(mask, contour_boundary)
    for contour in contours:
        xcent = np.mean(contour[:, 1])
        ycent = np.mean(contour[:, 0])
        rads = [ np.sqrt((x-xcent)**2 + (y-ycent)**2) for x,y in zip(contour[:, 1], contour[:, 0]) ]
        rad = np.mean(rads)
        radii.append(rad)
        circle = plt.Circle((xcent, ycent), rad, color='r', fill=False)
        ax.add_patch(circle)
    return radii

def in_reduced_zone(a, b):
    return np.abs(a) <= 1 and np.abs(b) <= 1 and a <= 1 - b and a >= -b - 1

def extend_zone(u, v):
    a = v - 1/np.sqrt(3)*u
    b = 2/np.sqrt(3)*u
    aofflst = [0, 1, -1, 1, -1, 2, -2]
    bofflst = [0, 1, -1, -2, 2, -1, 1]
    for i in range(len(aofflst)):
        aoff = aofflst[i]
        boff = bofflst[i]
        if in_reduced_zone(a - aoff, b - boff):
            v = (a - aoff) + (b - boff)*1/2
            u = (b - boff)*np.sqrt(3)/2
            return u, v
    print('error too large for zone unfolding')
    exit()

def call_counter(func):
    def helper(x):
        helper.calls += 1
        return func(x)
    helper.calls = 0
    return helper

# write a file
def writefile(filenm, data_name, data):
    with open(filenm, 'w') as f:
        f.write('{}\n'.format(data_name))
        if isinstance(data,dict):
            for k, v in data.items():
                f.write('{} : {}\n'.format(k, v))
        else:
            for el in data:
                f.write('{}\n'.format(el))

# for manual defined cropping
def manual_define_triangles(img):
    plt.close('all')
    fig, ax = plt.subplots()
    traingles = []
    @call_counter
    def click_event(click):
        x,y = click.xdata, click.ydata
        counter = click_event.calls - 1
        print('counter {}'.format(counter))
        print('({},{})'.format(x, y))
        ax.scatter(x,y,color='k')
        if counter%3 == 0: traingles.append([[x,y]])
        else: traingles[-1].append([x,y])
        if counter%3 == 1:
            ax.plot([traingles[-1][-1][0], traingles[-1][-2][0]], [traingles[-1][-1][1], traingles[-1][-2][1]], color='k')
        if counter%3 == 2:
            ax.plot([traingles[-1][-1][0], traingles[-1][-3][0]], [traingles[-1][-1][1], traingles[-1][-3][1]], color='k')
            ax.plot([traingles[-1][-1][0], traingles[-1][-2][0]], [traingles[-1][-1][1], traingles[-1][-2][1]], color='k')
        fig.canvas.draw()
    print("please click to define the moire triangles. click three times for each moire triangle")
    ax.imshow(img, origin='lower')
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    print('finished with manual definition')
    print(traingles)
    return traingles

def getAdjacencyMatrixManual(img, points=None, adjacency_matrix=None):
    if points is not None: points, adjacency_matrix = manual_remove_AA(img, points, adjacency_matrix)
    points, adjacency_matrix = manual_define_AA(img, points, adjacency_matrix)
    adjacency_matrix = manual_define_SP(img, points, adjacency_matrix, adj_type=0, title='click to remove SP')
    adjacency_matrix = manual_define_SP(img, points, adjacency_matrix, adj_type=1, title='click to def SP1 (c)')
    adjacency_matrix = manual_define_SP(img, points, adjacency_matrix, adj_type=2, title='click to def SP2 (m)')
    adjacency_matrix = manual_define_SP(img, points, adjacency_matrix, adj_type=3, title='click to def SP3 (y)')
    return points, adjacency_matrix 

def manual_define_SP(img, points, adjacency_matrix, adj_type, title):

    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_title(title)
    colors = ['k','c', 'm', 'y']
    adj_color = colors[adj_type]
    lastpoint_queue = []

    def getclosestpoint(x,y):
        dists = [(p[0]-x)**2 + (p[1]-y)**2 for p in points]
        return points[np.argmin(dists)], np.argmin(dists)

    @call_counter
    def sp_click_event(click):
        counter = sp_click_event.calls - 1
        x,y = click.xdata, click.ydata
        point, point_id = getclosestpoint(x,y)
        lastpoint_queue.append(point_id)
        x,y = point[:]
        ax.scatter(x,y,color=adj_color)
        if len(lastpoint_queue)==2:
            point2_id = lastpoint_queue.pop(0)
            point2 = points[point2_id]
            point_id = lastpoint_queue.pop(0)
            point = points[point_id]
            ax.plot([point[0], point2[0]], [point[1], point2[1]], color=adj_color)
            adjacency_matrix[point_id, point2_id] = adj_type
            adjacency_matrix[point2_id, point_id] = adj_type
        fig.canvas.draw()

    print("please click to define/remove the sp{} connections ({}), close figure when done".format(adj_type, adj_color))
    ax.imshow(img, origin='lower')
    for point in points: ax.scatter(point[0], point[1], color='w')
    for n in range(adjacency_matrix.shape[0]):
        for m in range(n, adjacency_matrix.shape[0]):
            if adjacency_matrix[n,m] > 0:
                ax.plot([points[n][0], points[m][0]], [points[n][1], points[m][1]], color=colors[int(adjacency_matrix[n,m])])
    cid = fig.canvas.mpl_connect('button_press_event', sp_click_event)
    plt.show()
    print('finished with manual sp{} definition'.format(adj_type))
    return adjacency_matrix

# for manual defined cropping
def manual_adjust_points(img, regions, points):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_title('click to adjust points')
    def getclosestpoint(x,y):
        dists = [(p[0]-x)**2 + (p[1]-y)**2 for p in points]
        return points[np.argmin(dists)], np.argmin(dists)
    def click_event(click):
        x,y = click.xdata, click.ydata
        point, point_id = getclosestpoint(x,y)
        x,y = point[:]
        ax.scatter(x,y,color='r')
        x,y = click.xdata, click.ydata
        points[point_id] = [x,y]
        ax.scatter(x,y,color='k')
        fig.canvas.draw()
    print("please click to adjust points")
    ax.imshow(img, origin='lower')
    ax.set_xlim([-50, img.shape[0]+50])
    ax.set_ylim([-50, img.shape[1]+50])
    for p in points: ax.scatter(p[0],p[1],color='k')
    for i in range(len(regions)):
        polygon = points[regions[i]]
        ax.fill(*zip(*polygon), alpha=0.4)
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    print('finished with manual aa definition')
    return points

# for manual defined cropping
def manual_remove_AA(img, points, adj_mat):
    plt.close('all')

    fig, ax = plt.subplots()
    ax.set_title('click to remove centers - DONT DO IF WATERSHED')
    adjcolors = ['c', 'm', 'y']

    def getclosestpoint(x,y):
        dists = [(p[0]-x)**2 + (p[1]-y)**2 for p in points]
        return points[np.argmin(dists)], np.argmin(dists)

    def aa_click_event(click):
        x,y = click.xdata, click.ydata
        point, point_id = getclosestpoint(x,y)
        x,y = point[:]
        points[point_id] = [-1,-1]
        ax.scatter(x,y,color='k')
        for n in range(adj_mat.shape[0]):
            if adj_mat[point_id,n] > 0:  ax.plot([point[0], points[n][0]], [point[1], points[n][1]], color='k')
            adj_mat[point_id,n] = 0
            adj_mat[n, point_id] = 0
        fig.canvas.draw()

    print("please click to remove AA centers, close figure when done. black centers and connections are removed")
    ax.imshow(img, origin='lower')
    for p in points: ax.scatter(p[0],p[1],color='w')
    for n in range(adj_mat.shape[0]):
        for m in range(n, adj_mat.shape[0]):
            if adj_mat[n,m] > 0:
                ax.plot([points[n][0], points[m][0]], [points[n][1], points[m][1]], color=adjcolors[int(adj_mat[n,m]) - 1])
    cid = fig.canvas.mpl_connect('button_press_event', aa_click_event)
    plt.show()
    print('finished with manual aa definition')
    return points, adj_mat

def manual_cropping(img, vmax=None):
    plt.close('all')
    fig, ax = plt.subplots()
    vertices = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        vertices.append([x,y])
        print('vertex {} at ({},{})'.format(len(vertices), x, y))
        ax.scatter(x,y,color='w')
        if len(vertices) > 1:
            ax.plot([vertices[-1][0], vertices[-2][0]], [vertices[-1][1], vertices[-1][1]], color='w')
            ax.plot([vertices[-1][0], vertices[-1][0]], [vertices[-1][1], vertices[-2][1]], color='w')
            ax.plot([vertices[-1][0], vertices[-2][0]], [vertices[-2][1], vertices[-2][1]], color='w')
            ax.plot([vertices[-2][0], vertices[-2][0]], [vertices[-1][1], vertices[-2][1]], color='w')
        fig.canvas.draw()
        if len(vertices) == 2:
            sleep(1)
            fig.canvas.mpl_disconnect(cid)
            plt.close('all')
    print("please click twice to define rectangular region")
    if vmax is None: ax.imshow(img, origin='lower')
    else: ax.imshow(img, origin='lower',vmax=vmax)
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    print('finished with manual region definition')
    vertices = np.array(vertices)
    ymin = int(np.min(vertices[:,0]))
    xmin = int(np.min(vertices[:,1]))
    ymax = int(np.max(vertices[:,0]))
    xmax = int(np.max(vertices[:,1]))
    crop_img = img[xmin:xmax, ymin:ymax]
    fig, ax = plt.subplots(1,2);
    ax[0].imshow(img, origin='lower', vmax=np.max(crop_img.flatten()), vmin=np.min(crop_img.flatten()));
    ax[0].plot([vertices[-1][0], vertices[-2][0]], [vertices[-1][1], vertices[-1][1]], color='w')
    ax[0].plot([vertices[-1][0], vertices[-1][0]], [vertices[-1][1], vertices[-2][1]], color='w')
    ax[0].plot([vertices[-1][0], vertices[-2][0]], [vertices[-2][1], vertices[-2][1]], color='w')
    ax[0].plot([vertices[-2][0], vertices[-2][0]], [vertices[-1][1], vertices[-2][1]], color='w')
    ax[1].imshow(crop_img, origin='lower');
    plt.show()
    return xmin, xmax, ymin, ymax

# for manual defined cropping
def manual_define_AA(img, points=None, adj_mat=None):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_title('click to define new centers - DONT DO IF WATERSHED')
    adjcolors = ['c', 'm', 'y']
    if points is None: points = []
    def aa_click_event(click):
        x,y = click.xdata, click.ydata
        print('({},{})'.format(x, y))
        points.append([x, y])
        ax.scatter(x,y,color='w')
        fig.canvas.draw()
    print("please click to define the AA centers, close figure when done")
    ax.imshow(img, origin='lower')
    for p in points: ax.scatter(p[0],p[1],color='w')
    for n in range(adj_mat.shape[0]):
        for m in range(n, adj_mat.shape[0]):
            if adj_mat[n,m] > 0:
                ax.plot([points[n][0], points[m][0]], [points[n][1], points[m][1]], color=adjcolors[int(adj_mat[n,m]) - 1])
    cid = fig.canvas.mpl_connect('button_press_event', aa_click_event)
    plt.show()
    print('finished with manual aa definition')
    points = [p for p in points if p is not None and p[0] is not None]
    adjacency_matrix = np.zeros((len(points), len(points)))
    adjacency_matrix[:adj_mat.shape[0],:adj_mat.shape[0]] = adj_mat[:,:]
    return points, adjacency_matrix

# for manual defined cropping
def manual_define_points(img):
    plt.close('all')
    fig, ax = plt.subplots()
    points = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        print('({},{})'.format(x, y))
        points.append([x, y])
        ax.scatter(x,y,color='k')
        fig.canvas.draw()
    print("please click to define the points in inner diffraction ring, close figure when done")
    ax.imshow(img, origin='lower')
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    print('finished with manual disk definition')
    return points

# for manual defined angle determinination. given image, ask user to pick 3 points. angle between returned in radians
def manual_extract_angle(img):
    # quick helper - get angle from v1 through v2 to v3.
    def get_angle(pts):
        v1, v2, v3 = pts[0], pts[1], pts[2] 
        l1 = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
        l2 = ((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)**0.5
        l3 = ((v3[0] - v2[0])**2 + (v3[1] - v2[1])**2)**0.5
        return np.arccos((l1**2 + l3**2 - l2**2)/(2*l1*l3))
    # the meat
    plt.close('all')
    fig, ax = plt.subplots()
    points = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        print('({},{})'.format(x, y))
        points.append([x, y])
        ax.scatter(x,y,color='k')
        fig.canvas.draw()
        if len(points) == 3: # quit at 3 points
            sleep(1)
            fig.canvas.mpl_disconnect(cid)
            plt.close('all')
    print("please click to define three points, will compute angle. close figure when done")
    ax.imshow(img, origin='lower')
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    angle = get_angle(points)
    print('finished. angle is {} degrees, {} radians'.format(angle*180/np.pi, angle))
    return angle #radians

# uses depth first search to get all triangles in the graph
def get_triangles(graph, n=3):
    marked = [False] * graph.shape[0]
    visited = [False] * graph.shape[0]
    cycles = []
    for i in range(graph.shape[0]):
        cycles = DFS(graph, marked, visited, n-1, i, i, cycles)
        marked[i] = True
    return cycles

def DFS(graph, marked, visited, n, vert, start, paths):
    marked[vert] = True # don't wanna trace back on self
    visited[vert] = True
    if n == 0:
        if graph[vert][start] == 1: # if closed
            path = [i for i in range(len(visited)) if visited[i]]
            if path not in paths: paths.append(path)
            marked[vert] = False
            visited[vert] = False
            return paths
        else:
            marked[vert] = False
            visited[vert] = False
            return paths
    for i in range(graph.shape[0]): # checking all paths
        if marked[i] == False and graph[vert][i] == 1:
            paths = DFS(graph, marked, visited, n-1, i, start, paths)
    marked[vert] = False
    visited[vert] = False
    return paths

####################################################################################################
# returns the u,v (displacement in x and in y) values at the position (x,y) and all neighboring pixels
# where neighbors are defined to include the pixels (x+1, y), (x-1, y), (x, y+1), and (x, y-1)
####################################################################################################
def get_neighbors(x, y, uvecs, nx, ny):
    near_u = []
    near_v = []
    if x < 0:
        near_u.append(uvecs[x-1,y,0]) # nearest x neighbors
        near_v.append(uvecs[x-1,y,1])
    if x < nx-1:
        near_u.append(uvecs[x+1,y,0]) # nearest x neighbors
        near_v.append(uvecs[x+1,y,1])
    if y > 0:
        near_u.append(uvecs[x,y-1,0]) # nearest y neighbors
        near_v.append(uvecs[x,y-1,1])
    if y < ny-1:
        near_u.append(uvecs[x,y+1,0]) # nearest x neighbors
        near_v.append(uvecs[x,y+1,1])
    near_u.append(uvecs[x,y,0]) # add (x,y) spot too, self included in neighbor list
    near_v.append(uvecs[x,y,1])
    return near_u, near_v

####################################################################################################
# ----> LEGACY, not using this function ATM
# monte carlo phase unwrapping algorithm utility functions
####################################################################################################
def find_equivalents_C2(x_magnitude, y_magnitude):
    return [[-x_magnitude, -y_magnitude]]

def find_equivalents_C3(x_magnitude, y_magnitude):
    equivs = []
    equivs.append(rotate2d([x_magnitude, y_magnitude], 120/180*np.pi))
    equivs.append(rotate2d([x_magnitude, y_magnitude], 240/180*np.pi))
    return equivs

def find_equivalents_C6(x_magnitude, y_magnitude):
    equivs = []
    equivs.append(rotate2d([x_magnitude, y_magnitude], 60/180*np.pi))
    equivs.append(rotate2d([x_magnitude, y_magnitude], 120/180*np.pi))
    equivs.append(rotate2d([x_magnitude, y_magnitude], 180/180*np.pi))
    equivs.append(rotate2d([x_magnitude, y_magnitude], 240/180*np.pi))
    equivs.append(rotate2d([x_magnitude, y_magnitude], 300/180*np.pi))
    return equivs

####################################################################################################
# ----> LEGACY, not using this function ATM
# replaces anamolous displacment vectors with median of neigbor's displacment vectors
####################################################################################################
def median_neighbor_scan(uvecs, nx, ny, plotbool=False):
    # want to do interior replacement first, larger number of neighbors
    # so less suceptible to erroneous replacements
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            median_u, median_v = get_neighbors(x, y, uvecs, nx, ny)
            neighbors_median_u = np.median(median_u) # get median of neighbors and self
            neighbors_median_v = np.median(median_v)
            # replace with median if more than 0.5 off
            if np.abs(neighbors_median_u - uvecs[x,y,0]) + np.abs(neighbors_median_v - uvecs[x,y,1]) > 0.5:
                uvecs[x,y,0] = neighbors_median_u
                uvecs[x,y,1] = neighbors_median_v
    # now exterior
    for x in range(nx):
        for y in [0, ny-1]:
            median_u, median_v = get_neighbors(x, y, uvecs, nx, ny)
            neighbors_median_u = np.median(median_u)
            neighbors_median_v = np.median(median_v)
            # replace with median if more than 0.5 off
            if np.abs(neighbors_median_u - uvecs[x,y,0]) + np.abs(neighbors_median_v - uvecs[x,y,1]) > 0.5:
                uvecs[x,y,0] = neighbors_median_u
                uvecs[x,y,1] = neighbors_median_v
    for x in [0, nx-1]:
        for y in range(ny):
            median_u, median_v = get_neighbors(x, y, uvecs, nx, ny)
            neighbors_median_u = np.median(median_u)
            neighbors_median_v = np.median(median_v)
            # replace with median if more than 0.5 off
            if np.abs(neighbors_median_u - uvecs[x,y,0]) + np.abs(neighbors_median_v - uvecs[x,y,1]) > 0.5:
                uvecs[x,y,0] = neighbors_median_u
                uvecs[x,y,1] = neighbors_median_v
    if plotbool:
        f, ax = plt.subplots()
        U = uvecs[:,:,0].reshape(nx, ny)
        V = uvecs[:,:,1].reshape(nx, ny)
        ax.quiver(U, V)
        plt.show()
    return uvecs

####################################################################################################
# magnitude of a vector
####################################################################################################
def mag(v): return (v[0]**2 + v[1]**2)**0.5

####################################################################################################
# denoising using bregman filter, see scipy documentation for more info
####################################################################################################
def denoise(dat): return denoise_tv_bregman(dat, weight=10)

####################################################################################################
# distance between two vertices
####################################################################################################
def get_dist(v1, v2): return ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5

####################################################################################################
# get xy components given a length and angle
####################################################################################################
def get_vec(len, ang):
    x = len*np.cos(ang)
    y = len*np.sin(ang)
    return np.array([x,y])

####################################################################################################
# points within a given distance of eachother are combined, being replaced by their average
# spots is an array of length n_spots, each array element is a length 3 array of (x,y,r)
# combine_criterion sets how far apart points should be in order to be combined
####################################################################################################
def combine_nearby_spots(spots, combine_criterion, weights=None):
    n = len(spots)
    bool_arr = np.ones((n,1))
    # calculate distances between all points
    distance_table = 100 * np.ones((n,n))
    for i in range(n):
        for j in range(i+1, n):
            distance_table[i,j] = get_dist([spots[i][0], spots[i][1]], [spots[j][0], spots[j][1]])
    # using this distance table combine nearby spots whose distance is less than the combine_criterion
    for i in range(n):
        d = np.min(distance_table[i,:])
        if d < combine_criterion:
            j = np.argmin(distance_table[i,:])
            spot_i = spots[i]
            spot_j = spots[j]
            if weights==None:
                spots[i] = [ (spot_i[0]+spot_j[0])/2, (spot_i[1]+spot_j[1])/2 ]
            else:
                spots[i] = [ (spot_i[0]*weights[i]+spot_j[0]*weights[j])/(weights[i] + weights[j]),
                             (spot_i[1]*weights[i]+spot_j[1]*weights[j])/(weights[i] + weights[j]),  ]
            spots[j] = spots[i] # set both spots equal to the new spot will later filer out duplicates
            bool_arr[i] = 0
    # now get all unique spots since process created duplicates
    new_spots = []
    for i in range(len(spots)):
        if bool_arr[i]: new_spots.append(spots[i])
    spots = new_spots
    return spots

####################################################################################################
# reads in a series of formatted matrices for each disk from a multislice disp field scan
####################################################################################################
def read_txt_dispvector(foldername):
    dat = np.zeros([12, 41, 41])
    for n in range(12):
        os.path.join(foldername,'disk{}.txt'.format(n+1))
        lines = []
        with open(filenm) as f:
           line = f.readline()
           lines.append(line)
           while line:
               line = f.readline()
               lines.append(line)
        lines = [l for l in lines if len(l.strip()) > 0]
        for i in range(41):
            line = lines[i]
            line_el = [el.strip() for el in line.split(',')]
            dat[n,:,i] = line_el
    return dat

def is_connected(contour, tolerance=5.0):
    start_pt = contour[0, :]
    end_pt   = contour[-1, :]
    dist = get_dist(start_pt, end_pt)
    return dist < tolerance

####################################################################################################
# bin the data by bin_w, such that the resultant data pizels correspond to bin_w x bin_w pixel avgs
####################################################################################################
def bin(data, bin_w, size_retain=False, method=np.median):
    sx, sy = data.shape[0], data.shape[1] #len(data)
    sx_bin, sy_bin = sx//bin_w, sy//bin_w
    data_binned = np.zeros((sx_bin, sy_bin))
    for i in range(sx_bin):
        for j in range(sy_bin):
            d = data[ i*bin_w:(i+1)*bin_w , j*bin_w:(j+1)*bin_w ]
            if len(d) > 0: data_binned[i, j] = method(d)
            else: data_binned[i, j] = np.nan
    if not size_retain: return data_binned
    else: return unbin(data_binned, bin_w)

####################################################################################################
####################################################################################################
def window_filter(data, bin_w, method=np.nanmean):
    nx, ny = data.shape
    for i in range(nx):
        for j in range(ny):
            data[i, j] = method(data[i:np.min([i+bin_w,nx]), j:np.min([j+bin_w,ny])])
    return data

####################################################################################################
# enlarge, recovering original size after binning
####################################################################################################
def unbin(data, bin_w):
    sx, sy = data.shape[0], data.shape[1]
    sx_bin, sy_bin = sx*bin_w, sy*bin_w
    data_unbinned = np.zeros((sx_bin, sy_bin))
    for i in range(sx):
        for j in range(sy):
            data_unbinned[ i*bin_w:(i+1)*bin_w , j*bin_w:(j+1)*bin_w ] = data[i, j]
    return data_unbinned

####################################################################################################
# read in dm4 file and save as h5 file
####################################################################################################
def convert_dm4(scan_shape, datapath=os.path.join('..','data')):
    fullpath = os.path.join(datapath, 'Diffraction SI.dm4')
    if not os.path.exists(fullpath):
        print("ERROR: couldn't find Diffraction SI.dm4")
        exit()
    data = read(fullpath)
    data.set_scan_shape(scan_shape[0],scan_shape[1])
    py4DSTEM.io.save(os.path.join(datapath,'dp.h5'),data,overwrite=True)

####################################################################################################
# read in dm4 files and save as h5 files for a batch of datasets, each stored within folders
# titled 1_100x100... for dataset 1 and scan shape 100x100 etc
####################################################################################################
def convert_all_h5(datapath):
    for d in [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath,el))]:
        if d == 'before_chunksplit': continue
        m_dir = os.path.join(datapath,d)
        datasetnum = d.split("_")[0]
        print("working on dataset {}".format(datasetnum))
        tmp = d.split("_")[1]
        scan_shape = [ int(tmp.split("x")[0]), int(tmp.split("x")[1]) ]
        if not datasetnum.isdigit(): continue
        if os.path.isfile(os.path.join(m_dir,'dp.h5')):
            print('dp.h5 already exists, skipping')
        else: convert_dm4(scan_shape, m_dir)

####################################################################################################
# normalize data to be in range [0,1]
####################################################################################################
def normalize(d):
    d = d - min(d.flatten())
    d = d/max(d.flatten())
    return d

####################################################################################################
# rotate a vector in 2d about origin
####################################################################################################
def rotate2d(v, ang):
    r = np.zeros((2,2))
    r[0,0] = np.cos(ang)
    r[0,1] = -np.sin(ang)
    r[1,0] = np.sin(ang)
    r[1,1] = np.cos(ang)
    return np.dot(r, v)

####################################################################################################
# use 3rd highest intensity (or whatever disk_to_use is) bragg peak to get disk radius
####################################################################################################



####################################################################################################
# gets real space probe by summing a series of files often saved as 'misc k3 images'
####################################################################################################
def sum_probe_imgs(datapath):
    if os.path.exists(os.path.join(datapath,"sum_probe.dm3")):
        print('dp.h5 already exists, skipping')
        return True
    tot_probe = None
    for d in [el for el in os.listdir(datapath) if el[0:2] == 'K3']:
        dm3_file = os.path.join(datapath,d)
        probe = dm.dmReader(dm3_file)['data']
        if tot_probe is None: tot_probe = probe
        else: tot_probe = tot_probe + probe
    py4DSTEM.io.save(os.path.join(datapath,"sum_probe.dm3"),tot_probe)
    return tot_probe

####################################################################################################
# return real space probe kernel given a vacuum scan .dm3 file
####################################################################################################
def get_probe_dm4(dm4_file, mask_threshold=0.1, sigma_probe_scale=2, plotflag=False):
    data = read(dm4_file)
    prefix = dm4_file.split('.')[0]
    dm3_file = "{}.dm3".format(prefix)
    py4DSTEM.io.save(os.path.join(datapath, dm3_file), data)
    return get_probe(dm3_file, mask_threshold, sigma_probe_scale, plotflag)

def get_probe(dm3_file, mask_threshold=0.1, sigma_probe_scale=2, plotflag=False):
    probe = dm.dmReader(dm3_file)['data']
    probe = get_probe_from_vacuum_2Dimage(probe, mask_threshold=mask_threshold)
    beamcenter = get_beamcenter(probe)
    probe_kernel = py4DSTEM.process.diskdetection.get_probe_kernel_subtrgaussian(probe, sigma_probe_scale=sigma_probe_scale)
    probe_kernel_FT = np.conj(np.fft.fft2(probe_kernel))
    if plotflag: f, ax = plt.subplots()
    if plotflag:
        ax.imshow(probe, cmap='gray')
        circle = plt.Circle((beamcenter[1], beamcenter[0]), 5, color='r')
        ax.add_patch(circle)
        plt.axis('off')
        plt.savefig(os.path.join('..','plots',"probe.png"),  dpi=300)
        plt.close()
    return probe_kernel, probe_kernel_FT, beamcenter

####################################################################################################
# determine com beam center from probe dp
####################################################################################################
def get_beamcenter(probe, contour_boundary=0.25):
    contours = measure.find_contours(normalize(probe), contour_boundary)
    max_l = 0
    for contour in contours:
        if len(contour) > max_l:
            max_l = len(contour)
            bs_contour = contour
    xcent = np.mean(bs_contour[:, 0])
    ycent = np.mean(bs_contour[:, 1])
    center = (xcent, ycent)
    return center
