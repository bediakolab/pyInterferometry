
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy.ndimage import gaussian_filter
from visualization import overlay_vdf
from diskset import DiskSet
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path


def make_legend(ax, inc3layer=False, abt_offset=False, f=1):
    #xrange = np.arange(-1.50, 1.51, 0.005)
    xrange = np.arange(-0.50, 0.51, 0.005)
    nx = len(xrange)
    U, V = np.meshgrid(xrange, xrange)
    displacement_colorplot(ax, U, V, inc3layer, abt_offset, f)
    ax.axis('off')
    ax.set_xlim([-15, nx+15])
    ax.set_ylim([-15, nx+15])

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

def plot_hexagon(ax, nx, ny, data, orientation=0, radius=1/2):
    hex = RegularPolygon((nx/2, ny/2), numVertices=6, radius=radius*nx, fc='none', edgecolor='k', lw=2, orientation=orientation)
    verts = hex.get_path().vertices
    trans = hex.get_patch_transform()
    points = trans.transform(verts)
    for i in range(len(points)):
        old_pt = points[i]
        points[i] = [old_pt[1], old_pt[0]]
    mask = make_contour_mask(nx, ny, points)
    if len(data.shape) == 3:   data[mask <= 0,:] = [1.0, 1.0, 1.0]
    elif len(data.shape) == 2: data[mask <= 0] = 0.0
    ax.imshow(data)
    ax.add_patch(hex)
    return data

def displacement_colorplot(ax, Ux, Uy, inc3layer, abt_offset, f):
    nx, ny = Ux.shape
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1, g2, g1-g2 ]
    gvecs2 = [ g1+g2, 2*g2-g1, 2*g1-g2 ]
    colors1 = np.zeros((nx, ny, 3))
    maxr1, maxr2 = 0,0
    minr1, minr2 = np.inf, np.inf
    for i in range(nx):
        for j in range(ny):
            u = [Ux[i,j], Uy[i,j]]
            if abt_offset:
            	u = np.array(u) + np.array([0, 1/np.sqrt(3)]) 
            r1, r2 = 0, 0
            if inc3layer:
            	for n in range(len(gvecs1)): 
            		r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 * 3/4
            		r1 += ((np.cos(np.pi * np.dot(gvecs1[n], np.array(u)*f))))**2 * 1/4
            	for n in range(len(gvecs2)):  
            		r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 * 3/4
            		r2 += ((np.cos(np.pi * np.dot(gvecs2[n], np.array(u)*f))))**2 * 1/4
            else:
            	for n in range(len(gvecs1)): r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 
            	for n in range(len(gvecs2)): r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 
            r1, r2 = r1/3, r2/3
            if abt_offset: r1 = 1 - r1
            colors1[i,j,0] = r1 
            colors1[i,j,1] = r1/2 + r2/2 
            colors1[i,j,2] = r2 
    f = 2 * np.max(Ux) * g2[0]
    colors1 = plot_hexagon(ax, nx, ny, colors1, radius=1/(2*f), orientation=0) 
    for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(1)

def displacement_lineplot(ax1, ax2, Ux, Uy, abt_offset, f):
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    nx = len(Ux)
    ring1 = np.zeros((nx))
    ring2 = np.zeros((nx))
    color = np.zeros((nx, nx, 3))
    gvecs1 = [ g1, g2, g1-g2 ]
    gvecs2 = [ g1+g2, 2*g2-g1, 2*g1-g2 ]
    for i in range(nx):
        u = [Ux[i], Uy[i]]
        if abt_offset: u = np.array(u) + np.array([0, 1/np.sqrt(3)]) 
        r1, r2 = 0, 0
        for n in range(len(gvecs1)): r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 
        for n in range(len(gvecs2)): r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 
        r1, r2 = r1/3, r2/3
        if abt_offset: r1 = 1 - r1
        ring1[i] = r1 
        ring2[i] = r2 
    ring1 = (ring1 - np.min(ring1))/(np.max(ring1)-np.min(ring1)) 
    ring2 = (ring2 - np.min(ring2))/(np.max(ring2)-np.min(ring2)) 
    for i in range(nx):
	    color[:,i, 0] = ring1[i] 
	    color[:,i, 1] = ring1[i]/2 + ring2[i]/2 
	    color[:,i, 2] = ring2[i] 
    ax1.plot(nx * ring1, c='r')
    ax1.plot(nx * ring1, c='r')
    ax1.plot(nx * ring2, c='k')
    ax1.imshow(color, origin='lower')
    ax2.plot(ring1, c='r')
    ax2.plot(ring2, c='k')
    for i in range(nx): print('{} {} {}'.format(i, ring1[i], ring2[i]))
    #for i in range(nx): ax.scatter(i,0,color=color[i, :]) 

def load_3disk_vdf(filepath):
	with open(filepath, 'rb') as f: diskset = pickle.load(f)
	vdf = overlay_vdf(diskset, plotflag= False)
	dfs = diskset.df_set()
	g = diskset.d_set() 
	ringnos = diskset.determine_rings()
	avgring1 = np.zeros((vdf.shape[0],vdf.shape[1]))  
	avgring2 = np.zeros((avgring1.shape[0], avgring1.shape[1]))
	if dfs.shape[0] != 12: print('warning! not 12 disks for {}'.format(filepath))
	for i in range(dfs.shape[0]):
		if ringnos[i] == 1:
			avgring1 += dfs[i,:avgring1.shape[0],:avgring1.shape[1]]
		elif ringnos[i] == 2:
			avgring2 += dfs[i,:avgring1.shape[0],:avgring1.shape[1]]
	avgring1 = gaussian_filter(avgring1,1)
	avgring1 = avgring1 - np.min(avgring1.flatten())
	avgring1 = avgring1/np.max(avgring1.flatten())
	avgring2 = gaussian_filter(avgring2,1)
	avgring2 = avgring2 - np.min(avgring2.flatten())
	avgring2 = avgring2/np.max(avgring2.flatten())
	return avgring1, avgring2

def manual_define_2pt(img):
    plt.close('all')
    fig, ax = plt.subplots()
    vertices = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        vertices.append([x,y])
        ax.scatter(x,y,color='k')
        fig.canvas.draw()
        if len(vertices) == 2:
        	fig.canvas.mpl_disconnect(cid)
        	plt.close('all')
    print("please click point")
    ax.imshow(img, cmap='gray')
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    return vertices

def make_linecut(img, ptA, ptB, nm_per_pix, ax):
	ax[0].imshow(img)
	ax[0].scatter(ptA[0], ptA[1], c='r')
	ax[0].scatter(ptB[0], ptB[1], c='r')
	ax[0].plot([ptA[0], ptB[0]], [ptA[1], ptB[1]], c='r')
	N = 200
	x, y = np.linspace(ptA[0], ptB[0], N), np.linspace(ptA[1], ptB[1], N)
	path_len = nm_per_pix * ((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2 ) ** 0.5
	zi = scipy.ndimage.map_coordinates(np.transpose(img), np.vstack((x,y))) # extract values along line w/ cubic interpolation
	d = np.linspace(0, path_len, N)
	ax[1].plot(d, zi, 'k')
	ax[1].set_title('nm')
	return x,y,d,zi
	

if __name__ == "__main__":

	f,ax = plt.subplots(3,2)
	ax = ax.flatten()
	make_legend(ax[4], inc3layer=False, abt_offset=True)
	make_legend(ax[5], inc3layer=False, abt_offset=False)
	xrange = np.arange(0, 3, 0.01)
	U, V = np.meshgrid(xrange, xrange)
	halfway = int(U.shape[0]/2)
	displacement_lineplot(ax[0], ax[2], U[:,halfway], V[:,halfway], abt_offset=True, f=1)
	displacement_lineplot(ax[1], ax[3], U[:,halfway], V[:,halfway], abt_offset=False, f=1)
	plt.show()
	exit()

	filepath = '/Users/isaaccraig/Desktop/TLGproj/{}/dat_ds{}.pkl'.format('ABt-nd1', 8) #c7, 2
	avgring1, avgring2 = load_3disk_vdf(filepath)
	nm_per_pix = 0.5
	pts = manual_define_2pt(avgring1)
	fig, ax = plt.subplots(2,2)
	x,y,d,I1 = make_linecut(avgring1, pts[0], pts[1], nm_per_pix, ax[0,:])
	x,y,d,I2 = make_linecut(avgring2, pts[0], pts[1], nm_per_pix, ax[1,:])
	plt.show()
	N = 200
	for i in range(N): print("{} {} {} {} {}".format(x[i], y[i], d[i], I1[i], I2[i]))



