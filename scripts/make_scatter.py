
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy.ndimage import gaussian_filter
from matplotlib.patches import RegularPolygon
from visualization import overlay_vdf
from diskset import DiskSet
from masking import make_contour_mask

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

def make_histoscatter(I1, I2, fig, ax):

	N = 80
	counts = np.zeros((N+1,N+1))	
	for x in range(I1.shape[0]):
		for y in range(I1.shape[1]):
			if not np.isnan(I1[x,y]):
				I1_ind = int(np.round(I1[x,y] * N, 1))
				I2_ind = int(np.round(I2[x,y] * N, 1))
				counts[I1_ind, I2_ind] += 1

	im = ax.imshow(counts, cmap='inferno', origin='lower')
	ax.set_xlabel('I1')
	ax.set_ylabel('I2')
	fig.colorbar(im, ax=ax)

def rigid_map(inc3layer=False, abt_offset=True, f=1):
    xrange = np.arange(-3.5, 3.55, 0.01)
    nx = len(xrange)
    Ux, Uy= np.meshgrid(xrange, xrange)
    nx, ny = Ux.shape
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1, g2, g1-g2 ]
    gvecs2 = [ g1+g2, 2*g2-g1, 2*g1-g2 ]
    colors1 = np.zeros((nx, ny, 3))
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
            
            r1, r2 = (r1-3/4)/(3-3/4), (r2-3/4)/(3-3/4)
            if abt_offset: r1 = 1 - r1	
            colors1[i,j,0] = r1 
            colors1[i,j,1] = r1/2 + r2/2 
            colors1[i,j,2] = r2 

    f = 2 * np.max(Ux) * g2[0]
    #colors1 = crop_hexagon(nx, ny, colors1, radius=( 1/(2*f) ), orientation=0) 
    return colors1

def crop_hexagon(nx, ny, data, orientation=0, radius=1/2):
    hex = RegularPolygon((nx/2, ny/2), numVertices=6, radius=radius*nx, fc='none', edgecolor='k', lw=2, orientation=orientation)
    verts = hex.get_path().vertices
    trans = hex.get_patch_transform()
    points = trans.transform(verts)
    for i in range(len(points)):
        old_pt = points[i]
        points[i] = [old_pt[1], old_pt[0]]
    mask = make_contour_mask(nx, ny, points)
    data[mask <= 0,:] = np.nan
    return data
	
if __name__ == "__main__":
	
	filepath = '/Users/isaaccraig/Desktop/TLGproj/data/{}/dat_ds{}.pkl'.format('ABt-nd1', 8) 
	avgring1, avgring2 = load_3disk_vdf(filepath)
	fig, ax = plt.subplots(1,1)
	make_histoscatter(avgring1, avgring2, fig, ax)
	plt.savefig("/Users/isaaccraig/Desktop/histo-abt.svg", dpi=300)

	filepath = '/Users/isaaccraig/Desktop/TLGproj/data/{}/dat_ds{}.pkl'.format('c7', 2) 
	avgring1, avgring2 = load_3disk_vdf(filepath)
	fig, ax = plt.subplots(1,1)
	make_histoscatter(avgring1, avgring2, fig, ax)
	plt.savefig("/Users/isaaccraig/Desktop/histo-ata.svg", dpi=300)

	exit()
	colors = rigid_map(inc3layer=False, abt_offset=True, f=1)
	avgring1, avgring2 = colors[:,:,0], colors[:,:,2]
	fig, ax = plt.subplots(1,1)
	ax.imshow(colors)
	plt.savefig("/Users/isaaccraig/Desktop/rigid-abt.svg", dpi=300)
	fig, ax = plt.subplots(1,1)
	make_histoscatter(avgring1, avgring2, fig, ax)
	plt.savefig("/Users/isaaccraig/Desktop/histo-rigid-abt.svg", dpi=300)
	plt.show()






