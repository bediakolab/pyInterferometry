
import pickle
import numpy as np
import matplotlib.pyplot as plt
from new_utils import parse_filepath, crop_image, manual_define_pt
import os
import sys
import gc
from visualization import overlay_vdf, plot_hexagon
from diskset import DiskSet
from scipy.ndimage import gaussian_filter
from masking import make_contour_mask

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path

def make_legend_thresh(ax, thresh1, thresh2):
    xrange = np.arange(-0.50, 0.51, 0.005)
    nx = len(xrange)
    U, V = np.meshgrid(xrange, xrange)
    displacement_colorplot_thresh(ax, U, V, thresh1, thresh2)
    ax.axis('off')
    ax.set_xlim([-15, nx+15])
    ax.set_ylim([-15, nx+15])

def make_legend_figures():

	f,ax = plt.subplots(1,2)
	ax = ax.flatten()
	#make_legend(ax[0], inc3layer=True, f=4)
	#make_legend(ax[1], inc3layer=True, f=6)
	#make_legend(ax[2], inc3layer=True, f=8)
	#make_legend(ax[3], inc3layer=True, f=10)
	make_legend(ax[0], inc3layer=False, abt_offset=True)
	make_legend(ax[1], inc3layer=False, abt_offset=False)
	plt.savefig("/Users/isaaccraig/Desktop/legend.svg", dpi=300)
	plt.show()

	#legend = np.zeros((50, 50, 3))
	#for i in range(50):
	#	for j in range(50):
	#		legend[i,j,0] = i/50
	#		legend[i,j,2] = j/50
	#		legend[i,j,1] = 0.5*i/50 + 0.5*j/50				
	#f,ax = plt.subplots(1,1)
	#ax.imshow(legend, origin='lower')
	#plt.savefig("/Users/isaaccraig/Desktop/legend.svg", dpi=300)

def make_legend(ax, inc3layer=False, abt_offset=False, f=1):
    #xrange = np.arange(-1.50, 1.51, 0.005)
    xrange = np.arange(-0.50, 0.51, 0.005)
    nx = len(xrange)
    U, V = np.meshgrid(xrange, xrange)
    displacement_colorplot(ax, U, V, inc3layer, abt_offset, f)
    ax.axis('off')
    ax.set_xlim([-15, nx+15])
    ax.set_ylim([-15, nx+15])

def displacement_colorplot_thresh(ax, Ux, Uy, thresh1=0.5, thresh2=0.5):
    nx, ny = Ux.shape
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1, g2, g1-g2 ]
    gvecs2 = [ g1+g2, 2*g2-g1, 2*g1-g2 ]
    colors1 = np.zeros((nx, ny, 3))
    maxr1, maxr2 = 0,0
    for i in range(nx):
        for j in range(ny):
            u = [Ux[i,j], Uy[i,j]]
            umag = (u[0]**2 + u[1]**2)**0.5
            r1, r2 = 0, 0
            for n in range(len(gvecs1)): r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 
            for n in range(len(gvecs2)): r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 
            r1, r2 = r1/3, r2/3
            if r1 > maxr1: maxr1 = r1
            if r2 > maxr2: maxr2 = r2
            if r1 > thresh1:    
                if r2 > thresh2:
                    colors1[i,j,:] = [169/255,169/255,169/255] #w
                else:
                    colors1[i,j,:] = [255/255, 165/255, 0/255] #w
            else: 
                if r2 > thresh2: colors1[i,j,:] = [0,0,1] #b
                else: colors1[i,j,:] = [0,0,0] #k
    print(maxr1,maxr2)
    f = 2 * np.max(Ux) * g2[0]
    colors1 = plot_hexagon(ax, nx, ny, colors1, radius=1/(2*f), orientation=0)
    for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)

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
            
            r1, r2 = (r1-3/4)/(3-3/4), (r2-3/4)/(3-3/4)
            if abt_offset: r1 = 1 - r1	
            colors1[i,j,0] = r1 
            colors1[i,j,1] = r1/2 + r2/2 
            colors1[i,j,2] = r2 
    if False: #rescales
	    for i in range(nx):
	        for j in range(ny):
	            #minr1, minr2, maxr1, maxr2 = 0.0, 0.0, 0.85, 1.0
	            minr1, minr2, maxr1, maxr2 = 3/4, 3/4, 3, 3
	            colors1[i,j,0] = (colors1[i,j,0] - minr1)/(maxr1-minr1)
	            colors1[i,j,2] = (colors1[i,j,2] - minr2)/(maxr2-minr2)
	            colors1[i,j,1] = (colors1[i,j,0] + colors1[i,j,2])/2

    print(np.max(colors1[:,:,2].flatten()),np.max(colors1[:,:,0].flatten()), np.min(colors1[:,:,2].flatten()), np.min(colors1[:,:,0].flatten()))
    f = 2 * np.max(Ux) * g2[0]
    #colors1 = plot_hexagon(ax, nx, ny, colors1, radius=3*1/(2*f), orientation=0) 
    colors1 = plot_hexagon(ax, nx, ny, colors1, radius=1/(2*f), orientation=0) 
    for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(1)

area_stats = True #if false just makes colored else shows all stat stuff

make_legend_figures()

if False: # plot stacking areas from 1/2 overlap region

	thresh1 = 0.5
	thresh2 = 0.5
	sample = '6c' #'a5'
	alldsnum = [12,11,4,5,7,10] #[1,2,3,4,5,6,7,8,10,11,12,13,15,16,17,19,20]
	cAA, cAB, cABSP, cAASP = [1,1,1], [0,0,1], [0,0,0], [1, 165/255, 0]
	print('ds \t\t aa \t\t ab \t\t ab-sp \t\t aa-sp')

	for dsnum in alldsnum: #alldsnum: 
		
		# indeces for the 2disk overlap of the bigger moire

		filepath = '/Users/isaaccraig/Desktop/TLGproj/{}/ring12-masks/vdf-ds{}-rings12.pkl'.format(sample, dsnum)
		with open(filepath, 'rb') as f: dfs = pickle.load(f)

		if sample == 'a5':
		
			if dsnum == 1:
				ring1 = [22,8,20, 10,11,24]
				ring2 = [2,4,15, 16,26,28]
				dfs = dfs[:,:-10,10:]
				dfs = dfs[:,25:,:]
			elif dsnum == 2:
				ring1 = [24,20,11,7,10,22]
				ring2 = [15,27,28,16,2,4]
				dfs[:,40:60,100:120] = np.nan * np.ones((20,20))
				dfs = dfs[:,25:,:]
			elif dsnum == 3:
				ring1 = [8,11,21,20,24,10]
				ring2 = [15,16,2,4,27,28]
				dfs[:,50:70,70:90] = np.nan * np.ones((20,20))
			elif dsnum == 4:
				ring1 = [8,10,11,20,21,24]
				ring2 = [15,16,2,4,26,28]
			elif dsnum == 5:
				ring1 = [20,10,7,23,11] #missing disk!
				ring2 = [15,26,27,16,2,4]
			elif dsnum == 6: 
				ring1 = [10,7,11,20,22,24]
				ring2 = [27,15,2,4,16,28]
			elif dsnum == 7:
				ring1 = [11,22,24,20,10,8]
				ring2 = [15,16,2,4,26,28]
				dfs = dfs[:,50:,:] #crop edge weird
			elif dsnum == 8: 
				ring1 = [7,9,22,20,24,11] 
				ring2 = [3,4,16,15,27] #missing disk!
			elif dsnum == 10:
				ring1 = [8,11,22,20,10,24] 
				ring2 = [2,4,16,28,26,15]
				dfs = dfs[:,50:,:] #crop edge weird
			elif dsnum == 11:
				ring1 = [7,10,11,20,22,24] 
				ring2 = [2,4,15,16,28,26]
			elif dsnum == 12:
				ring1 = [8,11,21,20,24,10]
				ring2 = [16,27,28,15,4,3]
			elif dsnum == 13:
				ring1 = [7,10,22,20,11,24]
				ring2 = [4,3,16,15,27,28]
			elif dsnum == 15:
				ring1 = [8,9,11,20,22,24]
				ring2 = [3,4,15,16,27,28]
			elif dsnum == 16:
				ring1 = [8,10,11,20,21,24]
				ring2 = [15,16,3,4,27,28]
			elif dsnum == 17:
				ring1 = [7,10,11,20,21,24]
				ring2 = [15,16,3,4,27,28]
			elif dsnum == 19:
				ring1 = [7,10,11,20,21,23] 
				ring2 = [2,4,15,16,27,28]
			elif dsnum == 20:
				ring1 = [7,10,11,20,22,24]
				ring2 = [3,4,15,16,27,28]
			else:
				ring1 = None
				ring2 = None

		elif sample == '6c':
			if dsnum == 4:
				ring1 = [9,10,15,22,27,29]
				ring2 = [1,5,16,21,31,35]
			elif dsnum == 5:
				ring1 = [9,10,15,22,27,29]
				ring2 = [1,6,16,21,31,34]
			elif dsnum == 7:
				ring1 = [9,10,15,22,27,28]
				ring2 = [1,6,16,21,31,36]
			elif dsnum == 10:
				ring1 = [30,10,15] #[9,10,15,22,27,30]
				ring2 = [1,5,16,21,31,35]
			elif dsnum == 11:
				ring1 = [8,15,27]#[8,10,15,22,27,28]
				ring2 = [6,21,35,1,6,16,21,31,35]
			elif dsnum == 12:
				ring1 = [10,22,30,27,15,9]
				ring2 = [31,16,24,21,5,1]

		avgring1 = np.zeros((dfs.shape[1], dfs.shape[2]))
		avgring2 = np.zeros((dfs.shape[1], dfs.shape[2]))
		stack_assign = np.zeros((dfs.shape[1], dfs.shape[2], 3))
		stack_assign2 = np.zeros((dfs.shape[1], dfs.shape[2], 3))
		stack_assign3 = np.zeros((dfs.shape[1], dfs.shape[2], 3))
		stack_assign4 = np.zeros((dfs.shape[1], dfs.shape[2], 3))

		ata_mask = np.zeros((dfs.shape[1], dfs.shape[2]))
		atb_mask = np.zeros((dfs.shape[1], dfs.shape[2]))
		sp_mask = np.zeros((dfs.shape[1], dfs.shape[2]))

		legend2 = np.zeros((50, 50, 3))

		if ring1 == None:

			filepath = '/Users/isaaccraig/Desktop/TLGproj/{}/ring12-masks/mask_ds{}-rings12.pkl'.format(sample, dsnum)
			with open(filepath, 'rb') as f: mask = pickle.load(f)
			f, ax = plt.subplots(1,1)
			ax.imshow(mask)
			plt.show()
			ndisks = dfs.shape[0]
			f, ax = plt.subplots(int(np.ceil(ndisks**0.5)), int(np.ceil(ndisks**0.5)))
			ax = ax.flatten()
			for i in range(ndisks):
				ax[i].imshow(dfs[i,:,:])
				ax[i].set_title('{}'.format(i))
			plt.show()
			exit()

		else:

			for i in ring1: avgring1 += dfs[i,:,:]
			for i in ring2: avgring2 += dfs[i,:,:]

			avgring1 = gaussian_filter(avgring1,1)
			avgring1 = avgring1 - np.nanmin(avgring1.flatten())
			avgring1 = avgring1/np.nanmax(avgring1.flatten())
			avgring2 = gaussian_filter(avgring2,1)
			avgring2 = avgring2 - np.nanmin(avgring2.flatten())
			avgring2 = avgring2/np.nanmax(avgring2.flatten())

			stack_assign[:,:,0] = avgring1[:,:] # r channel
			stack_assign[:,:,2] = avgring2[:,:] # b channel
			stack_assign[:,:,1] = 0.5 * avgring1[:,:] + 0.5 * avgring2[:,:]

			Ntot, NAB, NAA, NAASP, NABSP = 0,0,0,0,0

			for i in range(stack_assign.shape[0]):
				for j in range(stack_assign.shape[1]):
					if (avgring1[i,j]) > thresh1: 
						stack_assign3[i,j,:] = [1,1,1]
						if (avgring2[i,j]) > thresh2: 
							stack_assign2[i,j,:] = cAA
							stack_assign4[i,j,:] = cAA
							NAA+=1
							ata_mask[i,j] = 1
						else: 
							stack_assign2[i,j,:] = cAASP
							NAASP+=1
							sp_mask[i,j] = 1
					else: 
						if (avgring2[i,j]) > thresh2: 
							stack_assign2[i,j,:] = cAB
							stack_assign4[i,j,:] = cAB
							NAB+=1
							atb_mask[i,j] = 1
						else: 
							stack_assign2[i,j,:] = cABSP
							NABSP+=1
							sp_mask[i,j] = 1
					Ntot+=1

			legend = np.zeros((50, 50, 3))
			for i in range(50):
				for j in range(50):
					legend[i,j,0] = i/50
					legend[i,j,2] = j/50
					legend[i,j,1] = 0.5*i/50 + 0.5*j/50				
					if (i/50) > thresh1: 
						if (j/50) > thresh2: legend2[i,j,:] = cAA
						else: legend2[i,j,:] = cAASP
					else: 
						if (j/50) > thresh2: legend2[i,j,:] = cAB
						else: legend2[i,j,:] = cABSP

			if area_stats:
				f,ax = plt.subplots(2,4)
				ax = ax.flatten()
				ax[0].imshow(avgring1, origin='lower',cmap='gray')
				ax[1].imshow(avgring2, origin='lower',cmap='gray')
				ax[2].imshow(stack_assign, origin='lower')
				ax[3].imshow(legend, origin='lower')
				#make_legend(ax[3], thresh1, thresh2)
				ax[4].imshow(stack_assign2, origin='lower')
				ax[3].axis('off')
				ax[5].imshow(legend2, origin='lower')
				ax[6].imshow(stack_assign3, origin='lower')
				ax[7].imshow(stack_assign4, origin='lower')
				print('{} \t\t {} \t\t {} \t\t {} \t\t {}'.format(dsnum,NAA/Ntot,NAB/Ntot,NABSP/Ntot,NAASP/Ntot))
				plt.show()

				plt.close('all')
				f,ax = plt.subplots(2,2)
				ax[0,0].imshow(ata_mask)
				ax[0,1].imshow(atb_mask)
				ax[1,0].imshow(sp_mask)
				plt.savefig('/Users/isaaccraig/Desktop/stacking{}masks.png'.format(dsnum), dpi=150)
				fp = '/Users/isaaccraig/Desktop/stacking{}masks.pkl'.format(dsnum)
				with open(fp, 'wb') as f: pickle.dump({'ata':ata_mask, 'atb': atb_mask, 'sp': sp_mask}, f )

			else:
				f,ax = plt.subplots()
				ax.imshow(stack_assign, origin='lower')
				plt.savefig('/Users/isaaccraig/Desktop/stacking{}.svg'.format(dsnum), dpi=1200)

if False: # plot stacking areas from 1/2/3 overlap region 

	thresh1 = [0.15, 0.65] # ring 1 - for ABC   vs ABB  vs ABC
	thresh2 = [0.15, 0.65] # ring 2 - for SP-SP vs SP   vs non-SP
	sample = 'a5'#'c7'
	a5ds = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,19]
	c6ds = [1,2,3,4,5,6,7,8,12,13,16]
	dsnums = a5ds

	print('ds \t\t aaa \t\t aab \t\t abc \t\t spaa \t\t spab \t\t spsp')

	for dsnum in a5ds:

		#try:
		filepath = '/Users/isaaccraig/Desktop/TLGproj/{}/dat_ds{}.pkl'.format(sample, dsnum)
		with open(filepath, 'rb') as f: diskset = pickle.load(f)
		vdf = overlay_vdf(diskset, plotflag= False)
		dfs = diskset.df_set()
		g = diskset.d_set() #diskset.clean_normgset() 
		f,ax = plt.subplots(1,4)
		ringnos = diskset.determine_rings()
		avgring1 = np.zeros((vdf.shape[0],vdf.shape[1]))  #for a5#20 used np.zeros((vdf.shape[0],150))
		avgring2 = np.zeros((avgring1.shape[0], avgring1.shape[1]))
		stack_assign = np.zeros((avgring1.shape[0], avgring1.shape[1], 3))
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

		Ntot, N_AAA, N_ABA, N_ABC, N_SPSP, N_SPAA, N_SPAB = 0, 0, 0, 0, 0, 0, 0
		cAAA, cABA, cABC, cSPSP, cSPAA, cSPAB = [1,1,1], [0,0,1], [0,0,0.5], [0,0,0], [1, 165/255, 0], [0.5,0,0]

		stack_assign2 = np.zeros((stack_assign.shape[0], vdf.shape[1], 3))
		stack_assign3 = np.zeros((stack_assign.shape[0], stack_assign.shape[1], 3))
		stack_assign4 = np.zeros((stack_assign.shape[0], stack_assign.shape[1], 3))
		legend2 = np.zeros((50, 50, 3))

		def colorpartition(avr1, avr2):
			if avr1 > thresh1[1]: 
				if avr2 > thresh2[1]:   return cAAA
				elif avr2 > thresh2[0]: return cSPAA
				else:                   return cSPSP
			elif avr1 > thresh1[0]:
				if   avr2 > thresh2[1]: return cABA
				elif avr2 > thresh2[0]: return cSPAB
				else:                   return cSPSP
			else:                       return cABC

		def countpartition(avr1, avr2, counts):
			counts[0] += 1
			if avr1 > thresh1[1]: 
				if avr2 > thresh2[1]:    counts[1] += 1
				elif avr2 > thresh2[0]:  counts[5] += 1
				else:                    counts[4] += 1
			elif avr1 > thresh1[0]:
				if   avr2 > thresh2[1]:  counts[2] += 1
				elif avr2 > thresh2[0]:  counts[6] += 1
				else:                    counts[4] += 1
			else:                        counts[3] += 1
			return counts

		counts = [Ntot, N_AAA, N_ABA, N_ABC, N_SPSP, N_SPAA, N_SPAB]
		for i in range(stack_assign.shape[0]):
			for j in range(stack_assign.shape[1]):
				stack_assign2[i,j,:] = colorpartition(avgring1[i,j], avgring2[i,j])
				counts = countpartition(avgring1[i,j], avgring2[i,j], counts)
		Ntot, N_AAA, N_ABA, N_ABC, N_SPSP, N_SPAA, N_SPAB = counts[:]

		legend = np.zeros((50, 50, 3))
		for i in range(50):
			for j in range(50):	
				avr1 = (i/50)	
				avr2 = (j/50)
				legend2[i,j,:] = colorpartition(avr1, avr2)

		stack_assign[:,:,0] = avgring1[:,:] # r channel
		stack_assign[:,:,2] = avgring2[:,:] # b channel
		stack_assign[:,:,1] = 0.5 * avgring1[:,:] + 0.5 * avgring2[:,:]

		if area_stats:
			f,ax = plt.subplots(2,4)
			ax = ax.flatten()
			ax[0].imshow(avgring1, origin='lower',cmap='gray')
			ax[1].imshow(avgring2, origin='lower',cmap='gray')
			ax[4].imshow(stack_assign2, origin='lower')
			#make_legend_thresh(ax[3], thresh1, thresh2)
			
			ax[5].imshow(legend2, origin='lower')
			ax[0].set_title('aaa-{}'.format(N_AAA/Ntot))
			ax[1].set_title('aab-{}'.format(N_ABA/Ntot))
			ax[2].set_title('abc-{}'.format(N_ABC/Ntot))
			ax[3].set_title('ds-{}'.format(dsnum))
			ax[4].set_title('spaa-{}'.format(N_SPAA/Ntot))
			ax[5].set_title('spab-{}'.format(N_SPAB/Ntot))
			ax[6].set_title('spsp-{}'.format(N_SPSP/Ntot))

			print('{} \t\t {} \t\t {} \t\t {} \t\t {} \t\t {} \t\t {}'.format(dsnum, N_AAA/Ntot, 
				 N_ABA/Ntot, N_ABC/Ntot, N_SPAA/Ntot, N_SPAB/Ntot, N_SPSP/Ntot))
			
			plt.savefig('/Users/isaaccraig/Desktop/stacking_{}_ds{}_gpart.png'.format(sample, dsnum), dpi=300)
			plt.close('all')
			#exit()

		else:
			f,ax = plt.subplots()
			ax.imshow(stack_assign, origin='lower')	
			#ax.axis('off')
			plt.savefig('/Users/isaaccraig/Desktop/stacking_{}_ds{}_3layer.svg'.format(sample, dsnum), dpi=1200)

		#ax[0].imshow(avgring1, origin='lower',cmap='inferno')
		#ax[1].imshow(avgring2, origin='lower',cmap='inferno')
		#ax[2].imshow(stack_assign, origin='lower')
		#ax[3].imshow(legend, origin='lower')
		#plt.show()   
		
if False: #plot stacking from 1/2/3 overlap region after thresholding AtA, AtB with 2disk overlap

	thresh1 = 0.5
	thresh2 = 0.5
	sample = '6c'

	for dsnum in [4,5,7,10,11] : # 

		#try:
		filepath = '/Users/isaaccraig/Desktop/TLGproj/{}/dat_ds{}.pkl'.format(sample, dsnum)
		with open(filepath, 'rb') as f: diskset = pickle.load(f)
		vdf = overlay_vdf(diskset, plotflag= False)
		dfs = diskset.df_set()
		g = diskset.d_set() #diskset.clean_normgset() 
		f,ax = plt.subplots(1,4)
		ringnos = diskset.determine_rings()
		if dfs.shape[0] != 12: print('warning! not 12 disks for {}'.format(filepath))

		if sample == 'a5':
			if dsnum == 1:
				vdf = vdf[:-10,10:]
				dfs = dfs[:,:-10,10:]
				vdf = vdf[25:,:]
				dfs = dfs[:,25:,:]
			elif dsnum == 2:
				vdf = vdf[25:,:]
				dfs = dfs[:,25:,:]
			elif dsnum == 7:
				vdf = vdf[50:,:] #crop edge weird
				dfs = dfs[:,50:,:]
			elif dsnum == 10:
				vdf = vdf[50:,:] #crop edge weird
				dfs = dfs[:,50:,:]

		maskpath = '/Users/isaaccraig/Desktop/TLGproj/{}/stacking{}masks.pkl'.format(sample, dsnum)
		with open(maskpath, 'rb') as f: masks = pickle.load(f)
		stack_types = masks.keys()

		avgring1 = np.zeros((vdf.shape[0], vdf.shape[1]))
		avgring2 = np.zeros((avgring1.shape[0], avgring1.shape[1]))
		for i in range(dfs.shape[0]):
			if ringnos[i] == 1:   avgring1 += dfs[i,:avgring1.shape[0],:avgring1.shape[1]]
			elif ringnos[i] == 2: avgring2 += dfs[i,:avgring1.shape[0],:avgring1.shape[1]]
		avgring1 = gaussian_filter(avgring1,1)
		avgring2 = gaussian_filter(avgring2,1)

		for stack_type in stack_types:

			stack_assign = np.zeros((avgring1.shape[0], avgring1.shape[1], 3))
			mask = masks[stack_type]

			avgring1_m = np.zeros((avgring1.shape))
			avgring2_m = np.zeros((avgring1.shape))

			for i in range(avgring1.shape[0]):
				for j in range(avgring2.shape[0]):
					if mask[i,j]:
						avgring1_m[i,j] = avgring1[i,j]
						avgring2_m[i,j] = avgring2[i,j]
					else:
						avgring1_m[i,j] = np.nan
						avgring2_m[i,j] = np.nan

			avgring1_m = avgring1_m - np.nanmin(avgring1_m.flatten())
			avgring2_m = avgring2_m - np.nanmin(avgring2_m.flatten())
			avgring1_m = avgring1_m/np.nanmax(avgring1_m.flatten())
			avgring2_m = avgring2_m/np.nanmax(avgring2_m.flatten())

			Ntot, NAB, NAA, NAASP, NABSP = 0,0,0,0,0
			stack_assign2 = np.zeros((stack_assign.shape[0], vdf.shape[1], 3))
			stack_assign4 = np.zeros((stack_assign.shape[0], stack_assign.shape[1], 3))
			for i in range(stack_assign.shape[0]):
				for j in range(stack_assign.shape[1]):
					if np.isnan(avgring1_m[i,j]): 
						stack_assign2[i,j,:] = [1,1,1]
					elif (avgring1_m[i,j]) > thresh1: 
						Ntot+=1
						if (avgring2_m[i,j]) > thresh2: 
							stack_assign2[i,j,:] = [0.8,0.8,0.8]
							NAA+=1
						else:
							stack_assign2[i,j,:] = [1, 165/255, 0]
							NAASP+=1
					else: 
						Ntot+=1
						if (avgring2_m[i,j]) > thresh2: 
							stack_assign2[i,j,:] = [0,0,1]
							NAB+=1
						else: 
							stack_assign2[i,j,:] = [0,0,0]
							NABSP+=1
	
			stack_assign[:,:,0] = avgring1_m[:,:] # r channel
			stack_assign[:,:,2] = avgring2_m[:,:] # b channel
			stack_assign[:,:,1] = 0.5 * avgring1_m[:,:] + 0.5 * avgring2_m[:,:]

			if True:
				print('within {} of ds{}:  aa \t\t ab \t\t sp'.format(stack_type, dsnum))
				print(NAA/Ntot, NAB/Ntot, NAASP/Ntot, NABSP/Ntot)
				f,ax = plt.subplots(1,2)
				ax[0].imshow(stack_assign, origin='lower')
				ax[1].imshow(stack_assign2, origin='lower')
				ax[0].set_title(stack_type)
				plt.savefig('/Users/isaaccraig/Desktop/within-Ata-Atb/{}_ds{}_{}.png'.format(sample, dsnum, stack_type), dpi=200)
				plt.close('all')
				#exit()

			#ax[0].imshow(avgring1, origin='lower',cmap='inferno')
			#ax[1].imshow(avgring2, origin='lower',cmap='inferno')
			#ax[2].imshow(stack_assign, origin='lower')
			#ax[3].imshow(legend, origin='lower')
			#plt.show()   

			#f,ax = plt.subplots()
			#ax.imshow(stack_assign, origin='lower')
			#plt.savefig('/Users/isaaccraig/Desktop/stacking_c7ds{}_3layer.png'.format(dsnum), dpi=1200)
			#except:print('failed with ds{}'.format(dsnum))

if False: #loading_diskset_obj:

	filepath = '/Users/isaaccraig/Desktop/TLGproj/ds13/dat_ds13.pkl'
	with open(filepath, 'rb') as f: diskset = pickle.load(f)
	vdf = overlay_vdf(diskset, plotflag= False)
	dfs = diskset.df_set()
	g = diskset.d_set() #diskset.clean_normgset() 
	x,y = manual_define_pt(vdf) #30,50 
	f,ax = plt.subplots(4,4)
	ax = ax.flatten()
	ax[0].imshow(vdf, origin='lower')
	ax[0].scatter(x,y,c='r')
	for i in range(12):
	    ax[i+1].imshow(dfs[i,:,:], origin='lower', cmap='inferno', vmin=0, vmax=2e3)
	    ax[i+1].scatter(x,y,c='r')
	    ax[i+1].set_title(i)
	    gvec = g[i]
	    ax[-1].scatter(gvec[0], gvec[1], c=dfs[i,y,x], vmin=0, vmax=2e3, cmap='inferno')
	    ax[-1].text(gvec[0], gvec[1], i)
	plt.show()

if False: #loading_vdf_obj:

	path = "/Users/isaaccraig/Desktop/TLGproj/ds13/"
	maskpath = os.path.join(path, "mask.pkl")
	vdfpath  = os.path.join(path, "vdf.pkl")
	with open(maskpath, 'rb') as f: mask  = pickle.load(f)
	with open(vdfpath, 'rb') as f: dfs = pickle.load(f)

	nx, ny = dfs.shape[1], dfs.shape[2]
	alldisk = np.zeros((nx,ny))
	localdp = np.zeros((mask.shape[0],mask.shape[1]))
	for i in range(dfs.shape[0]): alldisk[:,:] += dfs[i,:,:]

	x,y = 30,50
	f,ax = plt.subplots(5,5)
	ax = ax.flatten()
	for i in range(dfs.shape[0]):
	    ax[i].imshow(dfs[i,:,:], origin='lower', cmap='inferno', vmin=0, vmax=10)
	    ax[i].scatter(x,y,c='r')
	    ax[i].set_title(i)
	    localdp += (mask == i) * dfs[i,y,x]
	f,ax = plt.subplots(1,3)
	ax[0].imshow(alldisk, origin='lower', cmap='gray')
	ax[0].scatter(x,y,c='r')    
	ax[1].imshow(localdp, origin='lower', cmap='inferno', vmin=0, vmax=10)
	ax[2].imshow(mask, origin='lower', cmap='inferno')
	plt.show()
			
if False: # heterostrain gaus fit from 1/2/3 overlap  

	thresh1 = 0.5
	thresh2 = 0.5
	sample = '6c' # 'ABt-nd1'#'c7' #'ABt-nd1' #'a5'
	area_stats = True #if false just makes colored

	#done:2,5,11,16
	#tbd:1,4,6,7,8,9,10,12,13
	# missing disk err 8,9

	for dsnum in [8]:# [8,9] : # 13,6,12,16,17,20

		if sample == '6c':
			if dsnum in [1,3,4,6,7,9,10]: SS = 1
			elif dsnum in [12,13]: SS = 2
			else: SS = 0.5
		if sample == '6a':
			if dsnum in [6,7,8]: SS = 1
			else: SS = 0.5
		else: SS=0.5

		filepath = '/Users/isaaccraig/Desktop/TLGproj/{}/dat_ds{}.pkl'.format(sample, dsnum)
		with open(filepath, 'rb') as f: diskset = pickle.load(f)
		vdf = overlay_vdf(diskset, plotflag= False)
		dfs = diskset.df_set()
		g = diskset.d_set() #diskset.clean_normgset() 
		f,ax = plt.subplots(1,4)
		ringnos = diskset.determine_rings()
		avgring1 = np.zeros((vdf.shape[0],vdf.shape[1]))  #for a5#20 used np.zeros((vdf.shape[0],150))
		avgring2 = np.zeros((vdf.shape[0],vdf.shape[1]))  #for a5#20 used np.zeros((vdf.shape[0],150))
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
		
		vdf = avgring1
		f,ax = plt.subplots(1,2)
		ax = ax.flatten()
		ax[0].imshow(vdf, origin='lower',cmap='gray')
		ax[1].imshow(avgring2, origin='lower',cmap='gray')
		plt.show()

		from heterostrain_new import extract_heterostrain_vdf
		from STMtriangulate import main_vdf
		extract_heterostrain_vdf(vdf, dsnum, prefix='/Users/isaaccraig/Desktop/TLGproj/{}'.format(sample), ss=SS)
		#main_vdf(filepath, vdf, FOV_len=SS*vdf.shape[0])

if False: # heterostrain gaus fit from 1/2 overlap  

	sample = 'a5'
	for dsnum in [7]:#[6,12,13,16,15,17]:

		if sample == 'a5':
			if dsnum == 6:
				ring1 = [10,7,11,20,22,24]
				ring2 = [27,15,2,4,16,28]
				SS = 0.5
			elif dsnum == 12:
				ring1 = [8,11,21,20,24,10]
				ring2 = [16,27,28,15,4,3]
				SS = 0.5
			elif dsnum == 13:
				ring1 = [7,10,22,20,11,24]
				ring2 = [4,3,16,15,27,28]
				SS = 0.5
			elif dsnum == 15:
				ring1 = [8,9,11,20,22,24]
				ring2 = [3,4,15,16,27,28]
				SS = 0.5
			elif dsnum == 16:
				ring1 = [8,10,11,20,21,24]
				ring2 = [15,16,3,4,27,28]
				SS = 0.5
			elif dsnum == 17:
				ring1 = [7,10,11,20,21,24]
				ring2 = [15,16,3,4,27,28]
				SS = 0.5
			elif dsnum == 20:
				ring1 = [7,10,11,20,22,24]
				ring2 = [3,4,15,16,27,28]
				SS = 0.5
			elif dsnum == 11:
				ring1 = [7,10,11,20,22,24] 
				ring2 = [2,4,15,16,28,26]
				SS = 0.5
			elif dsnum == 7:
				ring1 = [11,22,24,20,10,8]
				ring2 = [15,16,2,4,26,28]
				SS = 0.5
		elif sample == '6c':
			if dsnum == 4:
				ring1 = [9,10,15,22,27,29]
				ring2 = [1,5,16,21,31,35]
				SS = 1
			elif dsnum == 7:
				ring1 = [9,10,15,22,27,28]
				ring2 = [1,6,16,21,31,36]
				SS = 1
			elif dsnum == 10:
				ring1 = [30,10,15] #[9,10,15,22,27,30]
				ring2 = [1,5,16,21,31,35]
				SS = 1
			elif dsnum == 12:
				ring1 = [10,22,30,27,15,9]
				ring2 = [31,16,24,21,5,1]
				SS = 2

		filepath = '/Users/isaaccraig/Desktop/TLGproj/{}/ring12-masks/vdf-ds{}-rings12.pkl'.format(sample, dsnum)
		with open(filepath, 'rb') as f: dfs = pickle.load(f)

		avgring1 = np.zeros((dfs.shape[1], dfs.shape[2]))
		for i in ring1: avgring1 += dfs[i,:,:]
		avgring1 = gaussian_filter(avgring1,1)
		avgring1 = avgring1 - np.min(avgring1.flatten())
		avgring1 = avgring1/np.max(avgring1.flatten())

		avgring2 = np.zeros((dfs.shape[1], dfs.shape[2]))
		for i in ring2: avgring2 += dfs[i,:,:]
		avgring2 = gaussian_filter(avgring2,1)
		avgring2 = avgring2 - np.min(avgring2.flatten())
		avgring2 = avgring2/np.max(avgring2.flatten())

		vdf = avgring1
		f,ax = plt.subplots(1,2)
		ax = ax.flatten()
		ax[0].imshow(vdf, origin='lower',cmap='gray')
		ax[1].imshow(avgring2, origin='lower',cmap='gray')
		plt.show()

		from heterostrain_new import extract_heterostrain_vdf, extract_twist_vdf
		from STMtriangulate import main_vdf
		extract_heterostrain_vdf(vdf, dsnum, prefix='/Users/isaaccraig/Desktop/TLGproj/{}'.format(sample)) #manual
		#extract_twist_vdf(vdf, dsnum, prefix='/Users/isaaccraig/Desktop/TLGproj/{}'.format(sample)) #manul length only 
		#main_vdf(filepath, vdf, FOV_len=SS*vdf.shape[0]) #auto


				
