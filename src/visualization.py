
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.colors as mplc
import colorsys
from matplotlib.patches import RegularPolygon
from scipy.spatial import Voronoi, ConvexHull, Delaunay
import matplotlib.colors as mcolors
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from masking import *
import glob
from basis_utils import lv_to_rzcartesian
import pickle
from diskset import DiskSet
from new_utils import import_uvector, import_diskset, import_unwrap_uvector, normNeighborDistance, anom_filter, import_uvectors, fit_ellipse
from utils import writefile


def make_coloredvdf(avgring1, avgring2, gaussian_sigma=1):
    stack_assign = np.zeros((avgring1.shape[0], avgring1.shape[1], 3))
    if gaussian_sigma is not None: 
        avgring1 = gaussian_filter(avgring1,gaussian_sigma)
    avgring1 = avgring1 - np.nanmin(avgring1.flatten())
    avgring1 = avgring1/np.nanmax(avgring1.flatten())
    if gaussian_sigma is not None:
        avgring2 = gaussian_filter(avgring2,gaussian_sigma)
    avgring2 = avgring2 - np.nanmin(avgring2.flatten())
    avgring2 = avgring2/np.nanmax(avgring2.flatten())
    stack_assign[:,:,0] = avgring1[:,:] # r channel
    stack_assign[:,:,2] = avgring2[:,:] # b channel
    stack_assign[:,:,1] = 0.5 * avgring1[:,:] + 0.5 * avgring2[:,:]
    return stack_assign

def displacement_colorplot_ttlg_thresh(ax, Ux, Uy, thresh1=0.5, thresh2=0.5):
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

def displacement_colorplot_ttlg(ax, Ux, Uy, inc3layer, abt_offset, f):
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


def colored_quiver(ax, u_x, u_y, sample_angle=0):
    colors = displacement_colorplot(None, u_x, u_y)
    uxrot = np.cos(sample_angle) * u_x - np.sin(sample_angle) * u_y
    uyrot = np.cos(sample_angle) * u_y + np.sin(sample_angle) * u_x
    nx, ny = u_x.shape
    x, y, ux, uy, c = [], [], [], [], []
    for i in range(nx):
        for j in range(ny):
            x.append(j)
            y.append(i)
            ux.append(uxrot[i,j])
            uy.append(uyrot[i,j])
            c.append(colors[i,j,:])
    ax.quiver(x, y, ux, uy, color=c)
 
##################################################################
# plots voroni regions used in unwrapping procedure
##################################################################
def plot_voroni(points, nmcenters, regions, vertices, ax):
    for i in range(len(regions)):
        region = regions[i]
        polygon = vertices[region]
        ax.scatter(points[i][0], points[i][1], color='k')
        ax.text(points[i][0], points[i][1], "{}:[{},{}]".format(i, nmcenters[i,0], nmcenters[i,1]))
        ax.fill(*zip(*polygon), alpha=0.4)

##################################################################
# plots connectivity between AA (or analogous) centers, color
# coded for the types (SP1, SP2, SP3) of connectivity
##################################################################
def plot_adjacency(img, centers, adjacency_type, ax=None, colored=True):
    if ax is None: f, ax  = plt.subplots()
    if img is not None: ax.imshow(img, origin='lower')
    for i in range(len(centers)):
        if colored: ax.scatter(centers[i][1], centers[i][0], color='w')
        else: ax.scatter(centers[i][1], centers[i][0], color='grey', s=0.75)
        for j in range(i):
            if   adjacency_type[i, j] == 1:
                if colored: ax.plot([centers[i][1], centers[j][1]], [centers[i][0], centers[j][0]], color="c")
                else: ax.plot([centers[i][1], centers[j][1]], [centers[i][0], centers[j][0]], color="grey", linewidth=0.5)
            elif adjacency_type[i, j] == 2:
                if colored: ax.plot([centers[i][1], centers[j][1]], [centers[i][0], centers[j][0]], color="m")
                else: ax.plot([centers[i][1], centers[j][1]], [centers[i][0], centers[j][0]], color="grey", linewidth=0.5)
            elif adjacency_type[i, j] == 3:
                if colored: ax.plot([centers[i][1], centers[j][1]], [centers[i][0], centers[j][0]], color="y")
                else: ax.plot([centers[i][1], centers[j][1]], [centers[i][0], centers[j][0]], color="grey", linewidth=0.5)
            elif adjacency_type[i, j] == 4:
                if colored: ax.plot([centers[i][1], centers[j][1]], [centers[i][0], centers[j][0]], color="k")
                else: ax.plot([centers[i][1], centers[j][1]], [centers[i][0], centers[j][0]], color="k", linewidth=0.5)


def new_rgb_to_hsv(r, g, b):
 
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
 
    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.
 
    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0
     
    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
 
    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
 
    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360
 
    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100
 
    # compute v
    v = cmax * 100
    return h, s, v

##################################################################
# plots displacements (cartesian basis) with cosine colorplot
##################################################################
def displacement_colorplot_exp(ax, Ux, Uy=None, sample_angle=0, plot_hexagon_bool=False, quiverbool=True, debugplot=False):

    if Uy is None: Ux, Uy = Ux[:,:,0], Ux[:,:,1] # different way of entering U as a nx,ny,2 object
    uvecs = np.zeros((Ux.shape[0], Ux.shape[1], 2))
    uvecs[:,:,0], uvecs[:,:,1] = Ux, Uy
    from basis_utils import cartesian_to_rz_WZ
    #uvecs = cartesian_to_rz_WZ(uvecs, sign_wrap=False)
    #Ux, Uy = uvecs[:,:,0], uvecs[:,:,1]
    
    nx, ny = Ux.shape
    g1 = np.array([ 0, 2/np.sqrt(3)]) 
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1, g2, g1-g2 ]
    f = 3.0
    linear_scale, cos2_scale = True, False
    cvecs =  [[f, f*0.293014986, f*0.293014986], [0, f*0.6198545861, 0], [f*0.4,f*0.4,f]] 
    #cvecs =  [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    def luminance(v): 
        r,g,b = v[:]
        return (0.2126*r + 0.7152*g + 0.0722*b)
    import colorsys

    def hue(r,g,b):
        M = np.argmax([r,g,b])
        c = np.max([r,g,b]) - np.min([r,g,b])

    h1,s1,v1 = colorsys.rgb_to_hsv(*cvecs[0]);
    h2,s2,v2 = colorsys.rgb_to_hsv(*cvecs[1]);
    h3,s3,v3 = colorsys.rgb_to_hsv(*cvecs[2]);
    h12_diff = np.max([h1,h2]) - np.min([h1,h2])
    h32_diff = np.max([h3,h2]) - np.min([h3,h2])
    h13_diff = np.max([h1,h3]) - np.min([h1,h3])
    if h12_diff > 0.5: h12_diff = 1 - h12_diff 
    if h32_diff > 0.5: h32_diff = 1 - h32_diff 
    if h13_diff > 0.5: h13_diff = 1 - h13_diff 
    #assert(np.abs(h12_diff - h32_diff) < 1e-5)
    #assert(np.abs(h32_diff - h13_diff) < 1e-5)
    #assert(np.abs(luminance(cvecs[0]) - luminance(cvecs[1])) < 1e-5)
    #assert(np.abs(luminance(cvecs[0]) - luminance(cvecs[2])) < 1e-5)
    colors1 = np.zeros((nx, ny, 3))
    hmat = np.zeros((nx, ny))
    smat = np.zeros((nx, ny))
    vmat = np.zeros((nx, ny))
    lmat = np.zeros((nx, ny))
    lmat2 = np.zeros((nx, ny))
    umag = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            umag[i,j] = (Ux[i,j]**2 + Uy[i,j]**2)**0.5
    umag[i,j] = umag[i,j] * np.sqrt(3)

    from hsluv import rgb_to_hpluv, hpluv_to_rgb

    for i in range(nx):
        for j in range(ny):
            for n in range(len(gvecs1)):
                #coef_cos2scale   = 1 - (np.cos(np.pi * np.dot(gvecs1[n], [Ux[i,j], Uy[i,j]])))**2 #(between 0 and 1)
                coef_linearscale = np.abs(np.dot(gvecs1[n], [Ux[i,j], Uy[i,j]])) #(between 0 and 1)
                if linear_scale: coef = coef_linearscale
                #elif cos2_scale: coef = coef_cos2scale
                colors1[i,j,:] += coef * np.array(cvecs[n])

            if np.isnan(Ux[i,j]) or np.isnan(Uy[i,j]):
                continue

            uang = np.arctan(Uy[i,j]/(1e-7 + Ux[i,j])) # cartesian!
            uang = (uang + np.pi/2) * 1/np.pi # in the range of 0 to 1
            r, g, b = hpluv_to_rgb([uang*360, 95, umag[i,j]*150])

            def ang2sat(ang):
                a = ang%120
                if a > 60: a = 120-a 
                return a*100/60
            def ang2hue(ang):
                if ang < 120 and ang >= 0:   return 270
                if ang < 240 and ang >= 120: return 120
                if ang < 360 and ang >= 240: return 240
            #r, g, b = hpluv_to_rgb([ang2hue(uang*360), ang2sat(uang*360), 200*umag[i,j]])
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            h, s, l = rgb_to_hpluv([r,g,b]); #h, s, l = h/360, s/100, l/100;
            colors1[i,j,:] = r, g, b 
            hmat[i,j] = h
            smat[i,j] = s
            vmat[i,j] = v
            lmat[i,j] = l # 100*luminance([r,g,b])
            #lmat2[i,j] = l_m# # (0.2126*r + 0.7152*g + 0.0722*b)
    if debugplot:
        f, ax = plt.subplots(2,3)
        axes = ax.flatten()
        plot_hexagon(axes[0], nx, ny, colors1, radius=1/(2*(2 * np.max(Ux) * g2[0])), orientation=0)
        plot_hexagon(axes[1], nx, ny, hmat, radius=1/(2*(2 * np.max(Ux) * g2[0])), orientation=0)
        axes[1].set_title('hue')
        axes[2].set_title('sat')
        axes[3].set_title('val')
        axes[4].set_title('lum1')
        #axes[5].set_title('lum2')
        plot_hexagon(axes[2], nx, ny, smat, radius=1/(2*(2 * np.max(Ux) * g2[0])), orientation=0)
        plot_hexagon(axes[3], nx, ny, vmat, radius=1/(2*(2 * np.max(Ux) * g2[0])), orientation=0)
        plot_hexagon(axes[4], nx, ny, lmat, radius=1/(2*(2 * np.max(Ux) * g2[0])), orientation=0)
        #plot_hexagon(axes[5], nx, ny, lmat2, radius=1/(2*(2 * np.max(Ux) * g2[0])), orientation=0)
        for ax in axes:
            for s in ['top','bottom','left','right']: 
                ax.spines[s].set_linewidth(2)
        plt.show()
        exit()

    if ax is not None:
        if plot_hexagon_bool:
            f = 2 * np.max(Ux) * g2[0]
            colors1 = plot_hexagon(ax, nx, ny, colors1, radius=1/(2*f), orientation=0)
        else: ax.imshow(colors1, origin='lower')
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
        if not plot_hexagon_bool and quiverbool: 
            uxrot = np.cos(sample_angle) * Ux - np.sin(sample_angle) * Uy
            uyrot = np.cos(sample_angle) * Uy + np.sin(sample_angle) * Ux
            ax.quiver(uxrot, uyrot)
    return colors1

def displacement_colorplot(ax, Ux, Uy=None, sample_angle=0, plot_hexagon_bool=False, quiverbool=True, debugplot=False, quivert=False):
    if Uy is None: Ux, Uy = Ux[:,:,0], Ux[:,:,1] # different way of entering U as a nx,ny,2 object
    nx, ny = Ux.shape
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1, g2, g1-g2 ]
    cvecs =  [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # r, g, b
    # solitons will be [1, 0, 1], [1, 1, 0], [0, 1, 1] which are cmy
    colors1 = np.zeros((nx, ny, 3))
    for i in range(nx):
        for j in range(ny):
            for n in range(len(gvecs1)):
                u = [Ux[i,j], Uy[i,j]]
                cf = 1 - ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2
                colors1[i,j,:] += cf * np.array(cvecs[n])
    if ax is not None:
        if plot_hexagon_bool:
            f = 2 * np.max(Ux) * g2[0]
            colors1 = plot_hexagon(ax, nx, ny, colors1, radius=1/(2*f), orientation=0)
        else: ax.imshow(colors1, origin='lower')
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
        if not plot_hexagon_bool and quiverbool: 
            uxrot = np.cos(sample_angle) * Ux - np.sin(sample_angle) * Uy
            uyrot = np.cos(sample_angle) * Uy + np.sin(sample_angle) * Ux
            if quivert: ax.quiver(uyrot, uxrot)
            else: ax.quiver(uxrot, uyrot)
    return colors1

def displacement_bivariate2ring(ax, Ux, Uy=None, plot_hexagon_bool=True):
    if Uy is None: Ux, Uy = Ux[:,:,0], Ux[:,:,1] 
    nx, ny = Ux.shape
    colors1 = np.zeros((nx, ny, 3))
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1, g2, g1-g2 ]
    gvecs2 = [ g1+g2, 2*g2-g1, 2*g1-g2 ]
    A1, A2, B1, B2 = 1/3, 1/3, 0, 0
    for i in range(nx):
        for j in range(ny):
            u = [Ux[i,j], Uy[i,j]]
            I1 = A1*(np.cos(np.pi*np.dot(gvecs1[0], u)))**2 + B1
            I1 += A1*(np.cos(np.pi*np.dot(gvecs1[1], u)))**2 
            I1 += A1*(np.cos(np.pi*np.dot(gvecs1[2], u)))**2 
            I2 = A2*(np.cos(np.pi*np.dot(gvecs2[0], u)))**2 + B2
            I2 += A2*(np.cos(np.pi*np.dot(gvecs2[1], u)))**2 
            I2 += A2*(np.cos(np.pi*np.dot(gvecs2[2], u)))**2 
            colors1[i,j,0] = I1;
            colors1[i,j,1] = (I1+I2)*0.5;
            colors1[i,j,2] = I2;
    if ax is not None:
        if plot_hexagon_bool:
            f = 2 * np.max(Ux) * g2[0]
            colors1 = plot_hexagon(ax, nx, ny, colors1, radius=1/(2*f), orientation=0)
        else: ax.imshow(colors1, origin='lower')
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
    return colors1

##################################################################
# plots displacements (cartesian basis) with asymmetric colorplot
##################################################################
def displacement_colorplot_asym(ax, Ux, Uy, plot_hexagon_bool=False):
    nx, ny = Ux.shape
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1 , g2 ]
    cvecs =  [[1, 0, 0], [0, 1, 0]]
    cvecs2 = [[0, 1, 1], [1, 0, 1]] # [[0, 1/2, 1/2], [1/2, 0, 1/2]]
    colors1 = np.zeros((nx, ny, 3))
    for i in range(nx):
        for j in range(ny):
            for n in range(len(gvecs1)):
                u = [Ux[i,j], Uy[i,j]]
                cf = 1 - (np.cos(np.pi * np.dot(gvecs1[n], u)))**2
                if np.dot(gvecs1[n], u) > 0: colors1[i,j,:] += cf * np.array(cvecs[n])
                else: colors1[i,j,:] += cf * np.array(cvecs2[n])
    if ax is not None:
        if plot_hexagon_bool:
            f = 2 * np.max(Ux) * g2[0]
            colors1 = plot_hexagon(ax, nx, ny, colors1, radius=1/(2*f), orientation=0)
        else: ax.imshow(colors1, origin='lower')
        ax.set_xticks(np.arange(0, np.round(nx/100)*100+1, 100))
        ax.set_yticks(np.arange(0, np.round(nx/100)*100+1, 100))
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
    return colors1

##################################################################
# overlays virtual dark fields
##################################################################
def overlay_vdf(diskset, dsnum=0, prefix=None, norm_bool=True, sub=False, ax=None, plotflag=True):
    if isinstance(diskset, DiskSet):
        tot_img = np.zeros((diskset.nx, diskset.ny))
        for n in range(diskset.size):
            if diskset.in_use(n):
                tot_img = tot_img + diskset.df(n)
    elif isinstance(diskset, np.ndarray):
        tot_img = np.zeros((diskset.shape[1], diskset.shape[2]))
        for n in range(diskset.shape[0]):
            tot_img = tot_img + diskset[n,:,:]
    else:
        # throw an error if given an unrecognized data format
        print('overlay_vdf only defined for instances of DiskSet or np.ndarray')
        exit(1)
    if norm_bool: tot_img = normalize(tot_img)
    if plotflag and ax == None:
        # generate and save plot
        f, ax = plt.subplots()
        cf = ax.imshow(tot_img, cmap='gray')
        ax.set_title("Sum of Selected Disk Virtual DFs")
        cb = plt.colorbar(cf, ax=ax, orientation='horizontal')
        ax.set_xticks(np.arange(0, np.round(diskset.nx/50)*50+1, 50))
        ax.set_yticks(np.arange(0, np.round(diskset.ny/50)*50+1, 50))
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
        if sub: plt.savefig("../plots/{}/ds_{}/df_sum_sub.png".format(prefix,dsnum), dpi=300)
        else: plt.savefig("../plots/{}/ds_{}/df_sum_nosub.png".format(prefix,dsnum), dpi=300)
        plt.close()
    elif plotflag:
        cf = ax.imshow(tot_img, cmap='gray')
        cb = plt.colorbar(cf, ax=ax, orientation='horizontal')
        ax.set_title("Sum of Selected Disk Virtual DFs")
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
        ax.set_xticks(np.arange(0, np.round(diskset.nx/50)*50+1, 50))
        ax.set_yticks(np.arange(0, np.round(diskset.ny/50)*50+1, 50))
    return tot_img

##################################################################
# displacement_colorplot_lvbasis wrapper that also saves stuff
##################################################################
def displacement_plt_lvbasis(uvecs, nx, ny, title, prefix=None, dsnum=0, savebool=False):
    f, ax = plt.subplots()
    displacement_colorplot_lvbasis(ax, uvecs)
    if savebool:
        path = os.path.join('..','plots',prefix,'ds_{}'.format(dsnum),'{}.png'.format('_'.join(title.split(' '))))
        plt.savefig(path, dpi=300)
        plt.close()
    else: plt.show()

##################################################################
# plots displacements (lattice vector basis) with cosine colorplot
##################################################################
def displacement_colorplot_lvbasis(ax, Ux, Uy=None, quiverbool=False):
    if Uy is None: Ux, Uy = Ux[:,:,0], Ux[:,:,1] # different way of entering U as a nx,ny,2 object
    nx, ny = Ux.shape
    g1 = np.array([0, 1])
    g2 = np.array([1, 0])
    gvecs1 = [ g1, g2, g1-g2 ]
    cvecs =  [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # r, g, b
    colors1 = np.zeros((nx, ny, 3))
    for i in range(nx):
        for j in range(ny):
            for n in range(len(gvecs1)):
                cf = 1 - (np.cos(np.pi * np.dot(gvecs1[n], np.array([Ux[i,j], Uy[i,j]]))))**2
                colors1[i,j,:] += cf * np.array(cvecs[n])
    if ax is not None:
        ax.imshow(colors1, origin='lower')
        ax.set_xticks(np.arange(0, np.round(nx/100)*100+1, 100))
        ax.set_yticks(np.arange(0, np.round(nx/100)*100+1, 100))
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
        if quiverbool: ax.quiver(Ux, Uy)
    return colors1

##################################################################
# displacement_colorplot wrapper that also saves stuff
##################################################################
def displacement_plt(uvecs, nx, ny, title, prefix=None, dsnum=0, savebool=False):
    f, ax = plt.subplots(1,2)
    displacement_colorplot(ax, uvecs)
    if savebool:
        path = os.path.join('..','plots',prefix,'ds_{}'.format(dsnum),'{}.png'.format('_'.join(title.split(' '))))
        plt.savefig(path, dpi=300)
        plt.close()
    else: plt.show()

##################################################################
# visualizes the AA/SP1/SP2/SP3 thresholds
##################################################################
def threshold_plot(uvecs, boundary, delta):
    f, ax = plt.subplots(2,2)
    nx, ny = uvecs.shape[0], uvecs.shape[1]
    U = uvecs[:,:,0].reshape(nx, ny)
    V = uvecs[:,:,1].reshape(nx, ny)
    displacement_colorplot(ax[0,0], U, V)
    aa_mask = get_aa_mask(uvecs, boundary=boundary)
    contours = measure.find_contours(aa_mask, 0.5)
    for contour in contours:
        ax[0,0].plot(contour[:,1], contour[:,0], 'w')
        ax[1,0].plot(contour[:,1], contour[:,0], 'k')
    sp1mask, sp2mask, sp3mask = get_sp_masks(uvecs, aa_mask, delta=delta, plotbool=False)
    for spmask, colorcode in zip([sp1mask, sp2mask, sp3mask], ['c', 'm', 'y']):
        contours = measure.find_contours(spmask, 0.5)
        for contour in contours:
            ax[0,0].plot(contour[:,1], contour[:,0], colorcode)
            ax[1,1].plot(contour[:,1], contour[:,0], colorcode)
    make_legend(ax=ax[0,1], boundary=boundary, delta=delta, plotflag=False)
    ax[1,0].axis('off')
    ax[1,1].axis('off')
    plt.show()

    ##################################################################

def disp_categorize_plot(ufit, ax): #want cartesian u
    ufit = ufit[1:-1, 1:-1, :]# truncate a little 
    nx, ny = ufit.shape[0], ufit.shape[1]
    umag = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            umag[i,j] = (ufit[i,j,0]**2 + ufit[i,j,1]**2)**0.5
    boundary = 0.5 * np.nanmax(umag.flatten()) 
    print('max is', np.nanmax(umag.flatten()), ' a0')
    aa_mask = get_aa_mask(ufit, boundary=boundary, smooth=None)
    aa_radii = []
    contours = measure.find_contours(aa_mask, 0.5)
    for contour in contours: 
        if len(contour[:,1]) < 10: continue
        if len(contour[:,1]) > 190: continue
        try: 
            hull = ConvexHull(contour)
            xpts, ypts = contour[hull.vertices,1], contour[hull.vertices,0]
            contour = np.zeros((len(xpts), 2))
            contour[:,1], contour[:,0] = xpts, ypts
        except: continue
        if np.min(contour[:,1]) <= 0 or np.min(contour[:,0]) <= 0 or np.max(contour[:,1]) >= nx-1 or np.max(contour[:,0]) >= ny-1: continue
        try: x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
        except: continue
        # exclude contours that go outside of FOV since biased
        if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
        if not aa_mask[int(y0), int(x0)]: continue
        ax.plot(xfit, yfit, 'grey')
        r = (ap+bp)/2 # average semi-major and semi-minor axis lengths
        aa_radii.append(r)
        ax.text(x0, y0, "{:.2f}".format(r), color='grey', fontsize='xx-small', horizontalalignment='center')

    sp1mask, sp2mask, sp3mask, mm_mask, xx_mask = get_sp_masks(ufit, aa_mask, plotbool=False, exclude_aa=True, include_aa=False, window_filter_bool=False)
    no_aa_sp1mask = ((sp1mask.astype(int) - aa_mask.astype(int)) > 0).astype(int)
    contours = measure.find_contours(no_aa_sp1mask, 0.5)
    sp_widths = []
    for contour in contours: 
        if len(contour[:,1]) < 25: continue
        try: 
            x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
            # exclude contours that go outside of FOV since biased
            if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
            ax.plot(xfit, yfit, 'c')
            sp_widths.append(bp) # average semi-minor axis 
            ax.text(x0, y0, "{:.2f}".format(bp), color='c', fontsize='xx-small', horizontalalignment='center')
        except: continue    
    no_aa_sp2mask = ((sp2mask.astype(int) - aa_mask.astype(int)) > 0).astype(int)
    contours = measure.find_contours(no_aa_sp2mask, 0.5)
    for contour in contours: 
        if len(contour[:,1]) < 25: continue
        try: 
            x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
            # exclude contours that go outside of FOV since biased
            if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
            ax.plot(xfit, yfit, 'm')
            sp_widths.append(bp) # average semi-minor axis 
            ax.text(x0, y0, "{:.2f}".format(bp), color='m', fontsize='xx-small', horizontalalignment='center')
        except: continue    
    no_aa_sp3mask = ((sp3mask.astype(int) - aa_mask.astype(int)) > 0).astype(int)
    contours = measure.find_contours(no_aa_sp3mask, 0.5)
    for contour in contours: 
        if len(contour[:,1]) < 25: continue
        try: 
            x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
            # exclude contours that go outside of FOV since biased
            if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
            ax.plot(xfit, yfit, 'y')
            sp_widths.append(bp) # average semi-minor axis 
            ax.text(x0, y0, "{:.2f}".format(bp), color='y', fontsize='xx-small', horizontalalignment='center')
        except: continue 
    
    spmask = ( (sp1mask + sp2mask + sp3mask) > 0 ).astype(int)
    n_aa, n_sp1, n_sp2, n_sp3, n_ab, n_xx, n_mm = 0, 0, 0, 0, 0, 0, 0
    colors = 225 * np.ones((nx, ny, 3)) #start white
    for i in range(nx):
        for j in range(ny):
            if aa_mask[i,j]:   
                colors[i,j,:] = [0, 0, 0] #k
                n_aa += 1
            elif sp1mask[i,j]: 
                colors[i,j,:] = [0, 225, 225] #c
                n_sp1 += 1
            elif sp2mask[i,j]: 
                colors[i,j,:] = [225, 0, 225] #m
                n_sp2 += 1
            elif sp3mask[i,j]: 
                colors[i,j,:] = [225, 225, 0] #y
                n_sp3 += 1
            elif mm_mask[i,j]: 
                colors[i,j,:] = [225, 0, 0] #r
                n_mm += 1
            elif xx_mask[i,j]: 
                colors[i,j,:] = [225, 225, 225] #white
                n_xx += 1

    n_tot = np.sum([n_aa, n_sp1, n_sp2, n_sp3, n_xx, n_mm])
    pAA, pSP1, pSP2, pSP3, pXX, pMM = 100 * n_aa/n_tot, 100 * n_sp1/n_tot, 100 * n_sp2/n_tot, 100 * n_sp3/n_tot, 100 * n_xx/n_tot, 100 * n_mm/n_tot
    aa_radii = [el for el in aa_radii if not np.isnan(el)]
    sp_widths = [el for el in aa_radii if not np.isnan(el)]
    rAA, eAA, wSP, eSP = np.mean(aa_radii), np.std(aa_radii, ddof=1) / np.sqrt(np.size(aa_radii)), np.mean(sp_widths), np.std(sp_widths, ddof=1) / np.sqrt(np.size(sp_widths))
    ax.imshow(colors, origin='lower')
    for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(2)
    return pAA, pSP1, pSP2, pSP3, pXX, pMM, rAA, eAA, wSP, eSP    

def displacement_categorize(ufit, ax0=None, ax1=None, ax2=None):
    ufit = ufit[1:-1, 1:-1, :]# truncate a little 
    nx, ny = ufit.shape[0], ufit.shape[1]
    displacement_colorplot_lvbasis(ax0, ufit)
    ufit = lv_to_rzcartesian(ufit)
    umag = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            umag[i,j] = (ufit[i,j,0]**2 + ufit[i,j,1]**2)**0.5
    boundary = 0.5 * np.max(umag.flatten()) #(3/(4*np.pi*np.pi))**(1/4)
    aa_mask = get_aa_mask(ufit, boundary=boundary, smooth=None)
    aa_radii = []
    contours = measure.find_contours(aa_mask, 0.5)
    for contour in contours: 

        if len(contour[:,1]) < 10: continue
        if len(contour[:,1]) > 190: continue
        try: 
            hull = ConvexHull(contour)
            xpts, ypts = contour[hull.vertices,1], contour[hull.vertices,0]
            contour = np.zeros((len(xpts), 2))
            contour[:,1], contour[:,0] = xpts, ypts
        except: continue

        if np.min(contour[:,1]) <= 0 or np.min(contour[:,0]) <= 0 or np.max(contour[:,1]) >= nx-1 or np.max(contour[:,0]) >= ny-1: continue
        #ax.plot(contour[:,1], contour[:,0], 'r')

        try: x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
        except: continue
        # exclude contours that go outside of FOV since biased
        if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
        if not aa_mask[int(y0), int(x0)]: continue
        ax1.plot(xfit, yfit, 'grey')
        r = (ap+bp)/2 # average semi-major and semi-minor axis lengths
        aa_radii.append(r)
        ax1.text(x0, y0, "{:.2f}".format(r), color='grey', fontsize='xx-small', horizontalalignment='center')

    sp1mask, sp2mask, sp3mask = get_sp_masks(ufit, aa_mask, plotbool=False, include_aa=False, window_filter_bool=False)

    no_aa_sp1mask = ((sp1mask.astype(int) - aa_mask.astype(int)) > 0).astype(int)
    contours = measure.find_contours(no_aa_sp1mask, 0.5)
    sp_widths = []
    for contour in contours: 
        if len(contour[:,1]) < 25: continue
        try: 
            x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
            # exclude contours that go outside of FOV since biased
            if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
            ax1.plot(xfit, yfit, 'c')
            sp_widths.append(bp) # average semi-minor axis 
            ax1.text(x0, y0, "{:.2f}".format(bp), color='c', fontsize='xx-small', horizontalalignment='center')
        except: continue    
    no_aa_sp2mask = ((sp2mask.astype(int) - aa_mask.astype(int)) > 0).astype(int)
    contours = measure.find_contours(no_aa_sp2mask, 0.5)
    for contour in contours: 
        if len(contour[:,1]) < 25: continue
        try: 
            x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
            # exclude contours that go outside of FOV since biased
            if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
            ax1.plot(xfit, yfit, 'm')
            sp_widths.append(bp) # average semi-minor axis 
            ax1.text(x0, y0, "{:.2f}".format(bp), color='m', fontsize='xx-small', horizontalalignment='center')
        except: continue    
    
    spmask = ( (sp1mask + sp2mask + sp3mask) > 0 ).astype(int)
    n_aa, n_sp1, n_sp2, n_sp3, n_ab = 0, 0, 0, 0, 0
    colors = 225 * np.ones((nx, ny, 3)) #start white
    for i in range(nx):
        for j in range(ny):
            if aa_mask[i,j]:   
                colors[i,j,:] = [0, 0, 0] #k
                n_aa += 1
            elif sp1mask[i,j]: 
                colors[i,j,:] = [0, 225, 225] #c
                n_sp1 += 1
            elif sp2mask[i,j]: 
                colors[i,j,:] = [225, 0, 225] #m
                n_sp2 += 1
            elif sp3mask[i,j]: 
                colors[i,j,:] = [225, 225, 0] #y
                n_sp3 += 1
            else: 
                colors[i,j,:] = [220,220,220] #grey
                n_ab += 1

    n_tot = np.sum([n_aa, n_sp1, n_sp2, n_sp3, n_ab])
    #ax0.set_title("{:.2f} % AA \n{:.2f} % AB \n{:.2f} % SP1 \n{:.2f} % SP2 \n{:.2f} % SP3".format(100 * n_aa/n_tot, 100 * n_ab/n_tot, 100 * n_sp1/n_tot, 100 * n_sp2/n_tot, 100 * n_sp3/n_tot))
    #ax1.set_title("{:.2f}+/-{:.2f} AA radius (pixels) \n{:.2f}+/-{:.2f} SP width (pixels)".format(np.mean(aa_radii), np.std(aa_radii, ddof=1) / np.sqrt(np.size(aa_radii)), np.mean(sp_widths), np.std(sp_widths, ddof=1) / np.sqrt(np.size(sp_widths))))
    pAA, pSP1, pSP2, pSP3, pAB = 100 * n_aa/n_tot, 100 * n_sp1/n_tot, 100 * n_sp2/n_tot, 100 * n_sp3/n_tot, 100 * n_ab/n_tot
    aa_radii = [el for el in aa_radii if not np.isnan(el)]
    sp_widths = [el for el in aa_radii if not np.isnan(el)]
    rAA, eAA, wSP, eSP = np.mean(aa_radii), np.std(aa_radii, ddof=1) / np.sqrt(np.size(aa_radii)), np.mean(sp_widths), np.std(sp_widths, ddof=1) / np.sqrt(np.size(sp_widths))
    ax1.imshow(colors, origin='lower')
    #ax1.set_xticks(np.arange(0, np.round(nx/100)*100+1, 100))
    #ax1.set_yticks(np.arange(0, np.round(nx/100)*100+1, 100))
    for axis in ['top','bottom','left','right']: ax1.spines[axis].set_linewidth(2)
    #make_legend_categorized(ax=ax2, plotflag=False, boundary=0.5)
    return pAA, pSP1, pSP2, pSP3, pAB, rAA, eAA, wSP, eSP

# function to make a legend for the colorplot visualization
##################################################################
def make_legend_categorized(ax=None, plotflag=True, boundary=0.5):
    xrange = np.arange(-0.50, 0.5 + 0.001, 0.001)#np.arange(-2.5, 2.52, 0.02)
    nx = len(xrange)
    U, V = np.meshgrid(xrange, xrange)
    f = -2 * np.max(U[:,:]) 
    radius=1/(2*f)
    if ax is None: f, ax = plt.subplots()
    ax.set_xlim([-15, nx+15])
    ax.set_ylim([-15, nx+15])
    ufit = np.zeros((nx, nx, 2))
    ufit[:,:,0] = U[:,:]
    ufit[:,:,1] = V[:,:]
    boundary = boundary * np.max(U[:,:]) 
    aa_mask = get_aa_mask(ufit, boundary=boundary)
    sp1mask, sp2mask, sp3mask = get_sp_masks(ufit, aa_mask, plotbool=False, include_aa=False, window_filter_bool=False)
    colors = 225 * np.ones((nx, nx, 3)) #start white
    for i in range(nx):
        for j in range(nx):
            if aa_mask[i,j]:   colors[i,j,:] = [0, 0, 0] #k
            elif sp1mask[i,j]: colors[i,j,:] = [0, 1, 1] #c
            elif sp2mask[i,j]: colors[i,j,:] = [1, 0, 1] #m
            elif sp3mask[i,j]: colors[i,j,:] = [1, 1, 0] #y
            else: colors[i,j,:] = [0.8,0.8,0.8] #grey
    colors = plot_hexagon(ax, nx, nx, colors, radius=np.abs(radius), orientation=0)  
    n_k, n_c, n_m, n_y, n_g = 0, 0, 0, 0, 0
    for i in range(nx):
        for j in range(nx):
            if colors[i,j,0] == 0   and colors[i,j,1] == 0   and colors[i,j,2] == 0:   n_k +=1 
            if colors[i,j,0] == 0   and colors[i,j,1] == 1   and colors[i,j,2] == 1:   n_c +=1 
            if colors[i,j,0] == 1   and colors[i,j,1] == 0   and colors[i,j,2] == 1:   n_m +=1 
            if colors[i,j,0] == 1   and colors[i,j,1] == 1   and colors[i,j,2] == 0:   n_y +=1 
            if colors[i,j,0] == 0.8 and colors[i,j,1] == 0.8 and colors[i,j,2] == 0.8: n_g +=1 
    sum_n = n_c+n_m+n_y+n_g+n_k
    print(n_k/sum_n, n_g/sum_n, (n_c+n_m+n_y)/sum_n)
    ax.axis('off')
    if plotflag: plt.show()

def make_examplefig_categorized(ax=None, plotflag=True, boundary=0.5):

    from basis_utils import cartesian_to_rz_WZ, cartesian_to_rzcartesian, cartesian_to_latticevec
    xrange = np.arange(-1.9, 1.9, 0.01) #0.05
    x, y = np.meshgrid(xrange, xrange)
    n = len(xrange)
    nx, ny = n, n
    u_cart = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            u_cart[i,j,0] = y[i,j]  
            u_cart[i,j,1] = -x[i,j]
    u_cart = cartesian_to_rz_WZ(u_cart.copy(), sign_wrap=False)
    if ax is None: f, ax = plt.subplots()
    boundary = boundary * np.max(u_cart[:,:,0].flatten()) 
    aa_mask = get_aa_mask(u_cart, boundary=boundary, smooth=None)
    aa_radii = []
    contours = measure.find_contours(aa_mask, 0.5)
    for contour in contours: 
        if len(contour[:,1]) < 20: continue
        #ax.plot(contour[:,1], contour[:,0], 'r')
        x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
        # exclude contours that go outside of FOV since biased
        if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
        ax.plot(xfit, yfit, 'grey')
        r = (ap+bp)/4 # average semi-major and semi-minor axis lengths
        aa_radii.append(r)
        ax.text(x0, y0, "{:.2f}".format(r), color='grey', fontsize='xx-small', horizontalalignment='center')

    sp1mask, sp2mask, sp3mask = get_sp_masks(u_cart, aa_mask, plotbool=False, include_aa=False, window_filter_bool=False)

    no_aa_sp1mask = ((sp1mask.astype(int) - aa_mask.astype(int)) > 0).astype(int)
    contours = measure.find_contours(no_aa_sp1mask, 0.5)
    sp_widths = []
    for contour in contours: 
        if len(contour[:,1]) < 20: continue
        try: 
            x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
            # exclude contours that go outside of FOV since biased
            if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
            ax.plot(xfit, yfit, 'c')
            sp_widths.append(bp) # average semi-minor axis 
            ax.text(x0, y0, "{:.2f}".format(bp), color='c', fontsize='xx-small', horizontalalignment='center')
        except: continue    
    no_aa_sp2mask = ((sp2mask.astype(int) - aa_mask.astype(int)) > 0).astype(int)
    contours = measure.find_contours(no_aa_sp2mask, 0.5)
    for contour in contours: 
        if len(contour[:,1]) < 20: continue
        try: 
            x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
            # exclude contours that go outside of FOV since biased
            if np.min(xfit) < 0 or np.min(yfit) < 0 or np.max(xfit) > nx-1 or np.max(yfit) > ny-1: continue
            ax.plot(xfit, yfit, 'm')
            sp_widths.append(bp) # average semi-minor axis 
            ax.text(x0, y0, "{:.2f}".format(bp), color='m', fontsize='xx-small', horizontalalignment='center')
        except: continue    

    colors = 225 * np.ones((nx, nx, 3)) #start white
    for i in range(nx):
        for j in range(nx):
            if aa_mask[i,j]:   colors[i,j,:] = [0, 0, 0] #k
            elif sp1mask[i,j]: colors[i,j,:] = [0, 1, 1] #c
            elif sp2mask[i,j]: colors[i,j,:] = [1, 0, 1] #m
            elif sp3mask[i,j]: colors[i,j,:] = [1, 1, 0] #y
            else: colors[i,j,:] = [1,1,1] #grey
    ax.imshow(colors, origin='lower')  
    ax.axis('off')
    if plotflag: plt.show()

def make_legend_bivariate_ttlg(ax=None):
    xrange = np.arange(-0.50, 0.52, 0.01)
    nx = len(xrange)
    U, V = np.meshgrid(xrange, xrange)
    if ax is None: f, ax = plt.subplots()
    ax.set_xlim([-15, nx+15])
    ax.set_ylim([-15, nx+15])
    displacement_bivariate2ring(ax, U, V, plot_hexagon_bool=True)
    ax.axis('off')

##################################################################
# function to make a legend for the colorplot visualization
##################################################################
def make_legend(ax=None, sym=True, boundary=None, delta=None, plotflag=True, debugplot=False):
    xrange = np.arange(-0.50, 0.52, 0.01)#np.arange(-2.5, 2.52, 0.02)
    nx = len(xrange)
    U, V = np.meshgrid(xrange, xrange)
    if ax is None: f, ax = plt.subplots()
    ax.set_xlim([-15, nx+15])
    ax.set_ylim([-15, nx+15])
    if sym: displacement_colorplot(ax, U, V, plot_hexagon_bool=True, debugplot=debugplot)
    else: displacement_colorplot_asym(ax, U, V, plot_hexagon_bool=True)
    if (boundary is not None) and (delta is not None):
        ufit = np.zeros((nx, nx, 2))
        ufit[:,:,0] = U[:,:]
        ufit[:,:,1] = V[:,:]
        aa_mask = get_aa_mask(ufit, boundary=boundary)
        contours = measure.find_contours(aa_mask, 0.5)
        for contour in contours: ax.plot(contour[:,1], contour[:,0], 'k')
        sp1mask, sp2mask, sp3mask = get_sp_masks(ufit, aa_mask, delta=delta, plotbool=False, include_aa=False, window_filter_bool=False)
        for spmask, colorcode in zip([sp1mask, sp2mask, sp3mask], ['c', 'm', 'y']):
            contours = measure.find_contours(spmask, 0.5)
            for contour in contours: ax.plot(contour[:,1], contour[:,0], colorcode)
    ax.axis('off')
    if plotflag: plt.show()

##################################################################
# visualizes contours drawn on image with a threshold value
##################################################################
def plot_contour(img, thresh, ax=None, plotflag=True):
    if ax is None: f, ax = plt.subplots()
    ax.imshow(img, origin='lower')
    contours = measure.find_contours(img, thresh)
    for contour in contours: ax.plot(contour[:,1], contour[:,0], 'k')
    ax.axis('off')
    if plotflag: plt.show()

##################################################################
# helper function to create hexagon for the legend plots
##################################################################
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

##################################################################
# shows the virtual dark fields for the different disks
##################################################################
def plot_disks(diskset, norm_bool=True):
    counter = 0
    nx, ny = int(np.ceil(diskset.size_in_use ** 0.5)), int(np.ceil(diskset.size_in_use ** 0.5))
    f, axes = plt.subplots(nx, ny)
    axes = axes.flatten()
    for n in range(diskset.size):
        if diskset.in_use(n):
            img = diskset.df(n)
            if norm_bool: img = normalize(img)
            ax = axes[counter]
            ax.set_title("Disk {}".format(n))
            cf = ax.imshow(img, cmap='gray')
            cb = plt.colorbar(cf, ax=ax, orientation='horizontal')
            counter += 1
    plt.subplots_adjust(hspace=0.55, wspace=0.3)
    plt.show()
    plt.close()

##################################################################
# shows the difference in intensity of Fridel pair disks
##################################################################
def plot_disk_asymmetry(diskset):
    counter = 0
    n = diskset.size_in_use//2
    f, axes = plt.subplots(3, n)
    df_set = diskset.df_set()
    pairs = diskset.get_pairs()
    for pair in pairs:
        d1 = normalize(df_set[pair[0],:,:])
        d2 = normalize(df_set[pair[1],:,:])
        axes[0,counter].set_title("Disk {} - Disk {}".format(pair[0], pair[1]))
        cf = axes[0,counter].imshow(np.abs(d1 - d2), cmap='gray', vmin=0, vmax=1)
        axes[1,counter].set_title("Disk {}".format(pair[0]))
        cf = axes[1,counter].imshow(d1, cmap='gray')
        axes[2,counter].set_title("Disk {}".format(pair[1]))
        cf = axes[2,counter].imshow(d2, cmap='gray')
        counter += 1
    plt.show()
    plt.close()

if __name__ == "__main__":

    if True:
        #f, ax = plt.subplots(1,2)
        #make_legend_categorized(ax=ax[0], plotflag=False)
        #make_examplefig_categorized(ax[0])
        f, ax = plt.subplots()
        make_legend_bivariate_ttlg(ax)
        #make_legend(ax, plotflag=False, debugplot=True)
        plt.show()
        #plt.savefig("/Users/isaaccraig/Desktop/hex.png", dpi=500)
        exit()

    replot_bool = boolquery("would you like to categorize the displacement fields for stats?")
    if replot_bool:
        summary_file = os.path.join('..','results', 'stacking_statisitcs.txt')
        write_lines = []
        write_legend = 'dataset\tpercent AA\tpercent SP1\tpercent SP2\tpercent SP3\tpercent SP\tpercent AB\tradius AA\tstd error AA radius\twidth SP\tstd error width SP'
        pickles = glob.glob(os.path.join('..','results','*','*pkl*refit'))
        for indx in range(len(pickles)):
            filenm = pickles[indx]
            print("working on ", filenm)
            filepath = os.path.join('..','results',filenm)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if len(data) == 2: uvecs = data[1]
                elif len(data) == 4: uvecs = data[3]
                elif len(data) == 5: uvecs = data[4]
            f, ax = plt.subplots(1,3)
            pAA, pSP1, pSP2, pSP3, pAB, rAA, eAA, wSP, eSP = displacement_categorize(uvecs, ax[0], ax[1], ax[2])
            write_lines.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(filenm, pAA, pSP1, pSP2, pSP3, pSP1+pSP2+pSP3, pAB, rAA, eAA, wSP, eSP))
            savefile = "{}categorized.png".format(filenm)
            print('saving to {}'.format(savefile))
            plt.savefig(savefile, dpi=300)
            plt.close('all')
            #plt.show(); exit()
        writefile(summary_file, write_legend, write_lines)

    #replotall_bool = boolquery("would you like to replot all displacement fields?")
    if False and replotall_bool:
        pickles = glob.glob(os.path.join('..','results','*','*pkl_fit'))
        for indx in range(len(pickles)):
            filenm = pickles[indx]
            filepath = os.path.join('..','results',filenm)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if len(data) == 2: uvecs = data[1]
                elif len(data) == 4: uvecs = data[3]
            f, ax = plt.subplots(1,2)
            nx, ny = uvecs.shape[0:2]
            U = uvecs[:,:,0].reshape(nx, ny)
            V = uvecs[:,:,1].reshape(nx, ny)
            displacement_colorplot(ax[0], U, V)
            make_legend(ax[1])
            plt.subplots_adjust(wspace=0.0)
            savefile = "{}replot.png".format(filenm)
            print('saving to {}'.format(savefile))
            plt.savefig(savefile, dpi=300)
            plt.close('all')
        exit()

    replot_bool = boolquery("would you like to replot a displacement field?")
    while replot_bool:
        uvecs, prefix, dsnum, binbool = import_uvector(all=True)
        nx, ny = uvecs.shape[0:2]
        U = uvecs[:,:,0].reshape(nx, ny)
        V = uvecs[:,:,1].reshape(nx, ny)
        f, ax = plt.subplots(1,2)
        displacement_colorplot_lvbasis(ax[0], U, V)
        make_legend(ax[1], sym=True, plotflag=False)
        plt.show()
        replot_bool = boolquery("would you like to replot another displacement field?")

    replot_bool = boolquery("would you like to replot a series of displacement fields side by side?")
    while replot_bool:
        uvecs, prefixes, dsnums, binbools = import_uvectors(all=True)
        n = len(uvecs)
        n1, n2 = int(round(n**0.5)), int(round(n**0.5))+1
        nx, ny = uvecs[0].shape[0:2]
        f, ax = plt.subplots(n1, n2)
        axes = ax.flatten()
        for i in range(n):
            displacement_colorplot_lvbasis(axes[i], uvecs[i])
            pf = prefixes[i].split('_')[1:-1]
            pf = '_'.join(pf)
            if binbools[i]: axes[i].set_title('binned {} DS {}'.format(pf, dsnums[i]))
            else: axes[i].set_title('{} DS {}'.format(pf, dsnums[i]))
            finali = i+1 
        for i in range(finali, len(axes)): axes[i].axis('off')
        plt.show()
        replot_bool = boolquery("would you like to replot another set of displacement fields?")

    replot_bool = boolquery("would you like to replot an unwrapped disp field?")
    while replot_bool:
        u, prefix, dsnum, centers, adjacency_type = import_unwrap_uvector()
        f, ax = plt.subplots(1,3)
        img = displacement_colorplot(ax[0], u);
        plot_adjacency(img, centers, adjacency_type, ax=ax[1])
        d = normNeighborDistance(u, norm=False)
        ax[2].imshow(d, origin='lower', vmax=0.2)
        plot_adjacency(None, centers, adjacency_type, ax=ax[2], colored=False)
        plt.show()
        replot_bool = boolquery("would you like to replot another?")

    replot_bool = query("would you like to replot a virtual df?")
    while replot_bool:
        diskset, prefix, dsnum = import_diskset()
        plot_disks(diskset)
        f, ax = plt.subplots()
        overlay_vdf(diskset, ax=ax)
        plt.show()
        replot_bool = boolquery("would you like to replot another virtual df?")

    replot_bool = boolquery("would you like to plot disk asymmetry?")
    while replot_bool:
        diskset, prefix, dsnum = import_diskset()
        plot_disk_asymmetry(diskset)
        boolquery("would you like to another plot disk asymmetry?")
