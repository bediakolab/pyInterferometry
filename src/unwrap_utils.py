
##################################################################
##################################################################
## series of utility functions to aid in unwrapping 
##################################################################
##################################################################
import numpy as np
import matplotlib.pyplot as plt
from visualization import displacement_colorplot, plot_voroni, plot_adjacency 
from basis_utils import adjust_zone, cartesian_to_rzcartesian, in_rz_cart, latticevec_to_cartesian, cart_to_zonebasis, single_lv_to_cart, zonebasis_to_cart, get_nearby_equivs, rotate_uvecs, adjust_zone_single
from gekko import GEKKO # for integer programming
import matplotlib
from time import time
from utils import bin, unbin, debugplot, boolquery, get_triangles, manual_define_points
from masking import get_aa_mask, get_sp_masks, get_sp_line_method2, get_region_centers
import scipy.ndimage as ndimage
from utils import combine_nearby_spots, tic, toc, getAdjacencyMatrixManual
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from new_utils import normNeighborDistance
from skimage import measure
from masking import convex_mask
from new_utils import anom_nan_filter, plot_images
from utils import nan_gaussian_filter, getAdjacencyMatrixManualAB, doIntersect, manual_define_points
from new_utils import get_lengths, get_angles
from scipy.spatial import Delaunay
from numpy import ones,vstack
from numpy.linalg import lstsq

def watershed_regions(u, boundarythresh=0.5, refine=True):
    
    if len(u.shape) > 2 and u.shape[2] > 1:
        image = np.zeros((u.shape[0], u.shape[1]))
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                image[i,j] = (u[i,j,0]**2 + u[i,j,1]**2) **0.5
    else:
        image = u

    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.isnan(image[i,j]): image[i,j] = np.min(image.flatten())

    images2 = nan_gaussian_filter(image, sigma=2)  
   
    thresh = images2
    thresh = thresh - np.min(thresh.flatten()) 
    thresh = thresh / np.max(thresh.flatten())
    usemanual = False

    while True:

        if usemanual:
            print('using manual center definition')
            centers = manual_define_points(images2, origin='upper')
            for i in range(len(centers)): 
                centers[i][0], centers[i][1] = centers[i][1], centers[i][0] # swapped convention 
            center_mask = np.zeros((images2.shape[0], images2.shape[1]))
            for i in range(len(centers)): center_mask[int(centers[i][0]), int(centers[i][1])] = 1

        else:
            print('using threshold of {}'.format(boundarythresh))
            AA_regions = (thresh < boundarythresh)
            labeled, _ = ndi.label(AA_regions)
            centers = []
            center_mask = np.zeros((images2.shape[0], images2.shape[1]))
            for index in range(1, np.max(labeled) + 1):
                label_mask = (labeled == index)
                region_indeces = label_mask.nonzero()
                avg_i = round(np.mean(region_indeces[0]))
                avg_j = round(np.mean(region_indeces[1]))
                centers.append([avg_i, avg_j])
                center_mask[int(avg_i), int(avg_j)] = 1

        markers, _ = ndi.label(center_mask)
        labels = watershed(images2, markers)
        region_masks = []
        WScenters = []
        for index in np.unique(labels):
            label_mask = (labels == index)
            region_indeces = label_mask.nonzero()
            avg_i = round(np.mean(region_indeces[0]))
            avg_j = round(np.mean(region_indeces[1]))
            WScenters.append([avg_i, avg_j])
            region_masks.append(label_mask)

        if refine:

            images = [images2, AA_regions, labeled, labels]
            f, ax = plt.subplots(1,len(images))
            for i in range(len(images)): ax[i].imshow(images[i], cmap=plt.cm.gray)
            for center in centers: 
                for i in range(len(images)): ax[i].scatter(center[1], center[0], c='r', s=0.5)
            plt.show()

            satisfied = boolquery('satisfied? (y/n')
            if not satisfied: 
                usemanual = boolquery('use manual? (y/n)')
                if not usemanual:
                    boundarythresh = float(input("ok, new threshold value?"))
            else: return labels, WScenters, region_masks
        else:
             return labels, WScenters, region_masks

# rotate u vectors until the sp connection closest to vertical is sp1 (since sp1,sp2,sp3 choice arbitrary)
def automatic_sp_rotation(u, centers, adjacency_type, eps=1e-4, transpose=False, plotting=False):
    # get possible orrientations: u, u2, and u3
    avg_theta, count = 0, 0
    # get all SP1 connected pairs
    for i in range(adjacency_type.shape[0]):
        for j in range(adjacency_type.shape[0]):
            if i>j and adjacency_type[i,j] == 1:
                if transpose: 
                    dx = centers[i][1] - centers[j][1] 
                    dy = centers[i][0] - centers[j][0] + eps
                else: 
                    dx = centers[i][0] - centers[j][0] 
                    dy = centers[i][1] - centers[j][1] + eps
                ang = np.arctan(dx/dy)
                #print("...", dx, '...', dy, '...', ang)
                avg_theta += ang
                count += 1
    avg_theta = (avg_theta)/count 
    rotations = [0, 1/3*np.pi, -1/3*np.pi]#, 2/3*np.pi, -2/3*np.pi] 
    rotang = rotations[np.argmin(np.abs([avg_theta - pos for pos in rotations]))]
    #print(avg_theta*180/np.pi)
    #print(rotang*180/np.pi)
    if plotting:
        f, ax = plt.subplots(1,2); 
        img = displacement_colorplot(ax[0], u, quiverbool=False)
        plot_adjacency(img, centers, adjacency_type, ax[0])
    u = rotate_uvecs(u, ang=(-rotang))
    # permute connections when rotation complete
    if rotang == -1/3*np.pi: 
        for i in range(adjacency_type.shape[0]):
            for j in range(adjacency_type.shape[0]):
                if adjacency_type[i,j] > 0: adjacency_type[i,j] = ((adjacency_type[i,j])%3)+1 #123->231
    elif rotang == 1/3*np.pi: 
        for i in range(adjacency_type.shape[0]):
            for j in range(adjacency_type.shape[0]):
                if adjacency_type[i,j] > 0: adjacency_type[i,j] = ((adjacency_type[i,j]+1)%3)+1 #123->312
    if plotting:
        img = displacement_colorplot(ax[1], u, quiverbool=False); 
        ax[1].set_title('rot by {}'.format(-rotang*180/np.pi)); 
        plot_adjacency(img, centers, adjacency_type, ax[1])
        plt.show(); 
    return u, -rotang, adjacency_type

def neighborDistFilter(u, thresh):
    ndist = normNeighborDistance(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if ndist[i,j] > thresh: u[i,j,:] = np.nan
    return u

def manual_adjust_voroni(u, regions, vertices):
    from utils import manual_adjust_points
    img = displacement_colorplot(None, u, quiverbool=False)
    manadjust = boolquery('manual adjust voroni?')
    while manadjust:
        vertices = manual_adjust_points(img, regions, vertices)
        if boolquery('satisfied?'): break
    return regions,vertices

def geometric_unwrap_tri(centers, adjacency_type, u_cart, voronibool):

    # AB WATERSHED
    u_rz_noshift = cartesian_to_rzcartesian(u_cart.copy(), sign_wrap=False, shift=False) 
    u_rz_shift = cartesian_to_rzcartesian(u_cart.copy(), sign_wrap=False, shift=True) 
    uang = np.zeros((u_cart.shape[0], u_cart.shape[1]))
    type1 = np.zeros((u_cart.shape[0], u_cart.shape[1]))
    type2 = np.zeros((u_cart.shape[0], u_cart.shape[1]))
    eps=1e-6
    for i in range(u_cart.shape[0]):
        for j in range(u_cart.shape[1]):
            #umag[i,j] = np.sqrt(u[i,j,1]**2 + u[i,j,0]**2) # cartesian!
            uang[i,j] = np.arctan(u_rz_noshift[i,j,1]/(eps + u_rz_noshift[i,j,0])) # cartesian!
            if uang[i,j] < 0: uang[i,j] += 2*np.pi # uang now between 0 and 2pi

            # for the AB, BA regions the angle is around pi/6, 3pi/6, 5pi/6, 7pi/6, 9pi/6, or 11pi/6
            # for the SP regions the angle is around 0, 2pi/6, 4pi/6, 6pi/6, 8pi/6, 10pi/6, or 12pi/6
            uang[i,j] = uang[i,j] * 6/np.pi

            # now for the AB, (BA) regions uang is around 1, (3), 5, (7), 9, (11)
            # now for the SP regions uang is around 0, 2, 4, 6, 8, 10, 12
            type1[i,j] = np.min( [np.abs(uang[i,j]-3), np.abs(uang[i,j]-7), np.abs(uang[i,j]-11)] )
            type2[i,j] = np.min( [np.abs(uang[i,j]-1), np.abs(uang[i,j]-5),  np.abs(uang[i,j]-9)] )
            uang[i,j] = np.min( [type1[i,j], type2[i,j]] )

    refineWS = True
    regions, centers, region_masks = watershed_regions(uang, 0.35, refineWS)

    # used to see if each region is an AB or BA type
    region_type = []
    for center in centers:
        if type1[center[0],center[1]] > type2[center[0],center[1]]: region_type.append(0)
        else: region_type.append(1)

    centers, adjacency_type = getAdjacencyMatrixAB(u_cart, boundary_val=0.3, delta_val=0.3, combine_crit=0, spdist=5, centers=centers, refine=refineConnect)
    
    vertices = None
    # get nm centers
    n = u_cart.shape[0]
    start_indx = np.argmin([(c[0]-n/2)**2 + (c[1]-n/2)**2 for c in centers]) # start at center closest to middle
    nmcenters, unwrap_warn  = unwrapBFS_AB(adjacency_type, centers, start=start_indx)

    #for i in range(len(centers)): u = rzSignAlign(region_masks[i], u, points[i])

    # get n,m zone adjustments for regions
    nmat, mmat = np.zeros((n, n)), np.zeros((n, n))
    for r in range(len(centers)):
        for i in range(n):
            for j in range(n):
                if region_masks[r][i,j]:
                    nmat[i,j] = nmcenters[r, 0]
                    mmat[i,j] = nmcenters[r, 1]
    # adjust zones 
    u_unwrapped, u_adjusts = adjust_zone(u_rz_shift, nmat, mmat)

    if False:
        f, ax = plt.subplots(2,2); 
        ax[0,0].quiver(u_rz_shift[:,:,0], u_rz_shift[:,:,1]); 
        ax[0,0].imshow(regions, origin='lower'); 
        ax[0,0].set_title('raw u')
        ax[0,1].quiver(u_unwrapped[:,:,0], u_unwrapped[:,:,1]); 
        ax[0,1].imshow(regions, origin='lower'); 
        ax[0,1].set_title('unwrap u')
        ax[1,0].imshow(nmat)
        ax[1,0].set_title('n offset')
        ax[1,1].imshow(mmat)
        ax[1,1].set_title('m offset')
        plt.show()   

    return u_unwrapped, nmat, mmat, unwrap_warn

def geometric_unwrap_tri_for_testing(centers, adjacency_type, u_cart, refineWS=False, refineConnect=False):

    # AB WATERSHED
    u_rz_noshift = cartesian_to_rzcartesian(u_cart.copy(), sign_wrap=False, shift=False) 
    u_rz_shift = cartesian_to_rzcartesian(u_cart.copy(), sign_wrap=False, shift=True) 
    uang = np.zeros((u_cart.shape[0], u_cart.shape[1]))
    type1 = np.zeros((u_cart.shape[0], u_cart.shape[1]))
    type2 = np.zeros((u_cart.shape[0], u_cart.shape[1]))
    eps=1e-6
    for i in range(u_cart.shape[0]):
        for j in range(u_cart.shape[1]):
            #umag[i,j] = np.sqrt(u[i,j,1]**2 + u[i,j,0]**2) # cartesian!
            uang[i,j] = np.arctan(u_rz_noshift[i,j,1]/(eps + u_rz_noshift[i,j,0])) # cartesian!
            if uang[i,j] < 0: uang[i,j] += 2*np.pi # uang now between 0 and 2pi

            # for the AB, BA regions the angle is around pi/6, 3pi/6, 5pi/6, 7pi/6, 9pi/6, or 11pi/6
            # for the SP regions the angle is around 0, 2pi/6, 4pi/6, 6pi/6, 8pi/6, 10pi/6, or 12pi/6
            uang[i,j] = uang[i,j] * 6/np.pi

            # now for the AB, (BA) regions uang is around 1, (3), 5, (7), 9, (11)
            # now for the SP regions uang is around 0, 2, 4, 6, 8, 10, 12
            type1[i,j] = np.min( [np.abs(uang[i,j]-3), np.abs(uang[i,j]-7), np.abs(uang[i,j]-11)] )
            type2[i,j] = np.min( [np.abs(uang[i,j]-1), np.abs(uang[i,j]-5),  np.abs(uang[i,j]-9)] )
            uang[i,j] = np.min( [type1[i,j], type2[i,j]] )

    regions, centers, region_masks = watershed_regions(uang, 0.35, refineWS)
    # GOOD TO NOW

    # used to see if each region is an AB or BA type
    region_type = []
    for center in centers:
        if type1[center[0],center[1]] > type2[center[0],center[1]]: region_type.append(0)
        else: region_type.append(1)

    centers, adjacency_type = getAdjacencyMatrixAB(u_cart, boundary_val=0.3, delta_val=0.3, combine_crit=0, spdist=5, centers=centers, refine=refineConnect)
    
    vertices = None
    # get nm centers
    n = u_cart.shape[0]
    start_indx = np.argmin([(c[0]-n/2)**2 + (c[1]-n/2)**2 for c in centers]) # start at center closest to middle
    nmcenters, unwrap_warn  = unwrapBFS_AB(adjacency_type, centers, start=start_indx)

    #for i in range(len(centers)): u = rzSignAlign(region_masks[i], u, points[i])

    # get n,m zone adjustments for voroni regions
    nmat, mmat = np.zeros((n, n)), np.zeros((n, n))
    for r in range(len(centers)):
        for i in range(n):
            for j in range(n):
                if region_masks[r][i,j]:
                    nmat[i,j] = nmcenters[r, 0]
                    mmat[i,j] = nmcenters[r, 1]
    # adjust zones 
    u_unwrapped, u_adjusts = adjust_zone(u_rz_shift, nmat, mmat)

    if False:
        f, ax = plt.subplots(2,2); 
        ax[0,0].quiver(u_rz_shift[:,:,0], u_rz_shift[:,:,1]); 
        ax[0,0].imshow(regions, origin='lower'); 
        ax[0,0].set_title('raw u')
        ax[0,1].quiver(u_unwrapped[:,:,0], u_unwrapped[:,:,1]); 
        ax[0,1].imshow(regions, origin='lower'); 
        ax[0,1].set_title('unwrap u')
        ax[1,0].imshow(nmat)
        ax[1,0].set_title('n offset')
        ax[1,1].imshow(mmat)
        ax[1,1].set_title('m offset')
        plt.show()   

    return u_unwrapped, nmat, mmat, unwrap_warn

def geometric_unwrap_for_testing(centers, adjacency_type, u, voronibool=False, plotting=False, ax=None, man_adust=False):

    # get voroni regions
    if voronibool:
        points = [ [c[1], c[0]] for c in centers ]
        vor = Voronoi(points)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        if man_adust: regions, vertices = manual_adjust_voroni(u, regions, vertices)
    else: # watershed
        regions, centers, region_masks = watershed_regions(u, refine=False)
        points = [ [c[1], c[0]] for c in centers ]
        centers, adjacency_type = getAdjacencyMatrix(u, boundary_val=0.3, delta_val=0.3, combine_crit=0, spdist=5, centers=centers, refine=False)
        vertices = None

    # get nm centers
    n = u.shape[0]
    start_indx = np.argmin([(c[0]-n/2)**2 + (c[1]-n/2)**2 for c in centers]) # start at AA center closest to middle
    nmcenters, unwrap_warn  = unwrapBFS(adjacency_type, centers, start=start_indx)
        
    # sign align each voroni region
    if voronibool:
        for i in range(len(regions)):
            p = Polygon(vertices[regions[i]])
            region_mask = np.zeros((n, n))
            for x in range(n):
                for y in range(n):
                    if p.contains(Point(x,y)): region_mask[y,x] = 1 # point func is transpose since above
            u = rzSignAlign(region_mask, u, points[i])
    else:
        for i in range(len(centers)):
            u = rzSignAlign(region_masks[i], u, points[i])
        if plotting: f, ax = plt.subplots(); ax.quiver(u[:,:,0], u[:,:,1]); ax.imshow(regions, origin='lower'); plt.show()

    # get n,m zone adjustments for voroni regions
    nmat, mmat = np.zeros((n, n)), np.zeros((n, n))
    if voronibool:
        for r in range(len(regions)):
            p = Polygon(vertices[regions[r]])
            for i in range(n):
                for j in range(n):
                    if p.contains(Point(i,j)):
                        nmat[j,i] = nmcenters[r, 0]
                        mmat[j,i] = nmcenters[r, 1]
    else: 
        for r in range(len(centers)):
            for i in range(n):
                for j in range(n):
                    if region_masks[r][i,j]:
                        nmat[i,j] = nmcenters[r, 0]
                        mmat[i,j] = nmcenters[r, 1]
    # adjust zones 
    u_unwrapped, u_adjusts = adjust_zone(u.copy(), nmat, mmat)

    return u_unwrapped, nmat, mmat, unwrap_warn

def geometric_unwrap(centers, adjacency_type, u, voronibool=False, plotting=False, ax=None, man_adust=True):

    # get voroni regions
    if voronibool:
        points = [ [c[1], c[0]] for c in centers ]
        vor = Voronoi(points)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        if man_adust: regions, vertices = manual_adjust_voroni(u, regions, vertices)
    else: # watershed
        regions, centers, region_masks = watershed_regions(u)
        points = [ [c[1], c[0]] for c in centers ]
        centers, adjacency_type = getAdjacencyMatrix(u, boundary_val=0.3, delta_val=0.3, combine_crit=0, spdist=5, centers=centers)
        vertices = None

    # get nm centers
    n = u.shape[0]
    start_indx = np.argmin([(c[0]-n/2)**2 + (c[1]-n/2)**2 for c in centers]) # start at AA center closest to middle
    nmcenters, unwrap_warn  = unwrapBFS(adjacency_type, centers, start=start_indx)
        
    # sign align each voroni region
    if voronibool:
        for i in range(len(regions)):
            p = Polygon(vertices[regions[i]])
            region_mask = np.zeros((n, n))
            for x in range(n):
                for y in range(n):
                    if p.contains(Point(x,y)): region_mask[y,x] = 1 # point func is transpose since above
            u = rzSignAlign(region_mask, u, points[i])
    else:
        for i in range(len(centers)):
            u = rzSignAlign(region_masks[i], u, points[i])
        f, ax = plt.subplots(); ax.quiver(u[:,:,0], u[:,:,1]); ax.imshow(regions, origin='lower'); plt.show()

    # get n,m zone adjustments for voroni regions
    nmat, mmat = np.zeros((n, n)), np.zeros((n, n))
    if voronibool:
        for r in range(len(regions)):
            p = Polygon(vertices[regions[r]])
            for i in range(n):
                for j in range(n):
                    if p.contains(Point(i,j)):
                        nmat[j,i] = nmcenters[r, 0]
                        mmat[j,i] = nmcenters[r, 1]
    else: 
        for r in range(len(centers)):
            for i in range(n):
                for j in range(n):
                    if region_masks[r][i,j]:
                        nmat[i,j] = nmcenters[r, 0]
                        mmat[i,j] = nmcenters[r, 1]
    # adjust zones 
    u_unwrapped, u_adjusts = adjust_zone(u.copy(), nmat, mmat)
    #if plotting and voronibool: 
    #    if ax == None: f, ax = plt.subplots()
    #    plot_voroni(points, nmcenters, regions, vertices, ax)
    #    ax.set_xlim(0-5, u.shape[0]+5)
    #    ax.set_ylim(0-5, u.shape[1]+5)
    #    plt.show()    

    if plotting and voronibool:
        f, ax = plt.subplots(2,3); 
        ax[0,0].quiver(u[:,:,0], u[:,:,1]); 
        ax[0,0].set_title('raw u')
        ax[0,1].quiver(u_unwrapped[:,:,0], u_unwrapped[:,:,1]); 
        ax[0,1].set_title('unwrap u')
        ax[1,0].imshow(nmat)
        ax[1,0].set_title('n offset')
        ax[1,1].imshow(mmat)
        ax[1,1].set_title('m offset')
        plot_voroni(points, nmcenters, regions, vertices, ax[0,2])
    elif plotting:
        f, ax = plt.subplots(2,2); 
        ax[0,0].quiver(u[:,:,0], u[:,:,1]); 
        ax[0,0].imshow(regions, origin='lower'); 
        ax[0,0].set_title('raw u')
        ax[0,1].quiver(u_unwrapped[:,:,0], u_unwrapped[:,:,1]); 
        ax[0,1].imshow(regions, origin='lower'); 
        ax[0,1].set_title('unwrap u')
        ax[1,0].imshow(nmat)
        ax[1,0].set_title('n offset')
        ax[1,1].imshow(mmat)
        ax[1,1].set_title('m offset')
    plt.show()  
    #exit()

    return u, u_unwrapped, u_adjusts, nmcenters, regions, vertices

# given some center locations and some lines, figure out if the lines connect 
# the centers (within a threshold spdist) 
def computeAdjacencyMatrixAA(centers, lines_sp1, lines_sp2, lines_sp3, spdist):
    adjacency_type  = np.zeros((len(centers), len(centers)))
    line_labels       = [1 for line in lines_sp1]
    line_labels.extend( [2 for line in lines_sp2])
    line_labels.extend( [3 for line in lines_sp3])
    lines = [line for line in lines_sp1]
    lines.extend(lines_sp2)
    lines.extend(lines_sp3)
    for i in range(len(lines)): # look at all lines
        line = lines[i]
        label = line_labels[i]
        last_indx = -1
        for xel, yel in zip(line[0], line[1]): # look at (x,y) the line traverses
            for i in range(len(centers)): # look at all centers
                center = centers[i] 
                # if the center is close (within spdist) to the line we say it passes through it
                if (((center[1] - xel)**2 + (center[0] - yel)**2)**0.5 < spdist): 
                    if last_indx != -1 and last_indx != i:
                        adjacency_type[i, last_indx] = label # so we're connecting these two centers
                        adjacency_type[last_indx, i] = label # and labeling the type of the connection
                    last_indx = i # keep track of last center we intersected for next connection
                    break 
    return adjacency_type

def computeAdjacencyMatrixAB(centers, lines_sp1, lines_sp2, lines_sp3, region_type=None):

    adjacency_type  = np.zeros((len(centers), len(centers)))
    line_labels       = [1 for line in lines_sp1]
    line_labels.extend( [2 for line in lines_sp2])
    line_labels.extend( [3 for line in lines_sp3])
    lines = [line for line in lines_sp1]
    lines.extend(lines_sp2)
    lines.extend(lines_sp3)

    def line_to_vert(line, plot=False):
        if plot: 
            f, ax = plt.subplots(1,2)
            ax[0].scatter(line[0], line[1])
        x_coords = []
        y_coords = [] 
        for x,y in zip(line[0], line[1]):
            if not np.isnan(x) and (not np.isnan(y)):
                x_coords.append(x)
                y_coords.append(y)
        if len(x_coords) > 0:
            # FAILS FOR VERT
            #A = vstack([x_coords,ones(len(x_coords))]).T
            #m, b = lstsq(A, y_coords)[0]
            #min_x2 = np.min(x_coords)
            #max_x2 = np.max(x_coords)
            #ax[1].plot([min_x2, max_x2], [min_x2*m + b, max_x2*m + b], c=colors[label])

            # FAILS FOR HORIZ
            y_coords, x_coords = x_coords, y_coords 
            A = vstack([x_coords,ones(len(x_coords))]).T
            m, b = lstsq(A, y_coords)[0]
            min_x2 = np.min(x_coords)
            max_x2 = np.max(x_coords)
            if plot: 
                ax[1].plot([min_x2*m + b, max_x2*m + b], [min_x2, max_x2])
            return True, [min_x2, min_x2*m + b], [max_x2, max_x2*m + b]
        return False, None, None

    def assignconnection(i1, i2):
        nintersect = 0
        labels = []
        for i in range(len(lines)):
            success, vertex1, vertex2 = line_to_vert(lines[i])
            if success and doIntersect(vertex1,vertex2, centers[i1], centers[i2]):
                nintersect += 1
                labels.append( line_labels[i] )
        if nintersect == 1 and (3 in labels): return 4 
        if nintersect == 2 and (3 in labels) and (2 in labels): return 1 
        if nintersect == 2 and (3 in labels) and (1 in labels): return 2 
        if nintersect == 2 and (2 in labels) and (1 in labels): return 3
        else: return 0

    for i1 in range(len(centers)):
        for i2 in range(i1+1, len(centers)):
            adjacency_type[i1, i2] = assignconnection(i1, i2)
            adjacency_type[i2, i1] = adjacency_type[i1, i2] 

    return adjacency_type

def getAdjacencyMatrixAuto(u, boundary_val, delta_val, combine_crit, spdist, centers=None, plt=False, AB=False, region_type=None):
    mask_aa = get_aa_mask(u, boundary=boundary_val)
    mask_sp1, mask_sp2, mask_sp3, _, _ = get_sp_masks(u, mask_aa, delta=delta_val)
    mask_sp1, lines_sp1 = get_sp_line_method2(mask_sp1, plotbool=False)
    mask_sp2, lines_sp2 = get_sp_line_method2(mask_sp2, plotbool=False)
    mask_sp3, lines_sp3 = get_sp_line_method2(mask_sp3, plotbool=False)
    labeled, aa_regions = ndimage.label(mask_aa)
    if centers == None:
        centers = get_region_centers(labeled, mask_aa)
        centers = combine_nearby_spots(centers, combine_criterion=combine_crit)

    if not AB:
        adjacency_type = computeAdjacencyMatrixAA(centers, lines_sp1, lines_sp2, lines_sp3, spdist)
    else:
        adjacency_type = computeAdjacencyMatrixAB(centers, lines_sp1, lines_sp2, lines_sp3, region_type)

    img = displacement_colorplot(None, u, quiverbool=False)
    plot_adjacency(img, centers, adjacency_type)
    if plt: plt.show()
    return centers, adjacency_type 

def getAdjacencyMatrix(u, boundary_val=None, delta_val=None, combine_crit=None, spdist=None, centers=None, refine=True, AB=False, region_type=None):
    img = displacement_colorplot(None, u)
    centers, adjacency_type = getAdjacencyMatrixAuto(u, boundary_val, delta_val, combine_crit, spdist, centers, AB=AB, region_type=region_type)
    if refine: 
        centers, adjacency_type = getAdjacencyMatrixManual(img, [[c[1], c[0]] for c in centers], adjacency_type) # manual adjust
        centers = [[c[1], c[0]] for c in centers]
    kept_center_indx = [i for i in range(len(centers)) if centers[i][0] != -1 ]
    kept_adjacency_type = np.zeros((len(kept_center_indx), len(kept_center_indx)))
    for i in range(len(kept_center_indx)):
        for j in range(len(kept_center_indx)):
            indx1, indx2 = kept_center_indx[i], kept_center_indx[j]
            kept_adjacency_type[i,j] = kept_adjacency_type[j,i] = adjacency_type[indx1, indx2]
    return [c for c in centers if c[0] != -1], kept_adjacency_type    

def getAdjacencyMatrixAB(u, boundary_val=None, delta_val=None, combine_crit=None, spdist=None, centers=None, refine=True):
    img = displacement_colorplot(None, u)
    adjacency_type  = np.zeros((len(centers), len(centers)))
    centers, adjacency_type = getAdjacencyMatrixAuto(u, boundary_val, delta_val, combine_crit, spdist, centers=centers, AB=True)
    if refine: 
        centers, adjacency_type = getAdjacencyMatrixManualAB(img, [[c[1], c[0]] for c in centers], adjacency_type) # manual adjust
        centers = [[c[1], c[0]] for c in centers]
    kept_center_indx = [i for i in range(len(centers)) if centers[i][0] != -1 ]
    kept_adjacency_type = np.zeros((len(kept_center_indx), len(kept_center_indx)))
    for i in range(len(kept_center_indx)):
        for j in range(len(kept_center_indx)):
            indx1, indx2 = kept_center_indx[i], kept_center_indx[j]
            kept_adjacency_type[i,j] = kept_adjacency_type[j,i] = adjacency_type[indx1, indx2]
    return [c for c in centers if c[0] != -1], kept_adjacency_type 

def get_neighbors(x, y, mat, diag=False):
    Nx, Ny = mat.shape[0], mat.shape[1]
    neighbors = [mat[x,y]]
    if x > 0: neighbors.append(mat[x-1, y])
    if y > 0: neighbors.append(mat[x, y-1])
    if x < Nx-1: neighbors.append(mat[x+1, y]) 
    if y < Ny-1: neighbors.append(mat[x,y+1])
    if diag and x > 0 and y > 0:  neighbors.append(mat[x-1, y-1]) 
    if diag and x > 0 and y < Ny-1: neighbors.append(mat[x-1, y+1])
    if diag and x < Nx-1 and y > 0:  neighbors.append(mat[x+1, y-1]) 
    if diag and x < Nx-1 and y < Ny-1: neighbors.append(mat[x+1, y+1])
    return neighbors    

def neighborfixed(Nx, Ny, x, y, variable_region, diag=False):
    nfixed = False
    if x > 0: nfixed = nfixed or variable_region[x-1, y]==0 
    if y > 0: nfixed = nfixed or variable_region[x, y-1]==0
    if x < Nx-1: nfixed = nfixed or variable_region[x+1, y]==0 
    if y < Ny-1: nfixed = nfixed or variable_region[x,y+1]==0
    if diag and x > 0 and y > 0:       nfixed = nfixed or variable_region[x-1, y-1]==0 
    if diag and x > 0 and y < Ny-1:    nfixed = nfixed or variable_region[x-1, y+1]==0
    if diag and x < Nx-1 and y > 0:    nfixed = nfixed or variable_region[x+1, y-1]==0 
    if diag and x < Nx-1 and y < Ny-1: nfixed = nfixed or variable_region[x+1, y+1]==0
    return nfixed

# similar results w/ and w/o fixneighboronly
def median_fit(uslice, variable_region, count=0, fixneighboronly=True, plotting=False, verbose=False, ndist_crit=None, extend=False, L1=False):
    nx, ny = uslice.shape[0], uslice.shape[1]
    med_ux = np.nanmedian(uslice[:,:,0])
    med_uy = np.nanmedian(uslice[:,:,1])
    if plotting:
        f, ax = plt.subplots(2)
        displacement_colorplot(ax[0], uslice[:,:,0], uslice[:,:,1])
    for i in range(nx):
        for j in range(ny):
            if variable_region[i,j]:
                equivs = get_nearby_equivs(uslice[i,j,:], extend=extend)
                dis_from_med = [ (u[0,0]-med_ux)**2 + (u[0,1] - med_uy)**2 for u in equivs ]
                new_u = equivs[np.argmin(dis_from_med)]
                uslice[i,j,:] = new_u[:]
                if fixneighboronly:
                    nfixed = neighborfixed(nx, ny, i, j, variable_region)
                    if nfixed: variable_region[i,j] = 0
                else: variable_region[i,j] = 0
    if plotting: displacement_colorplot(ax[1], uslice[:,:,0], uslice[:,:,1])
    return uslice, variable_region, count+1, None

def mip_fit   (uslice, variable_region, count=0, fixneighboronly=True, plotting=False, verbose=False, ndist_crit=None, extend=False, L1=True):
    
    if (np.sum(variable_region.flatten()) == 0): 
        return uslice, variable_region, count, None   
    orig_variable_region = variable_region.copy()     

    # variable region is a nx by ny boolean array
    fc1, fc2, n, m = cart_to_zonebasis(uslice)
    sign = np.ones((fc1.shape[0], fc1.shape[1]))
    solver  = GEKKO(remote=False)
    Nx, Ny  = fc1.shape[0], fc1.shape[1]
    for x in range(Nx):
        for y in range(Ny):
            if np.isnan(uslice[x,y,0]): 
                #print('turn off opt of ', x, y, ' since nan')
                variable_region[x,y] = 0
    if (np.sum(variable_region.flatten()) == 0): 
        return uslice, variable_region, count, None   

    def help_get_u(x, y, n_mat, m_mat, s_mat, value, fitvals=True):
        vmat = np.matrix([[-1, 1/2],[0, np.sqrt(3)/2]])
        if fitvals and variable_region[x,y]:
            if value: 
                v1 = (2 * s_mat[x][y].value[0] - 1)*fc1[x,y]+n_mat[x][y].value[0]
                v2 = (2 * s_mat[x][y].value[0] - 1)*fc2[x,y]+m_mat[x][y].value[0]
            else:
                v1 = (2 * s_mat[x][y] - 1)*fc1[x,y]+n_mat[x][y]
                v2 = (2 * s_mat[x][y] - 1)*fc2[x,y]+m_mat[x][y]
            return np.matmul(vmat, np.array([v1, v2]))
        else: # if fitvals False will always use guess
            v1 = sign[x,y]*fc1[x,y]+n[x,y]
            v2 = sign[x,y]*fc2[x,y]+m[x,y]
            return np.matmul(vmat, np.array([v1, v2]))

    def neigbor_dist_penalty(Nx, Ny, n_mat, m_mat, s_mat, value, fitvals=True):
        tot_pen = 0
        for x in range(Nx):
            for y in range(Ny):
                u_c = help_get_u(x, y, n_mat, m_mat, s_mat, value, fitvals)
                if ( not variable_region[x,y] and np.isnan(u_c[0,0]) ): continue
                if y < Ny-1: # add sq euclidean distance from bottom neighbor
                    u_b = help_get_u(x, y+1, n_mat, m_mat, s_mat, value, fitvals)
                    skipbool = ( not variable_region[x,y+1] and np.isnan(u_b[0,0]) ) 
                    if not L1 and not skipbool: tot_pen += (u_c[0,0] - u_b[0,0])**2 + (u_c[0,1] - u_b[0,1])**2
                    elif not skipbool: tot_pen += np.abs(u_c[0,0] - u_b[0,0]) + np.abs(u_c[0,1] - u_b[0,1])
                if x < Nx-1: # add sq euclidean distance from right neighbor
                    u_r = help_get_u(x+1, y, n_mat, m_mat, s_mat, value, fitvals)
                    skipbool = ( not variable_region[x+1,y] and  np.isnan(u_r[0,0]) ) 
                    if not L1 and not skipbool: tot_pen += (u_c[0,0] - u_r[0,0])**2 + (u_c[0,1] - u_r[0,1])**2
                    elif not skipbool: tot_pen += np.abs(u_c[0,0] - u_r[0,0]) + np.abs(u_c[0,1] - u_r[0,1])
        return tot_pen

    n_mat = [[solver.Var(lb=n[i,j]-2, ub=n[i,j]+2, integer=True) if variable_region[i,j] else n[i,j] for j in range(Ny)] for i in range(Nx)]
    m_mat = [[solver.Var(lb=m[i,j]-2, ub=m[i,j]+2, integer=True) if variable_region[i,j] else m[i,j] for j in range(Ny)] for i in range(Nx)]
    s_mat = [[solver.Var(lb=0, ub=1, integer=True) if variable_region[i,j] else 1 for j in range(Ny)] for i in range(Nx)] # 0,1 -> -1,1

    tot_pen =  neigbor_dist_penalty(Nx, Ny, n_mat, m_mat, s_mat, value=False)
    solver.Obj(tot_pen) # obective function
    solver.options.SOLVER = 1 # integer solution
    if verbose: solver.solve()
    else: solver.solve(disp=False, debug=0)

    uvecs_new = np.zeros((Nx, Ny, 2))
    nms_new = np.zeros((Nx, Ny, 3))
    new_variable_region = orig_variable_region.copy()    

    for x in range(Nx):
        for y in range(Ny):
            u = help_get_u(x, y, n_mat, m_mat, s_mat, True)
            uvecs_new[x,y,:] = u[0,0], u[0,1]
            if variable_region[x,y]: 
                nms_new[x,y,:] = n_mat[x][y].value[0], m_mat[x][y].value[0], (2 * s_mat[x][y].value[0] - 1)
                new_variable_region[x,y] = 0 # no longer variable after assigned
            else: 
                nms_new[x,y,:] = n_mat[x][y], m_mat[x][y], (2 * s_mat[x][y] - 1)

    if plotting:
        f, ax = plt.subplots(2,5)
        displacement_colorplot(ax[0,0], uslice[:,:,0], uslice[:,:,1])
        displacement_colorplot(ax[0,1], uvecs_new[:,:,0], uvecs_new[:,:,1])
        for x in range(Nx):
            for y in range(Ny): 
                if variable_region[x,y]: ax[0,1].scatter(y,x,c='k')
        ax[0,2].imshow(((nms_new[:,:,0]-n)!=0) + ((nms_new[:,:,1]-m)!=0) + ((nms_new[:,:,2]-sign)!=0) , origin='lower')
        ax[1,0].imshow(nms_new[:,:,0]-n, origin='lower')
        ax[1,1].imshow(nms_new[:,:,1]-m, origin='lower')
        ax[1,2].imshow(nms_new[:,:,2]-sign, origin='lower')
        #ax[0,3].imshow(old_ndist, origin='lower')
        #ax[1,3].imshow(ndist, origin='lower')
        ax[0,4].imshow(orig_variable_region, origin='lower')
        ax[1,4].imshow(new_variable_region, origin='lower')
        plt.show()

    return uvecs_new, new_variable_region, count+1, nms_new     

def BFS_from_center_uwrap(bin_w, variable_region, uvecs_raw, centers, xwid, ywid, ovlp, fittype, debug, ndist_crit=None, whole_region=False, extend=False, L1=True):

    if bin_w > 1:      
        uxnew    = bin(uvecs_raw[:,:,0], bin_w, method=np.median)
        uynew    = bin(uvecs_raw[:,:,1], bin_w, method=np.median)
        uvecs = np.zeros((uxnew.shape[0], uynew.shape[1], 2))
        uvecs[:,:,0], uvecs[:,:,1] = uxnew, uynew
        variable_region = bin(variable_region, bin_w, method=np.max)
        centers         = [[ center[0]/bin_w, center[1]/bin_w ] for center in centers]
    else: uvecs = uvecs_raw    

    if fittype == 'ip': fitfunc    = mip_fit    
    elif fittype == 'median': fitfunc = median_fit    

    nx, ny = uvecs.shape[0], uvecs.shape[1]
    queue = []
    for center in centers: queue.append(center)
    assert(len(queue) > 0)

    if debug:
        f, ax = plt.subplots()
        f2, ax3 = plt.subplots(2)
        plt.ion()

    # while there are still areas to assign... doing BFS-like search on variable_region starting from centers.
    # everythng is asigned when matrix is all zeros
    visited = np.zeros((nx, ny))
    counter = 0
    ipcount = 0
    while np.max(variable_region.flatten()) > 0 and len(queue) > 0:

        counter += 1
        f_assign = (1-np.sum(variable_region.flatten())/(nx*ny))
        f_visit = np.sum(visited.flatten())/(nx*ny)
        if counter%1000 == 0: 
            print(fittype,' fit ',100*f_assign,' % assigned, iter ',counter,' ',100*f_visit,' % visited ')

        # pull out next vertex
        center = queue.pop(0)
        cx, cy = int(np.round(center[0])), int(np.round(center[1]))
        if visited[cx, cy]: continue
        visited[cx, cy] = 1

        # 4 regions nearby, we're gonna asign at the 4 regions that share corner at center
        # then add these 4 corners to the queue to search outwards from them later, will visit after higher prio stuff 
        lc = int(np.round(np.max([center[0]-xwid, 0])))
        bc = int(np.round(np.max([center[1]-ywid, 0])))
        rc = int(np.round(np.min([center[0]+xwid, nx-1])))
        tc = int(np.round(np.min([center[1]+ywid, ny-1])))
        region_coords = [ (lc, cx, cy, tc), (lc, cx, bc, cy), (cx, rc, cy, tc), (cx, rc, bc, cy) ]
        new_centers = [ (np.min([rc-ovlp, nx-1]), np.max([bc+ovlp, 0])), 
                        (np.min([rc-ovlp, nx-1]), np.min([tc-ovlp, ny-1])), 
                        (np.max([lc+ovlp, 0]), np.max([bc+ovlp, 0])), 
                        (np.max([lc+ovlp, 0]), np.min([tc-ovlp, ny-1])) ]
        
        for c in new_centers: 
            lc = int(np.round(np.max([c[0]-xwid, 0])))
            bc = int(np.round(np.max([c[1]-ywid, 0])))
            rc = int(np.round(np.min([c[0]+xwid, nx-1])))
            tc = int(np.round(np.min([c[1]+ywid, ny-1])))
            if not visited[c[0], c[1]]:# and np.max(variable_region[bc:tc+1, lc:rc+1].flatten()) > 0:
                queue.append(c)

        # todo, adding centers in already assigned regions stop doing this it slows down crazy at the end !!     
        if debug:
            ax.scatter(cx, cy, c='b')
            for c in new_centers: ax.scatter(c[0], c[1], c='r')
            for coord_set in region_coords:     
                xmin, xmax, ymin, ymax = coord_set[:]
                rect = matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            ax.imshow(variable_region, origin='lower')
            plt.show()
            plt.pause(0.00000001)

        # now asign each of these 4 squares sharing a vertex at the center
        for coord_set in region_coords:        

            xmin, xmax, ymin, ymax = coord_set[:]              # pull out coords to define the square region
            uslice = uvecs[ymin:ymax+1, xmin:xmax+1, :]        # pull out uvectors of interest 
            region = variable_region[ymin:ymax+1, xmin:xmax+1] # and bools to tell which cells are constrained here
            if (np.max(region.flatten()) == 0): continue       # something needs to be variable (1), also way of checking vistied in BFS
            if (np.min(region.flatten()) != 0): 
                print('skipping region since nothing constrained')
                continue       # something needs to be constrained (0), something failed if not.
            if debug: 
                rect = matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='b', facecolor='none')
                ax.scatter(cx, cy, c='b')
                ax.add_patch(rect)
                ax.imshow(variable_region, origin='lower')
                plt.show()
                plt.pause(0.00000001)
                for axis in ax3.flatten(): axis.clear()
                displacement_colorplot(ax3[0],uslice[:,:,0], uslice[:,:,1])

            #try:
            if True:
                uslice, newregion, ipcount, nms_new = fitfunc(uslice, region.copy(), count=ipcount, ndist_crit=ndist_crit, extend=extend, L1=L1)    # the meat - do the integer/median program
                if whole_region:
                    print('flipping whole region to IP soltion')
                    n,m,c = 0,0,0
                    for x in range(region.shape[0]):
                        for y in range(region.shape[1]): 
                            if region[x,y]: 
                                n += nms_new[x,y,0]
                                c += 1 
                                m += nms_new[x,y,1]
                    n, m = n/c, m/c
                    print(nms_new[:,:,0])
                    print(nms_new[:,:,1])
                    for x in range(variable_region.shape[0]):
                        for y in range(variable_region.shape[1]): 
                            if variable_region[x,y]:
                                uvecs[x,y] = adjust_zone_single(uvecs[x,y], n, m)
                    variable_region = np.zeros((variable_region.shape))

                else:
                    uvecs[ymin:ymax+1, xmin:xmax+1, :] = uslice[:, :]                                 # update uvectors now
                    variable_region[ymin:ymax+1, xmin:xmax+1] = newregion[:,:]                           # set as assigned, no longer variable. way of saying vertex traversed

            #except KeyboardInterrupt: exit()
            #except: print('region fit failed wait on other regions')

            if debug:
                displacement_colorplot(ax3[1],uslice[:,:,0], uslice[:,:,1])
                plt.show()
                plt.pause(0.000000001)

    ux, uy, variable_region = unbin(uvecs[:,:,0], bin_w), unbin(uvecs[:,:,1], bin_w), unbin(variable_region, bin_w)
    uvecs_return = np.zeros((ux.shape[0], ux.shape[1], 2))
    uvecs_return[:,:,0], uvecs_return[:,:,1] = ux, uy
    #print('ran ', ipcount, ' times')
    return uvecs_return, variable_region   

def refit_by_region(u, refit_regions, maxsize=100, minsize=0, width=2, onlysmall=True, fittype='ip'):
    labeled, regions = ndimage.label(refit_regions)
    for r in range(1, regions):
        region = (labeled == r)
        size = np.sum(region.flatten())
        xavg, yavg, c = 0,0,0
        for x in range(region.shape[0]):
            for y in range(region.shape[1]):
                if region[y,x]: 
                    xavg, yavg, c = xavg+x, yavg+y, c+1
        xavg, yavg = xavg/c, yavg/c
        if size > minsize and size < 2:  
            u, reg = BFS_from_center_uwrap(1, region, u.copy(), extend=False, centers=[[xavg, yavg]], xwid=2, ywid=2, ovlp=1, fittype=fittype, L1=False, debug=False) 
        elif size > minsize and size < 6: 
            u, reg = BFS_from_center_uwrap(1, region, u.copy(), extend=False, centers=[[xavg, yavg]], xwid=3, ywid=3, ovlp=1, fittype=fittype, L1=False, debug=False)
        elif size > minsize and size < maxsize: 
            u, reg = BFS_from_center_uwrap(1, region, u.copy(), extend=False, centers=[[xavg, yavg]], xwid=width, ywid=width, ovlp=1, fittype=fittype, L1=False, debug=False)
    return u, np.zeros((refit_regions.shape))

def unwrap_from_boundary(u, region_mask, fitfunc, L1, width=1):
    while np.max(region_mask.flatten()) > 0:
        u, region_mask = optimize_boundary(u, region_mask, fitfunc, L1, width)
    return u

def optimize_boundary(u, region_mask, fitfunc, L1, w):
    nx, ny = u.shape[0], u.shape[1]
    contour = measure.find_contours(region_mask, 0.5)
    contour = contour[0]
    queue = []
    dist_from_center = []
    for i in range(contour.shape[0]):
        xpt, ypt = contour[i,0], contour[i,1]
        dist_from_center.append((ypt - ny//2)**2 + (xpt - nx//2)**2)
        queue.append([ypt, xpt])

    queue = [queue[i] for i in np.argsort(dist_from_center)]
    
    while len(queue) > 0: 

        center = queue.pop(0)
        xmin = int(np.round(np.max([center[0]-w, 0])))
        ymin = int(np.round(np.max([center[1]-w, 0])))
        xmax = int(np.round(np.min([center[0]+w, nx-1])))
        ymax = int(np.round(np.min([center[1]+w, ny-1])))
        uslice = u[ymin:ymax+1, xmin:xmax+1, :]        
        region = region_mask[ymin:ymax+1, xmin:xmax+1]

        if False:
            f, ax = plt.subplots(1,2)
            ax[0].imshow(region_mask, origin='lower')
            ax[1].imshow(region, origin='lower')
            rect = matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none')
            ax[0].scatter(center[0], center[1])
            ax[0].add_patch(rect)
            plt.show()
            
        #print('fitting...')
        try:
            uslice, region, _, _ = fitfunc(uslice.copy(), region.copy(), L1=L1)  
            u[ymin:ymax+1, xmin:xmax+1, :] = uslice[:,:,:]       
            region_mask[ymin:ymax+1, xmin:xmax+1] = region[:,:]
        except:
            uslice, region, _, _ = fitfunc(uslice.copy(), region.copy(), L1=L1, verbose=True)  
        if np.max(region_mask.flatten()) == 0: break

    return u, region_mask

def test_unwrap_boundary():
    u = np.zeros((50,50,2))
    r = np.zeros((50,50))
    r[10:18,10:18] = 1
    def fitfunc(u, r, L1): 
        r[:,:] = 0
        return u, r, None, None
    u = unwrap_from_boundary(u, r, fitfunc, True, 1)

def refit_convex_regions(u, refit_regions, maxsize=100, minsize=0, width=1, fittype='ip'):
    refit_regions = bin(refit_regions, bin_w=2, size_retain=True, method=np.max)
    labeled, regions = ndimage.label(refit_regions)
    if fittype=='ip': 
        fitfunc=mip_fit
    else: 
        fitfunc=median_fit
    for r in range(1, regions): # first region is background
        region = (labeled == r)
        size = np.sum(region.flatten())
        if size >= minsize and size <= maxsize:
            region = convex_mask(region)
            u = unwrap_from_boundary(u, region, fitfunc=fitfunc, L1=False, width=width)
    return u

# didnt help much
def refit_outliers(bin_w, u, points, xwid, ywid, ovlp, refitbool=None, ndist_crit=None, fittype='median', offset=None, debug=False, extend=False, L1=True):
    nx, ny = u.shape[0], u.shape[1]
    variable_region = np.zeros((nx, ny))
    middle = [[int(nx/2), int(ny/2)]]
    if refitbool is None:
        ndist = normNeighborDistance(u)
        #f, ax = plt.subplots(1,3)
        #ax[2].imshow(ndist, origin='lower', vmax=0.05, vmin=0)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                mini = np.max([i-offset, 0])
                maxi = np.min([i+offset, u.shape[0]-1])
                minj = np.max([j-offset, 0])
                maxj = np.min([j+offset, u.shape[1]-1])
                if np.max(ndist[mini:maxi, minj:maxj].flatten()) > ndist_crit: variable_region[i,j] = 1
    else: 
        variable_region = refitbool   
        #f, ax = plt.subplots(1,2)
    #displacement_colorplot(ax[0], u)
    #ax[1].imshow(variable_region, origin='lower')
    #plt.show()
    return BFS_from_center_uwrap(bin_w, variable_region.copy(), u.copy(), ndist_crit=ndist_crit, extend=extend, L1=L1, centers=middle, xwid=xwid, ywid=ywid, ovlp=ovlp, fittype=fittype, debug=debug) 

# takes a chunk of the displacements all in the same 'zone' and flips until they wind around the center 
# accomplished by using that displacements with y > y_center should have a negative x component,
# those with y < y_center should have a positive x component, those with x > x_cent should have a pos y, 
# and those with x < x_cent should have a neg y. 
# --> maximize (y - y_cent)*(-ux) + (x - x_cent)*uy (better to maximize curl? nonlocal... harder...)
def rzSignAlign(region, uvecs, center):
    ycent, xcent = center[0], center[1]
    uvecs_orriented = np.zeros((uvecs.shape[0], uvecs.shape[1], 2))
    testing = np.zeros((uvecs.shape[0], uvecs.shape[1]))
    testing2 = np.zeros((uvecs.shape[0], uvecs.shape[1]))
    for x in range(region.shape[0]):
        for y in range(region.shape[1]):
            if region[x,y]: 
                uy, ux = uvecs[x,y,:] 
                #assert(in_rz_cart(uvecs[x,y,:]))
                xcrit  =  (x-xcent)*uy
                ycrit  = -(y-ycent)*ux 
                if   xcrit < 0 and ycrit < 0:  uvecs_orriented[x,y,:] = -uy, -ux
                elif xcrit > 0 and ycrit > 0:  uvecs_orriented[x,y,:] =  uy,  ux
                elif xcrit < 0 and np.abs(x-xcent) > np.abs(y-ycent): uvecs_orriented[x,y,:] = -uy, -ux
                elif ycrit < 0 and np.abs(x-xcent) < np.abs(y-ycent): uvecs_orriented[x,y,:] = -uy, -ux
                elif xcrit < 0 and np.abs(x-xcent) < np.abs(y-ycent): uvecs_orriented[x,y,:] =  uy,  ux
                elif ycrit < 0 and np.abs(x-xcent) > np.abs(y-ycent): uvecs_orriented[x,y,:] =  uy,  ux
                else: uvecs_orriented[x,y,:] = uy, ux
            else: uvecs_orriented[x,y,:] = uvecs[x,y,:] 

    if False:
        f,ax = plt.subplots(2,3)
        ax[0,0].imshow(region, origin='lower')
        ax[0,0].scatter(center[0], center[1])
        ax[0,1].imshow(testing, origin='lower')
        ax[1,1].imshow(np.sign(uvecs_orriented[:,:,0]), origin='lower')
        ax[0,1].scatter(center[0], center[1])
        ax[0,2].imshow(testing2, origin='lower')
        ax[1,2].imshow(np.sign(uvecs_orriented[:,:,1]), origin='lower')
        ax[0,2].scatter(center[0], center[1])
        ax[1,0].quiver(uvecs_orriented[:,:,0], uvecs_orriented[:,:,1])
        f, ax = plt.subplots()
        displacement_colorplot(ax, uvecs_orriented[:,:,0], uvecs_orriented[:,:,1])
        plt.show()
        exit()

    return uvecs_orriented

def normDistToNearestCenter(nx, ny, centers):
    dists = np.ones((nx, ny)) * np.inf
    for x in range(nx):
        for y in range(ny):
            for center in centers: dists[x,y] = np.min([(x-center[0])**2 + (y-center[1])**2, dists[x,y]])
    dists = dists/np.max(dists.flatten())
    return dists  

def unwrapBFS_AB(adjacency_type, centers, start): 
    visited   = np.zeros((adjacency_type.shape[0], 1))  # bool for if vertex is visited
    nmcenters = np.zeros((adjacency_type.shape[0], 2))  # what we're determining - the unwrapped offsets
    visited[start] = 1
    nmcenters[start, :] = [0,0]
    queue = []
    queue.append(start) 

    unwrap_warn = False

    while len(queue) > 0: # visit each vertex
        vert = queue.pop(0) # remove visted vertex from queue
        neighborlist = [i for i in range(adjacency_type.shape[0]) if adjacency_type[i, vert] in [1,2,3,4]]
        for neighbor in neighborlist:
            if not visited[neighbor]: # scan through unvisted neighbors
                visited[neighbor] = 1 
                # only check sp1 and sp2 type adjacency since sp3 is a combo of these... redundant basis
                # use sp3 to check work later?
                if adjacency_type[neighbor, vert] == 1: # sp1 type adjacency
                    if centers[neighbor][0] > centers[vert][0]: sign = -1
                    else: sign = 1 # if increasing in y we're adding v1 so n here increases by 1, else decrease by 1
                    nmcenters[neighbor, :] = nmcenters[vert, :] + [sign*1,0]
                if adjacency_type[neighbor, vert] == 2: # sp2 type adjacency
                    if centers[neighbor][1] > centers[vert][1]: sign = -1
                    else: sign = 1 # if increasing in x we're adding v2 so m here increases by 1, else decrease by 1
                    nmcenters[neighbor, :] = nmcenters[vert, :] + [0,sign*1]
                if adjacency_type[neighbor, vert] == 3: # sp3 type adjacency, v3 = v2-v1 here sign convention
                    if centers[neighbor][1] > centers[vert][1]: sign = -1
                    else: sign = 1 # if increasing in x we're adding v3 so m here increases by 1, else decrease by 1
                    nmcenters[neighbor, :] = nmcenters[vert, :] + [sign*1,sign*1] #[sign*-1,sign*1]
                if adjacency_type[neighbor, vert] == 4:
                    nmcenters[neighbor, :] = nmcenters[vert, :] # same cell!
                queue.append(neighbor) # now put neighbor on queue to check it 
            else: 
                # assert neighbor assignment agrees with own connection
                if adjacency_type[neighbor, vert] == 1: # sp1 type adjacency
                    if centers[neighbor][0] > centers[vert][0]: sign = -1
                    else: sign = 1 # if increasing in y we're adding v1 so n here increases by 1, else decrease by 1
                    expect = nmcenters[vert, :] + [sign*1,0]
                    if (nmcenters[neighbor, 0] != expect[0] or nmcenters[neighbor, 1] != expect[1]):
                        print('WARNING: inconsistency in the center adjacencies, please make sure connections are good and try again')
                        print('error occured connecting {} and {} with SP{}'.format(vert, neighbor, 1))
                        print('algorithm might still work but could misattribute the lattice vector offsets added to each zone')
                        unwrap_warn = True
                if adjacency_type[neighbor, vert] == 2: # sp2 type adjacency
                    if centers[neighbor][1] > centers[vert][1]: sign = -1
                    else: sign = 1 # if increasing in x we're adding v2 so m here increases by 1, else decrease by 1
                    expect = nmcenters[vert, :] + [0,sign*1]
                    if (nmcenters[neighbor, 0] != expect[0] or nmcenters[neighbor, 1] != expect[1]):    
                        print('WARNING: inconsistency in the center adjacencies, please make sure connections are good and try again')
                        print('error occured connecting {} and {} with SP{}'.format(vert, neighbor, 2))
                        print('algorithm might still work but could misattribute the lattice vector offsets added to each zone')
                        unwrap_warn = True
    for vert in range(adjacency_type.shape[0]):
        if not visited[vert]:
            print('Error, found erroneous AB center unconnected to neighbors, please remove and try again')
            exit()
    return nmcenters, unwrap_warn

# function for BFS on an adjacency matrix, matrix element is 'type' of neighbor (connected by sp1, sp2 or sp3)
# will find the integers n,m for each region so that the displacements in this region will be added to n*v1 + m*v2 (v1, v2 lattice vecs)
# in order to get the overall unwrapped displacements
def unwrapBFS(adjacency_type, centers, start): 
    visited   = np.zeros((adjacency_type.shape[0], 1))  # bool for if vertex is visited
    nmcenters = np.zeros((adjacency_type.shape[0], 2))  # what we're determining - the unwrapped offsets
    visited[start] = 1
    nmcenters[start, :] = [0,0]
    queue = []
    queue.append(start) 
    unwrap_warn = False

    # more confident about sp1, sp2 connections so do those first
    while len(queue) > 0: # visit each vertex
        vert = queue.pop(0) # remove visted vertex from queue
        neighborlist = [i for i in range(adjacency_type.shape[0]) if adjacency_type[i, vert] in [1,2,3]]
        for neighbor in neighborlist:
            if not visited[neighbor]: # scan through unvisted neighbors
                visited[neighbor] = 1 
                # only check sp1 and sp2 type adjacency since sp3 is a combo of these... redundant basis
                # use sp3 to check work later?
                if adjacency_type[neighbor, vert] == 1: # sp1 type adjacency
                    if centers[neighbor][0] > centers[vert][0]: sign = -1
                    else: sign = 1 # if increasing in y we're adding v1 so n here increases by 1, else decrease by 1
                    nmcenters[neighbor, :] = nmcenters[vert, :] + [sign*1,0]
                if adjacency_type[neighbor, vert] == 2: # sp2 type adjacency
                    if centers[neighbor][1] > centers[vert][1]: sign = -1
                    else: sign = 1 # if increasing in x we're adding v2 so m here increases by 1, else decrease by 1
                    nmcenters[neighbor, :] = nmcenters[vert, :] + [0,sign*1]
                if adjacency_type[neighbor, vert] == 3: # sp3 type adjacency, v3 = v2-v1 here sign convention
                    if centers[neighbor][1] > centers[vert][1]: sign = -1
                    else: sign = 1 # if increasing in x we're adding v3 so m here increases by 1, else decrease by 1
                    nmcenters[neighbor, :] = nmcenters[vert, :] + [sign*1,sign*1] #[sign*-1,sign*1]
                queue.append(neighbor) # now put neighbor on queue to check it 
            else: 
                # assert neighbor assignment agrees with own connection
                if adjacency_type[neighbor, vert] == 1: # sp1 type adjacency
                    if centers[neighbor][0] > centers[vert][0]: sign = -1
                    else: sign = 1 # if increasing in y we're adding v1 so n here increases by 1, else decrease by 1
                    expect = nmcenters[vert, :] + [sign*1,0]
                    if (nmcenters[neighbor, 0] != expect[0] or nmcenters[neighbor, 1] != expect[1]):
                        print('WARNING: inconsistency in the center adjacencies, please make sure connections are good and try again')
                        print('error occured connecting {} and {} with SP{}'.format(vert, neighbor, 1))
                        print('algorithm might still work but could misattribute the lattice vector offsets added to each zone')
                        unwrap_warn = True
                if adjacency_type[neighbor, vert] == 2: # sp2 type adjacency
                    if centers[neighbor][1] > centers[vert][1]: sign = -1
                    else: sign = 1 # if increasing in x we're adding v2 so m here increases by 1, else decrease by 1
                    expect = nmcenters[vert, :] + [0,sign*1]
                    if (nmcenters[neighbor, 0] != expect[0] or nmcenters[neighbor, 1] != expect[1]):    
                        print('WARNING: inconsistency in the center adjacencies, please make sure connections are good and try again')
                        print('error occured connecting {} and {} with SP{}'.format(vert, neighbor, 2))
                        print('algorithm might still work but could misattribute the lattice vector offsets added to each zone')
                        unwrap_warn = True
    for vert in range(adjacency_type.shape[0]):
        if not visited[vert]:
            print('Error, found erroneous AA center unconnected to neighbors, please remove and try again.')
            exit()
    return nmcenters, unwrap_warn

def generalBFS(start, getneighbors, do_a_thing, thing): 
    visited   = np.zeros((n, 1))  # bool for if vertex is visited
    visited[start] = 1
    queue = []
    queue.append(start) 
    while len(queue) > 0: # visit each vertex
        vert = queue.pop(0) # remove visted vertex from queue
        thing = do_a_thing(thing, vert)
        for neighbor in getneighbors(vert):
          if not visited[neighbor]: # scan through unvisted neighbors
            visited[neighbor] = 1 
            queue.append(neighbor) # now put neighbor on queue to check it 
    return thing

# from stackexchange did not write this!
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    """
    if vor.points.shape[1] != 2:raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:radius = vor.points.ptp().max()
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


if __name__ == '__main__':
    test_unwrap_boundary()