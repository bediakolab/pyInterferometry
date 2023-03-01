
from utils import *
from masking import *
import matplotlib
import numpy as np
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import glob
import pickle
from visualization import *
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
import warnings
import numpy.random as random
#from  interferometry_fitting import *
from new_utils import import_uvector, get_lengths, get_angles, get_area, crop_displacement
from visualization import displacement_colorplot, make_legend
from basis_utils import latticevec_to_cartesian
from scipy.spatial import Delaunay
from utils import manual_define_triangles, manual_define_lengths



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

    return a12, a13, a23

def ask_user():
    print('WARNING: Assuming GRAPHENE, ss=0.5nm! ')
    return False, 0.246, 0.5, 0.19, None, None
    #return is_hbl, a, pixelsize, poissonratio, delta, twist

def getColors(thetas, het_strains):
    # get colormaps
    colormap_func = matplotlib.cm.get_cmap('plasma')
    theta_colors = [t if not np.isnan(t) else 0 for t in thetas]
    norm = Normalize()
    norm.autoscale(theta_colors)
    theta_colors = colormap_func(norm(theta_colors))
    het_strain_colors = [t if not np.isnan(t) else 0 for t in het_strains]
    norm = Normalize()
    norm.autoscale(het_strain_colors)
    het_strain_colors = colormap_func(norm(het_strain_colors))
    return theta_colors, het_strain_colors

def plotTriangleQuantity(triangles, tri_centers, values, colors, ax, title, centers, manual=False, use_tris=None):

    if use_tris is None: use_tris = [True for tri in triangles]
    for n in range(len(triangles)):
        triangle = triangles[n]
        val = values[n]
        if not np.isnan(val) and use_tris[n]:
            trix = []
            triy = []
            tri_center = tri_centers[n]
            for i in range(3):
                if not manual: 
                    trix.append(centers[triangle[i]][1])
                    triy.append(centers[triangle[i]][0])
                    ax.plot([centers[triangle[(i+1)%3]][1], centers[triangle[i]][1]], [centers[triangle[(i+1)%3]][0], centers[triangle[i]][0]], color="grey", linewidth=0.5)
                else:
                    trix.append(triangle[i][0])
                    triy.append(triangle[i][1])
                    ax.plot([triangle[(i+1)%3][0], triangle[i][0]], [triangle[(i+1)%3][1], triangle[i][1]], color="grey", linewidth=0.5)    
            ax.fill(trix, triy, color=colors[n])
            ax.text(tri_center[0], tri_center[1], "{:.2f}".format(val), color='grey', fontsize='xx-small', horizontalalignment='center')
    ax.axis('off')
    ax.set_title(title)

def plotTris(triangles, ax, centers, manual=False, use_tris=None):
    if use_tris is None: use_tris = [True for tri in triangles]
    for n in range(len(triangles)):
        triangle = triangles[n]
        if use_tris[n]:
            for i in range(3):
                if not manual: 
                    ax.plot([centers[triangle[(i+1)%3]][1], centers[triangle[i]][1]], [centers[triangle[(i+1)%3]][0], centers[triangle[i]][0]], color="grey", linewidth=0.5)
                else:
                    ax.plot([triangle[(i+1)%3][0], triangle[i][0]], [triangle[(i+1)%3][1], triangle[i][1]], color="grey", linewidth=0.5)    
    ax.axis('off')
    
def computeAllTris(triangles, centers, pixelsize, poissonratio, a, delta, is_hbl, given_twist, manual=False):
    tri_centers, thetas, het_strains, area_fracts, angs_all = [], [], [], [], []
    for triangle in triangles:
        if not manual:
            dists = [(centers[triangle[0]][0]**2 + centers[triangle[0]][1]**2), 
                     (centers[triangle[1]][0]**2 + centers[triangle[1]][1]**2), 
                     (centers[triangle[2]][0]**2 + centers[triangle[2]][1]**2)]
            order = np.argsort(dists)
            point1 = centers[triangle[order[0]]]
            point2 = centers[triangle[order[1]]]
            point3 = centers[triangle[order[2]]]
            center_x = np.mean([point1[1], point2[1], point3[1]])
            center_y = np.mean([point1[0], point2[0], point3[0]])
        else: # different syntax slightly for man def triangles
            dists = [(triangle[0][0]**2 + triangle[0][1]**2), (triangle[1][0]**2 + triangle[1][1]**2), (triangle[2][0]**2 + triangle[2][1]**2)]
            order = np.argsort(dists)
            point1 = triangle[order[0]]
            point2 = triangle[order[1]]
            point3 = triangle[order[2]]
            center_y = np.mean([point1[1], point2[1], point3[1]])
            center_x = np.mean([point1[0], point2[0], point3[0]])

        l1, l2, l3 = get_lengths(point1, point2, point3)
        a12, a23, a31 = get_angles(l1,l2,l3)
        a1, a2, a3 = get_angles_from_vertical(point1, point2, point3)
        angs = np.sort(np.array([a1, a2, a3])) * 180/np.pi
        max_a = np.abs(np.max([a12, a23, a31])) * 180/np.pi
        l1, l2, l3 = l1*pixelsize, l2*pixelsize, l3*pixelsize
        if max_a < 160:
            theta_t, theta_s, eps = fit_heterostrain(l1, l2, l3, poissonratio, a, delta, is_hbl, given_twist)
            tri_centers.append([center_x, center_y])
            thetas.append(theta_t)
            angs_all.append(angs)
            het_strains.append(eps)
            area_fracts.append(get_area(l1, l2, l3))
        else:
            thetas.append(np.nan)
            het_strains.append(np.nan)
            tri_centers.append([np.nan, np.nan])
            area_fracts.append(np.nan)
            angs_all.append(np.nan)        
    return tri_centers, thetas, het_strains, area_fracts, angs_all

def extract_twist_vdf(img, dsnum, prefix):

    #matplotlib.use('Qt5Agg')
    vecs = manual_define_lengths(img)
    vecs = vecs[0]
    point1 = vecs[0]
    point2 = vecs[1]
    _, a, pixelsize, _, _, _ = ask_user()
    L = ((point1[0] - point2[0]) **2 + (point1[1] - point2[1]) **2)**0.5
    L = L*pixelsize/a
    # 0.5/L = sin(theta/2)
    twist = np.arcsin(0.5/L) * 2
    twist = 180/np.pi * twist 
    print(twist)    

def extract_heterostrain_vdf(img, dsnum, prefix, ss=0.5):

    #matplotlib.use('Qt5Agg')
    triangles = manual_define_triangles(img)
    f, ((ax6, ax3), (ax4, ax5)) = plt.subplots(2,2)
    is_hbl, a, pixelsize, poissonratio, delta, given_twist = ask_user()
    pixelsize = ss
    centers = None 
    manual = True
    tri_centers, thetas, het_strains, area_fracts, angs_all = computeAllTris(triangles, centers, pixelsize, poissonratio, a, delta, is_hbl, given_twist, manual)
    theta_colors, het_strain_colors = getColors(thetas, het_strains)

    plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax4, '$<\\theta_m> = {:.2f}^o$'.format(np.nanmean(thetas)), centers, manual)
    plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax5, '$<\epsilon> = {:.2f}%$'.format(np.nanmean(het_strains)), centers, manual)
    
    writefile('{}/{}_angles_from_vert.txt'.format(prefix, dsnum), 'angles (deg) from vert for {} (manual), any can be += 180'.format(dsnum), angs_all)
    writefile('{}/{}_angles.txt'.format(prefix, dsnum), 'twist angles for {} (manual)'.format(dsnum), thetas)
    print('{}/{}_angles.txt'.format(prefix, dsnum))
    for t in thetas: print(t)
    writefile('{}/{}_hetstrains.txt'.format(prefix, dsnum), 'heterostrains for {} (manual)'.format(dsnum), het_strains)
    print('{}/{}_hetstrains.txt'.format(prefix, dsnum))
    for t in het_strains: print(t)
    writefile('{}/{}_areafracts.txt'.format(prefix, dsnum), 'area fractions for {}'.format(dsnum), area_fracts)

    ax3.axis('off')
    ax6.imshow(img, origin='lower', cmap='plasma')
    ax6.axis('off')
    ax6.set_title('vdf sum')
    plt.savefig("{}/{}_hetstrains_manualdef_recent.png".format(prefix, dsnum), dpi=600)
    plt.show()

def auto_triangles_vdf(vdf):    
    GLOBAL_BOUNDARYVAL, GLOBAL_DELTAVAL, GLOBAL_COMBINE_CRIT, GLOBAL_COMBINE_CRIT = params[:]
    boundary_val = GLOBAL_BOUNDARYVAL
    delta_val = GLOBAL_DELTAVAL
    combine_crit = GLOBAL_COMBINE_CRIT
    spdist = GLOBAL_SPDIST
    nx, ny = ufit.shape[0:2]
    mask_aa = (vdf > boundary_val)
    labeled, aa_regions = ndimage.label(mask_aa)
    centers = get_region_centers(labeled, mask_aa)
    centers = combine_nearby_spots(centers, combine_criterion=combine_crit)
    triangles = Delaunay(centers)
    return triangles, centers

# determine heterostrain and twist angle given lengths of moire triangle
# delta = lattice mismatch NOT poisson ratio
def fit_heterostrain(l1, l2, l3, poissonratio, a, delta, hbl=False, given_twist=None, guess_theta_t=0.2, guess_theta_s=25, guess_hs=0.05):
    if not hbl:
        def cost_func(L, theta_t, theta_s, eps):
            k = 4*np.pi/(np.sqrt(3)*a)
            K = np.array( [[k, 0], [k*0.5, k*0.86602540378], [-k*0.5, k*0.86602540378]] )
            R_t = np.array([[np.cos(theta_t), -np.sin(theta_t)], [np.sin(theta_t), np.cos(theta_t)]])
            R_s = np.array([[np.cos(theta_s), -np.sin(theta_s)], [np.sin(theta_s), np.cos(theta_s)]])
            R_ns = np.array([[np.cos(-theta_s), -np.sin(-theta_s)], [np.sin(-theta_s), np.cos(-theta_s)]])
            E = np.array([[1/(1+eps), 0],[0, 1/(1-poissonratio*eps)]])
            M = R_t - np.matmul(R_ns, np.matmul(E, R_s))
            Y = [0,0,0]
            for i in range(3):
                v = np.matmul(M, K[i,:])
                l = (np.dot(v,v))**0.5
                Y[i] = 4*np.pi/(np.sqrt(3)*l)
            return [y - l for y,l in zip(Y,L)]
        def _cost_func(vars):
            L = np.array([l1,l2,l3])
            return cost_func(L, vars[0], vars[1], vars[2])
        guess_prms = [guess_theta_t * np.pi/180, guess_theta_s * np.pi/180, guess_hs/100]
        opt = least_squares(_cost_func, guess_prms, verbose=0)
        theta_t, theta_s, eps = opt.x[:]
        return np.abs(theta_t) * 180/np.pi, None, np.abs(eps*100)
    elif hbl and given_twist is None:
        L = np.mean([l1,l2,l3])
        eps = (np.max([l1,l2,l3]) - np.min([l1,l2,l3]))/np.max([l1,l2,l3])
        delta_only_L = (1+delta)*a/delta
        costheta = 1 + ((delta**2)/(2+2*delta)) - ((a/L)**2)*((1+delta)/2)
        if costheta > 1:
            print('********************************************************************************************************************************************')
            print('** WARNING: Moire ({} nm) is larger than what would expect for just lattice mismatch ({} nm)'.format(L, delta_only_L))
            print('**   Either provided lattice constants are wrong or sample is reconstructing to decrease mismatch.')
            print('**   Returning twist angle of 0.0 degrees for this triangle. Likely not accurate since lattice mismatch is incorrect.')
            print('********************************************************************************************************************************************')
            theta_t = 0.0 
        else:
            theta_t = np.arccos(costheta) # delta = lattice mismatch NOT poisson ratio
        return np.abs(theta_t) * 180/np.pi, None, np.abs(eps*100)
    elif hbl and given_twist is not None:
        L = np.mean([l1,l2,l3])
        eps = (np.max([l1,l2,l3]) - np.min([l1,l2,l3]))/np.max([l1,l2,l3])
        b = np.cos(given_twist * np.pi/180)
        denom = a**2 - L**2
        num1 = - (a**2 + b*L**2 - L**2)
        num2 = np.sqrt(L**2 * (a**2 + L**2 * b**2 - L**2))
        comp_delta = (num1-num2)/denom
        if comp_delta < 0: delta = (num1+num2)/denom
        return (comp_delta)*100, None, np.abs(eps*100)    


if __name__ == "__main__":

    if True:

        import sys
        dsnum = int(sys.argv[1])
        sample = '6a'#'a5' #'ABt-nd1'#'c7' #
        filepath = '/Users/isaaccraig/Desktop/TLGproj/{}/dat_ds{}.pkl'.format(sample,dsnum)

        with open(filepath, 'rb') as f: ds = pickle.load(f)
        vdf = overlay_vdf(ds, dsnum, prefix='/Users/isaaccraig/Desktop/TLGproj/{}/'.format(sample), plotflag=False)
        extract_heterostrain_vdf(vdf, dsnum, prefix='/Users/isaaccraig/Desktop/TLGproj/{}/'.format(sample))
        #extract_twist_vdf(vdf, dsnum, prefix='/Users/isaaccraig/Desktop/TLGproj/{}/'.format(sample))


    if False:

        filepath = '/Users/isaaccraig/Desktop/TLGproj/a5/ring12-masks/vdf-ds6-rings12.pkl'
        with open(filepath, 'rb') as f: vdf = pickle.load(f)
        f,ax = plt.subplots()
        ax.imshow(vdf)
        plt.show()
        #exit()
        #extract_heterostrain_vdf(vdf, dsnum=6, prefix='/Users/isaaccraig/Desktop/TLGproj/a5/ring12-masks')
        ss = 0.5 #nm
        main_vdf(filepath, vdf, FOV_len=ss*vdf.shape[0])




    

