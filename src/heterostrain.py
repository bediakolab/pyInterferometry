
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

"""
def determine_material(prefix, dsnum):
    bits = prefix.split('_')
    material = bits[0]
    orrientation = bits[1]
    print('WARNING: Assuming ss=0.5nm! ')
    if material == 'HBL':
        a1,a2 = 0.315, 0.328
        aL, aS = np.max([a1,a2]), np.min([a1,a2])
        delta = (aS/aL) - 1
        a = aL
        return True, a, 0.5, None, delta, None
    elif material == 'MoS2':
        if orrientation == 'P':    return False, 0.315, 0.5, 0.234, None, None
        elif orrientation == 'AP': return False, 0.315, 0.5, 0.247, None, None
    elif material == 'MoSe2':
        if orrientation == 'P':    return False, 0.329, 0.5, 0.223, None, None
        elif orrientation == 'AP': return False, 0.329, 0.5, 0.230, None, None
    elif material == 'MoTe2':
        if orrientation == 'P':    return False, 0.353, 0.5, 0.248, None, None
        elif orrientation == 'AP': return False, 0.353, 0.5, 0.242, None, None
    else: 
        print('PARSE ERROR')
        exit()

def ask_user():

    #print('WARNING: Assuming AP MoS2, ss=0.5nm! ')
    #return False, 0.315, 0.5, 0.247, None, None #is_hbl, a, pixelsize, poissonratio, delta, twist
    #print('WARNING: Assuming AP MoS2, ss=0.5nm! ')
    #return False, 0.315, 0.5, 0.247, None
    #print('WARNING: Assuming P MoTe2, ss=0.5nm! ')
    #return False, 0.335, 0.5, 0.25, None
    #print('WARNING: Assuming P MoSe2, ss=0.5nm! ')
    #return False, 0.332, 0.5, 0.223, None
    #print('WARNING: Assuming AP MoSe2, ss=0.5nm! ')
    #return False, 0.332, 0.5, 0.230, None
    #print('WARNING: Assuming AP MoTe2, ss=0.5nm! ')
    #return False, 0.335, 0.5, 0.240, None
    #print('WARNING: Assuming MoS2/WSe2, ss=0.5nm! ')
    #a1, a2 = 0.332, 0.320
    #return True, np.min([a1,a2]), 0.5, None, ( np.max([a1,a2]) - np.min([a1,a2]) ) / np.min([a1,a2]) 

    is_hbl = boolquery("is this a heterobilayer?")
    lattice_info = "\n\t MoS2 = 0.315nm [1] \n\t MoSe2 = 0.329nm [1] \n\t MoTe2 = 0.353nm [1] \n\t WSe2 = 0.328nm [1] \n\t Ref [1]: HQ Graphene"
    if not is_hbl:
        pr_info = "\n\tP MoS2=0.234 [1] \n\tAP MoS2=0.247 [1] \n\tP MoSe2=0.223 [1] \n\tAP MoSe2=0.230 [1] \n\tP MoTe2= 0.248 [1] \n\tAP MoTe2=0.242 [1]"
        pr_ref = "\n\tRef [1]: Assuming AP all XMMX, P all XM, ratios from DFT (optB88)\n\tValues from Zeng, Fan, Wei-Bing Zhang, and Bi-Yu Tang. Chinese Physics B 24.9 (2015): 097103."
        print('*****************************************************************************************************************************************')
        poissonratio = float(input("enter poisson ratio for this material \n ---> Note:  {}{}\n ---> ".format(pr_info, pr_ref)))
        a = float(input("enter lattice constant for this material in nm \n ---> Note: {}\n ---> ".format(lattice_info)))
        delta = None
        twist = None
        print('*****************************************************************************************************************************************')
    else:
        given_delta = boolquery("calculate with a given lattice mismatch (y) or a given angle (n)?")
        if given_delta:
            print('*****************************************************************************************************************************************')
            print('WARNING: heterobilayer twist angle calculation assumes no heterostrain, will plot deformation to validate this assumption. ')
            print('WARNING: twist calc also assumes a given value for lattice mismatch, if the material deforms away from this mismatch the calculation will not be accurate.')
            a1 = float(input("---> Enter a first lattice constant for this material in nm \n---> Note: {}\n---> ".format(lattice_info)))
            a2 = float(input("---> Enter another lattice constant for this material in nm \n---> Note: {}\n---> ".format(lattice_info)))
            aL, aS = np.max([a1,a2]), np.min([a1,a2])
            delta = (aS/aL) - 1
            a = aL
            print('----> Entered lattice mismatch is {}%'.format(100*delta))
            print('*****************************************************************************************************************************************')
            poissonratio = None
            twist = None
        else:
            print('*****************************************************************************************************************************************')
            print('WARNING: heterobilayer lattice mismatch calculation assumes no heterostrain, will plot deformation to validate this assumption. ')
            a1 = float(input("---> Enter a first lattice constant for this material in nm \n---> Note: {}\n---> ".format(lattice_info)))
            a2 = float(input("---> Enter another lattice constant for this material in nm \n---> Note: {}\n---> ".format(lattice_info)))
            aL, aS = np.max([a1,a2]), np.min([a1,a2])
            delta = (aS/aL) - 1
            a = aL
            print('----> Expect rigid lattice mismatch of {}%'.format(100* delta))
            twist = float(input("---> Enter a twist angle for this sample in degrees \n---> "))
            print('*****************************************************************************************************************************************')
            poissonratio = None

    pixelsize = float(input("what's the pixel size in nm? ---> "))
    return is_hbl, a, pixelsize, poissonratio, delta, twist

"""

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

##### WORK ON !!
def matrixFromTriangleQuantity(triangles, centers, values, N):
    matrix = np.zeros((N,N))
    for n in range(len(triangles)):
        triangle = triangles[n]
        val = values[n]
        if not np.isnan(val):
            mask = make_contour_mask(N, N, contour=[centers[tri] for tri in triangle])
            for i in range(N):
                for j in range(N):
                    if mask[i,j]: matrix[i,j] = val
    return matrix

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
    ax.set_aspect('equal')
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
    tri_centers, thetas, het_strains, area_fracts = [], [], [], []
    for triangle in triangles:
        if not manual:
            point1 = centers[triangle[0]]
            point2 = centers[triangle[1]]
            point3 = centers[triangle[2]]
            center_x = np.mean([point1[1], point2[1], point3[1]])
            center_y = np.mean([point1[0], point2[0], point3[0]])
        else: # different syntax slightly for man def triangles
            point1 = triangle[0]
            point2 = triangle[1]
            point3 = triangle[2]
            center_y = np.mean([point1[1], point2[1], point3[1]])
            center_x = np.mean([point1[0], point2[0], point3[0]])
        l1, l2, l3 = get_lengths(point1, point2, point3)
        a12, a23, a31 = get_angles(l1,l2,l3)
        max_a = np.abs(np.max([a12, a23, a31])) * 180/np.pi
        l1, l2, l3 = l1*pixelsize, l2*pixelsize, l3*pixelsize
        if max_a < 160:
            theta_t, theta_s, eps = fit_heterostrain(l1, l2, l3, poissonratio, a, delta, is_hbl, given_twist)
            tri_centers.append([center_x, center_y])
            thetas.append(theta_t)
            het_strains.append(eps)
        else:
            thetas.append(np.nan)
            het_strains.append(np.nan)
            tri_centers.append([np.nan, np.nan])
            #thetas, het_strains if homoBL
            #thetas, tri anisotropy if heteroBL rigid delta
            #delta, tri anisotropy  if heteroBL rigid twist
    return tri_centers, thetas, het_strains 

def extract_twist_hetstrain(ds):
    adjacency_table = (ds.adjacency_type > 0).astype(int)
    triangles = get_triangles(adjacency_table)
    a = ds.extract_parameter("LatticeConstant", update_if_unset=True, param_type=float)
    pixelsize = ds.extract_parameter("PixelSize", update_if_unset=True, param_type=float)
    delta = ds.extract_parameter("LatticeMismatch", update_if_unset=True, param_type=float)
    is_hbl = ( delta != 0.0 )
    if is_hbl:
        given_twist = ds.extract_parameter("DiffractionPatternTwist", update_if_unset=True, param_type=float)
        poissonratio = None
        tri_centers, deltas, het_strain_proxys = computeAllTris(triangles, ds.centers, pixelsize, poissonratio, a, delta, is_hbl, given_twist)
        return tri_centers, None, None, deltas, het_strain_proxys
    else:
        poissonratio = ds.extract_parameter("PoissonRatio", update_if_unset=True, param_type=float)
        tri_centers, thetas, het_strains = computeAllTris(triangles, ds.centers, pixelsize, poissonratio, a, delta, is_hbl, None)
        return tri_centers, thetas, het_strains, None, None

def plot_twist_hetstrain(ds, ax1, ax2, thetas, het_strains, tri_centers, N):
    is_hbl = ( ds.extract_parameter("LatticeMismatch", force_set=True, param_type=float) != 0.0 )
    if is_hbl:
        given_twist = ds.extract_parameter("DiffractionPatternTwist", update_if_unset=True, param_type=float)
    else:
        given_twist = None
    theta_colors, het_strain_colors = getColors(thetas, het_strains)
    adjacency_table = (ds.adjacency_type > 0).astype(int)
    triangles = get_triangles(adjacency_table)
    centers = ds.centers
    def stder(v): 
        v = np.array([e for e in v if not np.isnan(e)])
        return np.std(v, ddof=1) / np.sqrt(np.size(v))
    if is_hbl and given_twist is None:
        t = '$<\\theta_m> = {:.2f} +/- {:.2f}^o$'.format(np.nanmean(thetas), stder(thetas))
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax1, t, centers)
        t = '$<deform> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(het_strains), stder(het_strains))
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colextract_twist_hetstrainors, ax2, t, centers)
        return np.nanmean(thetas), np.nanmean(het_strains)
    elif is_hbl and given_twist is not None:
        t = '$<\delta> = {:.2f} +/- {:.2f}\%$'.format(np.nanmean(thetas), stder(thetas))
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax1, t, centers)
        t = '$<deform> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(het_strains), stder(het_strains))
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax2, t, centers)
        return given_twist, np.nanmean(het_strains)
    elif not is_hbl:
        t = '$<\\theta_m> = {:.2f} +/- {:.2f}^o$'.format(np.nanmean(thetas), stder(thetas))
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax1, t, centers)
        t = '$<\epsilon> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(het_strains), stder(het_strains))
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax2, t, centers)
        return np.nanmean(thetas), np.nanmean(het_strains)

def extract_heterostrain(ufit, filenm, params, manual=False):

    matplotlib.use('Qt5Agg') # need for user input on images BUT not thread safe
    nx, ny = ufit.shape[0:2]
    img = displacement_colorplot(None, ufit)
    if not manual: triangles, centers = auto_triangles(ufit)
    else: triangles = manual_define_triangles(img)
    f, ((ax6, ax3), (ax4, ax5)) = plt.subplots(2,2)
    is_hbl, a, pixelsize, poissonratio, delta, given_twist = ask_user()

    tri_centers, thetas, het_strains, area_fracts = computeAllTris(triangles, centers, pixelsize, poissonratio, a, delta, is_hbl, given_twist, manual)
    theta_colors, het_strain_colors = getColors(thetas, het_strains)

    if is_hbl and given_twist is None:
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax4, '$<\\theta_m> = {:.2f}^o$'.format(np.nanmean(thetas)), centers, manual)
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax5, '$<deform> = {:.2f}%$'.format(np.nanmean(het_strains)), centers, manual)
        writefile('../results/{}_angles.txt'.format(filenm), 'twist angles for {} (manual)'.format(filenm), thetas)
        writefile('../results/{}_hetstrains.txt'.format(filenm), 'deforms for {} (manual)'.format(filenm), het_strains)
        writefile('../results/{}_areafracts.txt'.format(filenm), 'area fractions for {}'.format(filenm), area_fracts)
    elif is_hbl and given_twist is not None:
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax4, '$<\\delta> = {:.2f}^o$'.format(np.nanmean(thetas)), centers, manual)
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax5, '$<deform> = {:.2f}%$'.format(np.nanmean(het_strains)), centers, manual)    
        writefile('../results/{}_deltas.txt'.format(filenm), 'lattice mismatch for {} (manual)'.format(filenm), thetas)
        writefile('../results/{}_hetstrains.txt'.format(filenm), 'deforms for {} (manual)'.format(filenm), het_strains)
        writefile('../results/{}_areafracts.txt'.format(filenm), 'area fractions for {}'.format(filenm), area_fracts)
    elif not is_hbl:
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax4, '$<\\theta_m> = {:.2f}^o$'.format(np.nanmean(thetas)), centers, manual)
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax5, '$<\epsilon> = {:.2f}%$'.format(np.nanmean(het_strains)), centers, manual)
        writefile('../results/{}_angles.txt'.format(filenm), 'twist angles for {} (manual)'.format(filenm), thetas)
        writefile('../results/{}_hetstrains.txt'.format(filenm), 'heterostrains for {} (manual)'.format(filenm), het_strains)
        writefile('../results/{}_areafracts.txt'.format(filenm), 'area fractions for {}'.format(filenm), area_fracts)

    ax3.axis('off')
    ax6.imshow(img, origin='lower')
    ax6.axis('off')
    ax6.set_title('displacement field')
    plt.savefig("../plots/{}_hetstrains_manualdef.png".format(filenm), dpi=600)
    plt.show()

def extract_heterostrain_vdf(img, filenm):

    matplotlib.use('Qt5Agg') # need for user input on images BUT not thread safe
    triangles = manual_define_triangles(img)
    f, ((ax6, ax3), (ax4, ax5)) = plt.subplots(2,2)
    is_hbl, a, pixelsize, poissonratio, delta, given_twist = ask_user()
    centers = None 
    manual = True
    tri_centers, thetas, het_strains, area_fracts = computeAllTris(triangles, centers, pixelsize, poissonratio, a, delta, is_hbl, given_twist, manual)
    theta_colors, het_strain_colors = getColors(thetas, het_strains)

    if is_hbl and given_twist is None:
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax4, '$<\\theta_m> = {:.2f}^o$'.format(np.nanmean(thetas)), centers, manual)
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax5, '$<deform> = {:.2f}%$'.format(np.nanmean(het_strains)), centers, manual)
        writefile('../results/{}_angles.txt'.format(filenm), 'twist angles for {} (manual)'.format(filenm), thetas)
        writefile('../results/{}_hetstrains.txt'.format(filenm), 'deforms for {} (manual)'.format(filenm), het_strains)
        writefile('../results/{}_areafracts.txt'.format(filenm), 'area fractions for {}'.format(filenm), area_fracts)
    elif is_hbl and given_twist is not None:
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax4, '$<\\delta> = {:.2f}^o$'.format(np.nanmean(thetas)), centers, manual)
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax5, '$<deform> = {:.2f}%$'.format(np.nanmean(het_strains)), centers, manual)    
        writefile('../results/{}_deltas.txt'.format(filenm), 'lattice mismatch for {} (manual)'.format(filenm), thetas)
        writefile('../results/{}_hetstrains.txt'.format(filenm), 'deforms for {} (manual)'.format(filenm), het_strains)
        writefile('../results/{}_areafracts.txt'.format(filenm), 'area fractions for {}'.format(filenm), area_fracts)
    elif not is_hbl:
        plotTriangleQuantity(triangles, tri_centers, thetas, theta_colors, ax4, '$<\\theta_m> = {:.2f}^o$'.format(np.nanmean(thetas)), centers, manual)
        plotTriangleQuantity(triangles, tri_centers, het_strains, het_strain_colors, ax5, '$<\epsilon> = {:.2f}%$'.format(np.nanmean(het_strains)), centers, manual)
        writefile('../results/{}_angles.txt'.format(filenm), 'twist angles for {} (manual)'.format(filenm), thetas)
        writefile('../results/{}_hetstrains.txt'.format(filenm), 'heterostrains for {} (manual)'.format(filenm), het_strains)
        writefile('../results/{}_areafracts.txt'.format(filenm), 'area fractions for {}'.format(filenm), area_fracts)

    ax3.axis('off')
    ax6.imshow(img, origin='lower', cmap='plasma')
    ax6.axis('off')
    ax6.set_title('vdf sum')
    plt.savefig("../plots/{}_hetstrains_manualdef.png".format(filenm), dpi=600)
    plt.show()

def auto_triangles(ufit):    
    GLOBAL_BOUNDARYVAL, GLOBAL_DELTAVAL, GLOBAL_COMBINE_CRIT, GLOBAL_COMBINE_CRIT = params[:]
    boundary_val = GLOBAL_BOUNDARYVAL
    delta_val = GLOBAL_DELTAVAL
    combine_crit = GLOBAL_COMBINE_CRIT
    spdist = GLOBAL_SPDIST
    nx, ny = ufit.shape[0:2]
    img = displacement_colorplot(None, ufit[:,:,0].reshape(nx, ny), ufit[:,:,1].reshape(nx, ny))
    img, uvecs = crop_displacement(img, ufit)
    mask_aa = get_aa_mask(ufit, boundary=boundary_val)
    mask_sp1, mask_sp2, mask_sp3 = get_sp_masks(ufit, mask_aa, delta=delta_val)
    mask_sp1, lines_sp1 = get_sp_line_method2(mask_sp1, plotbool=False)
    mask_sp2, lines_sp2 = get_sp_line_method2(mask_sp2, plotbool=False)
    mask_sp3, lines_sp3 = get_sp_line_method2(mask_sp3, plotbool=False)
    labeled, aa_regions = ndimage.label(mask_aa)
    centers = get_region_centers(labeled, mask_aa)
    centers = combine_nearby_spots(centers, combine_criterion=combine_crit)
    adjacency_table = np.zeros((len(centers), len(centers)))
    lines = lines_sp1
    lines.extend(lines_sp2)
    lines.extend(lines_sp3)
    for line in lines:
        last_indx = -1
        for xel, yel in zip(line[0], line[1]):
            for i in range(len(centers)):
                center = centers[i]
                if ((center[1] - xel)**2 + (center[0] - yel)**2)**0.5 < spdist:
                    if last_indx != -1 and last_indx != i:
                        adjacency_table[i, last_indx] = 1
                        adjacency_table[last_indx, i] = 1
                    last_indx = i
                    break
    triangles = get_triangles(adjacency_table)
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
        costheta = 1 + ((delta**2)/(2+2*delta)) - ((a/L)**2)*((1+delta)/2) # 1+delta equation, using delta = aS/aL - 1 
        if costheta > 1:
            show_warning = False
            if show_warning: print('********************************************************************************************************************************************')
            if show_warning: print('** WARNING: Moire ({} nm) is larger than what would expect for just lattice mismatch ({} nm)'.format(L, delta_only_L))
            if show_warning: print('**   Either provided lattice constants are wrong or sample is reconstructing to decrease mismatch.')
            if show_warning: print('**   Returning twist angle of 0.0 degrees for this triangle. Likely not accurate since lattice mismatch is incorrect.')
            if show_warning: print('********************************************************************************************************************************************')
            theta_t = 0.0 
        else:
            theta_t = np.arccos(costheta) # delta = lattice mismatch NOT poisson ratio
        return np.abs(theta_t) * 180/np.pi, None, np.abs(eps*100)
    elif hbl and given_twist is not None:
        L = np.mean([l1,l2,l3])
        #print('moire wl was {}'.format(L))
        #print('moire twist {} degrees'.format(given_twist))
        #print('moire twist -> wl of {}'.format(1/(2*np.sin(given_twist * np.pi/180 * 0.5))))
        eps = (np.max([l1,l2,l3]) - np.min([l1,l2,l3]))/np.max([l1,l2,l3])
        #b = np.cos(given_twist * np.pi/180)
        denom = a**2 - L**2
        #num1 = - (a**2 + b*L**2 - L**2)
        #num2 = np.sqrt(L**2 * (a**2 + L**2 * b**2 - L**2))
        #comp_delta = (num1-num2)/denom
        #if comp_delta > 0: delta = (num1-num2)/denom
        gamma = a**2 - L**2 + (L**2 * np.cos(given_twist * np.pi/180))
        comp_delta = (- gamma + np.sqrt(L**2 * gamma))/denom
        if comp_delta > 0: # using delta = aS/aL - 1 so want < 0
            comp_delta = (- gamma - np.sqrt(L**2 * gamma))/denom
        #print('lattice mismatch therefore {}%'.format(comp_delta*100))
        return (comp_delta)*100, None, np.abs(eps*100)    


def test_sensitivity_to_pr(L, deform, PR):
    theta, _, hs = fit_heterostrain(L, L, (L + deform*L), PR, a=1, delta=None, hbl=False)
    return theta*180/np.pi, np.abs(hs)*100

if __name__ == "__main__":

    if False:

        t, hs = test_sensitivity_to_pr(4.0, 0.5, 0.22)
        t2, hs2 = test_sensitivity_to_pr(4.0, 0.5, 0.26)
        print('{} +/- {}'.format(np.mean([t, t2])), np.abs(t - np.mean([t, t2]))/2)
        print('{} +/- {}'.format(np.mean([hs, hs2])), np.abs(hs - np.mean([hs, hs2]))/2)
        t, hs =test_sensitivity_to_pr(4.0, 1.5, 0.22)
        t2, hs2 =test_sensitivity_to_pr(4.0, 1.5, 0.26)
        print('{} +/- {}'.format(np.mean([t, t2])), np.abs(t - np.mean([t, t2]))/2)
        print('{} +/- {}'.format(np.mean([hs, hs2])), np.abs(hs - np.mean([hs, hs2]))/2)
        t, hs =test_sensitivity_to_pr(4.0, 2.5, 0.22)
        t2, hs2 = test_sensitivity_to_pr(4.0, 2.5, 0.26)
        print('{} +/- {}'.format(np.mean([t, t2])), np.abs(t - np.mean([t, t2]))/2)
        print('{} +/- {}'.format(np.mean([hs, hs2])), np.abs(hs - np.mean([hs, hs2]))/2)
        t, hs =test_sensitivity_to_pr(8.0, 5, 0.22)
        t2, hs2 = test_sensitivity_to_pr(8.0, 5, 0.26)
        print('{} +/- {}'.format(np.mean([t, t2])), np.abs(t - np.mean([t, t2]))/2)
        print('{} +/- {}'.format(np.mean([hs, hs2])), np.abs(hs - np.mean([hs, hs2]))/2)
        exit()


    phsbool = boolquery("would you extract hetero-strain from a saved dataset?")
    while phsbool:
        uvecs, prefix, dsnum, _ = import_uvector()
        uvecs = latticevec_to_cartesian(uvecs)
        filenm = os.path.join(prefix,'ds_{}'.format(dsnum))
        parbool = boolquery("is the dataset parallel (y) or anti-parallel (n)?")
        if parbool: # found these to be good for parallel datasets
            GLOBAL_BOUNDARYVAL = 0.75 
            GLOBAL_DELTAVAL = 0.1
            GLOBAL_COMBINE_CRIT = 15.0
            GLOBAL_SPDIST = 10.0
        else:  # found these to be good for anti-parallel datasets
            GLOBAL_BOUNDARYVAL = 0.5
            GLOBAL_DELTAVAL = 0.01
            GLOBAL_COMBINE_CRIT = 5.0
            GLOBAL_SPDIST = 5.0
        print('*****************************************************************************************************************************************')
        print('Using the following hyperparameters to find soliton walls in the data. Please edit them if automatic detection behaving poorly.')
        print("They're Defined at bottom of heterostrain.py. And they don't matter if you do it manually. ")
        print('   boundaryval={} --- threshold to find u=0 (colored black) centroids. '.format(GLOBAL_BOUNDARYVAL))
        print('   deltaval={} --- threshold to find soliton walls. SPs found if angle of u is within this many radians of an expected angle'.format(GLOBAL_DELTAVAL))
        print('   combinecrit={} --- threshold to combine nearby centroids. centers within this many pixels of eachother are merged '.format(GLOBAL_COMBINE_CRIT))
        print("   spdist={} --- controls how close the u=0 centers need to be to the soliton walls (in pixels) in order to register them as 'connecting' to it ".format(GLOBAL_SPDIST))
        print('*****************************************************************************************************************************************')
        nx, ny = uvecs.shape[0:2]
        params = [GLOBAL_BOUNDARYVAL, GLOBAL_DELTAVAL, GLOBAL_COMBINE_CRIT, GLOBAL_COMBINE_CRIT]
        manbool = boolquery("would you like to define moire triangles manually?")
        extract_heterostrain(uvecs, filenm, params, manbool)
        phsbool = boolquery("would you extract hetero-strain from another saved dataset?")

    phsbool = boolquery("would you extract hetero-strain from a saved disket (from vdf)?")
    while phsbool:
        ds, prefix, dsnum = import_diskset()
        filenm = os.path.join(prefix,'ds_{}'.format(dsnum))
        vdf = overlay_vdf(ds, dsnum, prefix, plotflag=False)
        extract_heterostrain_vdf(vdf, filenm)
        phsbool = boolquery("would you extract hetero-strain from another saved diskset?")
