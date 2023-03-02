import numpy as np
import matplotlib.pyplot as plt
import colorsys
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.colors as mplc
from scipy import ndimage

from basis_utils import cartesian_to_rz_WZ, getclosestequiv
from visualization import displacement_colorplot
from unwrap_utils import automatic_sp_rotation
from unwrap_utils import getAdjacencyMatrix
from basis_utils import rotate_uvecs

global_vmat = np.matrix([[np.sqrt(3),0], [np.sqrt(3)/2, 3/2]]).T
global_g1 = np.array([0, 2/3]) # this set of g corresponds to [a,0] and [a/2, sqrt(3)a/2]
global_g2 = np.array([1/np.sqrt(3),1/3])

def plot_connections(axis, i, j, N, vec, el, c, vec1, vec2, is_mat=False):
    vec_R = vec1 * (i+1 - N/2 + el[1]) + vec2 * (j - N/2 + el[2]) 
    vec_U = vec1 * (i - N/2 + el[1]) + vec2 * (j+1 - N/2 + el[2])
    vec_D = vec1 * (i+1 - N/2 + el[1]) + vec2 * (j-1 - N/2 + el[2]) 
    if not is_mat:
        if i < N-1: axis.plot([vec[0], vec_R[0]], [vec[1], vec_R[1]], color=c, linewidth=0.5)
        if j < N-1: axis.plot([vec[0], vec_U[0]], [vec[1], vec_U[1]], color=c, linewidth=0.5)
        if i < N-1 and j > 0: axis.plot([vec[0], vec_D[0]], [vec[1], vec_D[1]], color=c, linewidth=0.5)
    else:
        if i < N-1: axis.plot([vec[0,0], vec_R[0,0]], [vec[0,1], vec_R[0,1]], color=c, linewidth=0.5)
        if j < N-1: axis.plot([vec[0,0], vec_U[0,0]], [vec[0,1], vec_U[0,1]], color=c, linewidth=0.5)
        if i < N-1 and j > 0: axis.plot([vec[0,0], vec_D[0,0]], [vec[0,1], vec_D[0,1]], color=c, linewidth=0.5)

def main():
    # mos2
    a = 1
    c = 1.0

    """
    basis = [[42, 1/3, 2/3, 1/4],
             [42, 2/3, 1/3, 3/4],
             [16, 2/3, 1/3, 0.14482600],
             [16, 2/3, 1/3, 0.35517400],
             [16, 1/3, 2/3, 0.64482600],
             [16, 1/3, 2/3, 0.85517400]]
    color_map = { 16: 'y', 42: 'b' } # sulfur is y moly is blue
    """

    basis = [[1, 1/3, 2/3, 1/4], # in hexagonal LV basis
             [2, 1/3, 2/3, 3/4]] # twisting about an AA
    color_map = { 1: 'k', 2: 'r' } # hexagonal, bottom black top grey
    N = 20
    f, axes = plt.subplots(2,3)
    axes = axes.flatten()
    print('on angle 1')
    make_twisted_structure(a, c, 6,   basis, N, color_map, axes[:4])
    make_hbl_structure(a, c, 1.1, basis, N, color_map, axes[3:])
    plt.show()

# basis expected to be in the form [Z_atom, x_atom, y_atom, z_atom] for each atom
# it's a list of 4 element arrays
def make_hbl_structure(a, c, scale, basis, N, color_map, ax):

    # hexagonal lattice vectors for first layerl
    v1 = np.array([a,0,0])
    v2 = np.array([1/2*a,a* np.sqrt(3)/2,0])
    v3 = np.array([0, 0, c])

    # hexagonal lattice vectors for second layer
    mat = np.matrix([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    v1_2   = np.matmul(mat, v1).reshape(3)
    v2_2   = np.matmul(mat, v2).reshape(3)

    # place atoms according to provided bases
    xlayer1, ylayer1, zlayer1, colorslayer1 = [], [], [], []
    xlayer2, ylayer2, zlayer2, colorslayer2 = [], [], [], []
    dispvecs_x, dispvecs_y = [], []
    for i in range(N):
        for j in range(N):
            for el in basis:
                if el[3] > 0.5:
                    vec = v1 * (i - N/2 + el[1]) + v2 * (j - N/2 + el[2]) + v3 * el[3]
                    xlayer1.append(vec[0])
                    ylayer1.append(vec[1])
                    zlayer1.append(vec[2])
                    colorslayer1.append(color_map[el[0]])
                    #plot_connections(ax[0], i, j, N, vec, el, color_map[el[0]], v1, v2)
                else:
                    vec = v1_2 * (i - N/2 + el[1]) + v2_2 * (j - N/2 + el[2]) + v3 * el[3]
                    xlayer2.append(vec[0,0])
                    ylayer2.append(vec[0,1])
                    zlayer2.append(vec[0,2])
                    vec_ref = v1 * (i - N/2 + el[1]) + v2 * (j - N/2 + el[2]) + v3 * el[3]
                    dispvecs_x.append(vec[0,0] - vec_ref[0])
                    dispvecs_y.append(vec[0,1] - vec_ref[1])
                    colorslayer2.append(color_map[el[0]])
                    #plot_connections(ax[0], i, j, N, vec, el, color_map[el[0]], v1_2, v2_2, is_mat=True)

    xlayer1, ylayer1, xlayer2, ylayer2 = np.array(xlayer1), np.array(ylayer1), np.array(xlayer2), np.array(ylayer2)
    dispvecs_x, dispvecs_y = np.array(dispvecs_x), np.array(dispvecs_y)
    dispvecs_x, dispvecs_y = dispvecs_x.reshape(N,N), dispvecs_y.reshape(N,N)
    u_cart = np.zeros((dispvecs_x.shape[0], dispvecs_x.shape[1], 2))
    
    
    u_cart[:,:,0], u_cart[:,:,1] =  dispvecs_x[:,:], dispvecs_y[:,:]
    colored_scatter(ax[0], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))
    
    dispvecs_x[:,:], dispvecs_y[:,:] = u_cart[:,:,0], u_cart[:,:,1]
    u_cartrot = rotate_uvecs(u_cart, ang=(-np.pi/3))
    dispvecs_x[:,:], dispvecs_y[:,:] = u_cartrot[:,:,0], u_cartrot[:,:,1]
    colored_scatter(ax[1], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))
    
    dispvecs_x[:,:], dispvecs_y[:,:] = u_cart[:,:,0], u_cart[:,:,1]
    u_cartrot = rotate_uvecs(u_cart, ang=(np.pi/3))
    dispvecs_x[:,:], dispvecs_y[:,:] = u_cartrot[:,:,0], u_cartrot[:,:,1]
    colored_scatter(ax[2], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))

    for axis in ax: axis.axis('off')  
    plt.show(); exit()

    u_zcart = cartesian_to_rz_WZ(u_cart, sign_wrap=False)
    ax[0].scatter(xlayer1, ylayer1, color=colorslayer1, s=5, edgecolors=colorslayer1)
    ax[0].scatter(xlayer2, ylayer2, color=colorslayer2, s=5, edgecolors=colorslayer2)
    colored_scatter(ax[2], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))
    for axis in ax: axis.axis('off')    

# basis expected to be in the form [Z_atom, x_atom, y_atom, z_atom] for each atom
# it's a list of 4 element arrays
def make_twisted_structure(a, c, angle, basis, N, color_map, ax):

    # hexagonal lattice vectors for first layerl
    v1 = np.array([a,0,0])
    v2 = np.array([1/2*a,a* np.sqrt(3)/2,0])
    v3 = np.array([0, 0, c])

    # hexagonal lattice vectors for second layer
    theta  = angle * np.pi/180
    rotmat = np.matrix([[np.cos(-theta), -np.sin(-theta), 0], [np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    v1_2   = np.matmul(rotmat, v1).reshape(3)
    v2_2   = np.matmul(rotmat, v2).reshape(3)

    # place atoms according to provided bases
    xlayer1, ylayer1, zlayer1, colorslayer1 = [], [], [], []
    xlayer2, ylayer2, zlayer2, colorslayer2 = [], [], [], []
    dispvecs_x, dispvecs_y = [], []
    for i in range(N):
        for j in range(N):
            for el in basis:
                if el[3] > 0.5:
                    vec = v1 * (i - N/2 + el[1]) + v2 * (j - N/2 + el[2]) + v3 * el[3]
                    xlayer1.append(vec[0])
                    ylayer1.append(vec[1])
                    zlayer1.append(vec[2])
                    colorslayer1.append(color_map[el[0]])
                    #plot_connections(ax[0], i, j, N, vec, el, color_map[el[0]], v1, v2)
                else:
                    vec = v1_2 * (i - N/2 + el[1]) + v2_2 * (j - N/2 + el[2]) + v3 * el[3]
                    xlayer2.append(vec[0,0])
                    ylayer2.append(vec[0,1])
                    zlayer2.append(vec[0,2])
                    vec_untwist = v1 * (i - N/2 + el[1]) + v2 * (j - N/2 + el[2]) + v3 * el[3]
                    dispvecs_x.append(vec[0,0] - vec_untwist[0])
                    dispvecs_y.append(vec[0,1] - vec_untwist[1])
                    colorslayer2.append(color_map[el[0]])
                    #plot_connections(ax[0], i, j, N, vec, el, color_map[el[0]], v1_2, v2_2, is_mat=True)

    xlayer1, ylayer1, xlayer2, ylayer2 = np.array(xlayer1), np.array(ylayer1), np.array(xlayer2), np.array(ylayer2)
    dispvecs_x, dispvecs_y = np.array(dispvecs_x), np.array(dispvecs_y)
    dispvecs_x, dispvecs_y = dispvecs_x.reshape(N,N), dispvecs_y.reshape(N,N)
    u_cart = np.zeros((dispvecs_x.shape[0], dispvecs_x.shape[1], 2))
    
    
    u_cart[:,:,0], u_cart[:,:,1] =  dispvecs_x[:,:], dispvecs_y[:,:]
    colored_quiver(ax[0], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))
    
    dispvecs_x[:,:], dispvecs_y[:,:] = u_cart[:,:,0], u_cart[:,:,1]
    u_cartrot = rotate_uvecs(u_cart, ang=(-np.pi/3))
    dispvecs_x[:,:], dispvecs_y[:,:] = u_cartrot[:,:,0], u_cartrot[:,:,1]
    colored_quiver(ax[1], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))
    
    dispvecs_x[:,:], dispvecs_y[:,:] = u_cart[:,:,0], u_cart[:,:,1]
    u_cartrot = rotate_uvecs(u_cart, ang=(np.pi/3))
    dispvecs_x[:,:], dispvecs_y[:,:] = u_cartrot[:,:,0], u_cartrot[:,:,1]
    colored_quiver(ax[2], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))

    for axis in ax: axis.axis('off')  
    plt.show(); exit()

    colored_quiver(ax[1], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))
    u_cart = np.zeros((dispvecs_x.shape[0], dispvecs_x.shape[1], 2))
    u_cart[:,:,0], u_cart[:,:,1] =  dispvecs_x[:,:], dispvecs_y[:,:]
    u_zcart = cartesian_to_rz_WZ(u_cart, sign_wrap=False)
    #u_zcart = rotate_uvecs(u_zcart, ang=(-np.pi/3))
    dispvecs_x[:,:], dispvecs_y[:,:] = u_zcart[:,:,0], u_zcart[:,:,1]
    ax[0].scatter(xlayer1, ylayer1, color=colorslayer1, s=5, edgecolors=colorslayer1)
    ax[0].scatter(xlayer2, ylayer2, color=colorslayer2, s=5, edgecolors=colorslayer2)
    colored_scatter(ax[2], xlayer1, ylayer1, np.array(dispvecs_x), np.array(dispvecs_y))
    for axis in ax: axis.axis('off')    

def colored_scatter(ax, x, y, u_x, u_y):
    colors = displacement_colorplot(None, u_x, u_y)
    colors = colors.reshape(len(x),3)
    for i in range(len(x)):
        ax.scatter(x[i], y[i], color=colors[i,:], s=15)

def colored_quiver(ax, x, y, u_x, u_y):
    colors = displacement_colorplot(None, u_x, u_y)
    colors = colors.reshape(len(x),3)
    ax.quiver(x, y, u_x.flatten(), u_y.flatten(), color=colors)

if __name__ == '__main__':
    main()
