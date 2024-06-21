
##################################################################
##################################################################
## series of utility functions to convert between various bases 
## of interest, cartesian <-> lattice vector <-> reduced zone, etc
##################################################################
##################################################################
import numpy as np
from utils import rotate2d
import matplotlib.pyplot as plt


def get_smallest_dist(u1, u2, extend=True):
    equivs = get_nearby_equivs(u, extend=extend)
    dist = [ (ue[0,0]-uref[0])**2 + (ue[0,1] - uref[1])**2 for ue in equivs ]
    return dist[np.argmin(np.abs(dist))]

def getclosestequiv(u, uref, extend=True): 
    equivs = get_nearby_equivs(u, extend=extend)
    dist = [ (ue[0,0]-uref[0])**2 + (ue[0,1] - uref[1])**2 for ue in equivs ]
    return equivs[np.argmin(np.abs(dist))]

##################################################################
# takes a value c1 and adds integer multiples of offset to find 
# the value of lowest possible absolute value. when offset=1 
# the returned value is |v| <= 1/2
##################################################################
def rz_helper(c1, offset=1): 
    if c1 >= 0:
        c1 = c1%(offset)
        c1_option2 = c1 - (offset)
        if np.abs(c1) < np.abs(c1_option2): returnc = c1
        else: returnc = c1_option2
    else:
        c1 = c1%(-(offset))
        c1_option2 = c1 + (offset)
        if np.abs(c1) < np.abs(c1_option2): returnc = c1
        else: returnc = c1_option2
    return (returnc)

##################################################################
# takes two values and adds integers until both are |v| <= 1/2
# if sign_wrap will also flip sign of both to insure first value > 0
##################################################################
def rz_helper_pair(c1, c2, sign_wrap=True):
    c1 = rz_helper(c1) # want |a| < 1/2
    c2 = rz_helper(c2) # want |b| < 1/2
    if sign_wrap and c1 < 0: c1, c2 = -c1, -c2
    return c1, c2

def rz_helper_hex(c1, c2):
    c1 = rz_helper(c1) # want |a| < 1/2
    c2 = rz_helper(c2) # want |b| < 1/2
    c3 = c1 + c2
    return c1, c2, c3

def rotate_uvecs(uvecs_cart, ang):
    uvecs_cart_rot = np.zeros((uvecs_cart.shape[0], uvecs_cart.shape[1], 2))
    for i in range(uvecs_cart.shape[0]):
        for j in range(uvecs_cart.shape[1]):
            rotu = rotate2d([uvecs_cart[i,j,0], uvecs_cart[i,j,1]], ang)
            uvecs_cart_rot[i,j,:] = rotu[:]
    return uvecs_cart_rot

##################################################################
# moves the cartesian vectors into a new 'zone' by adding/removing
# lattice vectors. number to add/remove given by matrices n, m
##################################################################
def adjust_zone(uvecs_cart, n, m):
    adjusts_lv = np.zeros((uvecs_cart.shape[0], uvecs_cart.shape[1], 2))
    for i in range(uvecs_cart.shape[0]):
        for j in range(uvecs_cart.shape[1]):
            adjusts_lv[i,j,:] = n[i,j], m[i,j]
    adjustments_cart = latticevec_to_cartesian(adjusts_lv)
    return uvecs_cart+adjustments_cart, adjustments_cart

def adjust_zone_single(u, n, m):
    adjustments_cart = single_lv_to_cart(np.array([n,m]))
    return u+adjustments_cart

##################################################################
# moves from reduced zone basis back into cartesian basis,
# inverse of the cart_to_zonebasis function.
##################################################################
def zonebasis_to_cart(f1, f2, n, m, signs=None):
    ulv = np.zeros((f1.shape[0], f1.shape[1], 2))
    if signs is None: signs = np.ones((f1.shape[0], f1.shape[1])) 
    for x in range(f1.shape[0]):
        for y in range(f2.shape[1]):
            try: ulv[x,y,:] = signs[x,y]*f1[x,y] + n[x,y], signs[x,y]*f2[x,y] + m[x,y] 
            except: ulv[x,y,:] = signs[x][y]*f1[x,y] + n[x][y], signs[x][y]*f2[x,y] + m[x][y]
    return latticevec_to_cartesian(ulv)

##################################################################
# moves from the cartesian basis to the reduced zone basis
# for a single vector. see cart_to_zonebasis. 
##################################################################
def single_cart2zb(ucart):
    ulv = single_cart2lv(ucart)
    f1, f2 = ulv[0,0], ulv[0,1]
    f1red, f2red = rz_helper(f1), rz_helper(f2)
    n, m = f1-f1red, f2-f2red 
    return f1red, f2red, n, m

##################################################################
# return all possible nearby lattice vectors that are deamed 
# equivalent/indistinguishable by the method. 
# possible changes: add/subtract integer from n, m, or n+m
# with or without sign flips
##################################################################
def get_nearby_equivs(ucart, extend=False):
    fc1, fc2, n, m = single_cart2zb(ucart)
    offsets = [[0,0], [0,1], [1,0], [1,1]]
    equivs  = []
    for offset in offsets: equivs.append(( fc1+n+offset[0], fc2+m+offset[1]))
    for offset in offsets: equivs.append((-fc1+n+offset[0],-fc2+m+offset[1]))
    for offset in offsets: equivs.append(( fc1+n-offset[0], fc2+m-offset[1]))
    for offset in offsets: equivs.append((-fc1+n-offset[0],-fc2+m-offset[1]))
    if extend:
        offsets = [[0,2], [2,0], [1,2], [2,1], [2,2], [1,-1]]
        for offset in offsets: equivs.append(( fc1+n+offset[0], fc2+m+offset[1]))
        for offset in offsets: equivs.append((-fc1+n+offset[0],-fc2+m+offset[1]))
        for offset in offsets: equivs.append(( fc1+n-offset[0], fc2+m-offset[1]))
        for offset in offsets: equivs.append((-fc1+n-offset[0],-fc2+m-offset[1]))
    equivs = [single_lv_to_cart(ulv) for ulv in equivs]
    return equivs

##################################################################
# moves from cartesian basis into reduced zone basis, keeping
# track of the lattice vectors added/removed to end up in this zone
# so u_cart = lv_to_cart(f1+n, f2+m) with n, m integers. from 
# adding integer multiples of the lattice vectors
##################################################################
def cart_to_zonebasis(uvecs_cart):
    uvecs_lv   = cartesian_to_latticevec(uvecs_cart)
    f1mat   = np.zeros((uvecs_cart.shape[0], uvecs_cart.shape[1]))
    f2mat   = np.zeros((uvecs_cart.shape[0], uvecs_cart.shape[1]))
    nmat    = np.zeros((uvecs_cart.shape[0], uvecs_cart.shape[1]))
    mmat    = np.zeros((uvecs_cart.shape[0], uvecs_cart.shape[1]))
    for x in range(uvecs_cart.shape[0]):
        for y in range(uvecs_cart.shape[1]):
            f1orig, f2orig = uvecs_lv[x,y,:]
            f1mat[x,y], f2mat[x,y] = rz_helper(f1orig), rz_helper(f2orig)
            nmat[x,y], mmat[x,y] = f1orig - f1mat[x,y], f2orig - f2mat[x,y]
    return f1mat, f2mat, nmat, mmat

##################################################################
# moves into reduced zone, staying in lattice vector basis 
# (reduced zone is when coefificents c are |c|<=1/2 in lattice 
# vector basis)
##################################################################
def lv_to_rzlv(uvecs_lv, sign_wrap=True, shift=False):
    uvecs_rzlv = np.zeros(uvecs_lv.shape) 
    for i in range(uvecs_lv.shape[0]):
        for j in range(uvecs_lv.shape[1]):
            c1, c2 = rz_helper_pair(uvecs_lv[i,j,0], uvecs_lv[i,j,1], sign_wrap)
            if shift:
                # want to go from -0.5, 0.5 --> 0, 1
                if c1 < 0: c1 += 1
                if c2 < 0: c2 += 1
            uvecs_rzlv[i,j,0], uvecs_rzlv[i,j,1] = c1, c2 
    return uvecs_rzlv

##################################################################
# returns true if cartesian basis vectors are in reduced zone
##################################################################
def in_rz_cart(uvec_cart):
    vmat = np.matrix([[-1, 1/np.sqrt(3)],[0, 2/np.sqrt(3)]])
    c = vmat @ uvec_cart
    return (np.abs(c) <= 0.5).all()

##################################################################
# chain rule to turn gradients in lattice vector basis to cartesian
##################################################################
def lvstrain_to_cartesianstrain(du1dx, du1dy, du2dx, du2dy):
	# dux/dx = dux/du1 * du1/dx + dux/du2 * du2/dx
	# ux = -u1 + u2/2 and uy = root3*u2/2 
	duxdu1, duxdu2, duydu1, duydu2 = -1, 1/2, 0, np.sqrt(3)/2
	dux_dx = du1dx * duxdu1 + du2dx * duxdu2 
	duy_dx = du1dx * duydu1 + du2dx * duydu2
	dux_dy = du1dy * duxdu1 + du2dy * duxdu2 
	duy_dy = du1dy * duydu1 + du2dy * duydu2
	return dux_dx, duy_dx, dux_dy, duy_dy

##################################################################
# moves from a cartesian basis to reduced zone cartesian
# (reduced zone is when coefificents c are |c|<=1/2 in lattice 
# vector basis, cartesian reduced zone from changing basis back 
# to cartesian following this)
# with shift flag, will instead do c in [0,1]
##################################################################
def cartesian_to_rzcartesian(uvecs_cart, sign_wrap=True, shift=False):
    uvecs_lv = cartesian_to_latticevec(uvecs_cart)
    return lv_to_rzcartesian(uvecs_lv, sign_wrap, shift)

##################################################################
# moves from a lattice vector basis to reduced zone cartesian
##################################################################
def lv_to_rzcartesian(uvecs_lv, sign_wrap=True, shift=False):
    uvecs_rzlv = lv_to_rzlv(uvecs_lv, sign_wrap, shift) 
    return latticevec_to_cartesian(uvecs_rzlv)

##################################################################
# moves from a conventional (hexagonal) lattice vector to 
# cartesian basis, for a single vector
##################################################################
def single_lv_to_cart(u_lv):
    vmat = np.matrix([[-1, 1/2],[0, np.sqrt(3)/2]])
    return vmat @ np.array([u_lv[0], u_lv[1]]) # [-c1 + c2/2, np.sqrt(3)*c2/2] v

##################################################################
# moves from a conventional (hexagonal) lattice vector to 
# cartesian basis, for a matrix of vectors
##################################################################
def latticevec_to_cartesian(uvecs_lv):
    uvecs_cart = np.zeros(uvecs_lv.shape) 
    vmat = np.matrix([[-1, 1/2],[0, np.sqrt(3)/2]]) # [-1,0] or [1/2, np.sqrt(3)/2]
    for i in range(uvecs_lv.shape[0]):
        for j in range(uvecs_lv.shape[1]):
            c1, c2 = uvecs_lv[i,j,0], uvecs_lv[i,j,1]
            uvecs_cart[i,j,:] = vmat @ np.array([c1,c2]) # [-c1 + c2/2, np.sqrt(3)*c2/2] v
    return uvecs_cart

##################################################################
# for each vector in u, replace with an equivalent vector that's
# closest to the value in the references, u_ref
##################################################################
def flip_to_reference(u, uref):
    unew = u.copy()
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            equivs = get_nearby_equivs(u[i,j], extend=True)
            dist = [ (equiv[0,0]-uref[i,j,0])**2 + (equiv[0,1] - uref[i,j,1])**2 for equiv in equivs ]
            unew[i,j,:] = equivs[np.argmin(dist)][:]
    return unew

##################################################################
# moves from cartesian to a conventional (hexagonal) lattice 
# vector basis, for a single vector of shape 2
##################################################################
def single_cart2lv(u):
    vmat = np.matrix([[-1, 1/np.sqrt(3)],[0, 2/np.sqrt(3)]])
    return vmat @ np.array([u[0], u[1]]) # [-cx + cy/np.sqrt(3), 2*cy/np.sqrt(3)]

##################################################################
# moves from cartesian to a conventional (hexagonal) lattice 
# vector basis, for a set of vectors - Nx by Ny by 2
##################################################################
def cartesian_to_latticevec(uvecs_cart):
    uvecs_lv = np.zeros(uvecs_cart.shape)
    vmat = np.matrix([[-1, 1/np.sqrt(3)],[0, 2/np.sqrt(3)]])
    for i in range(uvecs_lv.shape[0]):
        for j in range(uvecs_lv.shape[1]):
            cx, cy = uvecs_cart[i,j,0], uvecs_cart[i,j,1]
            uvecs_lv[i,j,:] = vmat @ np.array([cx,cy]) # [-cx + cy/np.sqrt(3), 2*cy/np.sqrt(3)]
    return uvecs_lv

##################################################################
# moves from cartesian to a reduced zone basis that's a wigner
# seitz cell rather than a conventional (hexagonal) unit cell
##################################################################
def cartesian_to_rz_WZ(uvecs_cart, sign_wrap):
    uvecs_cart = cartesian_to_rzcartesian(uvecs_cart, sign_wrap)
    uvecs_rzcart = np.zeros((uvecs_cart.shape[0], uvecs_cart.shape[1], 2))
    f = 1
    def xytolvperp(cx, cy): return f*(-cy-cx/np.sqrt(3)),   f*(-cx*2/np.sqrt(3))    # perpindicular to a used in lattice vec basis 
    def lvperptoxy(c1, c2): return 1/f*(-np.sqrt(3)/2*c2),  1/f*(-c1+c2/2)          # and scaled by 1/root3
    crit = 1/(np.sqrt(3))
    offsets = [[1,0], [-1,0], [1/2, np.sqrt(3)/2], [-1/2, -np.sqrt(3)/2], [1/2, -np.sqrt(3)/2], [-1/2, np.sqrt(3)/2], [3/2, -np.sqrt(3)/2]]
    for i in range(uvecs_cart.shape[0]):
        for j in range(uvecs_cart.shape[1]):
            cx, cy = uvecs_cart[i,j,0], uvecs_cart[i,j,1] #uvecs_rzcart[i,j,0], uvecs_rzcart[i,j,1] #
            c1, c2 = xytolvperp(cx, cy)
            c3 = -c1+c2
            counter = 0
            while np.abs(c1) > crit or np.abs(c2) > crit or np.abs(c3) > crit: 
                offset  = offsets[counter]
                cxnew, cynew = cx + offset[0], cy + offset[1]
                c1, c2 = xytolvperp(cxnew, cynew)
                c3 = -c1+c2
                counter += 1
                if counter == len(offsets): break
            if counter < len(offsets):
                cxnew, cynew = lvperptoxy(c1, c2) 
                uvecs_rzcart[i,j,:] = cxnew, cynew
            else: uvecs_rzcart[i,j,:] = 0, 0

    #f,ax = plt.subplots(1,2); ax[0].imshow(uvecs_rzcart[:,:,0]); ax[1].imshow(uvecs_rzcart[:,:,1]); plt.show(); 
    return uvecs_rzcart
