
import os
import py4DSTEM.io
import glob
import pickle
import gc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from masking import make_contour_mask
from basis_utils import get_nearby_equivs, getclosestequiv


def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    coeffs = np.concatenate((ak, T @ ak)).ravel() # a, b, c, d, e, f
    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    a,b,c,d,f,g = coeffs[0], coeffs[1] / 2, coeffs[2], coeffs[3] / 2, coeffs[4] / 2, coeffs[5]
    den = b**2 - a*c
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den # The location of the ellipse centre.
    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    ap = np.sqrt(num / den / (fac - a - c)) # The semi-major and semi-minor axis lengths (these are not sorted).
    bp = np.sqrt(num / den / (-fac - a - c))
    width_gt_height = True
    if ap < bp: 
        width_gt_height = False
        ap, bp = bp, ap
    r = (bp/ap)**2 
    if r > 1: r = 1/r
    e = np.sqrt(1 - r) # The eccentricity.
    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0: phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c: phi += np.pi/2
    if not width_gt_height: phi += np.pi/2
    phi = phi % np.pi
    # A grid of the parametric variable, t.
    t = np.linspace(0, 2*np.pi, 100)
    xfit = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    yfit = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x0, y0, ap, bp, e, phi, xfit, yfit
    

def plot_images(images):
    n = len(images)
    f, ax = plt.subplots(1,n)
    if n > 1:
        for i in range(n): ax[i].imshow(images[i], cmap=plt.cm.gray)
    else: ax.imshow(images[0], cmap=plt.cm.gray)
    plt.show()

def anom_nan_filter(u, remove_filter_crit=None, extend=True):
    print('denoising...')
    neighdist = wrapNeighborDistance(u, extend) #f,ax = plt.subplots(); ax.imshow(neighdist); plt.show();
    nx, ny = u.shape[0], u.shape[1]
    filtered_u = np.zeros((nx, ny, 2))
    filtered_u[0, :,:] = np.nan
    filtered_u[-1,:,:] = np.nan
    filtered_u[:, 0,:] = np.nan
    filtered_u[:,-1,:] = np.nan
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            if remove_filter_crit != None and (neighdist[x,y] > remove_filter_crit):
                filtered_u[x,y,:] = np.nan, np.nan
            else: filtered_u[x,y,:] = u[x,y,:]
    return filtered_u, neighdist

def anom_filter(u, remove_filter_crit, neighdist=None, extend=True, projection=np.median):
    print('denoising...')
    if neighdist == None: neighdist = wrapNeighborDistance(u, extend)
    nx, ny = u.shape[0], u.shape[1]
    filtered_u = np.zeros((nx, ny, 2))
    filtered_u[0, :,:] = np.nan
    filtered_u[-1,:,:] = np.nan
    filtered_u[:, 0,:] = np.nan
    filtered_u[:,-1,:] = np.nan
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            if (neighdist[x,y] > remove_filter_crit):
                filtered_u[x,y,:] = np.nan, np.nan
                u1 = getclosestequiv(u[x+1,y,:], uref=u[x,y,:])
                u2 = getclosestequiv(u[x-1,y,:], uref=u[x,y,:])
                u3 = getclosestequiv(u[x,y+1,:], uref=u[x,y,:])
                u4 = getclosestequiv(u[x,y-1,:], uref=u[x,y,:])
                filtered_u[x,y,0] = projection([u1[0,0], u2[0,0], u3[0,0], u4[0,0]])
                filtered_u[x,y,1] = projection([u1[0,1], u2[0,1], u3[0,1], u4[0,1]])
            else: filtered_u[x,y,:] = u[x,y,:]
    return filtered_u, neighdist

def project_down(u, w=2, extend=True, projection=np.median):
    nx, ny = u.shape[0], u.shape[1]
    filtered_u = np.zeros((nx//w, ny//w, 2))

    for x in range(1,nx-1):
        for y in range(1,ny-1):
            if (neighdist[x,y] > remove_filter_crit):
                filtered_u[x,y,:] = np.nan, np.nan
                u1 = getclosestequiv(u[x+1,y,:], uref=u[x,y,:])
                u2 = getclosestequiv(u[x-1,y,:], uref=u[x,y,:])
                u3 = getclosestequiv(u[x,y+1,:], uref=u[x,y,:])
                u4 = getclosestequiv(u[x,y-1,:], uref=u[x,y,:])
                filtered_u[x,y,0] = projection([u1[0,0], u2[0,0], u3[0,0], u4[0,0]])
                filtered_u[x,y,1] = projection([u1[0,1], u2[0,1], u3[0,1], u4[0,1]])
            else: filtered_u[x,y,:] = u[x,y,:]
    return filtered_u, neighdist

def project_down(uvecs, bin_w=2, method='median'):
    if method == 'median':
        def projectfunc(dat):
            return np.median(dat[:,:,0]), np.median(dat[:,:,1])
    elif method == 'L2':
        def projectfunc(dat): # find u closest to those given defined as what minimizes L2 norm
            return np.mean(dat[:,:,0]), np.mean(dat[:,:,1])
    sx, sy = uvecs.shape[0], uvecs.shape[1] #len(data)
    sx_bin, sy_bin = sx//bin_w, sy//bin_w
    data_binned = np.zeros((sx_bin, sy_bin,2))
    for i in range(sx_bin):
        if i%10==0: print(i/sx_bin * 100, '% done with projection')
        for j in range(sy_bin):
            d = uvecs[ i*bin_w:(i+1)*bin_w , j*bin_w:(j+1)*bin_w, :]
            # force all same zone (fails if too noisy)
            for ii in range(d.shape[0]):
                for jj in range(d.shape[1]):
                    if not (ii==jj==1):
                        d[ii,jj,:] = getclosestequiv(d[ii,jj,:], uref=d[1,1,:], extend=True)
            binu = projectfunc(d[:,:,:])
            data_binned[i,j,:] = binu[:]
    return data_binned

def wrapNeighborStrain(u, ss, extend=True, forward=True, eno=False, unwrapped=False):
    def disthelp(u1, u2):
        if not unwrapped:
            equivs = get_nearby_equivs(u2, extend=extend)
            dist = [ (ue[0,0]-u1[0])**2 + (ue[0,1] - u1[1])**2 for ue in equivs ]
            indx = np.argmin(np.abs(dist))
            ue = equivs[indx]
            return (ue[0,0]-u1[0])/ss, (ue[0,1] - u1[1])/ss
        else:
            return (u2[0]-u1[0])/ss, (u2[1] - u1[1])/ss
    nx, ny = u.shape[0], u.shape[1]
    exx = np.zeros((nx, ny))
    exy = np.zeros((nx, ny))
    eyx = np.zeros((nx, ny))
    eyy = np.zeros((nx, ny))
    if forward:
        for x in range(nx-1):
            for y in range(ny-1):
                exy[x,y], eyy[x,y] = disthelp(u[x,y,:], u[x,y+1,:])
                exx[x,y], eyx[x,y] = disthelp(u[x,y,:], u[x+1,y,:])
    elif eno:
        for x in range(1,nx-1):
            for y in range(1,ny-1):
                exyf, eyyf = disthelp(u[x,y,:], u[x,y+1,:])
                exxf, eyxf = disthelp(u[x,y,:], u[x+1,y,:])
                exyb, eyyb = disthelp(u[x,y-1,:], u[x,y,:])
                exxb, eyxb = disthelp(u[x-1,y,:], u[x,y,:])
                def minabs(lst): return lst[np.argmin([np.abs(e) for e in lst])]
                exy[x,y] = minabs([exyb, exyf])
                eyy[x,y] = minabs([eyyb, eyyf])
                exx[x,y] = minabs([exxb, exxf])
                eyx[x,y] = minabs([eyxb, eyxf])

    e_off = 0.5*(exy+eyx)
    gamma = np.zeros((nx, ny))
    for i in range(exx.shape[0]):
        for j in range(exx.shape[1]):
            e = np.matrix([[exx[i,j], e_off[i,j]], [e_off[i,j], eyy[i,j]]])
            v, u = np.linalg.eig(e)
            emax, emin = np.max(v), np.min(v)
            gamma[i,j] = emax - emin
    return exx, exy, eyx, eyy, gamma

def wrapNeighborDistance(u, extend=True, method=np.median):
    def disthelp(u1, u2):
        equivs = get_nearby_equivs(u2, extend=extend)
        dist = [ (ue[0,0]-u1[0])**2 + (ue[0,1] - u1[1])**2 for ue in equivs ]
        return np.min(np.abs(dist))
    nx, ny = u.shape[0], u.shape[1]
    neighdist = np.zeros((nx, ny))
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            d1 = disthelp(u[x,y,:], u[x+1,y,:])
            d2 = disthelp(u[x,y,:], u[x-1,y,:])
            d3 = disthelp(u[x,y,:], u[x,y+1,:])
            d4 = disthelp(u[x,y,:], u[x,y-1,:])
            neighdist[x,y] = method([d1, d2, d3, d4])
    return neighdist

def normNeighborDistance(u, norm=True, method=np.median):
    def disthelp(u1, u2):      return (u2[0]-u1[0])**2 + (u2[1] - u1[1])**2
    nx, ny = u.shape[0], u.shape[1]
    neighdist = np.zeros((nx, ny))
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            d1 = disthelp(u[x,y,:], u[x+1,y,:])
            d2 = disthelp(u[x,y,:], u[x-1,y,:])
            d3 = disthelp(u[x,y,:], u[x,y+1,:])
            d4 = disthelp(u[x,y,:], u[x,y-1,:])
            neighdist[x,y] = method([d1, d2, d3, d4])
    return neighdist

def manual_define_region(img):
    plt.close('all')
    fig, ax = plt.subplots()
    vertices = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        vertices.append([x,y])
        print('vertex {} at ({},{})'.format(len(vertices), x, y))
        ax.scatter(x,y,color='k')
        if len(vertices) > 1: ax.plot([vertices[-1][0], vertices[-2][0]], [vertices[-1][1], vertices[-2][1]], color='k')
        fig.canvas.draw()
    print("please click where desired region vertices should be for cropping (close figure if you dont wanna crop anything or if youre done)")
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    print('finished with manual region definition')
    return vertices

def dump_matrix(mat, savepath):
    mat = np.matrix(mat)
    with open(savepath,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f,')

def crop_displacement(img, u):
    nx, ny = u.shape[0:2]
    cropbool = input("crop displacement field? (y/n) ").lower().strip()[0] == 'y'
    if cropbool:
        vertices = manual_define_region(img)
        if len(vertices) > 0:
            mask = make_contour_mask(nx, ny, vertices, transpose=True)
            f, ax = plt.subplots()
            for i in range(nx):
                for j in range(ny):
                    if mask[i,j]:
                        img[i,j,:] = np.nan
                        u[i,j,:] = np.nan
            ax.imshow(img)
            plt.show()
    return img, u

def get_lengths(v1, v2, v3):
    l1 = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
    l2 = ((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)**0.5
    l3 = ((v3[0] - v2[0])**2 + (v3[1] - v2[1])**2)**0.5
    return l1, l2, l3

def get_angles(l1, l2, l3):
    a12 = np.arccos((l1**2 + l2**2 - l3**2)/(2*l1*l2))
    a23 = np.arccos((l2**2 + l3**2 - l1**2)/(2*l3*l2))
    a31 = np.arccos((l1**2 + l3**2 - l2**2)/(2*l1*l3))
    return a12, a23, a31

def get_area(a, b, c):
    s = (a + b + c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))

def parse_filepath(path):
    tmp        = path.replace('\\','/').replace('//','/').split('/')
    tmp        = [el for el in tmp if len(el.strip()) > 0]
    prefix     = tmp[-2]
    dsnum      = int(tmp[-1].split("_")[0])
    tmp        = tmp[-1].split("_")[1]
    scan_shape = [ int(tmp.split("x")[0]), int(tmp.split("x")[1]) ]
    return prefix, dsnum, scan_shape

def get_diskset_index_options():
    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl'))
    return [i for i in range(len(pickles))]

def import_probe(indx=None):
    probes = glob.glob(os.path.join('..', 'data', 'probe*'))
    if indx == None:
        for i in range(len(probes)): print('{}:    {}'.format(i, probes[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = probes[indx]
    return filepath

def import_diskset(indx=None):
    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl'))
    if indx == None:
        for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = pickles[indx]
    print('reading from {}'.format(filepath))
    tmp = filepath.replace('\\','/').replace('//','/').split('/')
    tmp = [el for el in tmp if len(el.strip()) > 0]
    prefix = tmp[-2]
    dsnum = int(tmp[-1].split('ds')[1].split('.')[0])
    with open(filepath, 'rb') as f: diskset = pickle.load(f)
    return diskset, prefix, dsnum

def import_unwrap_uvector(indx=None, adjust=False):
    if not adjust: pickles = glob.glob(os.path.join('..', 'results', '*', '*unwrap'))
    else: pickles = glob.glob(os.path.join('..', 'results', '*', 'adjust*unwrap'))
    if indx == None:
        for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = pickles[indx]
    print('reading from {}'.format(filepath))
    tmp = filepath.replace('\\','/').replace('//','/').split('/')
    tmp = [el for el in tmp if len(el.strip()) > 0]
    prefix = tmp[-2]
    dsnum = int(tmp[-1].split('ds')[1].split('.')[0])
    with open(filepath, 'rb') as f: d = pickle.load(f)
    u, centers, adjacency_type = d[0], d[1], d[2]
    return u, prefix, dsnum, centers, adjacency_type, filepath

def update_adjacencies(u, prefix, dsnum, centers, adjacency_type):
    filepath = os.path.join('..', 'results', prefix, "adjust_ds{}.pkl_unwrap".format(dsnum))
    print('updated adjacencies saved to {}'.format(filepath))
    with open(filepath, 'wb') as f: pickle.dump([u, centers, adjacency_type, None], f) 

def import_disket_uvector(indx=None):
    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl_fit*'))
    if indx == None:
        for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = pickles[indx]
    print('reading from {}'.format(filepath))
    try:
        tmp = filepath.replace('\\','/').replace('//','/').split('/')
        tmp = [el for el in tmp if len(el.strip()) > 0]
        prefix = tmp[-2]
        dsnum = int(tmp[-1].split('ds')[1].split('.')[0])
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            diskset = data[0]
            if len(data) == 2:
                uvecs = data[1]
            elif len(data) == 4:
                coefs = data[1:3]
                uvecs = data[3]
            elif len(data) == 5:
                coefs = data[1:4]
                uvecs = data[4]
        return uvecs, prefix, dsnum, coefs, diskset
    except:
        print('failed reading ', filepath)
        return None, None, None

def import_uvectors(all=False):
    if not all: pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl*asym*')) 
    else: pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl*refit*'))
    pickles.sort() 
    for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
    indxs = [int(i) for i in input("which to use? enter separated by commas").lower().strip().split(',')]
    uvecs = []
    prefixes = []
    dsnums = []
    binbools = []
    for i in indxs:
        u, pf, n, b = import_uvector(i, all)
        uvecs.append(u)
        prefixes.append(pf)
        dsnums.append(n)
        binbools.append(b)
    return uvecs, prefixes, dsnums, binbools

def import_uvector(indx=None, all=False):
    if not all: pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl*asym*')) 
    else: pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl*refit*')) 
    pickles.sort() 
    if indx == None:
        for i in range(len(pickles)): print('{}:    {}'.format(i, pickles[i]))
        indx = int(input("which to use? ").lower().strip())
    filepath = pickles[indx]
    print('reading from {}'.format(filepath))
    try:
        isbinned = (filepath[-4:-1] == 'bin')
        tmp = filepath.replace('\\','/').replace('//','/').split('/')
        tmp = [el for el in tmp if len(el.strip()) > 0]
        prefix = tmp[-2]
        dsnum = int(tmp[-1].split('ds')[1].split('.')[0])
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            diskset = data[0]
            #print(diskset.determine_rings())
            if len(data) == 2:
                uvecs = data[1]
            elif len(data) == 4:
                A = data[1]
                B = data[2]
                #print(A, B)
                uvecs = data[3]
            elif len(data) == 5:
                A = data[1]
                B = data[2]
                C = data[3]
                #print(A, B, C)
                uvecs = data[4]
        return uvecs, prefix, dsnum, isbinned
    except:
        print('failed reading ', filepath)
        return None, None, None, False

def chunk_split(datapath, chunksize):
    offset = 0
    for d in [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, el))]:
        m_dir = os.path.join(datapath,d)
        os.makedirs(os.path.join(datapath,'before_chunksplit'), exist_ok=True)
        new_dir = os.path.join(datapath,'before_chunksplit',d)
        os.rename(m_dir, new_dir)
        splitd = d.split("_")
        tmp = splitd[1]
        scan_shape = [ int(tmp.split("x")[0]), int(tmp.split("x")[1]) ]
        gc.collect()
        if np.max(scan_shape) > chunksize:
            nx = int(np.ceil(scan_shape[0]/chunksize))
            ny = int(np.ceil(scan_shape[1]/chunksize))
            data = py4DSTEM.io.read( os.path.join(new_dir, 'Diffraction SI.dm4') )
            data.set_scan_shape(scan_shape[0],scan_shape[1])
            count = 0
            for i in range(nx):
                for j in range(ny):
                    splitd = d.split("_")
                    splitd[0] = str(1 + count + offset)
                    upperx = min((i+1)*chunksize, scan_shape[0])
                    uppery = min((j+1)*chunksize, scan_shape[1])
                    splitd[1] = "{}x{}".format(upperx-i*chunksize,uppery-j*chunksize)
                    new_dir = os.path.join(datapath, "_".join(splitd))
                    os.makedirs(new_dir, exist_ok=True)
                    dat = data.data[i*chunksize:upperx, j*chunksize:uppery, :, :]
                    datcubetmp = py4DSTEM.io.datastructure.datacube.DataCube(dat)
                    py4DSTEM.io.save(os.path.join(new_dir, 'dp.h5'), datcubetmp,overwrite=True)
                    count += 1
            offset += count
