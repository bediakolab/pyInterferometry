
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.colors as mplc
import colorsys
from scipy import ndimage
from visualization import displacement_plt
import pickle
from utils import boolquery
from skimage.restoration import denoise_tv_bregman
from masking import *
import random

global_vmat = np.matrix([[np.sqrt(3),0], [np.sqrt(3)/2, 3/2]]).T
global_g1 = np.array([0, 2/3]) # this set of g corresponds to [a,0] and [a/2, sqrt(3)a/2]
global_g2 = np.array([1/np.sqrt(3),1/3])

def wrap_helper(c1, offset=1):
    # offset = 1 produces c in [-1/2, 1/2]
    # offset = 1/2 produces c in [-1/4, 1/4]
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

def wrap_helper_pair(c1, c2, sign_wrap=True):
    c1 = wrap_helper(c1) # want |a| < 1/2
    c2 = wrap_helper(c2) # want |b| < 1/2
    if sign_wrap and c1 < 0: c1, c2 = -c1, -c2
    return c1, c2

def wrap(dispvecs_x, dispvecs_y, sign_wrap=True):
    a = 1
    vmat = global_vmat
    for i in range(len(dispvecs_x)):
        c = np.matmul(np.linalg.inv(vmat), np.array([dispvecs_x[i], dispvecs_y[i]]))
        c1, c2 = wrap_helper_pair(c[0,0], c[0,1], sign_wrap)
        u = np.matmul(vmat, np.array([c1, c2]))
        dispvecs_x[i], dispvecs_y[i] = u[0,0], u[0,1]
    return dispvecs_x, dispvecs_y

def force_smooth_zone(fx, fy, debug=False):

    vmat = global_vmat # np.matrix([[1,0], [1/2, np.sqrt(3)/2]]).T
    fc1 = np.zeros(fx.shape)
    fc2 = np.zeros(fx.shape)
    # move to lattice vector space, reduced zone
    for i in range(fx.shape[0]):
        for j in range(fx.shape[1]):
            c = np.matmul(np.linalg.inv(vmat), np.array([fx[i, j], fy[i, j]]))
            #c1, c2 = wrap_helper_pair(c[0,0], c[0,1])
            fc1[i, j] = c[0,0]
            fc2[i, j] = c[0,1]

    for i in range(fx.shape[0]):
        for j in range(fx.shape[1]):
            method1, method2 = False, True
            if debug:
                print('****** start w *******')
                print(fc1[i, j])
                print(fc2[i, j])
            if method1: # only works if wrapped without sign!
                meanfc1 = np.sign(np.mean((fc1))) * np.mean(np.abs(fc1))
                meanfc2 = np.sign(np.mean((fc2))) * np.mean(np.abs(fc2))
                dc1 = wrap_helper(fc1[i, j] - meanfc1)
                dc2 = wrap_helper(fc2[i, j] - meanfc2)
                fc1[i, j] = meanfc1 + dc1
                fc2[i, j] = meanfc2 + dc2
            if method2:
                #fc2_sign_change = ( np.sign(np.max(fc2)) != np.sign(np.min(fc2)) )
                meanfc1 = np.sign(np.mean((fc1))) * np.mean(np.abs(fc1))
                meanfc2 = np.sign(np.mean((fc2))) * np.mean(np.abs(fc2))
                dc1_1, dc2_1 = wrap_helper(fc1[i, j] - meanfc1), wrap_helper(fc2[i, j] - meanfc2)
                d_1 = np.sqrt(dc1_1**2 + dc2_1**2)
                dc1_2, dc2_2 = wrap_helper(- fc1[i, j] - meanfc1), wrap_helper(- fc2[i, j] - meanfc2)
                d_2 = np.sqrt(dc1_2**2 + dc2_2**2)
                nanoutambgig = False
                if np.abs(d_2 - d_1) < 0.15 * np.max([np.std(np.abs(fc1)), np.std(np.abs(fc2))]) and nanoutambgig: # hard to tell which zone!
                    fc1[i, j] = np.nan
                    fc2[i, j] = np.nan
                elif d_2 < d_1 :
                    fc1[i, j] = meanfc1 + dc1_2
                    fc2[i, j] = meanfc2 + dc2_2
                else:
                    fc1[i, j] = meanfc1 + dc1_1
                    fc2[i, j] = meanfc2 + dc2_1
                if debug:
                    print('*****************')
                    print(np.mean((fc1)))
                    print(np.sign(np.mean((fc2))) * np.mean(np.abs(fc2)))
                    print('******************')
                    print(dc1_1, dc2_1)
                    print(d_1)
                    print('******************')
                    print(dc1_2, dc2_2)
                    print(d_2)
                    print('******* end w *********')
                    print(meanfc1 + dc1_2, meanfc2 + dc2_2)
                    print(meanfc1 + dc1_1, meanfc2 + dc2_1)
                    print('******************')

    if debug:
        print('******** fc1 fc2 *********')
        print(fc1)
        print(fc2)

    # move back cartesian
    for i in range(fx.shape[0]):
        for j in range(fx.shape[1]):
            u = np.matmul(vmat, np.array([fc1[i, j], fc2[i, j]]))
            fx[i, j] = u[0,0]
            fy[i, j] = u[0,1]
    if debug:
        print('******** fx fy *********')
        print(fx)
        print(fy)
    return fx, fy

def gausfilt(ux, uy, sigma, wrapfunc=force_smooth_zone):
    def convolution(oldimagex, oldimagey, kernel, wrapfunc=force_smooth_zone):
        image_h = oldimagex.shape[0]
        image_w = oldimagex.shape[1]
        kernel_h = kernel.shape[0]
        kernel_w = kernel.shape[1]
        if(len(oldimagex.shape) == 3):
            imagex_pad = np.pad(oldimagex, pad_width=( (kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2),(0,0)), mode='edge').astype(np.float32)
            imagey_pad = np.pad(oldimagey, pad_width=( (kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2),(0,0)), mode='edge').astype(np.float32)
        elif(len(oldimagex.shape) == 2):
            imagex_pad = np.pad(oldimagex, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2)), mode='edge').astype(np.float32)
            imagey_pad = np.pad(oldimagey, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2)), mode='edge').astype(np.float32)
        h = kernel_h // 2
        w = kernel_w // 2
        imagex_conv = np.zeros(imagex_pad.shape)
        imagey_conv = np.zeros(imagey_pad.shape)
        for i in range(h, imagex_pad.shape[0]-h):
            for j in range(w, imagex_pad.shape[1]-w):
                x = imagex_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w].copy()
                y = imagey_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w].copy()
                # to do, make sure these are all in the same zone!
                if wrapfunc is not None:
                    #if i-h == 7 and j-w == 4:
                    #    x, y = wrapfunc(x, y, debug=True)
                    #else:
                    x, y = wrapfunc(x, y, debug=False) # make sure all in the same zone
                x = x.flatten()*kernel.flatten()
                y = y.flatten()*kernel.flatten()
                imagex_conv[i][j] = x.sum()
                imagey_conv[i][j] = y.sum()
                #if i-h == 7 and j-w == 4:
                #    print(imagex_conv[i][j])
                #    print(imagey_conv[i][j])
                #    imagex_conv[i][j] = np.nan
                #    imagey_conv[i][j] = np.nan
        h_end = -h
        w_end = -w
        if(h == 0): return imagex_conv[h:,w:w_end], imagey_conv[h:,w:w_end]
        if(w == 0): return imagex_conv[h:h_end,w:], imagey_conv[h:h_end,w:]
        return imagex_conv[h:h_end,w:w_end], imagey_conv[h:h_end,w:w_end]
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
    uxnew, uynew = convolution(ux, uy, gaussian_filter, wrapfunc)
    return uxnew, uynew

def denoise(ux, uy, wrapfunc=force_smooth_zone, sigma=0.5):
    ux, uy = gausfilt(ux, uy, sigma, wrapfunc) # 1.0
    return ux, uy

def denoise_main(sigma, diskset, A, B, ufit):
    nx, ny = ufit.shape[0:2]
    uvecs = np.zeros((nx, ny,2))
    ux, uy = wrap(ufit[:,:,0].reshape(n*n), ufit[:,:,1].reshape(n*n))
    ux, uy = denoise(ux.reshape(n,n), uy.reshape(n,n), sigma=sigma)
    uvecs[:,:,0], uvecs[:,:,1] = ux, uy
    if sigma == 1: sigmakey = 1
    elif sigma == 0.5: sigmakey = 5
    savepath = os.path.join('..','results', prefix, 'dat_ds{}.{}.pkl_fit'.format(dsnum, sigmakey))
    with open(savepath, 'wb') as f: pickle.dump([diskset, A, B, ufit], f)
    return uvecs

if __name__ == "__main__":
    print('interface not yet set up, please smooth from strain.py interface for now')
    exit()
    phsbool = boolquery("want to smooth a saved dataset?")
    while phsbool:
        uvecs, prefix, dsnum, _ = import_uvector()
        sigma = float("sigma (0.5 and 1.0 work best) ")
        denoise_main(sigma, uvecs, prefix, dsnum)
        phsbool = boolquery("want to smooth another saved dataset?")
