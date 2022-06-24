
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
from skimage.restoration import denoise_tv_bregman
from masking import *
import random
from denoise import *
from basis_utils import lvstrain_to_cartesianstrain, cartesian_to_latticevec, rz_helper

def selectminabs(lst): return lst[np.argmin([np.abs(e) for e in lst])]
def minabs(vals): return vals[np.argmin([np.abs(v) for v in vals])]

# ENO forward/backward 
def gradient_wrapped3(f, h, wrap=True, signwrap=False):
    dfdx = np.zeros(f.shape)
    dfdy = np.zeros(f.shape)
    dfdx_alt = np.zeros(f.shape)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if i == 0:
                df = (f[i+1,j] - f[i,j]) # forward
                if wrap: 
                    df1 = rz_helper((f[i+1,j] - f[i,j]))
                    df2 = rz_helper((f[i+1,j] + f[i,j]))
                    dfs = [df1, df2]
                    df  = dfs[np.argmin([np.abs(df) for df in dfs])]
                dfdx[i,j] = df/(h)
            elif i == f.shape[0]-1:
                df = (f[i,j] - f[i-1,j]) # backward
                if wrap: 
                    df1 = rz_helper((f[i,j] - f[i-1,j]))
                    df2 = rz_helper((f[i,j] + f[i-1,j]))
                    dfs = [df1, df2]
                    df  = dfs[np.argmin([np.abs(df) for df in dfs])]
                dfdx[i,j] = df/(h)
            else:
                if wrap:
                    df1 = rz_helper((f[i+1,j] - f[i,j]))
                    df2 = rz_helper((f[i,j] - f[i-1,j]))
                    df3 = rz_helper((f[i+1,j] + f[i,j]))
                    df4 = rz_helper((f[i,j] + f[i-1,j]))
                    dfs = [df1, df2, df3, df4]
                else:
                    df1 = (f[i+1,j] - f[i,j]) # ENO forward v backward
                    df2 = (f[i,j] - f[i-1,j])
                    dfs = [df1, df2]
                df  = dfs[np.argmin([np.abs(df) for df in dfs])]
                dfdx[i,j] = df/(h)
            if j == 0:
                df = (f[i,j+1] - f[i,j])
                if wrap: 
                    df1 = rz_helper((f[i,j+1] - f[i,j]))
                    df2 = rz_helper((f[i,j+1] + f[i,j]))
                    dfs = [df1, df2]
                    df  = dfs[np.argmin([np.abs(df) for df in dfs])]
                dfdy[i,j] = df/(h)
            elif j == f.shape[0]-1:
                df = (f[i,j] - f[i,j-1])
                if wrap: 
                    df1 = rz_helper((f[i,j] - f[i,j-1]))
                    df2 = rz_helper((f[i,j] - f[i,j-1]))
                    dfs = [df1, df2]
                    df  = dfs[np.argmin([np.abs(df) for df in dfs])]
                dfdy[i,j] = df/(h)
            else:
                if wrap:
                    df1 = rz_helper((f[i,j+1] - f[i,j]))
                    df2 = rz_helper((f[i,j] - f[i,j-1]))
                    df3 = rz_helper((f[i,j+1] + f[i,j]))
                    df4 = rz_helper((f[i,j] + f[i,j-1]))
                    dfs = [df1, df2, df3, df4]
                else:
                    df1 = (f[i,j+1] - f[i,j]) # ENO forward v backward
                    df2 = (f[i,j] - f[i,j-1])
                    dfs = [df1, df2]
                df  = dfs[np.argmin([np.abs(df) for df in dfs])]
                dfdy[i,j] = df/(h)
    return dfdx, dfdy

def differentiate_lv(uvecs_lv, x, y, wrap=True):
    ss = y[1,1] - y[0,1]
    nx, ny = y.shape 
    du1dx, du1dy = gradient_wrapped3(uvecs_lv[:,:,0], ss, wrap) # du1/dx, du1/dy
    du2dx, du2dy = gradient_wrapped3(uvecs_lv[:,:,1], ss, wrap) # du2/dx, du2/dy
    dxx, dyx, dxy, dyy = lvstrain_to_cartesianstrain(du1dx, du1dy, du2dx, du2dy)
    return dxx, dyx, dxy, dyy

def differentiate(uvecs_cart, x, y, wrap=True):
    uvecs_lv = cartesian_to_latticevec(uvecs_cart)
    return differentiate_lv(uvecs_lv, x, y, wrap)
