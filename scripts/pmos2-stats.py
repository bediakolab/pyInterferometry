
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

# for the P MoS2 Datasets 
"""
data, in order, from :
MV_2.10.21_4d_ds2
MV_2.10.21_4d_ds3
MV_2.10.21_4d_ds4
MV_2.10.21_4d_ds5
MV_2.10.21_4d_ds8
MV_2.10.21_4d_ds10
MV_2.10.21_4d_ds11
MV_2.10.21_4d_ds12
MV_2.10.21_4d_ds13
MV_2.10.21_4d_ds14
MV_2.10.21_4d_ds15
MV_2.10.21_4d_ds16
MV_2.10.21_4d_ds17
MV_2.10.21_4d_ds18
MV_2.10.21_4d_ds19
MV_2.10.21_4d_ds20
MV_2.10.21_4d_ds21
MV_2.10.21_4d_ds22
MV_2.7.20_9_ds4
MV_2.7.20_9_ds7
MV_2.7.20_9_ds8
"""

def fit_func3(x, a, b, c, d):
	x = np.array(x)
	return a*x*x*x + b*x*x + c*x + d

def fit_func2(x, a, b, c):
	x = np.array(x)
	return a*x*x + b*x + c

def colored_3var_scatter(ax, x, y, z, xer, yer, label, fmt):
	z = np.array(z)
	ax.scatter(x, y, c=z, marker=fmt, cmap='viridis')

AApercent = [17.89, 17.44, 11.97, 15.98, 1.26,  1.7,   16.51, 17.9,  16.67, 16.37, 8.12, 15.37, 11.67, 23.14, 17.14, 21.78, 8.51, 11.12, 1.49, 7.09, 7.03]#, 6.98, 3.84]
ABpercent = [49.6,  50.17, 54.18, 51.56, 84.05, 80.62, 54.06, 51.82, 54.63, 55.08, 65.44, 54.45, 59.34, 46.51, 52.69, 48.13, 64.62, 58.4, 87.24, 65.45, 65.8]#, 62.44, 70.68]
SPpercent = [32.51, 32.39, 33.85, 32.47, 14.68, 17.68, 29.43, 30.28, 28.7, 28.54, 26.44, 30.18, 29, 30.35, 30.18, 30.09, 26.87, 30.47, 11.27, 27.46, 27.17]#, 30.59, 25.47]
AAradii   = [2.33,  2.285, 2.42,  2.505, 2.285,   2.565, 2.09,  2.285, 2.255, 2.295, 2.35, 2.28, 2.39, 2.385, 2.495, 2.37, 2.455, 2.475, 2.31, 2.91, 3.1]#, 3.095, 3.185]
AAradii = np.array(AAradii) * 0.5
AAr_stde  = [0.05,  0.04,  0.07,  0.035, 0.245,  0.09,  0.04,  0.03,  0.025, 0.045, 0.05, 0.04, 0.045, 0.025, 0.045, 0.025, 0.08, 0.08, 0.3, 0.105, 0.085]#, 0.08, 0.255]
AAr_stde = np.array(AAr_stde) * 0.5
SPwidths  = [1.275, 1.27,  1.46,  1.37,  1.545, 1.47,  1.11,  1.135, 1.155, 1.16, 1.42, 1.175, 1.265, 1.07, 1.2, 1.085, 1.325, 1.285, 1.81, 1.72, 1.765]#, 1.825, 1.795] 
SPw_stde  = [0.025, 0.025, 0.04,  0.02,  0.17,  0.12,  0.02,  0.02,  0.015, 0.025, 0.04, 0.025, 0.035, 0.015, 0.025, 0.015, 0.07, 0.04, 0.19, 0.03, 0.06]#, 0.04, 0.1]
twist     = [1.83,  1.78,  1.36,  1.57,  0.37,   0.52,  1.86,  1.82,  1.77, 1.72, 1.23, 1.64, 1.35, 1.98, 1.6, 1.92, 1.2, 1.2, 0.25, 0.86, 0.82]#, 1.48, 0.98]
twist_err = [0.02,  0.02,  0.02,  0.01,  0,     0.01,  0.01,  0.02,  0.01, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.02, 0.01, 0.01, 0.01]#, 0.08, 0.05]
hetstrain = [0.38,  0.46,  0.4,   0.44,  0.45,  0.14,  0.56,  0.48,  0.47, 0.85, 0.3, 0.97, 0.6, 1.38, 0.93, 1.32, 0.57, 1.32, 0.13, 0.62, 0.32]#, 2.11, 1.54] 
hetst_err = [0.03,  0.03,  0.04,  0.02,  0.04,   0.01,  0.04,  0.04,  0.03, 0.05, 0.03, 0.05, 0.05, 0.03, 0.05, 0.04, 0.07, 0.1, 0.01, 0.04, 0.02]#, 0.2, 0.26]

# Twist vs Area Fracs
f, ax = plt.subplots(1,2)
ax[0].errorbar(twist, AApercent, xerr=twist_err, fmt="o", color='k', capsize=1, elinewidth=1, label='MMXX')
popt, pcov = curve_fit(fit_func3, twist, AApercent)
ax[0].plot(np.arange(0, 2.5, 0.01), fit_func3(np.arange(0, 2.5, 0.01), *popt), 'k')
ax[0].errorbar(twist, ABpercent, xerr=twist_err, fmt="s", color='grey', capsize=1, elinewidth=1, label='XM')
popt, pcov = curve_fit(fit_func3, twist, ABpercent)
ax[0].plot(np.arange(0, 2.5, 0.01), fit_func3(np.arange(0, 2.5, 0.01), *popt), 'grey')
ax[0].errorbar(twist, SPpercent, xerr=twist_err, fmt="d", color='c', capsize=1, elinewidth=1, label='SP')
popt, pcov = curve_fit(fit_func3, twist, SPpercent)
ax[0].plot(np.arange(0, 2.5, 0.01), fit_func3(np.arange(0, 2.5, 0.01), *popt), 'c')
ax[0].set_xlabel('$\\theta_m (^o)$')
ax[0].set_ylabel('area $(\%)$')
rigid_MMXX, rigid_XM, rigid_SP = 30.20, 38.50, 31.31
ax[0].axhline(y=rigid_MMXX, c='k',linestyle='dashed')
ax[0].axhline(y=rigid_XM, c='grey', linestyle='dashed')
ax[0].axhline(y=rigid_SP, c='c', linestyle='dashed')
ax[0].set_xlim([0.2,2.25])

# Twist vs Radius/Width
ax[1].errorbar(twist, AAradii, xerr=twist_err, yerr=AAr_stde, fmt="o", color='k', capsize=1, elinewidth=1, label='MMXX diameter')
popt, pcov = curve_fit(fit_func2, twist, AAradii)
ax[1].plot(np.arange(0, 2.5, 0.01), fit_func2(np.arange(0, 2.5, 0.01), *popt), 'k')
ax[1].errorbar(twist, SPwidths, xerr=twist_err, yerr=SPw_stde, fmt="d", color='c', capsize=1, elinewidth=1, label='SP width')
popt, pcov = curve_fit(fit_func3, twist, SPwidths)
ax[1].plot(np.arange(0, 2.5, 0.01), fit_func3(np.arange(0, 2.5, 0.01), *popt), 'c')
ax[1].set_xlabel('$\\theta_m (^o)$')
ax[1].set_ylabel('nm')
def rigid_lambda(theta): return 0.315 / (2*np.sin(theta*np.pi/180 * 0.5)) #0.315A for MoS2
ax[1].plot(np.arange(0, 2.5, 0.01), 0.25 * rigid_lambda(np.arange(0, 2.5, 0.01)) * 0.5 , c='k',linestyle='dashed')
ax[1].plot(np.arange(0, 2.5, 0.01), (2 - np.sqrt(3))/2 * rigid_lambda(np.arange(0, 2.5, 0.01)), c='c', linestyle='dashed')
ax[1].set_xlim([0.2,2.25])
ax[1].set_ylim([0.75,2.55])
ax[1].yaxis.set_label_position('right')
ax[1].yaxis.tick_right()

for axis in ax.flatten(): axis.legend(loc='upper right', fontsize='small', frameon=False)

"""
# HS vs Area Fracs
ax[0,2].errorbar(hetstrain, AApercent, xerr=hetst_err, fmt="o", color='k', capsize=1, elinewidth=1, label='MMXX')
ax[0,2].errorbar(hetstrain, ABpercent, xerr=hetst_err, fmt="s", color='grey', capsize=1, elinewidth=1, label='XM')
ax[0,2].errorbar(hetstrain, SPpercent, xerr=hetst_err, fmt="d", color='c', capsize=1, elinewidth=1, label='SP')
ax[0,2].set_xlabel('$\\epsilon (\%)$')
ax[0,2].set_ylabel('area $(\%)$')

# HS vs Radius/Width
ax[0,3].errorbar(hetstrain, AAradii, xerr=hetst_err, yerr=AAr_stde, fmt="o", color='k', capsize=1, elinewidth=1, label='MMXX radius')
ax[0,3].errorbar(hetstrain, SPwidths, xerr=hetst_err, yerr=SPw_stde, fmt="d", color='c', capsize=1, elinewidth=1, label='SP width')
ax[0,3].set_xlabel('$\\epsilon (\%)$')
ax[0,3].set_ylabel('nm')
"""
f, ax = plt.subplots(2,3)
ax = ax.flatten()

# HS vs Twist w/ colored Area Fracs
colored_3var_scatter(ax[0], twist, hetstrain, AApercent, twist_err, hetst_err, 'MMXX', "o")
ax[0].set_ylabel('$\\epsilon (\%)$')
ax[0].set_xlabel('$\\theta_m (^o)$')
colored_3var_scatter(ax[1], twist, hetstrain, ABpercent, twist_err, hetst_err, 'XM', "s")
ax[1].set_ylabel('$\\epsilon (\%)$')
ax[1].set_xlabel('$\\theta_m (^o)$')
colored_3var_scatter(ax[2], twist, hetstrain, SPpercent, twist_err, hetst_err, 'SP', "d")
ax[2].set_ylabel('$\\epsilon (\%)$')
ax[2].set_xlabel('$\\theta_m (^o)$')
colored_3var_scatter(ax[3], twist, hetstrain, AAradii, twist_err, hetst_err, 'MMXX radius', "o")
ax[3].set_ylabel('$\\epsilon (\%)$')
ax[3].set_xlabel('$\\theta_m (^o)$')
colored_3var_scatter(ax[4], twist, hetstrain, SPwidths, twist_err, hetst_err, 'SP width', "d")
ax[4].set_ylabel('$\\epsilon (\%)$')
ax[4].set_xlabel('$\\theta_m (^o)$')


for axis in ax.flatten(): axis.legend(loc='upper right', fontsize='small', frameon=False)
plt.show()
