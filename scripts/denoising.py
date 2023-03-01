
import matplotlib.pyplot as plt
from new_utils import import_uvector, dump_matrix
from new_utils import anom_nan_filter, anom_filter, project_down
from visualization import displacement_colorplot
from basis_utils import latticevec_to_cartesian, cartesian_to_latticevec
from utils import merge_u, boolquery
import os
import pickle    
import glob
   
if __name__ == '__main__':

    pickles = glob.glob(os.path.join('..', 'results', '*', '*pkl_fit_asym'))
    for indx in range(len(pickles)):
        u, prefix, dsnum, isbinned = import_uvector(indx)
        if not isbinned:
            u = latticevec_to_cartesian(u)
            ufilt2 = project_down(u, 2, method='L2')
            ufilt4 = project_down(ufilt2, 2, method='L2')
            ufilt6 = project_down(ufilt2, 3, method='L2')
            f, ax = plt.subplots(1,4) 
            displacement_colorplot(ax[0], u, quiverbool=False)
            displacement_colorplot(ax[1], ufilt2, quiverbool=False)
            displacement_colorplot(ax[2], ufilt4, quiverbool=False)
            displacement_colorplot(ax[3], ufilt6, quiverbool=False)
            plt.savefig("../plots/{}/ds_{}/binning.png".format(prefix,dsnum), dpi=300)
            savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_fit_asym_bin2'.format(dsnum))
            ufilt2 = cartesian_to_latticevec(ufilt2)
            with open(savepath, 'wb') as f: pickle.dump([None, None, None, None, ufilt2], f)
            savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_fit_asym_bin4'.format(dsnum))
            ufilt4 = cartesian_to_latticevec(ufilt4)
            with open(savepath, 'wb') as f: pickle.dump([None, None, None, None, ufilt4], f)
            savepath = os.path.join('..','results', prefix, 'dat_ds{}.pkl_fit_asym_bin6'.format(dsnum))
            ufilt6 = cartesian_to_latticevec(ufilt6)
            with open(savepath, 'wb') as f: pickle.dump([None, None, None, None, ufilt6], f)


    
   