
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

def prune_data(df, label, material=None, orrient=None, dflag=None, fflag=None, pflag=None, uflag=None):
    use_row = [ True for entry in df["Material"] ]
    if material != None: 
        use_row = [ entry==material and use for entry,use in zip(df["Material"], use_row) ]
    if orrient != None:
        use_row = [ entry==orrient and use for entry,use in zip(df["Orientation"], use_row) ]
    if dflag != None:
        use_row = [ entry==dflag and use for entry,use in zip(df["DataQualityFlag"], use_row) ]
    if fflag != None:
        use_row = [ entry==fflag and use for entry,use in zip(df["FitQualityFlag"], use_row) ]
    if pflag != None:
        use_row = [ entry==pflag and use for entry,use in zip(df["PartitionQualityFlag"], use_row) ]
    if uflag != None:
        use_row = [ entry==uflag and use for entry,use in zip(df["UnwrapQualityFlag"], use_row) ]
    return [ entry for entry,use in zip(df[label], use_row) if use ]

def add_title(ax, material=None, orrient=None, dflag=None, fflag=None, pflag=None, uflag=None):
    if orrient == None:  orrient =  "(ap and p)"
    if material == None: material = "all materials"
    flags = ""
    if not (dflag == fflag == pflag == uflag == None): flags = "- only"
    if dflag != None: flags += " {} data".format(dflag)
    if fflag != None: flags += " {} fits".format(fflag)
    if pflag != None: flags += " {} geom partitions".format(pflag)
    if uflag != None: flags += " {} unwraps".format(uflag)
    title = "{} {} {}".format(orrient, material, flags)
    ax.set_title(title)

def add_dslabels(ax, x, y, dsets):
    for xel, yel, key in zip(x,y,dsets): 
        if not np.isnan(xel) and not np.isnan(yel): ax.text(xel, yel, key)

def compare_errbar(df, xaxis, yaxis, xerr=None, yerr=None, show_dslabels=False, material=None, orrient=None, dflag=None, fflag=None, pflag=None, uflag=None):
    x = prune_data(df, xaxis, material, orrient, dflag, fflag, pflag, uflag)
    y = prune_data(df, yaxis, material, orrient, dflag, fflag, pflag, uflag)
    if xerr != None: xerr = prune_data(df, xerr, material, orrient, dflag, fflag, pflag, uflag)
    if yerr != None: yerr = prune_data(df, yerr, material, orrient, dflag, fflag, pflag, uflag)
    dsets = prune_data(df, "DataSetPath", material, orrient, dflag, fflag, pflag, uflag)
    f, ax = plt.subplots()
    if xerr != None and yerr != None: ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color='k', capsize=1, elinewidth=1)
    elif xerr != None: ax.errorbar(x, y, xerr=xerr, fmt="o", color='k', capsize=1, elinewidth=1)
    elif yerr != None: ax.errorbar(x, y, yerr=yerr, fmt="o", color='k', capsize=1, elinewidth=1)
    else: ax.errorbar(x, y, fmt="o", color='k', capsize=1, elinewidth=1)
    if show_dslabels: add_dslabels(ax, x, y, dsets)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    add_title(ax, material, orrient, dflag, fflag, pflag, uflag)
    plt.show()   

def compare_hist(df, xaxis, material=None, orrient=None, dflag=None, fflag=None, pflag=None, uflag=None):
    x = prune_data(df, xaxis, material, orrient, dflag, fflag, pflag, uflag)
    dsets = prune_data(df, "DataSetPath", material, orrient, dflag, fflag, pflag, uflag)
    f, ax = plt.subplots()
    if not isinstance(x[0], float):
        names = np.unique(x)
        counts = [ np.sum([xel == name for xel in x]) for name in names ]
        ax.bar(names, counts)
    else: ax.hist(x)
    ax.set_xlabel(xaxis)
    ax.set_ylabel("Counts")
    add_title(ax, material, orrient, dflag, fflag, pflag, uflag)
    plt.show()       

def compare_scatter(df, xaxis, yaxis, caxis=None, show_dslabels=False, material=None, orrient=None, dflag=None, fflag=None, pflag=None, uflag=None):
    x = prune_data(df, xaxis, material, orrient, dflag, fflag, pflag, uflag)
    y = prune_data(df, yaxis, material, orrient, dflag, fflag, pflag, uflag)
    dsets = prune_data(df, "DataSetPath", material, orrient, dflag, fflag, pflag, uflag)
    f, ax = plt.subplots()
    if caxis is None: 
        ax.scatter(x, y)
    else:
        z = prune_data(df, caxis, material, orrient, dflag, fflag, pflag, uflag)
        if not isinstance(z[0], float):
            unique_labels = np.unique(z)
            for i in range(len(z)):
                if z[i] not in unique_labels: 
                    z[i] = np.nan 
                else: z[i] = np.where(unique_labels == z[i])[0][0]
            im = ax.scatter(x, y, c=z, cmap='tab20b')
            formatter = plt.FuncFormatter(lambda val, loc: unique_labels[val])
            plt.colorbar(im, ax=ax, orientation='vertical', ticks=np.arange(len(unique_labels)), format=formatter)
        else:
            im = ax.scatter(x, y, c=z, cmap='viridis')
            plt.colorbar(im, ax=ax, orientation='vertical')

    if show_dslabels: add_dslabels(ax, x, y, dsets)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    add_title(ax, material, orrient, dflag, fflag, pflag, uflag)
    plt.show()

if __name__ == '__main__':
    
    #df = compile_spreadsheet()
    df = pd.read_excel('summary.xlsx', index_col=0, comment='#')  

    # some example plots to make
    #compare_scatter(df, xaxis="AvgHeteroStrain", yaxis="AvgMoireTwist", show_dslabels=True,    material='mose2', orrient="ap")
    #compare_scatter(df, xaxis="AAPercent",       yaxis="AvgAAradius",   caxis="DataSetPrefix", show_dslabels=True, pflag="good")

    #compare_hist(df, xaxis="FoundUnwrap")
    #compare_hist(df, xaxis="AvgMoireTwist")

    compare_scatter(df, yaxis="AvgAAradius",  xaxis="AvgMoireTwist", material="mos2", orrient='p', caxis="DataQualityFlag", show_dslabels=False)

    """
    compare_errbar(df,  yaxis="AvgAAradius",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mos2', orrient="ap", show_dslabels=False)
    compare_errbar(df,  yaxis="AvgAAradius",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mos2', orrient="p", show_dslabels=False)
    compare_errbar(df,  yaxis="AvgAAradius",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mose2', orrient="ap", show_dslabels=False)
    compare_errbar(df,  yaxis="AvgAAradius",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mose2', orrient="p", show_dslabels=False)
    compare_errbar(df,  yaxis="AvgAAradius",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mote2', orrient="ap", show_dslabels=False)
    compare_errbar(df,  yaxis="AvgAAradius",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mote2', orrient="p", show_dslabels=False)
    compare_errbar(df,  yaxis="AvgAAradius",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mos2-wse2', orrient="ap", show_dslabels=False)
    compare_errbar(df,  yaxis="AvgAAradius",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mos2-wse2', orrient="p", show_dslabels=False)

    compare_errbar(df,  yaxis="AAPercent",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mos2', orrient="ap", show_dslabels=False)
    compare_errbar(df,  yaxis="AAPercent",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mos2', orrient="p", show_dslabels=False)
    compare_errbar(df,  yaxis="AAPercent",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mose2', orrient="ap", show_dslabels=False)
    compare_errbar(df,  yaxis="AAPercent",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mose2', orrient="p", show_dslabels=False)
    compare_errbar(df,  yaxis="AAPercent",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mote2', orrient="ap", show_dslabels=False)
    compare_errbar(df,  yaxis="AAPercent",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mote2', orrient="p", show_dslabels=False)
    compare_errbar(df,  yaxis="AAPercent",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mos2-wse2', orrient="ap", show_dslabels=False)
    compare_errbar(df,  yaxis="AAPercent",     yerr="ErrAAradius",    xaxis="AvgMoireTwist", xerr="ErrMoireTwist", material='mos2-wse2', orrient="p", show_dslabels=False)
    """


