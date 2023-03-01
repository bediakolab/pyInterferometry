
from heterostrain import hetstrain_from_adjacencies, hetstrain_from_adjacencies_noplot, plotTriangleQuantity, plotTris
from new_utils import import_unwrap_uvector, normNeighborDistance, update_adjacencies
from unwrap_utils import getAdjacencyMatrixManual
from utils import nan_gaussian_filter, boolquery, get_triangles, debugplot
from visualization import displacement_colorplot, plot_adjacency
from strain import plot_grad
import numpy as np
import matplotlib.pyplot as plt
from masking import make_contour_mask
from basis_utils import cartesian_to_rz_WZ
import os

def smooth(u, sigma):
    d = normNeighborDistance(u, norm=False)
    nx, ny = u.shape[0], u.shape[1]
    filtered_u = np.zeros((nx, ny, 2))
    filtered_u[0, :,:] = np.nan 
    filtered_u[-1,:,:] = np.nan 
    filtered_u[:, 0,:] = np.nan 
    filtered_u[:,-1,:] = np.nan
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            if (d[x,y] > 0.05):
                filtered_u[x,y,:] = np.nan, np.nan
            else: filtered_u[x,y,:] = u[x,y,:]
    ux, uy = filtered_u[:,:,0], filtered_u[:,:,1]
    ux = nan_gaussian_filter(ux, sigma) 
    uy = nan_gaussian_filter(uy, sigma) 
    filtered_u[:,:,0], filtered_u[:,:,1] = ux, uy
    return filtered_u


def het_strain_all():
    u, prefix, dsnum, centers, adjacency_type = import_unwrap_uvector(indx=indx, adjust=True)
    thetam, hetstrain, is_hbl, a, ss, poissonratio, delta, given_twist = hetstrain_from_adjacencies(centers, (adjacency_type > 0), ax[1], ax[2]) 

# for manual defined cropping
def manual_mark_tri(img, points, adjacency_type, centers, tris):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_title('click to remove bad triangle centers')
    use_tris = [1 for p in points]
    masks = []
    def plotself(ax, img, points, adjacency_type, centers):
        ax.imshow(img, origin='lower')
        for n in range(adjacency_type.shape[0]):
            for m in range(n, adjacency_type.shape[0]):
                if adjacency_type[n,m] > 0: 
                    ax.plot([centers[n][0], centers[m][0]], [centers[n][1], centers[m][1]], color='k')
    def getclosestpoint(x,y):
        dists = [(p[0]-x)**2 + (p[1]-y)**2 for p in points]
        return points[np.argmin(dists)], np.argmin(dists) 
    def click_event(click):
        x,y = click.xdata, click.ydata
        point, point_id = getclosestpoint(x,y)
        x,y = point[:]
        use_tris[point_id] = 0
        tri = tris[point_id]
        vertices = [centers[tri[0]], centers[tri[1]], centers[tri[2]]]
        mask = make_contour_mask(img.shape[0], img.shape[1], vertices, transpose=True)
        masks.append(mask)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i,j]: 
                    img[i,j] = np.max(img.flatten())
        plotself(ax, img, points, adjacency_type, centers)
        fig.canvas.draw()
    plotself(ax, img, points, adjacency_type, centers)
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    tot_mask = np.zeros((img.shape[0], img.shape[1]))
    for mask in masks: tot_mask += mask
    return tot_mask, use_tris

def manual_define_good_triangles(img, centers, adjacency_type):
    # get region centers from the adjacencies 
    tris = get_triangles((adjacency_type > 0), n=3)
    tri_centers = []
    for tri in tris:
        center_y = np.mean([centers[tri[0]][0], centers[tri[1]][0], centers[tri[2]][0]])
        center_x = np.mean([centers[tri[0]][1], centers[tri[1]][1], centers[tri[2]][1]])
        tri_centers.append([center_y, center_x])
    if False: #boolquery('manually remove regions?'):
        mask, use_tris = manual_mark_tri(img, tri_centers, adjacency_type, centers, tris)   
    else:
        use_tris = [True for tri in tris]
    use_mask = np.zeros((img.shape[0], img.shape[1]))
    for i in range(len(tris)):
        if use_tris[i]:
            tri = tris[i]
            vertices = [centers[tri[0]], centers[tri[1]], centers[tri[2]]]
            mask = make_contour_mask(img.shape[0], img.shape[1], vertices, transpose=True)
            use_mask += mask 
    use_mask = (use_mask > 0).astype(int)
    return (1-use_mask), use_tris, tris, tri_centers

if __name__ == '__main__':

    """
    indx = 0
    write_lines = []
    summary_file = os.path.join('..','results', 'summary_twists.txt')
    while True: # get all twists!!
        try: u, prefix, dsnum, centers, adjacency_type, filenm = import_unwrap_uvector(indx=indx, adjust=False)
        except: break
        write_lines.append("{} :".format(filenm))
        indx += 1
        f, ax = plt.subplots(1,2) 
        thetam, hetstrain, is_hbl, a, ss, poissonratio, delta, given_twist, write_lines = hetstrain_from_adjacencies(centers, (adjacency_type > 0), ax[0], ax[1], prefix, dsnum, write_lines)
        os.makedirs(os.path.join('..', 'plots', prefix), exist_ok=True)
        savepath = os.path.join('..','plots',prefix,'ds{}twist.png'.format(dsnum))
        counter = 1
        while os.path.exists(savepath):
            savepath = os.path.join('..','plots',prefix,'ds{}redo{}twist.png'.format(dsnum,counter))
            counter += 1
        plt.savefig(savepath, dpi=300)
        plt.close('all')
    writefile(summary_file, "", write_lines)
    exit()
    """

    sbool = boolquery("extract?")
    indx = 0
    while sbool:

        adjust = True
        try: u, prefix, dsnum, centers, adjacency_type = import_unwrap_uvector(indx=indx, adjust=True)
        except: break
        indx += 1
        img = displacement_colorplot(None, u)
        if not adjust:#True:#
            centers, adjacency_type = getAdjacencyMatrixManual(img, [[c[1], c[0]] for c in centers], adjacency_type)
            update_adjacencies(u, prefix, dsnum, [[c[1], c[0]] for c in centers], adjacency_type)
            centers = [[c[1], c[0]] for c in centers]

        filtered_u = smooth(u, 2)

        f, ax = plt.subplots(2,3)  
        ax = ax.flatten()
        thetam, hetstrain, is_hbl, a, ss, poissonratio, delta, given_twist = hetstrain_from_adjacencies(centers, (adjacency_type > 0), ax[1], ax[2])  
        img = displacement_colorplot(None, ux, uy)
        plot_adjacency(img, centers, adjacency_type, ax=ax[0], colored=False)        
        theta, gamma, dil = plot_grad(ux, uy, ax[3:5], centers, adjacency_type, thetam, a, ss)
        dil = 100*np.abs(dil)
        gamma = 100*gamma
        from visualization import displacement_categorize
        pAA, pSP1, pSP2, pSP3, pAB, rAA, eAA, wSP, eSP = displacement_categorize(filtered_u, ax0=None, ax1=ax[-1], ax2=None)

        t_1 = "{:.2f} % AA {:.2f} % AB {:.2f} % SP".format(pAA, pAB, pSP1+pSP2+pSP3)
        t_2 = "{:.2f}+/-{:.2f} AAr (pix)  {:.2f}+/-{:.2f} SPw (pix)".format(rAA, eAA, wSP, eSP)
        ax[-1].set_title("{}\n{}".format(t_1, t_2))

        #plt.show()
        os.makedirs(os.path.join('..', 'plots', prefix), exist_ok=True)
        plt.savefig(os.path.join('..','plots',prefix,'ds{}plt1.png'.format(dsnum)), dpi=300)
        plt.close('all')

        mask, use_tris, tris, tri_centers = manual_define_good_triangles(theta, ux, uy, [[c[1], c[0]] for c in centers], adjacency_type)
        print('fraction used : ', np.sum(use_tris)/len(use_tris))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]: 
                    gamma[i,j] = theta[i,j] = dil[i,j] = np.nan
                    img[i,j] = [1.0, 1.0, 1.0]

        tris, thetas, het_strains, theta_colors, het_strain_colors = hetstrain_from_adjacencies_noplot(centers, (adjacency_type > 0), is_hbl, a, ss, poissonratio, delta, given_twist)  
        
        print('average theta_r in whole shown FOV was {} degrees'.format(np.nanmean(theta.flatten())))
        print('--> suggests theta_moire should have actually been {} degrees??'.format(thetam + np.nanmean(theta.flatten())))
        print('average dil in whole shown FOV was {} percent'.format(np.nanmean(dil.flatten())))
        print('average gamma in whole shown FOV was {} percent'.format(np.nanmean(gamma.flatten())))

        f, axes = plt.subplots(4,3)
        axes = axes.flatten()
        axes[2].imshow(img,   origin='lower')
        axes[2].set_title('$u$') 
        plotTris(tris, axes[2], centers, manual=False, use_tris=use_tris)
        im = axes[4].imshow(theta, origin='lower', cmap='RdBu_r')#, vmax=5., vmin=-5.)
        plt.colorbar(im, ax=axes[4], orientation='vertical')
        axes[4].set_title('$\\theta_r$')
        plotTris(tris, axes[4], centers, manual=False, use_tris=use_tris)
        im = axes[5].imshow(dil, origin='lower', cmap='inferno')#, vmax=10., vmin=0.)
        plt.colorbar(im, ax=axes[5], orientation='vertical')
        plotTris(tris, axes[5], centers, manual=False, use_tris=use_tris)
        axes[5].set_title('$dil$')
        im = axes[3].imshow(gamma, origin='lower', cmap='inferno')#, vmax=5., vmin=0.)
        plt.colorbar(im, ax=axes[3], orientation='vertical')
        axes[3].set_title('$\gamma$')  
        plotTris(tris, axes[3], centers, manual=False, use_tris=use_tris)

        uvecs_cart = np.zeros((ux.shape[0], ux.shape[1], 2))
        uvecs_cart[:,:,0], uvecs_cart[:,:,1] = ux, uy
        uvecs_cart = cartesian_to_rz_WZ(uvecs_cart, sign_wrap=False)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]: 
                    uvecs_cart[i,j,0] = uvecs_cart[i,j,1] = np.nan

        start = 0.6
        spacing = 25
        xrang = np.arange(-start,start+(1/spacing),1/spacing)
        N = len(xrang)
        unique_u = np.zeros((N,N,2))
        avg_gamma = np.zeros((N,N))
        avg_theta = np.zeros((N,N))
        avg_dil = np.zeros((N,N))
        counter = np.zeros((N,N))
        unique_u[:,:,1], unique_u[:,:,0] = np.meshgrid(xrang, xrang)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not np.isnan(uvecs_cart[i,j,0]) and not np.isnan(uvecs_cart[i,j,1]):
                    ux_index = int(np.round((uvecs_cart[i,j,0]+start)*spacing))
                    uy_index = int(np.round((uvecs_cart[i,j,1]+start)*spacing))
                    avg_gamma[ux_index, uy_index] += gamma[i,j]
                    avg_theta[ux_index, uy_index] += theta[i,j]
                    avg_dil[ux_index, uy_index] += dil[i,j]
                    counter[ux_index, uy_index] += 1

        img = displacement_colorplot(None, unique_u)
        for i in range(N):
            for j in range(N):    
                if counter[i,j] > 0:
                    avg_gamma[i, j] /= counter[i,j]
                    avg_theta[i, j] /= counter[i,j]
                    avg_dil[i, j] /= counter[i,j]
                else:
                    avg_gamma[i, j] = avg_theta[i, j] = avg_dil[i, j] = counter[i,j] = np.nan

        def stder(v): return np.std(v, ddof=1) / np.sqrt(np.size(v))

        if is_hbl and given_twist is None:
            plotTriangleQuantity(tris, tri_centers, thetas, theta_colors, axes[0], '$<\\theta_m> = {:.2f} +/- {:.2f}^o$'.format(np.nanmean(thetas), stder(thetas)), centers, use_tris=use_tris)
            print('$<\\theta_m> = {:.2f} +/- {:.2f}^o$'.format(np.nanmean(thetas), stder(thetas)))
            plotTriangleQuantity(tris, tri_centers, het_strains, het_strain_colors, axes[1], '$<deform> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(het_strains), stder(het_strains)), centers, use_tris=use_tris)
            print('$<deform> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(het_strains), stder(het_strains)))
        elif is_hbl and given_twist is not None:
            plotTriangleQuantity(tris, tri_centers, thetas, theta_colors, axes[0], '$<\delta> = {:.2f}\% ({:.2f}\%)$'.format(np.abs(np.nanmean(thetas)), 100*np.abs(delta)), centers, use_tris=use_tris)
            plotTriangleQuantity(tris, tri_centers, het_strains, het_strain_colors, axes[1], '$<deform> = {:.2f}\%$'.format(np.nanmean(het_strains)), centers, use_tris=use_tris)
        elif not is_hbl:
            plotTriangleQuantity(tris, tri_centers, thetas, theta_colors, axes[0], '$<\\theta_m> = {:.2f} +/- {:.2f}^o$'.format(np.nanmean(thetas), stder(thetas)), centers, use_tris=use_tris)
            print('$<\\theta_m> = {:.2f} +/- {:.2f}^o$'.format(np.nanmean(thetas), stder(thetas)))
            plotTriangleQuantity(tris, tri_centers, het_strains, het_strain_colors, axes[1], '$<\epsilon> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(het_strains), stder(het_strains)), centers, use_tris=use_tris)
            print('$<\epsilon> = {:.2f} +/- {:.2f} \%$'.format(np.nanmean(het_strains), stder(het_strains)))

        im = axes[6].imshow(avg_theta, origin='lower', cmap='RdBu_r')#, vmax=5., vmin=-5.)
        plt.colorbar(im, ax=axes[6], orientation='vertical')
        im = axes[7].imshow(avg_gamma, origin='lower', cmap='inferno')#, vmax=5.0, vmin=0.0)
        plt.colorbar(im, ax=axes[7], orientation='vertical')
        im = axes[9].imshow(avg_dil, origin='lower', cmap='inferno')#, vmax=10.0, vmin=0.0)
        plt.colorbar(im, ax=axes[9], orientation='vertical')
        im = axes[8].imshow(counter, origin='lower', cmap='inferno')
        plt.colorbar(im, ax=axes[8], orientation='vertical')
        axes[9].set_title('$<dil>$')
        axes[6].set_title('$<\\theta_r>%$')       
        axes[7].set_title('$<\gamma>$')   
        axes[8].set_title('counts')

        for ax in axes: ax.axis('off')
        for ax in axes: ax.set_aspect('equal')
        #plt.show()
        plt.savefig(os.path.join('..','plots',prefix,'ds{}plt2.png'.format(dsnum)), dpi=300)
        plt.close('all')

        sbool = True#boolquery("another?")
