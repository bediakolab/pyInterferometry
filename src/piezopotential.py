
import numpy as np  
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2

# import g and u
# NOTE: need reference frame with positive direction of y-axis being 
# in-plane projection of a M-X vector in bottom layer!
# they expanded up to "80th reciprocal space star"

def compute_rs_from_ms(g, xi, yi, u_ms, adl_factor=lambda n: 1):
    valx, valy = 0, 0
    N = len(g)
    for n in range(N):
        # u_ms is N by 2 by 2, first dim is corresponding g, then x or y, then re or im
        f = adl_factor(n)
        gdotr = g[n][0] * xi + g[n][1] * yi
        valx += u_ms[n,0,0] * np.cos(gdotr) - u_ms[n,0,1] * np.sin(gdotr)
        valy += u_ms[n,1,0] * np.cos(gdotr) - u_ms[n,1,1] * np.sin(gdotr)
    return valx, valy

# inline equation before S9
def alpha(d_0, eps_p): 
    return d_0 * (eps_p - 1) / (4 * np.pi) # same units as d_0, so nm

def in_rs_unitcell(r, a0=1):
    a1 = a0 * [1,0]
    a2 = a0 * [1/2, np.sqrt(3)/2]
    # want r = c1 a1 + c2 a2, c1,c1 in [0,1)
    c2 = (2*r[1])/(np.sqrt(3)*a0)
    c1 = (r[0]/a0) - c2/2
    if not ( c1 >= 0 and c1 < 1 ):
        print('OUTSIDE UNIT CELL FOR ', c1, r)
    if not ( c2 >= 0 and c2 < 1 ):
        print('OUTSIDE UNIT CELL FOR ', c2, r)

def reduce_rs_unitcell(rs_val, a0=1):
    c2 = (2*rs_val[1])/(np.sqrt(3)*a0)
    c1 = (rs_val[0]/a0) - c2/2
    while c1 < 0: c1 += 1
    c1 = c1%1
    while c2 < 0: c2 += 1
    c2 = c2%1
    rs_val = [a0*c1 + a0*c2/2, a0*np.sqrt(3)*c2/2]
    in_rs_unitcell(rs_val, a0)
    return rs_val


def generate_fake_g(a_0, npts, visualize, include_zero=True):
    # hexagonal unit cell w/ a1 = a[1,0] and a2 = a[1/2, root3/2]
    b1 = ( np.pi * 2 / a_0 ) * np.array([1, -1/np.sqrt(3)])
    b2 = ( np.pi * 2 / a_0 ) * np.array([0,  2/np.sqrt(3)])
    gvecs = []
    gmags = []

    c = 2 * npts/6 
    lim = int(np.ceil((np.sqrt(4*c-1) - 1)/2))

    if visualize: f, ax = plt.subplots()
    for i in np.arange(-lim,lim+1,1):
        for j in np.arange(-lim,lim+1,1):
            if np.abs(i - j) > lim: continue
            if i==j==0 and not include_zero: continue
            g = i * b1 + j * b2
            gmag = np.dot(g,g) ** 0.5
            gmags.append(gmag)
            gvecs.append(g)
            
    while len(gvecs) > npts:
        i = gmags.index(np.max(gmags))
        gmags.remove(gmags[i])
        gvecs = [ gvecs[j] for j in range(len(gvecs)) if j != i ] 

    if visualize: 
        for g in gvecs:
            ax.scatter(g[0], g[1], c='k')
            #ax.text(g[0], g[1], "{}".format(gmag))

    if visualize: plt.show()
    return gvecs

def get_tiling_rs_unitcell(spacing, a0=1, visualize=False):
    a1 = a0 * [1,0]
    a2 = a0 * [1/2, np.sqrt(3)/2]
    N = int(1/spacing)
    xmat = np.zeros((N,N))
    ymat = np.zeros((N,N))
    if visualize: f, ax = plt.subplots()
    for i in np.arange(N):
        for j in np.arange(N):
            c1, c2 = spacing*i, spacing*j
            xmat[i,j] = c1 * a1[0] + c2 * a2[0]
            ymat[i,j] = c1 * a1[1] + c2 * a2[1]
            in_rs_unitcell([xmat[i,j], ymat[i,j]], a0=1)
            if visualize: 
                ax.scatter(xmat[i,j], ymat[i,j], c='k')
    if visualize: plt.show()
    return xmat, ymat

# if generate u_g from u_r, make sure ms2rs(u_g)=u_r!
def sanitycheck2(tol=1e-5):

    N = 20 
    spacing = 1/N
    g = generate_fake_g(a_0=1, npts=N*N, visualize=False)
    x, y = get_tiling_rs_unitcell(spacing, visualize=False)
    Nx, Ny = x.shape[0], x.shape[1]
    u_rs = np.zeros((Nx, Ny, 2))
    
    if False:
        for i in range(Nx):
            for j in range(Ny): 
                u_rs[i,j,0] = 0.5
                u_rs[i,j,1] = 0.0
        
        u_ms = rs_to_ms_u(g, x, y, u_rs) 
        u_rs_2 = ms_to_rs_u(g, x, y, u_ms) 
        u_ms_2 = rs_to_ms_u(g, x, y, u_rs_2)

        # making sure that get u_ms perfectly from this data
        assert(np.max(np.abs((u_rs[:,:,0] - u_rs_2[:,:,0]))) < tol)
        assert(np.max(np.abs((u_rs[:,:,1] - u_rs_2[:,:,1]))) < tol)
        assert(np.max(np.abs((u_ms[:,0] - u_ms_2[:,0]))) < tol)
        assert(np.max(np.abs((u_ms[:,1] - u_ms_2[:,1]))) < tol)
        print('sanitycheck 2.1 passes')

    if False:
        for i in range(Nx):
            for j in range(Ny): 
                u_rs[i,j,0] = -0.15
                u_rs[i,j,1] = 0.04
        
        u_ms = rs_to_ms_u(g, x, y, u_rs) 
        u_rs_2 = ms_to_rs_u(g, x, y, u_ms) 
        u_ms_2 = rs_to_ms_u(g, x, y, u_rs_2)

        # making sure that get u_ms perfectly from this data
        assert(np.max(np.abs((u_rs[:,:,0] - u_rs_2[:,:,0]))) < tol)
        assert(np.max(np.abs((u_rs[:,:,1] - u_rs_2[:,:,1]))) < tol)
        assert(np.max(np.abs((u_ms[:,0] - u_ms_2[:,0]))) < tol)
        assert(np.max(np.abs((u_ms[:,1] - u_ms_2[:,1]))) < tol)
        print('sanitycheck 2.2 passes')

    if True:
        u_rs = expu()
        from visualization import displacement_colorplot
        f,ax = plt.subplots(4,2)   
        img = displacement_colorplot(ax[3,0], u_rs, quiverbool=False)
        u_rs[:,:,1] *= 0 # now u_rs represents Re and Im part of u_x

        g = generate_fake_g(a_0=1, npts=u_rs.shape[0]*u_rs.shape[1], visualize=False)
        u_ms = rs_to_ms_u(g, x, y, u_rs) 
        u_rs_2 = ms_to_rs_u(g, x, y, u_ms) 

        expect_dudx = np.gradient(u_rs[:,:,0], axis=0)
        expect_dudy = np.gradient(u_rs[:,:,0], axis=1)
        dudx, dudy = grads_from_ms(g, x, y, u_ms)

        ax[0,0].imshow(u_rs[:20,:20,0])
        ax[0,1].imshow(u_rs_2[:20,:20,0])
        ax[1,0].imshow(expect_dudx[:20,:20])
        ax[1,1].imshow(dudx[:20,:20,0])
        ax[2,0].imshow(expect_dudy[:20,:20])
        ax[2,1].imshow(dudy[:20,:20,0])

        plt.show()
        exit()

    if True:
        for i in range(Nx):
            for j in range(Ny): 
                u_rs[i,j,0] =  urs_top_x_rep_from_theory(i,j) # -0.15 + (0.2*i*i - 0.1*j*j)
                u_rs[i,j,1] = 0.0

        f, ax = plt.subplots()
        u_rs = np.zeros((100, 100, 2))
        for i in np.arange(0, 100):
            for j in np.arange(0, 100): 
                u_rs[i,j,0] =  urs_top_x_rep_from_theory(i*3,j*3) # -0.15 + (0.2*i*i - 0.1*j*j)
                u_rs[i,j,1] = 0.0        
        ax.imshow(u_rs[:,:,0], origin='lower')   
        plt.show()
        exit()     

        
        
        u_ms = rs_to_ms_u(g, x, y, u_rs) 
        u_rs_2 = ms_to_rs_u(g, x, y, u_ms) 
        u_ms_2 = rs_to_ms_u(g, x, y, u_rs_2)

        dudx, dudy = grads_from_ms(g, x, y, u_ms)

        # making sure that get u_ms perfectly from this data
        f,ax = plt.subplots(3,2)
        
        ax[0,0].imshow(u_rs[:,:,0])
        ax[0,1].imshow(u_rs_2[:,:,0])

        ax[1,0].imshow(expect_dudx)
        ax[1,1].imshow(dudx[:,:,0])

        ax[2,0].imshow(expect_dudy)
        ax[2,1].imshow(dudy[:,:,0])

        plt.show()
        assert(np.max(np.abs((u_rs[:,:,0] - u_rs_2[:,:,0]))) < tol)
        assert(np.max(np.abs((u_rs[:,:,1] - u_rs_2[:,:,1]))) < tol)
        assert(np.max(np.abs((u_ms[:,0] - u_ms_2[:,0]))) < tol)
        assert(np.max(np.abs((u_ms[:,1] - u_ms_2[:,1]))) < tol)
        print('sanitycheck 2.3 passes')

    exit()

def rs_to_ms_all_u(g, x, y, u_rs_x_t, u_rs_y_t, u_rs_x_b, u_rs_y_b):

    u_ms_x_t = rs_to_ms_u(g, x, y, u_rs_x_t) 
    u_ms_y_t = rs_to_ms_u(g, x, y, u_rs_y_t) 
    u_ms_x_b = rs_to_ms_u(g, x, y, u_rs_x_b) 
    u_ms_y_b = rs_to_ms_u(g, x, y, u_rs_y_b)
    Ng, Ndim = len(g), 2
    u_t = np.zeros((Ng, Ndim, 2)) # last axis is Re part, Im part
    u_b = np.zeros((Ng, Ndim, 2)) # last axis is Re part, Im part

    for n in range(Ng):
        u_t[n, 0, 0], u_t[n, 0, 1] = u_ms_x_t[n, 0], u_ms_x_t[n, 1]
        u_t[n, 1, 0], u_t[n, 1, 1] = u_ms_y_t[n, 0], u_ms_y_t[n, 1]
        u_b[n, 0, 0], u_b[n, 0, 1] = u_ms_x_b[n, 0], u_ms_x_b[n, 1]
        u_b[n, 1, 0], u_b[n, 1, 1] = u_ms_y_b[n, 0], u_ms_y_b[n, 1] 

    return u_t, u_b

def assert_eq_matrix(M1, M2, tol=1e-4):
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            assert_eq_val(M1[i,j], M2[i,j], tol=1e-4)

def assert_eq_val(v1, v2, tol=1e-4):
	if not (np.abs(v1-v2) < tol): print("NOT EQUAL ", v1, v2)         

def x_derivative(f, spacing, a0=1):
    return np.gradient(f, axis=0) * 1/a0 * 1/spacing 

def y_derivative(f, spacing, a0=1):
    return np.gradient(f, axis=1) * 1/a0 * 2/np.sqrt(3) * 1/spacing - np.gradient(f, axis=0) * 1/a0 * 1/np.sqrt(3) * 1/spacing

def sanitycheck3(tol=1e-8):

    if False:
        for spacing, nrings in zip([0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.1,0.1,0.1],[5,10,20,25,30,5,10,20,25,30]):
            g = generate_fake_g(a_0=1, lim=nrings, visualize=False)
            x, y, u_rs_x_t, u_rs_y_t, u_rs_x_b, u_rs_y_b = generate_fake_u_rs(spacing=spacing, visualize=False, a0=1)
            #u_t, u_b = rs_to_ms_all_u(g, x, y, u_rs_x_t, u_rs_y_t, u_rs_x_b, u_rs_y_b)
            u_ms    = rs_to_ms_u(g, x, y, u_rs_x_t) # FACTOR OFF
            u_rs_2  = ms_to_rs_u(g, x, y, u_ms) # RS that should be perfectly represented by the above MS
            u_ms_2  = rs_to_ms_u(g, x, y, u_rs_2) # FACTOR OFF
            print(spacing, nrings, "--->", u_rs_2[3,3,0]/u_rs_x_t[3,3,0], "  ", u_ms_2[5,0]/u_ms[5,0])

    spacing, nrings = 0.1, 25 # for spacing of greater than 0.1 (like 0.2) found large anomolies in gradient
    g = generate_fake_g(a_0=1, lim=nrings, visualize=False)
    x, y, u_rs_x_t, u_rs_y_t, u_rs_x_b, u_rs_y_b = generate_fake_u_rs(spacing=spacing, visualize=False, a0=1)
    u_t, u_b = rs_to_ms_all_u(g, x, y, u_rs_x_t, u_rs_y_t, u_rs_x_b, u_rs_y_b)        
    eps, alphaval, d = 2.9, alpha(0.65, 16.3), 0.65

    # ux, uy = -y, x^2/2
    dx_ux = x_derivative(u_rs_x_t[:,:,0], spacing) # 0
    dy_ux = y_derivative(u_rs_x_t[:,:,0], spacing) # -1
    dx_uy = x_derivative(u_rs_y_t[:,:,0], spacing) # x
    dy_uy = y_derivative(u_rs_y_t[:,:,0], spacing) # 0
    dxy_ux = x_derivative(dy_ux, spacing) # 0
    dxx_uy = x_derivative(dx_uy, spacing) # 1
    dyy_uy = y_derivative(dy_uy, spacing) # 0

    assert_eq_val(dxy_ux[3,3], 0.2)
    assert_eq_val(dxx_uy[3,3], 0.3)
    assert_eq_val(dyy_uy[3,3], 0.4)

    #u_rs_x = ms_to_rs_u(g, x, y, u_t[:,0,:])    
    #assert_eq_val(u_rs_x_t[3,3,0], u_rs_x[3,3,0])
    #u_rs_y = ms_to_rs_u(g, x, y, u_t[:,1,:])    
    #assert_eq_val(u_rs_y_t[3,3,0], u_rs_y[3,3,0])

    expect_piezo_top = eps * ( 2*dxy_ux + dxx_uy - dyy_uy )
    expect_piezo_tot = expect_piezo_top * 2
    #print(expect_piezo_top[2:-2, 2:-2])
    #rho_piezo_t = compute_rho_tot_t(x, y, g, d, 0.0, 0.0, eps, -eps, u_t, u_b) # set alpha=0 for just the piezo part
    #print(rho_piezo_t[2:-2, 2:-2, 0])

    val_dxy_ux, val_dxx_uy, val_dyy_uy, valx, valy = 0,0,0,0,0
    Ng = len(g)
    for n in range(Ng):
        gdotr = g[n][0] * x[3,3] + g[n][1] * y[3,3]
        uxval = (u_t[n,0,0] * np.cos(gdotr) - u_t[n,0,1] * np.sin(gdotr))/Ng
        uyval = (u_t[n,1,0] * np.cos(gdotr) - u_t[n,1,1] * np.sin(gdotr))/Ng
        val_dx_ux += g[n][0] * uxval / Ng
        val_dy_ux += g[n][1] * uxval / Ng
        valx += uxval
        valy += uyval

    
    print()


    assert_eq_val(0.2, val_dxy_ux)
    assert_eq_val(0.3, val_dxx_uy)
    assert_eq_val(0.4, val_dyy_uy)

    exit()



    # test 1, defined u s.t. expect rho_piezo_t[i,j] = rho_piezo_b[i,j]= - 0.18 * eps 
    rho_t = compute_rho_tot_t(x, y, g, d, alphaval, alphaval, eps, -eps, u_t, u_b)
    
    # expecting rho_piezo_t[i,j] = - 0.18 * eps 
    assert_eq_matrix(rho_piezo_t[:,:,0], - 0.18 * eps * np.ones((rho_piezo_t.shape[0], rho_piezo_t.shape[1]))) 

    dzz_varphi_rs_t_val = delzz_varphi_rs_t(x, y, g, d, alphaval, alphaval, eps, -eps, u_t, u_b)
    varphi_rs_t_val = varphi_rs_t(x, y, g, d, alphaval, alphaval, eps, -eps, u_t, u_b)
    dzz_varphi_rs_t_val, varphi_rs_t_val = dzz_varphi_rs_t_val[:,:,0], varphi_rs_t_val[:,:,0] # grab out re part
    dxx_varphi_rs_t_val = np.gradient(np.gradient(varphi_rs_t_val, axis=0), axis=0) * (1/spacing) * (1/spacing)
    dyy_varphi_rs_t_val = np.gradient(np.gradient(varphi_rs_t_val, axis=1), axis=1) * (1/spacing) * (1/spacing)
 
    # also want elementwise equal, sum_n phi_n_t gn^2 exp(i gn r) = 4 pi eps sum_n ( 2 gnx gny unx + gnx^2 uny - gny^2 uny) exp(i gn r)
    # expect phi_n_t = 4 pi eps ( 2 gnx gny unx + (gnx^2 - gny^2) uny) / ( gnx^2 + gny^2)
    # note uny = u_t[n,1,:]
    for n in range(len(g)):
        gnx, gny = g[n][0], g[n][1]
        u_nx_re, u_ny_re = u_t[n,0,0], u_t[n,1,0]
        u_nx_im, u_ny_im = u_t[n,0,1], u_t[n,1,1]
        a = (2 * gnx * gny) * 4 * np.pi * eps
        b = (gnx*gnx - gny*gny) * 4 * np.pi * eps
        c = (gnx*gnx + gny*gny)
        expect_re = (a*u_nx_re + b*u_ny_re)/c
        expect_im = (a*u_nx_re + b*u_ny_re)/c
        # expect phi_n_t = ( a unx + b uny) / c
        print(expect_re, expect_im)

        phi_n_t_value = varphi_t(g, n, 0.65, alphaval, alphaval, eps, -eps, u_t, u_b)
        phi_n_t_value2 = varphi_t(g, n, 0.65, 0, 0, eps, -eps, u_t, u_b)

        print(phi_n_t_value)
        print(phi_n_t_value2)    

    # check that dzz phi = -4pi (rho_piezo) when (dxx+dyy)phi=0 so rho_ind = 0
    assert_eq_matrix(dzz_varphi_rs_t_val, -4*np.pi*rho_t[:,:,0])

    if False:
        #same thing for b instead of t, unnecssisary since same pretty much... try again with different alpha to be sure
        u_t, u_b, eps = u_b, u_t, -eps
        rho_t = compute_rho_tot_t(x, y, g, 0.65, alphaval, alphaval, eps, -eps, u_t, u_b)
        dzz_varphi_rs_t_val = delzz_varphi_rs_t(x, y, g, 0.65, alphaval, alphaval, eps, -eps, u_t, u_b)
        varphi_rs_t_val = varphi_rs_t(x, y, g, 0.65, alphaval, alphaval, eps, -eps, u_t, u_b)
        # dzz_varphi_rs_t_val + dxx_varphi_rs_t_val + dyy_varphi_rs_t_val = -4pi * rho_t
        dzz_varphi_rs_t_val, varphi_rs_t_val = dzz_varphi_rs_t_val[:,:,0], varphi_rs_t_val[:,:,0] # grab out re part
        dxx_varphi_rs_t_val = np.gradient(np.gradient(varphi_rs_t_val, axis=0), axis=0) * (1/spacing) * (1/spacing)
        dyy_varphi_rs_t_val = np.gradient(np.gradient(varphi_rs_t_val, axis=1), axis=1) * (1/spacing) * (1/spacing)
        RHS = -4*np.pi*rho_t[:,:,0] # grab out re part
        LHS = dzz_varphi_rs_t_val + dxx_varphi_rs_t_val + dyy_varphi_rs_t_val 
        f, ax = plt.subplots(1,3)
        ax[0].imshow(RHS, origin='lower')
        ax[1].imshow(LHS, origin='lower')
        ax[2].imshow(RHS-LHS, origin='lower')
        print(RHS[0,7])
        print(LHS[0,7])
        plt.show()

def main():

    # constants 
    a_0 = 1
    d_0 = 0.65                       # interlayer distance minimum (expanded about) in nm for MoS2 - average sep over all stacking configs present
    d   = d_0                        # interlayer distance, nm. taking as minumum above. 
    z_b, z_t = -d/2, d/2             # set 0 halfway between the two layers.
    e_11_t  = 2.9                    # in C-1 m, for MoS2
    e_11_b  = -e_11_t                # for AP, e_b = -e_t
    alpha_t = alpha(d_0, eps_p=16.3) # in-plane 2d polarizability for MoS2
    alpha_b = alpha_t                # same in-plane 2d polarizability (homobilayer)
    g = generate_fake_g(a_0=a_0, npts=20, visualize=False)
    
    #sanitycheck1()
    sanitycheck2()
    sanitycheck3()

    x, y, u_rs_x_t, u_rs_y_t, u_rs_x_b, u_rs_y_b = generate_fake_u_rs(spacing=0.1, visualize=False, a0=1)
    u_t, u_b = rs_to_ms_all_u(g, x, y, u_rs_x_t, u_rs_y_t, u_rs_x_b, u_rs_y_b)
    rho = compute_rho_tot(x, y, g, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b)

    f, ax = plt.subplots(2,2)
    ax[0,0].imshow(rho[:,:,0])
    x, y, z = x[:,:].flatten(), y[:,:].flatten(), rho[:,:,0].flatten()
    im = ax[0,1].scatter(x,y,c=z,s=120,marker='h',edgecolors='k',cmap='seismic')
    ax[1,0].quiver(u_rs_x_t[:,:,0], u_rs_y_t[:,:,0])
    ax[1,1].quiver(u_rs_x_b[:,:,0], u_rs_y_b[:,:,0])
    plt.show()

def compute_rho_tot_t(x, y, g, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b):
    rho = np.zeros((x.shape[0], x.shape[1], 2))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r = np.array([x[i,j], y[i,j]])
            re_rho_t, im_rho_t = rho_t_tot(r, g, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b)
            rho[i,j,0], rho[i,j,1] = re_rho_t , im_rho_t 
    return rho

def rs_to_ms_u(g, x, y, u_rs):
    n_gvecs = len(g)
    u_ms = np.zeros((n_gvecs, 2)) # gvec, re or im
    N = x.shape[0]
    N_r = N * N
    for n in range(n_gvecs):
        for i in range(N):
            for j in range(N): 
                gdotr = g[n][0] * x[i,j] + g[n][1] * y[i,j]
                u_ms[n,0] += u_rs[i,j,0] * np.cos(gdotr) + u_rs[i,j,1] * np.sin(gdotr) 
                u_ms[n,1] += u_rs[i,j,1] * np.cos(gdotr) - u_rs[i,j,0] * np.sin(gdotr) 
    return u_ms 

def grads_from_ms(g, x, y, u_ms):
    n_gvecs = len(g)
    Nx, Ny = x.shape[0], x.shape[1]
    dudx = np.zeros((Nx, Ny, 2))  # x or y pixel, re or im
    dudy = np.zeros((Nx, Ny, 2))  # x or y pixel, re or im
    Ng = len(g)
    for i in range(Nx):
        for j in range(Ny):
            for n in range(Ng):
                gdotr = g[n][0] * x[i,j] + g[n][1] * y[i,j]
                # multiplying by i*gnx
                dudx[i,j,0] += - g[n][0] * u_ms[n,1] * np.cos(gdotr) - g[n][0] * u_ms[n,0] * np.sin(gdotr)
                dudx[i,j,1] +=   g[n][0] * u_ms[n,0] * np.cos(gdotr) - g[n][0] * u_ms[n,1] * np.sin(gdotr)
                dudy[i,j,0] += - g[n][1] * u_ms[n,1] * np.cos(gdotr) - g[n][1] * u_ms[n,0] * np.sin(gdotr)
                dudy[i,j,1] +=   g[n][1] * u_ms[n,0] * np.cos(gdotr) - g[n][1] * u_ms[n,1] * np.sin(gdotr)
    return dudx/Ng, dudy/Ng

def ms_to_rs_u(g, x, y, u_ms):
    Nx, Ny = x.shape[0], x.shape[1]
    u_rs = np.zeros((Nx, Ny, 2))  # x or y pixel, re or im
    Ng = len(g)
    for i in range(Nx):
        for j in range(Ny):
            for n in range(Ng):
                gdotr = g[n][0] * x[i,j] + g[n][1] * y[i,j]
                u_rs[i,j,0] += u_ms[n,0] * np.cos(gdotr) - u_ms[n,1] * np.sin(gdotr)
                u_rs[i,j,1] += u_ms[n,0] * np.sin(gdotr) + u_ms[n,1] * np.cos(gdotr)
    return u_rs/Ng

def generate_fake_u_rs(spacing, visualize=True, a0=1):
    # fake twisting displacement field
    # u_top = - u_bottom = u_interlayer / 2 for homobilayers

    x, y = get_tiling_rs_unitcell(spacing, visualize=False)
    Nx, Ny = x.shape[0], x.shape[1]
    u_rs_x_t = np.zeros((Nx, Ny, 2))
    u_rs_y_t = np.zeros((Nx, Ny, 2))
    a,b,c = 0.2, 0.3, 0.4
    
    for i in range(Nx):
        for j in range(Ny): 
            xval, yval = x[i,j], y[i,j]
            u_rs_x_t[i,j,0], u_rs_y_t[i,j,0] = a*xval*yval, b*0.5*xval*xval + c*0.5*yval*yval
   
    u_rs_x_b, u_rs_y_b= -1*u_rs_x_t, -1*u_rs_y_t

    if visualize: 
        f, ax = plt.subplots()
        ax.quiver(x, y, u_rs_x_t[:,:,0], u_rs_y_t[:,:,0])
        plt.show()

    return x, y, u_rs_x_t, u_rs_y_t, u_rs_x_b, u_rs_y_b

def varphi_rs_t(x, y, g, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b):
    varphi_t_ms = np.zeros((len(g), 2))
    for n in range(len(g)):
        varphi_t_ms[n,:] = varphi_t(g, n, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b)
    return ms_to_rs_u(g, x, y, varphi_t_ms)

def delzz_varphi_rs_t(x, y, g, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b):
    # 2 orders of magnitude off here
    varphi_t_ms = np.zeros((len(g), 2))
    for n in range(len(g)):
        val = varphi_t(g, n, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b)
        varphi_t_ms[n,0] = np.dot(g[n],g[n]) * val[0]
        varphi_t_ms[n,1] = np.dot(g[n],g[n]) * val[1]
    return ms_to_rs_u(g, x, y, varphi_t_ms)    

# equation 11 in SI for top layer
def varphi_t(g, n, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b):
    g_n = np.dot(g[n],g[n])**0.5
    a = 4 * np.pi * np.sinh(g_n * d) # unitless
    b_t = ( np.exp(g_n * d)  + g_n * alpha_t * a)
    b_b = ( np.exp(g_n * d)  + g_n * alpha_b * a)
    re_rho_t, im_rho_t = rho(g, n, u_t, e_11_t)
    re_rho_b, im_rho_b = rho(g, n, u_b, e_11_b)
    re_num_t = a * (re_rho_b + re_rho_b * b_t)
    im_num_t = a * (im_rho_b + im_rho_b * b_t)
    denom = g_n * (b_t * b_b - 1)
    return re_num_t / denom, im_num_t / denom

# equation 11 in SI for bottom layer
def varphi_b(g, n, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b):
    return varphi_t(g, n, d, alpha_b, alpha_t, e_11_b, e_11_t, u_b, u_t)

# equation 12 in SI for either layer - FT of rho_induced for a given layer
def rho(g, n, u, e_11):
    # u_t[n, x or y, re or im]
    gnx, gny = g[n][0], g[n][1]
    u_nx_re, u_ny_re = u[n,0,0], u[n,1,0]
    u_nx_im, u_ny_im = u[n,0,1], u[n,1,1]
    a = 2 * gnx * gny 
    b = gnx*gnx - gny*gny
    rho_t_re = e_11 * (a*u_nx_re + b*u_ny_re)
    rho_t_im = e_11 * (a*u_nx_im + b*u_ny_im)
    return rho_t_re, rho_t_im

# inline equation after eq12 in SI for top layer
def rho_t_tot(r, g, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b):
    Re_val = 0
    Im_val = 0
    for n in range(len(g)):
        gdotr = g[n][0] * r[0] + g[n][1] * r[1]
        g_n = np.dot(g[n],g[n])**0.5
        re_varphi_t, im_varphi_t = varphi_t(g, n, d, alpha_t, alpha_b, e_11_t, e_11_b, u_t, u_b)
        re_a = 4 * np.pi * g_n * g_n * alpha_t * re_varphi_t
        im_a = 4 * np.pi * g_n * g_n * alpha_t * im_varphi_t
        re_rho, im_rho = rho(g, n, u_t, e_11_t)
        Re_val += (re_rho - re_a) * np.cos( gdotr )
        Re_val += -1 * (im_rho - im_a) * np.sin( gdotr )
        Im_val += (re_rho - re_a) * np.sin( gdotr )
        Im_val += (im_rho - im_a) * np.cos( gdotr )  
    return Re_val, Im_val

def urs_top_x_rep_from_theory(x, y):
    return 6.25e24 * (-0.0000282235 * np.cos(0.0346396 * x - 0.179993 * y ) - 9.67386e-6 * np.cos(0.0115465 * x - 0.179993 * y ) - 8.13535e-7 * np.cos(0.0923723 * x - 0.159994 * y ) - 0.0000105172 * np.cos(0.0230931 * x - 0.159994 * y ) + 9.28058e-6 * np.cos(0.0461862 * x - 0.159994 * y ) + 0.0000200406 * np.cos(0.0692793 * x - 0.159994 * y ) + 0.0000256219 * np.cos(0.0577327 * x - 0.139994 * y ) - 7.92822e-8 * np.cos(0.0808258 * x - 0.139994 * y ) - 0.0000182285 * np.cos(0.103919 * x - 0.139994 * y ) - 0.0000156465 * np.cos(0.0115465 * x - 0.139994 * y ) + 0.0000163231 * np.cos(0.0346396 * x - 0.139994 * y ) + 0.0000193024 * np.cos(0.138559 * x - 0.119995 * y ) - 6.2663e-6 * np.cos(0.0692793 * x - 0.119995 * y ) - 0.0000270859 * np.cos(0.0923723 * x - 0.119995 * y ) - 0.0000116686 * np.cos(0.115465 * x - 0.119995 * y ) + 1.22552e-6 * np.cos(0.0230931 * x - 0.119995 * y ) + 0.0000213087 * np.cos(0.0461862 * x - 0.119995 * y ) - 6.20911e-7 * np.cos(0.0577327 * x - 0.099996 * y ) + 0.0000158419 * np.cos(0.127012 * x - 0.099996 * y ) + 8.77685e-6 * np.cos(0.150105 * x - 0.099996 * y ) + 4.31318e-6 * np.cos(0.0115465 * x - 0.099996 * y ) + 0.0000380148 * np.cos(0.0346396 * x - 0.099996 * y ) - 0.0000200167 * np.cos(0.0808258 * x - 0.099996 * y ) - 9.15251e-6 * np.cos(0.103919 * x - 0.099996 * y ) - 0.0000348491 * np.cos(0.0230931 * x - 0.0799968 * y ) - 0.0000110397 * np.cos(0.0461862 * x - 0.0799968 * y ) - 0.0000337806 * np.cos(0.0692793 * x - 0.0799968 * y ) + 0.0000143417 * np.cos(0.115465 * x - 0.0799968 * y ) + 5.40614e-7 * np.cos(0.138559 * x - 0.0799968 * y ) - 3.35613e-7 * np.cos(0.161652 * x - 0.0799968 * y ) - 7.30667e-6 * np.cos(0.0923723 * x - 0.0799968 * y ) + 8.12855e-6 * np.cos(0.173198 * x - 0.0599976 * y ) - 0.000015672 * np.cos(0.0808258 * x - 0.0599976 * y ) - 0.0000293692 * np.cos(0.0346396 * x - 0.0599976 * y ) + 0.0000175725 * np.cos(0.0577327 * x - 0.0599976 * y ) + 0.0000217848 * np.cos(0.103919 * x - 0.0599976 * y ) + 5.08751e-7 * np.cos(0.127012 * x - 0.0599976 * y ) - 6.12846e-6 * np.cos(0.150105 * x - 0.0599976 * y ) + 0.0000170476 * np.cos(0.0115465 * x - 0.0599976 * y ) + 0.0000128063 * np.cos(0.0923723 * x - 0.0399984 * y ) + 1.1585e-6 * np.cos(0.161652 * x - 0.0399984 * y ) + 0.0000513269 * np.cos(0. - 0.0399984 * y ) - 0.0000213422 * np.cos(0.0230931 * x - 0.0399984 * y ) - 4.68046e-6 * np.cos(0.0692793 * x - 0.0399984 * y ) - 0.000021645 * np.cos(0.0461862 * x - 0.0399984 * y ) + 5.62884e-6 * np.cos(0.115465 * x - 0.0399984 * y ) - 8.10622e-6 * np.cos(0.138559 * x - 0.0399984 * y ) + 0.0000182509 * np.cos(0.0808258 * x - 0.0199992 * y ) - 0.0000270003 * np.cos(0.0346396 * x - 0.0199992 * y ) - 3.81465e-6 * np.cos(0.103919 * x - 0.0199992 * y ) - 3.01692e-7 * np.cos(0.150105 * x - 0.0199992 * y ) - 3.78865e-6 * np.cos(0.173198 * x - 0.0199992 * y ) - 0.000021021 * np.cos(0.0115465 * x - 0.0199992 * y ) + 6.69117e-6 * np.cos(0.0577327 * x - 0.0199992 * y ) - 2.85823e-6 * np.cos(0.127012 * x - 0.0199992 * y ) - 1.42126e-6 * np.cos(0.184745 * x - 1.11022e-16 * y ) - 4.36699e-6 * np.cos(0.161652 * x - 9.71445e-17 * y ) + 5.73128e-6 * np.cos(0.138559 * x - 8.32667e-17 * y ) - 9.81921e-6 * np.cos(0.115465 * x - 6.93889e-17 * y ) + 5.06291e-6 * np.cos(0.0923723 * x - 5.55112e-17 * y ) + 0.0000300024 * np.cos(0.0692793 * x - 4.16334e-17 * y ) - 0.0000114088 * np.cos(0.0461862 * x - 2.77556e-17 * y ) - 0.0000258181 * np.cos(0.0230931 * x - 1.38778e-17 * y ) - 3.40342e-6 * np.cos(0.173198 * x + 0.0199992 * y ) - 1.49231e-8 * np.cos(0.150105 * x + 0.0199992 * y ) - 2.97923e-6 * np.cos(0.127012 * x + 0.0199992 * y ) - 3.40734e-6 * np.cos(0.103919 * x + 0.0199992 * y ) + 0.0000186391 * np.cos(0.0808258 * x + 0.0199992 * y ) + 1.33693e-6 * np.cos(0.0577327 * x + 0.0199992 * y ) - 0.0000243266 * np.cos(0.0346396 * x + 0.0199992 * y ) + 0.0000468391 * np.cos(0.0115465 * x + 0.0199992 * y ) + 1.28381e-6 * np.cos(0.161652 * x + 0.0399984 * y ) - 7.81833e-6 * np.cos(0.138559 * x + 0.0399984 * y ) + 0.0000132047 * np.cos(0.0923723 * x + 0.0399984 * y ) + 5.96639e-6 * np.cos(0.115465 * x + 0.0399984 * y ) - 7.12266e-6 * np.cos(0.0692793 * x + 0.0399984 * y ) - 0.0000237388 * np.cos(0.0461862 * x + 0.0399984 * y ) + 0.0000327509 * np.cos(0.0230931 * x + 0.0399984 * y ) + 9.87542e-6 * np.cos(0.173198 * x + 0.0599976 * y ) - 5.86263e-6 * np.cos(0.150105 * x + 0.0599976 * y ) + 1.12311e-6 * np.cos(0.127012 * x + 0.0599976 * y ) + 0.0000221118 * np.cos(0.103919 * x + 0.0599976 * y ) - 0.0000171195 * np.cos(0.0808258 * x + 0.0599976 * y ) + 0.000020308 * np.cos(0.0115465 * x + 0.0599976 * y ) - 6.3313e-7 * np.cos(0.0346396 * x + 0.0599976 * y ) + 0.0000165983 * np.cos(0.0577327 * x + 0.0599976 * y ) + 0.0000151377 * np.cos(0.115465 * x + 0.0799968 * y ) + 1.41496e-6 * np.cos(0.138559 * x + 0.0799968 * y ) + 1.04945e-6 * np.cos(0.161652 * x + 0.0799968 * y ) - 0.0000342001 * np.cos(0.0692793 * x + 0.0799968 * y ) - 6.85436e-6 * np.cos(0.0923723 * x + 0.0799968 * y ) + 0.0000118031 * np.cos(2.08167e-17 * x + 0.0799968 * y ) - 0.0000362116 * np.cos(0.0230931 * x + 0.0799968 * y ) + 5.9768e-6 * np.cos(0.0461862 * x + 0.0799968 * y ) + 0.0000100095 * np.cos(0.150105 * x + 0.099996 * y ) + 0.0000166457 * np.cos(0.127012 * x + 0.099996 * y ) - 8.2169e-6 * np.cos(0.103919 * x + 0.099996 * y ) - 0.0000184505 * np.cos(0.0808258 * x + 0.099996 * y ) + 2.4673e-6 * np.cos(0.0115465 * x + 0.099996 * y ) + 0.000037188 * np.cos(0.0346396 * x + 0.099996 * y ) + 0.0000104401 * np.cos(0.0577327 * x + 0.099996 * y ) + 0.000020095 * np.cos(0.138559 * x + 0.119995 * y ) - 0.0000104391 * np.cos(0.115465 * x + 0.119995 * y ) - 0.0000253203 * np.cos(0.0923723 * x + 0.119995 * y ) + 1.34028e-6 * np.cos(0.0230931 * x + 0.119995 * y ) + 0.000022996 * np.cos(0.0461862 * x + 0.119995 * y ) + 5.35016e-7 * np.cos(0.0692793 * x + 0.119995 * y ) - 0.0000438966 * np.cos(2.77556e-17 * x + 0.119995 * y ) - 0.0000162519 * np.cos(0.103919 * x + 0.139994 * y ) - 0.0000154648 * np.cos(0.0115465 * x + 0.139994 * y ) + 0.0000169708 * np.cos(0.0346396 * x + 0.139994 * y ) + 0.0000271008 * np.cos(0.0577327 * x + 0.139994 * y ) + 4.44627e-6 * np.cos(0.0808258 * x + 0.139994 * y ) - 1.95557e-6 * np.cos(4.16334e-17 * x + 0.159994 * y ) - 9.9793e-6 * np.cos(0.0230931 * x + 0.159994 * y ) + 0.0000103848 * np.cos(0.0461862 * x + 0.159994 * y ) + 0.0000216319 * np.cos(0.0692793 * x + 0.159994 * y ) + 2.2348e-6 * np.cos(0.0923723 * x + 0.159994 * y ) - 9.82631e-6 * np.cos(0.0115465 * x + 0.179993 * y ) - 0.0000291778 * np.cos(0.0346396 * x + 0.179993 * y ) - 0.0000645931 * np.cos(0.0346396 * x - 0.179993 * y ) - 0.0000887685 * np.cos(0.0115465 * x - 0.179993 * y ) + 0.000340008 * np.cos(0.0923723 * x - 0.159994 * y ) - 0.0000431591 * np.cos(0.0230931 * x - 0.159994 * y ) + 0.0000287417 * np.cos(0.0461862 * x - 0.159994 * y ) + 0.0000553735 * np.cos(0.0692793 * x - 0.159994 * y ) + 0.000156499 * np.cos(0.0577327 * x - 0.139994 * y ) + 0.000791894 * np.cos(0.0808258 * x - 0.139994 * y ) + 0.0000586398 * np.cos(0.103919 * x - 0.139994 * y ) + 0.0000168892 * np.cos(0.0115465 * x - 0.139994 * y ) + 0.00010627 * np.cos(0.0346396 * x - 0.139994 * y ) - 0.0000324721 * np.cos(0.138559 * x - 0.119995 * y ) + 0.00179172 * np.cos(0.0692793 * x - 0.119995 * y ) + 0.000158347 * np.cos(0.0923723 * x - 0.119995 * y ) + 0.0000434439 * np.cos(0.115465 * x - 0.119995 * y ) + 0.000292933 * np.cos(0.0230931 * x - 0.119995 * y ) + 0.000368179 * np.cos(0.0461862 * x - 0.119995 * y ) + 0.00396877 * np.cos(0.0577327 * x - 0.099996 * y ) - 0.0000125151 * np.cos(0.127012 * x - 0.099996 * y ) - 0.0000445286 * np.cos(0.150105 * x - 0.099996 * y ) + 0.000680708 * np.cos(0.0115465 * x - 0.099996 * y ) + 0.000762469 * np.cos(0.0346396 * x - 0.099996 * y ) + 0.000349856 * np.cos(0.0808258 * x - 0.099996 * y ) + 0.000113671 * np.cos(0.103919 * x - 0.099996 * y ) + 0.00116986 * np.cos(0.0230931 * x - 0.0799968 * y ) + 0.00874819 * np.cos(0.0461862 * x - 0.0799968 * y ) + 0.000675921 * np.cos(0.0692793 * x - 0.0799968 * y ) + 0.0000245917 * np.cos(0.115465 * x - 0.0799968 * y ) - 0.0000324911 * np.cos(0.138559 * x - 0.0799968 * y ) - 0.0000438212 * np.cos(0.161652 * x - 0.0799968 * y ) + 0.000246071 * np.cos(0.0923723 * x - 0.0799968 * y ) - 0.0000305109 * np.cos(0.173198 * x - 0.0599976 * y ) + 0.00046021 * np.cos(0.0808258 * x - 0.0599976 * y ) + 0.0194393 * np.cos(0.0346396 * x - 0.0599976 * y ) + 0.00144522 * np.cos(0.0577327 * x - 0.0599976 * y ) + 0.000104131 * np.cos(0.103919 * x - 0.0599976 * y ) - 7.26304e-6 * np.cos(0.127012 * x - 0.0599976 * y ) - 0.0000300059 * np.cos(0.150105 * x - 0.0599976 * y ) + 0.00250625 * np.cos(0.0115465 * x - 0.0599976 * y ) + 0.000221441 * np.cos(0.0923723 * x - 0.0399984 * y ) - 0.0000143911 * np.cos(0.161652 * x - 0.0399984 * y ) + 0.00342777 * np.cos(0. - 0.0399984 * y ) + 0.0445226 * np.cos(0.0230931 * x - 0.0399984 * y ) + 0.000721973 * np.cos(0.0692793 * x - 0.0399984 * y ) + 0.00172226 * np.cos(0.0461862 * x - 0.0399984 * y ) + 0.0000469058 * np.cos(0.115465 * x - 0.0399984 * y ) - 7.27204e-6 * np.cos(0.138559 * x - 0.0399984 * y ) - 0.000274953 * np.cos(0.0808258 * x - 0.0199992 * y ) + 0.00171564 * np.cos(0.0346396 * x - 0.0199992 * y ) + 0.0000867141 * np.cos(0.103919 * x - 0.0199992 * y ) - 1.67475e-6 * np.cos(0.150105 * x - 0.0199992 * y ) - 2.71933e-6 * np.cos(0.173198 * x - 0.0199992 * y ) + 0.118252 * np.cos(0.0115465 * x - 0.0199992 * y ) + 0.000782366 * np.cos(0.0577327 * x - 0.0199992 * y ) + 0.0000184919 * np.cos(0.127012 * x - 0.0199992 * y ) + 6.55296e-7 * np.cos(0.184745 * x - 1.11022e-16 * y ) + 4.63891e-7 * np.cos(0.161652 * x - 9.71445e-17 * y ) + 9.92188e-8 * np.cos(0.138559 * x - 8.32667e-17 * y ) + 1.44737e-7 * np.cos(0.115465 * x - 6.93889e-17 * y ) + 3.38063e-7 * np.cos(0.0923723 * x - 5.55112e-17 * y ) - 5.74087e-8 * np.cos(0.0692793 * x - 4.16334e-17 * y ) + 3.37911e-6 * np.cos(0.0461862 * x - 2.77556e-17 * y ) - 8.54991e-7 * np.cos(0.0230931 * x - 1.38778e-17 * y ) + 3.5484e-6 * np.cos(0.173198 * x + 0.0199992 * y ) + 2.1584e-6 * np.cos(0.150105 * x + 0.0199992 * y ) - 0.0000182661 * np.cos(0.127012 * x + 0.0199992 * y ) - 0.0000862325 * np.cos(0.103919 * x + 0.0199992 * y ) + 0.000275405 * np.cos(0.0808258 * x + 0.0199992 * y ) - 0.000779754 * np.cos(0.0577327 * x + 0.0199992 * y ) - 0.00171213 * np.cos(0.0346396 * x + 0.0199992 * y ) - 0.118253 * np.cos(0.0115465 * x + 0.0199992 * y ) + 0.0000150063 * np.cos(0.161652 * x + 0.0399984 * y ) + 7.77036e-6 * np.cos(0.138559 * x + 0.0399984 * y ) - 0.000221192 * np.cos(0.0923723 * x + 0.0399984 * y ) - 0.0000464436 * np.cos(0.115465 * x + 0.0399984 * y ) - 0.000720821 * np.cos(0.0692793 * x + 0.0399984 * y ) - 0.00172389 * np.cos(0.0461862 * x + 0.0399984 * y ) - 0.0445192 * np.cos(0.0230931 * x + 0.0399984 * y ) + 0.0000289401 * np.cos(0.173198 * x + 0.0599976 * y ) + 0.0000304808 * np.cos(0.150105 * x + 0.0599976 * y ) + 7.8392e-6 * np.cos(0.127012 * x + 0.0599976 * y ) - 0.000103809 * np.cos(0.103919 * x + 0.0599976 * y ) - 0.000459267 * np.cos(0.0808258 * x + 0.0599976 * y ) - 0.00250201 * np.cos(0.0115465 * x + 0.0599976 * y ) - 0.0194394 * np.cos(0.0346396 * x + 0.0599976 * y ) - 0.00144481 * np.cos(0.0577327 * x + 0.0599976 * y ) - 0.0000241522 * np.cos(0.115465 * x + 0.0799968 * y ) + 0.0000331289 * np.cos(0.138559 * x + 0.0799968 * y ) + 0.0000430134 * np.cos(0.161652 * x + 0.0799968 * y ) - 0.000675755 * np.cos(0.0692793 * x + 0.0799968 * y ) - 0.000246027 * np.cos(0.0923723 * x + 0.0799968 * y ) - 0.00144279 * np.cos(2.08167e-17 * x + 0.0799968 * y ) - 0.00116981 * np.cos(0.0230931 * x + 0.0799968 * y ) - 0.00874785 * np.cos(0.0461862 * x + 0.0799968 * y ) + 0.0000449473 * np.cos(0.150105 * x + 0.099996 * y ) + 0.0000131533 * np.cos(0.127012 * x + 0.099996 * y ) - 0.000113542 * np.cos(0.103919 * x + 0.099996 * y ) - 0.000349687 * np.cos(0.0808258 * x + 0.099996 * y ) - 0.000681403 * np.cos(0.0115465 * x + 0.099996 * y ) - 0.000762154 * np.cos(0.0346396 * x + 0.099996 * y ) - 0.00396862 * np.cos(0.0577327 * x + 0.099996 * y ) + 0.0000340822 * np.cos(0.138559 * x + 0.119995 * y ) - 0.0000431328 * np.cos(0.115465 * x + 0.119995 * y ) - 0.000158174 * np.cos(0.0923723 * x + 0.119995 * y ) - 0.000292515 * np.cos(0.0230931 * x + 0.119995 * y ) - 0.000368122 * np.cos(0.0461862 * x + 0.119995 * y ) - 0.00179162 * np.cos(0.0692793 * x + 0.119995 * y ) - 0.00020794 * np.cos(2.77556e-17 * x + 0.119995 * y ) - 0.0000580928 * np.cos(0.103919 * x + 0.139994 * y ) - 0.0000167525 * np.cos(0.0115465 * x + 0.139994 * y ) - 0.000105901 * np.cos(0.0346396 * x + 0.139994 * y ) - 0.000156188 * np.cos(0.0577327 * x + 0.139994 * y ) - 0.00079143 * np.cos(0.0808258 * x + 0.139994 * y ) + 0.00006562 * np.cos(4.16334e-17 * x + 0.159994 * y ) + 0.0000429959 * np.cos(0.0230931 * x + 0.159994 * y ) - 0.0000284376 * np.cos(0.0461862 * x + 0.159994 * y ) - 0.0000550914 * np.cos(0.0692793 * x + 0.159994 * y ) - 0.000339353 * np.cos(0.0923723 * x + 0.159994 * y ) + 0.000087542 * np.cos(0.0115465 * x + 0.179993 * y ) + 0.0000614122 * np.cos(0.0346396 * x + 0.179993 * y ))

def expu():
    u = np.zeros((30,30,2))
    u[0,0, :] = [-0.00276168,   0.54649261] 
    u[0,1, :] = [0.00338417 , 0.54128673] 
    u[0,2, :] = [0.44526793 , 0.28355622] 
    u[0,3, :] = [0.3916809  , 0.27011859] 
    u[0,4, :] = [0.34472951 , 0.23106405] 
    u[0,5, :] = [0.27984241 , 0.20605413] 
    u[0,6, :] = [0.23147835 , 0.1861648 ] 
    u[0,7, :] = [0.18087728 , 0.16861051] 
    u[0,8, :] = [0.11585181 , 0.13195933] 
    u[0,9, :] = [0.13758319 , 0.00725982] 
    u[0,10, :] = [-0.14026106,  -0.15264873] 
    u[0,11, :] = [-0.18764607,  -0.20993243] 
    u[0,12, :] = [-0.29196102,  -0.25269613] 
    u[0,13, :] = [-0.34687767,  -0.29228958] 
    u[0,14, :] = [0.10156268 , 0.51814996] 
    u[0,15, :] = [0.06064856 , 0.49887852] 
    u[0,16, :] = [0.03489905 , 0.49453625] 
    u[0,17, :] = [-0.00663491,   0.47377511] 
    u[0,18, :] = [-0.05336659,   0.45129635] 
    u[0,19, :] = [-0.09832781,   0.4330127 ] 
    u[0,20, :] = [-0.15195224,   0.41243414] 
    u[0,21, :] = [-0.25674888,   0.34885458] 
    u[0,22, :] = [-0.28021506,   0.30082194] 
    u[0,23, :] = [-0.31844746,   0.25141768] 
    u[0,24, :] = [-0.38286917,   0.19018645] 
    u[0,25, :] = [-0.42406405,   0.13152492] 
    u[0,26, :] = [-0.45447017,   0.07885997] 
    u[0,27, :] = [-0.47314925,   0.04650686] 
    u[0,28, :] = [ 0.48060373,  -0.00486242] 
    u[0,29, :] = [ 0.42659998,  -0.0865423 ] 
    u[1,0, :] = [0.01204998 , 0.56015718] 
    u[1,1, :] = [0.04125505 , 0.54485932] 
    u[1,2, :] = [0.40770592 , 0.29555641] 
    u[1,3, :] = [0.35276742 , 0.27098397] 
    u[1,4, :] = [0.28987582 , 0.25069523] 
    u[1,5, :] = [0.20946126 , 0.23684109] 
    u[1,6, :] = [0.15572448 , 0.22170614] 
    u[1,7, :] = [0.10717443 , 0.17669641] 
    u[1,8, :] = [0.05228931 , 0.13713823] 
    u[1,9, :] = [0.12622252 , 0.11503501] 
    u[1,10, :] = [-0.21669391,  -0.06043591] 
    u[1,11, :] = [-0.2122947 , -0.2176193] 
    u[1,12, :] = [-0.34468319,  -0.22088991] 
    u[1,13, :] = [-0.43113561,  -0.23926806] 
    u[1,14, :] = [0.04409794 , 0.51400864] 
    u[1,15, :] = [0.03273769 , 0.49296516] 
    u[1,16, :] = [-1.60983526e-04,  4.86909805e-01] 
    u[1,17, :] = [-0.02815518,   0.46913994] 
    u[1,18, :] = [-0.06246107,   0.46220407] 
    u[1,19, :] = [-0.12554365,   0.43623157] 
    u[1,20, :] = [-0.18696563,   0.38030376] 
    u[1,21, :] = [-0.23931642,   0.35099698] 
    u[1,22, :] = [-0.29375215,   0.2891848 ] 
    u[1,23, :] = [-0.34932879,   0.25666913] 
    u[1,24, :] = [-0.38605299,   0.19736201] 
    u[1,25, :] = [-0.42118769,   0.13650693] 
    u[1,26, :] = [-0.44608463,   0.09338416] 
    u[1,27, :] = [-0.47666779,   0.04041258] 
    u[1,28, :] = [0.46973602 , 0.00419339] 
    u[1,29, :] = [ 0.40354728,  -0.05971623] 
    u[2,0, :] = [-0.07949673,  -0.52111966] 
    u[2,1, :] = [0.38086554 , 0.32213188] 
    u[2,2, :] = [0.32472156 , 0.32027965] 
    u[2,3, :] = [0.27882193 , 0.335272  ] 
    u[2,4, :] = [0.21015594 , 0.31274804] 
    u[2,5, :] = [0.15220043 , 0.29485263] 
    u[2,6, :] = [0.10788615 , 0.26749453] 
    u[2,7, :] = [0.0827898  , 0.24209092] 
    u[2,8, :] = [-0.124001  ,   0.18507336] 
    u[2,9, :] = [-0.24852773,   0.00731461] 
    u[2,10, :] = [-0.28704701,  -0.01787204] 
    u[2,11, :] = [-0.33648704,  -0.04259035] 
    u[2,12, :] = [-0.40885938,  -0.11075763] 
    u[2,13, :] = [-0.44883514,  -0.17161268] 
    u[2,14, :] = [-0.46896406,  -0.23359579] 
    u[2,15, :] = [0.01235617 , 0.55211663] 
    u[2,16, :] = [0.00573788 , 0.52581613] 
    u[2,17, :] = [-0.05200239,   0.51277307] 
    u[2,18, :] = [-0.0957987 ,   0.48217275] 
    u[2,19, :] = [-0.14490144,   0.4330127 ] 
    u[2,20, :] = [-0.19314633,   0.37869528] 
    u[2,21, :] = [-0.26141648,   0.33268236] 
    u[2,22, :] = [-0.30970805,   0.27105931] 
    u[2,23, :] = [-0.36350676,   0.23641322] 
    u[2,24, :] = [-0.44354322,   0.19201536] 
    u[2,25, :] = [-0.48071766,   0.1666836 ] 
    u[2,26, :] = [-0.49373208,   0.13987872] 
    u[2,27, :] = [0.4913639  , 0.09726186] 
    u[2,28, :] = [0.46579501 , 0.03428235] 
    u[2,29, :] = [ 0.38953039,  -0.0411552 ] 
    u[3,0, :] = [0.33363217 , 0.35704308] 
    u[3,1, :] = [0.29625728 , 0.35083946] 
    u[3,2, :] = [0.26114706 , 0.39150374] 
    u[3,3, :] = [0.21508243 , 0.37387228] 
    u[3,4, :] = [0.17237628 , 0.34418871] 
    u[3,5, :] = [0.11917029 , 0.31461046] 
    u[3,6, :] = [0.08202929 , 0.29040267] 
    u[3,7, :] = [0.10004784 , 0.28028062] 
    u[3,8, :] = [-0.15550098,   0.22445509] 
    u[3,9, :] = [-0.19907125,   0.18825411] 
    u[3,10, :] = [-0.31765695,   0.04219387] 
    u[3,11, :] = [-0.37507407,  -0.01764145] 
    u[3,12, :] = [-0.41235348,  -0.10449575] 
    u[3,13, :] = [-0.47625155,  -0.16962455] 
    u[3,14, :] = [ 0.49376513,  -0.20352994] 
    u[3,15, :] = [ 0.48678921,  -0.26615038] 
    u[3,16, :] = [ 0.46998754,  -0.30535347] 
    u[3,17, :] = [-0.07973744,   0.52461673] 
    u[3,18, :] = [-0.10934471,   0.48550884] 
    u[3,19, :] = [-0.15068055,   0.4330127 ] 
    u[3,20, :] = [-0.23498514,   0.36445761] 
    u[3,21, :] = [-0.27815185,   0.34809019] 
    u[3,22, :] = [-0.31905238,   0.31341048] 
    u[3,23, :] = [-0.35092928,   0.25819807] 
    u[3,24, :] = [-0.43245455,   0.20726076] 
    u[3,25, :] = [0.49125079 , 0.17498678] 
    u[3,26, :] = [0.48678031 , 0.1579079 ] 
    u[3,27, :] = [0.46416641 , 0.12323244] 
    u[3,28, :] = [0.40794439 , 0.06301692] 
    u[3,29, :] = [0.36125174 , 0.00045398] 
    u[4,0, :] = [0.29277407 , 0.39750902] 
    u[4,1, :] = [0.2474012 , 0.4195723] 
    u[4,2, :] = [0.21765526 , 0.43175456] 
    u[4,3, :] = [0.15038797 , 0.39863545] 
    u[4,4, :] = [0.08856407 , 0.37995421] 
    u[4,5, :] = [0.05360551 , 0.33551411] 
    u[4,6, :] = [-0.06539511,   0.31435913] 
    u[4,7, :] = [-0.11666453,   0.2755576 ] 
    u[4,8, :] = [-0.19431033,   0.23313754] 
    u[4,9, :] = [-0.3050771 ,   0.12056503] 
    u[4,10, :] = [-0.34430831,   0.04202334] 
    u[4,11, :] = [-0.37876355,   0.00276247] 
    u[4,12, :] = [-0.41817284,  -0.06706313] 
    u[4,13, :] = [-0.48394094,  -0.11042956] 
    u[4,14, :] = [ 0.49827159,  -0.15694395] 
    u[4,15, :] = [ 0.48923254,  -0.21015902] 
    u[4,16, :] = [ 0.44946369,  -0.25048136] 
    u[4,17, :] = [ 0.40766396,  -0.30568938] 
    u[4,18, :] = [-0.11092468,   0.49902169] 
    u[4,19, :] = [-0.15481593,   0.46386089] 
    u[4,20, :] = [-0.19951342,   0.43303039] 
    u[4,21, :] = [-0.25      ,  0.4330127] 
    u[4,22, :] = [-0.32986607,   0.34428901] 
    u[4,23, :] = [-0.39165062,   0.27076106] 
    u[4,24, :] = [-0.42103889,   0.2348579 ] 
    u[4,25, :] = [-0.45872119,   0.2097965 ] 
    u[4,26, :] = [-0.49747263,   0.17570403] 
    u[4,27, :] = [0.46377822 , 0.14121379] 
    u[4,28, :] = [0.4350215  , 0.09591936] 
    u[4,29, :] = [0.37634495 , 0.04087645] 
    u[5,0, :] = [-0.16988114,  -0.46863257] 
    u[5,1, :] = [-0.21106828,  -0.4330127 ] 
    u[5,2, :] = [0.2003357 , .4230651] 
    u[5,3, :] = [0.1626024 , .4330127] 
    u[5,4, :] = [0.13634875 , 0.38096451] 
    u[5,5, :] = [0.10160654 , 0.33837874] 
    u[5,6, :] = [-0.05409124,   0.32360454] 
    u[5,7, :] = [-0.12681751,   0.29748715] 
    u[5,8, :] = [-0.19311723,   0.26290042] 
    u[5,9, :] = [-0.21395633,   0.22952411] 
    u[5,10, :] = [-0.3216344 ,   0.07483201] 
    u[5,11, :] = [-0.38280574,   0.02263939] 
    u[5,12, :] = [-0.43271803,  -0.02654194] 
    u[5,13, :] = [-0.44914577,  -0.0729521 ] 
    u[5,14, :] = [-0.48940741,  -0.15015755] 
    u[5,15, :] = [-0.49772221,  -0.18838469] 
    u[5,16, :] = [ 0.47750099,  -0.23515352] 
    u[5,17, :] = [-0.10132106,   0.48308835] 
    u[5,18, :] = [-0.14845377,   0.44826114] 
    u[5,19, :] = [-0.19752186,   0.43301271] 
    u[5,20, :] = [-0.23554097,   0.4330127 ] 
    u[5,21, :] = [-0.1904129 ,  0.4330127] 
    u[5,22, :] = [-0.35229885,   0.30589635] 
    u[5,23, :] = [-0.41805798,   0.24777785] 
    u[5,24, :] = [-0.4577706 ,   0.24266397] 
    u[5,25, :] = [-0.48805735,   0.22069308] 
    u[5,26, :] = [0.48577116 , 0.1947892 ] 
    u[5,27, :] = [0.44841273 , 0.17996198] 
    u[5,28, :] = [0.39184784 , 0.16221499] 
    u[5,29, :] = [0.32447241 , 0.05926318] 
    u[6,0, :] = [0.29216572 , 0.39796333] 
    u[6,1, :] = [0.26352538 , 0.41035368] 
    u[6,2, :] = [0.17129315 , 0.43050266] 
    u[6,3, :] = [0.13667592 , 0.39480665] 
    u[6,4, :] = [0.08064771 , 0.39599857] 
    u[6,5, :] = [-0.00152524,   0.38946432] 
    u[6,6, :] = [-0.07754219,   0.36787846] 
    u[6,7, :] = [-0.1333203 ,  0.3258377] 
    u[6,8, :] = [-0.2154897 ,   0.27100607] 
    u[6,9, :] = [-0.27942837,   0.23337808] 
    u[6,10, :] = [-0.34434853,   0.15551979] 
    u[6,11, :] = [-0.39434788,   0.07624361] 
    u[6,12, :] = [-0.41848328,   0.02262952] 
    u[6,13, :] = [-0.42924957,  -0.01414782] 
    u[6,14, :] = [-0.44184944,  -0.07417694] 
    u[6,15, :] = [ 0.49712581,  -0.14815199] 
    u[6,16, :] = [ 0.47546013,  -0.20353834] 
    u[6,17, :] = [ 0.40670592,  -0.31835892] 
    u[6,18, :] = [-0.13032571,   0.48960441] 
    u[6,19, :] = [-0.17957599,   0.46149313] 
    u[6,20, :] = [-0.27023456,   0.39796542] 
    u[6,21, :] = [-0.30507086,   0.35446166] 
    u[6,22, :] = [-0.36372132,   0.31087633] 
    u[6,23, :] = [-0.40611807,   0.28290191] 
    u[6,24, :] = [-0.44669522,   0.26433667] 
    u[6,25, :] = [-0.44354287,   0.2577876 ] 
    u[6,26, :] = [0.05727954 , 0.54059947] 
    u[6,27, :] = [-0.42689198,  -0.24969167] 
    u[6,28, :] = [0.39601223 , 0.18588793] 
    u[6,29, :] = [0.30070933 , 0.09033337] 
    u[7,0, :] = [0.18562993 , 0.4330127 ] 
    u[7,1, :] = [0.15521318 , 0.43299649] 
    u[7,2, :] = [0.11181712 , 0.4330127 ] 
    u[7,3, :] = [0.07052003 , 0.4330127 ] 
    u[7,4, :] = [0.01336854 , 0.42945043] 
    u[7,5, :] = [-0.03829084,   0.40256102] 
    u[7,6, :] = [-0.12283504,   0.34712883] 
    u[7,7, :] = [-0.16149213,   0.33505952] 
    u[7,8, :] = [-0.24076777,   0.28404143] 
    u[7,9, :] = [-0.28461477,   0.24747246] 
    u[7,10, :] = [-0.34061369,   0.1957762 ] 
    u[7,11, :] = [-0.3931885 ,   0.13197725] 
    u[7,12, :] = [-0.42896511,   0.05791463] 
    u[7,13, :] = [-0.43877497,   0.00370899] 
    u[7,14, :] = [-0.41755985,  -0.01319182] 
    u[7,15, :] = [ 0.43995465,  -0.10400159] 
    u[7,16, :] = [ 0.44393153,  -0.18572026] 
    u[7,17, :] = [ 0.38678927,  -0.2682068 ] 
    u[7,18, :] = [ 0.34110902,  -0.32221344] 
    u[7,19, :] = [ 0.28419596,  -0.38314196] 
    u[7,20, :] = [ 0.22226329,  -0.44454081] 
    u[7,21, :] = [-0.34639113,   0.37091974] 
    u[7,22, :] = [-0.40787626,   0.3336963 ] 
    u[7,23, :] = [-0.44629658,   0.30801145] 
    u[7,24, :] = [-0.47847641,   0.29739135] 
    u[7,25, :] = [0.46641756 , 0.29842827] 
    u[7,26, :] = [-0.42130597,  -0.32340952] 
    u[7,27, :] = [-0.42245778,  -0.27346182] 
    u[7,28, :] = [0.36815904 , 0.25348594] 
    u[7,29, :] = [0.29927954 , 0.25345247] 
    u[8,0, :] = [-0.35317874,  -0.36828815] 
    u[8,1, :] = [0.11084929 , 0.46449273] 
    u[8,2, :] = [0.07410062 , 0.45560413] 
    u[8,3, :] = [0.01351637 , 0.45057607] 
    u[8,4, :] = [-0.03153481,   0.4397318 ] 
    u[8,5, :] = [-0.072441  ,  0.4330127] 
    u[8,6, :] = [-0.13747199,   0.37614059] 
    u[8,7, :] = [-0.20546883,   0.34344519] 
    u[8,8, :] = [-0.25559106,   0.29195546] 
    u[8,9, :] = [-0.31316044,   0.25872954] 
    u[8,10, :] = [-0.36946097,   0.22610024] 
    u[8,11, :] = [-0.41712251,   0.14354802] 
    u[8,12, :] = [-0.44720686,   0.0914404 ] 
    u[8,13, :] = [-0.47470059,   0.04381987] 
    u[8,14, :] = [ 0.47934366,  -0.03577782] 
    u[8,15, :] = [ 0.45169979,  -0.08363318] 
    u[8,16, :] = [ 0.4217219 ,  -0.13547315] 
    u[8,17, :] = [ 0.36448048,  -0.21756182] 
    u[8,18, :] = [ 0.30714509,  -0.32456649] 
    u[8,19, :] = [ 0.24622925,  -0.36979302] 
    u[8,20, :] = [ 0.18447579,  -0.41530908] 
    u[8,21, :] = [ 0.09916396,  -0.4330127 ] 
    u[8,22, :] = [ 0.04665448,  -0.51763756] 
    u[8,23, :] = [-0.01697834,  -0.5198899 ] 
    u[8,24, :] = [0.08732533 , 0.51851853] 
    u[8,25, :] = [-0.37759805,  -0.34242947] 
    u[8,26, :] = [0.37905434 , 0.32720951] 
    u[8,27, :] = [0.36775307 , 0.34199671] 
    u[8,28, :] = [0.34961864 , 0.30580078] 
    u[8,29, :] = [-0.10070631,  -0.43956514] 
    u[9,0, :] = [-0.42463906,  -0.29359722] 
    u[9,1, :] = [0.02836631 , 0.5377767 ] 
    u[9,2, :] = [0.0290114  , 0.51954991] 
    u[9,3, :] = [-0.00339747,   0.49631447] 
    u[9,4, :] = [-0.05552448,   0.4915299 ] 
    u[9,5, :] = [-0.1088546 ,   0.45135262] 
    u[9,6, :] = [-0.16171203,   0.42320533] 
    u[9,7, :] = [-0.20616523,   0.38758182] 
    u[9,8, :] = [-0.26670081,   0.3291062 ] 
    u[9,9, :] = [-0.32713226,   0.27230763] 
    u[9,10, :] = [-0.38129799,   0.21236413] 
    u[9,11, :] = [-0.42431619,   0.17198285] 
    u[9,12, :] = [-0.44216908,   0.12060581] 
    u[9,13, :] = [-0.46695408,   0.07340638] 
    u[9,14, :] = [0.47689713 , 0.00311102] 
    u[9,15, :] = [ 0.44240902,  -0.06067589] 
    u[9,16, :] = [ 0.40011664,  -0.11427285] 
    u[9,17, :] = [ 0.31595815,  -0.24315785] 
    u[9,18, :] = [ 0.24787815,  -0.31252509] 
    u[9,19, :] = [ 0.19394982,  -0.33751758] 
    u[9,20, :] = [ 0.15137297,  -0.37141708] 
    u[9,21, :] = [ 0.07675958,  -0.42271478] 
    u[9,22, :] = [ 0.00214088,  -0.46168699] 
    u[9,23, :] = [-0.09268628,  -0.47536026] 
    u[9,24, :] = [0.16544032 , 0.47976795] 
    u[9,25, :] = [-0.19848533,  -0.4330127 ] 
    u[9,26, :] = [0.29947744 , 0.39419542] 
    u[9,27, :] = [0.3043306  , 0.36668676] 
    u[9,28, :] = [-0.13564468,  -0.4330127 ] 
    u[9,29, :] = [-0.0649271 , -0.4330127] 
    u[10,0, :] = [-0.44155961,  -0.2290218 ] 
    u[10,1, :] = [-0.44403766,  -0.27804256] 
    u[10,2, :] = [-0.46280774,  -0.30664246] 
    u[10,3, :] = [ 0.45653   ,  -0.29678769] 
    u[10,4, :] = [-0.08810453,   0.51446799] 
    u[10,5, :] = [-0.11790552,   0.43694931] 
    u[10,6, :] = [-0.18312034,   0.42773664] 
    u[10,7, :] = [-0.22775049,   0.38218281] 
    u[10,8, :] = [-0.30704013,   0.3342163 ] 
    u[10,9, :] = [-0.34353042,   0.28202105] 
    u[10,10, :] = [-0.38415132,   0.25874384] 
    u[10,11, :] = [-0.41349489,   0.21396673] 
    u[10,12, :] = [-0.45894355,   0.17169854] 
    u[10,13, :] = [-0.47364717,   0.12556124] 
    u[10,14, :] = [0.47991982 , 0.06892081] 
    u[10,15, :] = [ 0.43742313,  -0.0243378 ] 
    u[10,16, :] = [ 0.37800782,  -0.07712511] 
    u[10,17, :] = [ 0.32399384,  -0.07853601] 
    u[10,18, :] = [-0.19607656,   0.23055038] 
    u[10,19, :] = [-0.15511187,   0.27461802] 
    u[10,20, :] = [-0.11651153,   0.31407745] 
    u[10,21, :] = [-0.0420975 ,   0.37825559] 
    u[10,22, :] = [-0.08225582,  -0.45962679] 
    u[10,23, :] = [-0.12790879,  -0.47052927] 
    u[10,24, :] = [-0.17616477,  -0.46898536] 
    u[10,25, :] = [0.29615632 , 0.39382498] 
    u[10,26, :] = [-0.21424007,  -0.4330127 ] 
    u[10,27, :] = [-0.18695131,  -0.43300284] 
    u[10,28, :] = [0.1640131  , 0.33404494] 
    u[10,29, :] = [0.10221965 , 0.30312448] 
    u[11,0, :] = [-0.45186591,  -0.23500684] 
    u[11,1, :] = [-0.47702565,  -0.24533198] 
    u[11,2, :] = [-0.02111616,   0.54791046] 
    u[11,3, :] = [-0.06456013,   0.49909141] 
    u[11,4, :] = [-0.11016884,   0.48724005] 
    u[11,5, :] = [-0.16267583,   0.45488511] 
    u[11,6, :] = [-0.20861879,   0.40578739] 
    u[11,7, :] = [-0.24245109,   0.38055485] 
    u[11,8, :] = [-0.31146539,   0.32666217] 
    u[11,9, :] = [-0.35997983,   0.30548284] 
    u[11,10, :] = [-0.40943149,   0.24100377] 
    u[11,11, :] = [-0.4576757 ,   0.20564807] 
    u[11,12, :] = [-0.48648686,   0.17279195] 
    u[11,13, :] = [0.48143053 , 0.14024748] 
    u[11,14, :] = [0.45769114 , 0.09264992] 
    u[11,15, :] = [0.42204752 , 0.06045988] 
    u[11,16, :] = [ 0.35463792,  -0.02605864] 
    u[11,17, :] = [ 0.30045701,  -0.08598203] 
    u[11,18, :] = [-0.18193187,   0.19484347] 
    u[11,19, :] = [-0.10242234,   0.24568734] 
    u[11,20, :] = [-0.04459891,   0.28876819] 
    u[11,21, :] = [0.09043401 , 0.32180277] 
    u[11,22, :] = [-0.14273214,  -0.4330127 ] 
    u[11,23, :] = [-0.20540524,  -0.44028258] 
    u[11,24, :] = [0.24030004 , 0.41909497] 
    u[11,25, :] = [0.21362047 , 0.42609734] 
    u[11,26, :] = [0.16487235 , 0.4330127 ] 
    u[11,27, :] = [0.10167628 , 0.4330127 ] 
    u[11,28, :] = [0.04794479 , 0.37224157] 
    u[11,29, :] = [-0.06298407,   0.31478688] 
    u[12,0, :] = [-0.47185354,  -0.19193865] 
    u[12,1, :] = [ 0.47575132,  -0.22720784] 
    u[12,2, :] = [ 0.4499011 ,  -0.28667986] 
    u[12,3, :] = [-0.06473717,   0.53492756] 
    u[12,4, :] = [-0.09395725,   0.47860328] 
    u[12,5, :] = [-0.16636546,   0.44950226] 
    u[12,6, :] = [-0.25240808,   0.42884179] 
    u[12,7, :] = [-0.28322333,   0.37546821] 
    u[12,8, :] = [-0.35241307,   0.31437679] 
    u[12,9, :] = [-0.38746235,   0.29422549] 
    u[12,10, :] = [-0.41161684,   0.26548059] 
    u[12,11, :] = [-0.42562616,   0.26722355] 
    u[12,12, :] = [-0.49374885,   0.21582807] 
    u[12,13, :] = [0.4578978  , 0.18118921] 
    u[12,14, :] = [0.43397261 , 0.13132972] 
    u[12,15, :] = [0.3986072  , 0.07751657] 
    u[12,16, :] = [0.3369158  , 0.02105245] 
    u[12,17, :] = [ 0.28204305,  -0.0370766 ] 
    u[12,18, :] = [ 0.2514645 ,  -0.06471193] 
    u[12,19, :] = [ 0.33198544,  -0.04354835] 
    u[12,20, :] = [-0.09775492,  -0.29236982] 
    u[12,21, :] = [-0.12732324,  -0.34183285] 
    u[12,22, :] = [-0.17243623,  -0.38966449] 
    u[12,23, :] = [-0.23258661,  -0.41370967] 
    u[12,24, :] = [0.21892042 , 0.44756331] 
    u[12,25, :] = [0.16295025 , 0.45286778] 
    u[12,26, :] = [0.13012968 , 0.44484592] 
    u[12,27, :] = [0.07995751 , 0.43301298] 
    u[12,28, :] = [0.01285905 , 0.41148316] 
    u[12,29, :] = [-0.07308804,   0.38177401] 
    u[13,0, :] = [-0.47650236,  -0.19577631] 
    u[13,1, :] = [ 0.48653356,  -0.25782201] 
    u[13,2, :] = [ 0.47507843,  -0.2793969 ] 
    u[13,3, :] = [ 0.44904297,  -0.28488177] 
    u[13,4, :] = [-0.10099168,   0.51734906] 
    u[13,5, :] = [-0.14077049,   0.46617177] 
    u[13,6, :] = [-0.25399004,   0.42610176] 
    u[13,7, :] = [-0.27961044,   0.38172592] 
    u[13,8, :] = [-0.33485864,   0.3366751 ] 
    u[13,9, :] = [-0.3691185 ,   0.30087021] 
    u[13,10, :] = [-0.40525247,   0.27039238] 
    u[13,11, :] = [-0.43768447,   0.23988318] 
    u[13,12, :] = [0.49903621 , 0.19069099] 
    u[13,13, :] = [0.46045226 , 0.16657498] 
    u[13,14, :] = [0.40167437 , 0.13209158] 
    u[13,15, :] = [0.37672572 , 0.06763126] 
    u[13,16, :] = [0.3190824  , 0.00738739] 
    u[13,17, :] = [ 0.24018881,  -0.02478437] 
    u[13,18, :] = [ 0.19078929,  -0.03994067] 
    u[13,19, :] = [ 0.20251647,  -0.02424395] 
    u[13,20, :] = [-0.1324365 ,  -0.27637131] 
    u[13,21, :] = [-0.18376967,  -0.30143327] 
    u[13,22, :] = [-0.26575384,  -0.35005063] 
    u[13,23, :] = [-0.28242779,  -0.36743009] 
    u[13,24, :] = [-0.32883405,  -0.37992148] 
    u[13,25, :] = [0.13538957 , 0.46751755] 
    u[13,26, :] = [0.10694697 , 0.48113465] 
    u[13,27, :] = [0.04256567 , 0.46158192] 
    u[13,28, :] = [-0.03524717,   0.4330127 ] 
    u[13,29, :] = [-0.08276688,   0.4103571 ] 
    u[14,0, :] = [-0.47718229,  -0.13750662] 
    u[14,1, :] = [-0.49690941,  -0.19658678] 
    u[14,2, :] = [ 0.44114467,  -0.25583148] 
    u[14,3, :] = [ 0.41867858,  -0.33249623] 
    u[14,4, :] = [ 0.37482953,  -0.35385843] 
    u[14,5, :] = [ 0.31918953,  -0.38064243] 
    u[14,6, :] = [ 0.26257508,  -0.41123203] 
    u[14,7, :] = [-0.25      ,  0.4330127] 
    u[14,8, :] = [-0.32834659,   0.34336302] 
    u[14,9, :] = [-0.39341409,   0.27991121] 
    u[14,10, :] = [-0.42773782,   0.28459107] 
    u[14,11, :] = [-0.49385458,   0.22110779] 
    u[14,12, :] = [0.46465517 , 0.22068206] 
    u[14,13, :] = [0.43587945 , 0.19701832] 
    u[14,14, :] = [0.38837481 , 0.17292334] 
    u[14,15, :] = [0.33956244 , 0.09933918] 
    u[14,16, :] = [0.21586966 , 0.17465055] 
    u[14,17, :] = [0.16418076 , 0.17493034] 
    u[14,18, :] = [-0.10833462,  -0.14254144] 
    u[14,19, :] = [-0.10776645,  -0.17223573] 
    u[14,20, :] = [-0.1563202 ,  -0.19251232] 
    u[14,21, :] = [-0.25270158,  -0.22213563] 
    u[14,22, :] = [-0.28566366,  -0.26206632] 
    u[14,23, :] = [-0.34841367,  -0.32682486] 
    u[14,24, :] = [0.08776942 , 0.48239185] 
    u[14,25, :] = [0.06664954 , 0.49451839] 
    u[14,26, :] = [0.03478842 , 0.48340352] 
    u[14,27, :] = [-0.00268537,   0.46143177] 
    u[14,28, :] = [-0.05929276,   0.43510184] 
    u[14,29, :] = [-0.10721851,   0.4330127 ] 
    u[15,0, :] = [ 0.48026793,  -0.14327154] 
    u[15,1, :] = [ 0.45004798,  -0.20494744] 
    u[15,2, :] = [ 0.41409207,  -0.25977981] 
    u[15,3, :] = [ 0.37984812,  -0.31230003] 
    u[15,4, :] = [ 0.33587525,  -0.32932277] 
    u[15,5, :] = [ 0.29269161,  -0.38930373] 
    u[15,6, :] = [ 0.25387023,  -0.42630927] 
    u[15,7, :] = [-0.25      ,  0.4330127] 
    u[15,8, :] = [-0.38013982,   0.33678506] 
    u[15,9, :] = [-0.42995046,   0.29346512] 
    u[15,10, :] = [ 0.01398372,  -0.55397657] 
    u[15,11, :] = [0.48800247 , 0.29269081] 
    u[15,12, :] = [0.4341274  , 0.29425585] 
    u[15,13, :] = [0.3829107  , 0.26234954] 
    u[15,14, :] = [0.32312522 , 0.24325025] 
    u[15,15, :] = [0.24203818 , 0.21547927] 
    u[15,16, :] = [0.19279588 , 0.21235188] 
    u[15,17, :] = [0.14099156 , 0.17615966] 
    u[15,18, :] = [-0.13859331,  -0.02094257] 
    u[15,19, :] = [-0.18627368,  -0.03609419] 
    u[15,20, :] = [-0.25594457,  -0.0900299 ] 
    u[15,21, :] = [-0.27214768,  -0.19719952] 
    u[15,22, :] = [-0.32707657,  -0.2400024 ] 
    u[15,23, :] = [-0.38270483,  -0.26711497] 
    u[15,24, :] = [0.07202653 , 0.51065833] 
    u[15,25, :] = [0.0544368  , 0.51622497] 
    u[15,26, :] = [0.03337966 , 0.50179871] 
    u[15,27, :] = [-0.03529453,   0.47890225] 
    u[15,28, :] = [-0.09129052,   0.45373717] 
    u[15,29, :] = [-0.1329907 ,  0.4330127] 
    u[16,0, :] = [ 0.48872459,  -0.06850096] 
    u[16,1, :] = [ 0.47175607,  -0.13068136] 
    u[16,2, :] = [ 0.43676927,  -0.19084705] 
    u[16,3, :] = [ 0.39267302,  -0.26471436] 
    u[16,4, :] = [ 0.33302704,  -0.31074256] 
    u[16,5, :] = [ 0.28646754,  -0.36984908] 
    u[16,6, :] = [ 0.20712858,  -0.4330127 ] 
    u[16,7, :] = [ 0.14665827,  -0.46283248] 
    u[16,8, :] = [ 0.0848402 ,  -0.50018245] 
    u[16,9, :] = [ 0.02658303,  -0.52070713] 
    u[16,10, :] = [-0.02582762,  -0.53301218] 
    u[16,11, :] = [-0.04235965,  -0.54246104] 
    u[16,12, :] = [0.41965097 , 0.2912388 ] 
    u[16,13, :] = [0.3641514  , 0.26500838] 
    u[16,14, :] = [0.30144435 , 0.26864053] 
    u[16,15, :] = [0.23579403 , 0.23606734] 
    u[16,16, :] = [0.19382749 , 0.21494739] 
    u[16,17, :] = [0.11746216 , 0.17828536] 
    u[16,18, :] = [-0.17537475,  -0.01132522] 
    u[16,19, :] = [-0.22438691,  -0.03664512] 
    u[16,20, :] = [-0.29127643,  -0.10321387] 
    u[16,21, :] = [-0.30950332,  -0.10632097] 
    u[16,22, :] = [-0.33800385,  -0.19211614] 
    u[16,23, :] = [-0.38104813,  -0.26411958] 
    u[16,24, :] = [-0.42516259,  -0.28965544] 
    u[16,25, :] = [0.05273827 , 0.52528547] 
    u[16,26, :] = [0.01579557 , 0.4922015 ] 
    u[16,27, :] = [-0.03935431,   0.4827775 ] 
    u[16,28, :] = [-0.09957559,   0.47505041] 
    u[16,29, :] = [-0.15588502,   0.44542622] 
    u[17,0, :] = [-0.43179486,  -0.02062273] 
    u[17,1, :] = [ 0.46550533,  -0.13389275] 
    u[17,2, :] = [ 0.42302893,  -0.19084713] 
    u[17,3, :] = [ 0.35080876,  -0.26762768] 
    u[17,4, :] = [ 0.31492633,  -0.320557  ] 
    u[17,5, :] = [ 0.2804221 ,  -0.38032008] 
    u[17,6, :] = [ 0.19043932,  -0.4330127 ] 
    u[17,7, :] = [ 0.1424824 ,  -0.45695959] 
    u[17,8, :] = [ 0.06274744,  -0.49385057] 
    u[17,9, :] = [-0.00701299,  -0.51405035] 
    u[17,10, :] = [-0.0459527 ,  -0.52553423] 
    u[17,11, :] = [0.40985041 , 0.32194608] 
    u[17,12, :] = [0.34652529 , 0.30835419] 
    u[17,13, :] = [0.30944352 , 0.29560764] 
    u[17,14, :] = [0.26831759 , 0.27577532] 
    u[17,15, :] = [0.1856423  , 0.25251858] 
    u[17,16, :] = [0.13767803 , 0.23177451] 
    u[17,17, :] = [0.08470177 , 0.20210186] 
    u[17,18, :] = [-0.10700185,   0.1418095 ] 
    u[17,19, :] = [-0.24379437,   0.02054322] 
    u[17,20, :] = [-0.28270223,  -0.04938117] 
    u[17,21, :] = [-0.3214714 ,  -0.12495932] 
    u[17,22, :] = [-0.37650069,  -0.18080347] 
    u[17,23, :] = [-0.42695697,  -0.22885917] 
    u[17,24, :] = [-0.46492501,  -0.28236608] 
    u[17,25, :] = [-0.02624629,   0.51757559] 
    u[17,26, :] = [-0.05289915,   0.49930616] 
    u[17,27, :] = [-0.0993887 ,   0.46793349] 
    u[17,28, :] = [-0.15511668,   0.4330127 ] 
    u[17,29, :] = [-0.21999754,   0.4330127 ] 
    u[18,0, :] = [-0.48727115,  -0.05248626] 
    u[18,1, :] = [ 0.46581744,  -0.1107878 ] 
    u[18,2, :] = [ 0.40935372,  -0.16596298] 
    u[18,3, :] = [ 0.37506138,  -0.21640003] 
    u[18,4, :] = [ 0.30839645,  -0.27517317] 
    u[18,5, :] = [ 0.25952662,  -0.33654441] 
    u[18,6, :] = [ 0.19624268,  -0.37940966] 
    u[18,7, :] = [ 0.1406475 ,  -0.41490251] 
    u[18,8, :] = [ 0.00693624,  -0.46100718] 
    u[18,9, :] = [-0.08051006,  -0.47576686] 
    u[18,10, :] = [-0.11216665,  -0.49055183] 
    u[18,11, :] = [0.33702038 , 0.3586518 ] 
    u[18,12, :] = [0.29575841 , 0.34620058] 
    u[18,13, :] = [0.27493304 , 0.3603095 ] 
    u[18,14, :] = [0.1989071  , 0.32981302] 
    u[18,15, :] = [0.15217216 , 0.30298024] 
    u[18,16, :] = [-0.09409145,   0.27184612] 
    u[18,17, :] = [-0.10328168,   0.25643928] 
    u[18,18, :] = [-0.13467134,   0.21340959] 
    u[18,19, :] = [-0.19543616,   0.21683832] 
    u[18,20, :] = [-0.30631502,  -0.01454984] 
    u[18,21, :] = [-0.34350046,  -0.05476685] 
    u[18,22, :] = [-0.39414106,  -0.12412002] 
    u[18,23, :] = [-0.45085104,  -0.19364903] 
    u[18,24, :] = [-0.47379557,  -0.22698423] 
    u[18,25, :] = [ 0.4788569 ,  -0.27975822] 
    u[18,26, :] = [-0.0877596 ,   0.50866505] 
    u[18,27, :] = [-0.13129507,   0.45729313] 
    u[18,28, :] = [ 0.25      , -0.4330127] 
    u[18,29, :] = [-0.25858202,   0.418149  ] 
    u[19,0, :] = [-0.48151158,   0.03202288] 
    u[19,1, :] = [ 0.47511557,  -0.0431011 ] 
    u[19,2, :] = [ 0.43724547,  -0.10869404] 
    u[19,3, :] = [ 0.39185171,  -0.18731834] 
    u[19,4, :] = [ 0.27265218,  -0.24929155] 
    u[19,5, :] = [ 0.19904292,  -0.30726888] 
    u[19,6, :] = [ 0.14585634,  -0.34929559] 
    u[19,7, :] = [ 0.08865704,  -0.37175554] 
    u[19,8, :] = [-0.07654319,  -0.4330127 ] 
    u[19,9, :] = [-0.13981792,  -0.45482795] 
    u[19,10, :] = [-0.17170652,  -0.45897411] 
    u[19,11, :] = [0.31047657 , 0.39392481] 
    u[19,12, :] = [0.24159951 , 0.36996262] 
    u[19,13, :] = [0.2148265  , 0.36205681] 
    u[19,14, :] = [0.18274809 , 0.34581   ] 
    u[19,15, :] = [0.14207539 , 0.31563033] 
    u[19,16, :] = [-0.07262656,   0.30108409] 
    u[19,17, :] = [-0.10047052,   0.2789696 ] 
    u[19,18, :] = [-0.14771105,   0.23720112] 
    u[19,19, :] = [-0.18459613,   0.21657712] 
    u[19,20, :] = [-0.32021608,   0.00131354] 
    u[19,21, :] = [-0.35961189,  -0.03507137] 
    u[19,22, :] = [-0.40978942,  -0.11845411] 
    u[19,23, :] = [-0.47328005,  -0.17015199] 
    u[19,24, :] = [ 0.46935482,  -0.24922338] 
    u[19,25, :] = [ 0.4256095 ,  -0.30981718] 
    u[19,26, :] = [ 0.3745661 ,  -0.35348167] 
    u[19,27, :] = [-0.17262044,   0.46438642] 
    u[19,28, :] = [-0.25189808,   0.42972514] 
    u[19,29, :] = [-0.26803115,   0.40209624] 
    u[20,0, :] = [0.47250553 , 0.04588535] 
    u[20,1, :] = [ 0.41450088,  -0.01770694] 
    u[20,2, :] = [ 0.36602749,  -0.07129454] 
    u[20,3, :] = [ 0.31331827,  -0.10492676] 
    u[20,4, :] = [ 0.19856233,  -0.27824789] 
    u[20,5, :] = [ 0.14595903,  -0.32226196] 
    u[20,6, :] = [ 0.06951167,  -0.36497794] 
    u[20,7, :] = [-0.05400042,  -0.40458357] 
    u[20,8, :] = [-0.11300544,  -0.42255441] 
    u[20,9, :] = [-0.18178328,  -0.4300877 ] 
    u[20,10, :] = [-0.22436505,  -0.4330127 ] 
    u[20,11, :] = [0.22099008 , 0.44633899] 
    u[20,12, :] = [0.18859869 , 0.4330127 ] 
    u[20,13, :] = [0.15683455 , 0.4330127 ] 
    u[20,14, :] = [0.10817938 , 0.36855076] 
    u[20,15, :] = [0.05456005 , 0.3762295 ] 
    u[20,16, :] = [-0.06874613,   0.31717353] 
    u[20,17, :] = [-0.12501209,   0.28578026] 
    u[20,18, :] = [-0.16759395,   0.24655642] 
    u[20,19, :] = [-0.19056715,   0.22181176] 
    u[20,20, :] = [-0.34577238,   0.0485752 ] 
    u[20,21, :] = [-0.39758759,   0.00123208] 
    u[20,22, :] = [-0.44846405,  -0.08005469] 
    u[20,23, :] = [-0.48258204,  -0.13484004] 
    u[20,24, :] = [ 0.47346778,  -0.1949078 ] 
    u[20,25, :] = [ 0.43181725,  -0.28709009] 
    u[20,26, :] = [ 0.39581155,  -0.32010329] 
    u[20,27, :] = [-0.1454438 ,   0.48839815] 
    u[20,28, :] = [ 0.25      , -0.4330127] 
    u[20,29, :] = [-0.28145037,   0.37880477] 
    u[21,0, :] = [0.43854297 , 0.0706236 ] 
    u[21,1, :] = [0.39509205 , 0.02707945] 
    u[21,2, :] = [ 0.34934294,  -0.02710939] 
    u[21,3, :] = [ 0.29018928,  -0.07560679] 
    u[21,4, :] = [ 0.18172849,  -0.22478733] 
    u[21,5, :] = [ 0.13393868,  -0.2752507 ] 
    u[21,6, :] = [-0.04788102,  -0.34427603] 
    u[21,7, :] = [-0.10527038,  -0.34722649] 
    u[21,8, :] = [-0.17237402,  -0.37339909] 
    u[21,9, :] = [-0.21723698,  -0.41125768] 
    u[21,10, :] = [-0.23575618,  -0.41797455] 
    u[21,11, :] = [0.20842599 , 0.4330127 ] 
    u[21,12, :] = [0.15840094 , 0.4330127 ] 
    u[21,13, :] = [0.12345998 , 0.4330127 ] 
    u[21,14, :] = [0.07041797 , 0.39206641] 
    u[21,15, :] = [0.02584234 , 0.37528929] 
    u[21,16, :] = [-0.08439509,   0.30639998] 
    u[21,17, :] = [-0.12696046,   0.26808317] 
    u[21,18, :] = [-0.18337046,   0.22685448] 
    u[21,19, :] = [-0.19793846,   0.19684293] 
    u[21,20, :] = [-0.35050446,   0.061851  ] 
    u[21,21, :] = [-0.41745801,   0.00079926] 
    u[21,22, :] = [-0.48111486,  -0.09876667] 
    u[21,23, :] = [ 0.4960204 ,  -0.13336553] 
    u[21,24, :] = [ 0.49417085,  -0.1919069 ] 
    u[21,25, :] = [ 0.42033553,  -0.28267978] 
    u[21,26, :] = [ 0.37423083,  -0.33418023] 
    u[21,27, :] = [-0.15751926,   0.48361157] 
    u[21,28, :] = [-0.26040763,   0.42042681] 
    u[21,29, :] = [ 0.18914647,  -0.46144915] 
    u[22,0, :] = [0.44956284 , 0.13173721] 
    u[22,1, :] = [0.39259997 , 0.05369653] 
    u[22,2, :] = [ 0.33360614,  -0.00457906] 
    u[22,3, :] = [ 0.28461413,  -0.04615042] 
    u[22,4, :] = [ 0.16034779,  -0.17157649] 
    u[22,5, :] = [ 0.11302594,  -0.20734093] 
    u[22,6, :] = [-0.11719646,  -0.28662299] 
    u[22,7, :] = [-0.14511706,  -0.30981568] 
    u[22,8, :] = [-0.20166517,  -0.36892566] 
    u[22,9, :] = [-0.26338657,  -0.38996456] 
    u[22,10, :] = [0.2007698  , 0.45062284] 
    u[22,11, :] = [0.15661696 , 0.44993279] 
    u[22,12, :] = [0.11054831 , 0.44267194] 
    u[22,13, :] = [0.09297979 , 0.44747353] 
    u[22,14, :] = [0.05383519 , 0.4330127 ] 
    u[22,15, :] = [-0.04143535,   0.39972115] 
    u[22,16, :] = [-0.08935804,   0.34898254] 
    u[22,17, :] = [-0.14707868,   0.30321994] 
    u[22,18, :] = [-0.1972273 ,   0.26638124] 
    u[22,19, :] = [-0.26234813,   0.22345362] 
    u[22,20, :] = [-0.41317821,   0.15037975] 
    u[22,21, :] = [-0.45849282,   0.07189254] 
    u[22,22, :] = [-0.48402281,   0.0276733 ] 
    u[22,23, :] = [ 0.46616642,  -0.05860148] 
    u[22,24, :] = [ 0.43505373,  -0.11249023] 
    u[22,25, :] = [ 0.37932599,  -0.20901351] 
    u[22,26, :] = [ 0.33416181,  -0.28724018] 
    u[22,27, :] = [ 0.29821664,  -0.34949903] 
    u[22,28, :] = [ 0.22780109,  -0.43015185] 
    u[22,29, :] = [ 0.16876583,  -0.4330127 ] 
    u[23,0, :] = [0.38893815 , 0.16510911] 
    u[23,1, :] = [0.35042792 , 0.1103673 ] 
    u[23,2, :] = [0.29206048 , 0.03752906] 
    u[23,3, :] = [0.23189488 , 0.00075656] 
    u[23,4, :] = [ 0.16063247,  -0.06632575] 
    u[23,5, :] = [ 0.07944341,  -0.16397804] 
    u[23,6, :] = [-0.12271328,  -0.20842351] 
    u[23,7, :] = [-0.17796909,  -0.2660367 ] 
    u[23,8, :] = [-0.2471267 ,  -0.30331646] 
    u[23,9, :] = [-0.29674562,  -0.34546505] 
    u[23,10, :] = [-0.34160366,  -0.37842925] 
    u[23,11, :] = [0.1345941  , 0.46448375] 
    u[23,12, :] = [0.08249377 , 0.4373359 ] 
    u[23,13, :] = [0.06281236 , 0.44848955] 
    u[23,14, :] = [0.0127014 , .4330127] 
    u[23,15, :] = [-0.06285712,   0.41322292] 
    u[23,16, :] = [-0.12808088,   0.36846168] 
    u[23,17, :] = [-0.18477602,   0.32569566] 
    u[23,18, :] = [-0.23840524,   0.2999971 ] 
    u[23,19, :] = [-0.28638777,   0.24974785] 
    u[23,20, :] = [-0.40879631,   0.15796942] 
    u[23,21, :] = [-0.43502064,   0.11254757] 
    u[23,22, :] = [-0.47232606,   0.04799822] 
    u[23,23, :] = [ 0.49665994,  -0.00578515] 
    u[23,24, :] = [ 0.44763064,  -0.09070639] 
    u[23,25, :] = [ 0.40542065,  -0.16381623] 
    u[23,26, :] = [ 0.27459129,  -0.26716329] 
    u[23,27, :] = [ 0.24525406,  -0.30522386] 
    u[23,28, :] = [ 0.19590779,  -0.34840645] 
    u[23,29, :] = [ 0.13206079,  -0.40412988] 
    u[24,0, :] = [0.34115581 , 0.23973938] 
    u[24,1, :] = [0.24697543 , 0.21748779] 
    u[24,2, :] = [0.18492425 , 0.17709052] 
    u[24,3, :] = [0.13122998 , 0.13990479] 
    u[24,4, :] = [0.10516306 , 0.1319417 ] 
    u[24,5, :] = [0.15333973 , 0.18070731] 
    u[24,6, :] = [-0.19079744,  -0.23425944] 
    u[24,7, :] = [-0.25730222,  -0.27173176] 
    u[24,8, :] = [-0.34934779,  -0.32894158] 
    u[24,9, :] = [-0.39085449,  -0.33233531] 
    u[24,10, :] = [0.09020431 , 0.49687891] 
    u[24,11, :] = [0.07331348 , 0.4896847 ] 
    u[24,12, :] = [0.04151319 , 0.47351485] 
    u[24,13, :] = [-0.01906993,   0.45451147] 
    u[24,14, :] = [-0.07966907,   0.42139674] 
    u[24,15, :] = [-0.12733557,   0.37968851] 
    u[24,16, :] = [-0.19714935,   0.35035526] 
    u[24,17, :] = [-0.26451341,   0.31749192] 
    u[24,18, :] = [-0.33085867,   0.29296138] 
    u[24,19, :] = [-0.36721523,   0.25978455] 
    u[24,20, :] = [-0.42430783,   0.19014842] 
    u[24,21, :] = [-0.46211092,   0.15076603] 
    u[24,22, :] = [-0.48756788,   0.08660569] 
    u[24,23, :] = [0.45630418 , 0.02713164] 
    u[24,24, :] = [ 0.42140909,  -0.0350375 ] 
    u[24,25, :] = [ 0.3738982 ,  -0.06630976] 
    u[24,26, :] = [-0.31648606,   0.05376321] 
    u[24,27, :] = [-0.29182987,   0.04355112] 
    u[24,28, :] = [0.13162941 , 0.25353607] 
    u[24,29, :] = [0.08900836 , 0.26603056] 
    u[25,0, :] = [0.29108554 , 0.23295861] 
    u[25,1, :] = [0.26441432 , 0.17393246] 
    u[25,2, :] = [0.19512436 , 0.17049776] 
    u[25,3, :] = [0.11875313 , 0.13077108] 
    u[25,4, :] = [0.09695966 , 0.12323723] 
    u[25,5, :] = [0.13898144 , 0.16569023] 
    u[25,6, :] = [-0.20045655,  -0.19725271] 
    u[25,7, :] = [-0.25377929,  -0.22815589] 
    u[25,8, :] = [-0.34230267,  -0.26588786] 
    u[25,9, :] = [-0.38646631,  -0.31230252] 
    u[25,10, :] = [0.06432195 , 0.53455962] 
    u[25,11, :] = [0.04771054 , 0.52327498] 
    u[25,12, :] = [0.0102752 , .4956267] 
    u[25,13, :] = [-0.02936623,   0.4733074 ] 
    u[25,14, :] = [-0.07374057,   0.4330127 ] 
    u[25,15, :] = [-0.13549903,   0.39959   ] 
    u[25,16, :] = [-0.1880133 ,   0.38918006] 
    u[25,17, :] = [-0.27539258,   0.32478927] 
    u[25,18, :] = [-0.32498165,   0.31052094] 
    u[25,19, :] = [-0.3829345 ,   0.28697667] 
    u[25,20, :] = [-0.43114428,   0.22467972] 
    u[25,21, :] = [-0.47174339,   0.19273785] 
    u[25,22, :] = [0.49919598 , 0.14347226] 
    u[25,23, :] = [0.45521384 , 0.09561445] 
    u[25,24, :] = [0.36124849 , 0.0121947 ] 
    u[25,25, :] = [ 0.3295108 ,  -0.01872835] 
    u[25,26, :] = [-0.29306507,   0.04892295] 
    u[25,27, :] = [-0.29768281,   0.03664592] 
    u[25,28, :] = [0.11879929 , 0.26463564] 
    u[25,29, :] = [0.10278039 , 0.29124022] 
    u[26,0, :] = [0.25457582 , 0.28507878] 
    u[26,1, :] = [0.20710098 , 0.24892407] 
    u[26,2, :] = [0.15210682 , 0.2203798 ] 
    u[26,3, :] = [0.10825082 , 0.20733216] 
    u[26,4, :] = [-0.08140777,   0.12944168] 
    u[26,5, :] = [-0.18731467,  -0.01019671] 
    u[26,6, :] = [-0.24156444,  -0.05051626] 
    u[26,7, :] = [-0.31371542,  -0.13722458] 
    u[26,8, :] = [-0.35536895,  -0.20395553] 
    u[26,9, :] = [-0.39800427,  -0.24660773] 
    u[26,10, :] = [-0.41831017,  -0.30148382] 
    u[26,11, :] = [-0.42756526,  -0.3096218 ] 
    u[26,12, :] = [0.06084518 , 0.52899892] 
    u[26,13, :] = [-0.04212922,   0.48828734] 
    u[26,14, :] = [-0.09393348,   0.4330127 ] 
    u[26,15, :] = [-0.14412025,   0.39329575] 
    u[26,16, :] = [-0.1913955 ,   0.35494581] 
    u[26,17, :] = [-0.26900759,   0.31909016] 
    u[26,18, :] = [-0.34057884,   0.30051536] 
    u[26,19, :] = [-0.40739732,   0.27665128] 
    u[26,20, :] = [-0.46513892,   0.29323124] 
    u[26,21, :] = [-0.4841938 ,   0.22401252] 
    u[26,22, :] = [0.45806052 , 0.18547988] 
    u[26,23, :] = [0.41584335 , 0.11497308] 
    u[26,24, :] = [0.36227537 , 0.05337132] 
    u[26,25, :] = [0.30954918 , 0.012015  ] 
    u[26,26, :] = [ 0.27343976,  -0.03052393] 
    u[26,27, :] = [ 0.28752494,  -0.03803161] 
    u[26,28, :] = [-0.11453839,  -0.27915772] 
    u[26,29, :] = [-0.13241876,  -0.29276738] 
    u[27,0, :] = [0.20575841 , 0.27467486] 
    u[27,1, :] = [0.17007373 , 0.23757154] 
    u[27,2, :] = [0.12980869 , 0.21503015] 
    u[27,3, :] = [0.10186065 , 0.15395648] 
    u[27,4, :] = [-0.21238276,  -0.01577158] 
    u[27,5, :] = [-0.25578867,  -0.06004028] 
    u[27,6, :] = [-0.31592297,  -0.14533178] 
    u[27,7, :] = [-0.3621061 ,  -0.18815931] 
    u[27,8, :] = [-0.38233537,  -0.27774294] 
    u[27,9, :] = [-0.41158704,  -0.295696  ] 
    u[27,10, :] = [-0.44744962,  -0.31519896] 
    u[27,11, :] = [0.05158994 , 0.53897487] 
    u[27,12, :] = [-0.02155419,   0.50794884] 
    u[27,13, :] = [-0.06925317,   0.48219242] 
    u[27,14, :] = [-0.12938487,   0.42703731] 
    u[27,15, :] = [-0.18900719,   0.3774349 ] 
    u[27,16, :] = [-0.23272626,   0.36585103] 
    u[27,17, :] = [-0.30105527,   0.34458238] 
    u[27,18, :] = [-0.3543808 ,   0.30851481] 
    u[27,19, :] = [-0.39433133,   0.28466308] 
    u[27,20, :] = [-0.45350493,   0.27548881] 
    u[27,21, :] = [0.47787494 , 0.24104766] 
    u[27,22, :] = [0.44805214 , 0.21755873] 
    u[27,23, :] = [0.41770898 , 0.18099358] 
    u[27,24, :] = [0.35212304 , 0.10518772] 
    u[27,25, :] = [0.30646521 , 0.04536312] 
    u[27,26, :] = [0.23003667 , 0.00256805] 
    u[27,27, :] = [ 0.1840941 ,  -0.00407187] 
    u[27,28, :] = [-0.10108687,  -0.20593286] 
    u[27,29, :] = [-0.12810118,  -0.22622528] 
    u[28,0, :] = [0.18413825 , 0.31928691] 
    u[28,1, :] = [0.15385938 , 0.28451541] 
    u[28,2, :] = [-0.08293344,   0.24877497] 
    u[28,3, :] = [-0.08289674,   0.20620906] 
    u[28,4, :] = [-0.12818987,   0.17455072] 
    u[28,5, :] = [-0.18242357,   0.14169704] 
    u[28,6, :] = [-0.29445814,   0.03199402] 
    u[28,7, :] = [-0.35330625,  -0.0304495 ] 
    u[28,8, :] = [-0.408491  ,  -0.08939461] 
    u[28,9, :] = [-0.45559791,  -0.12964441] 
    u[28,10, :] = [-0.47154468,  -0.17823839] 
    u[28,11, :] = [-0.46645207,  -0.23650106] 
    u[28,12, :] = [ 0.4783567 ,  -0.24828762] 
    u[28,13, :] = [ 0.39091282,  -0.33592621] 
    u[28,14, :] = [-0.14012529,   0.49081475] 
    u[28,15, :] = [-0.19640288,   0.43995507] 
    u[28,16, :] = [-0.2838029 ,  0.4119232] 
    u[28,17, :] = [-0.33287043,   0.36492177] 
    u[28,18, :] = [-0.39341428,   0.32734682] 
    u[28,19, :] = [ 0.03300584,  -0.53812438] 
    u[28,20, :] = [-0.03694881,  -0.54419058] 
    u[28,21, :] = [0.42766947 , 0.28160266] 
    u[28,22, :] = [0.37603403 , 0.27646946] 
    u[28,23, :] = [0.30414294 , 0.23406261] 
    u[28,24, :] = [0.22153197 , 0.1987594 ] 
    u[28,25, :] = [0.19399086 , 0.19188467] 
    u[28,26, :] = [0.15910882 , 0.0612647 ] 
    u[28,27, :] = [0.09425918 , 0.11374106] 
    u[28,28, :] = [-0.13924607,  -0.11563919] 
    u[28,29, :] = [-0.18700354,  -0.15593129] 
    u[29,0, :] = [0.15294196 , 0.35904665] 
    u[29,1, :] = [0.1156559 , .3253921] 
    u[29,2, :] = [-0.08851492,   0.26995419] 
    u[29,3, :] = [-0.10865037,   0.24281636] 
    u[29,4, :] = [-0.14955382,   0.20979151] 
    u[29,5, :] = [-0.19830169,   0.18861409] 
    u[29,6, :] = [-0.31418937,   0.03514239] 
    u[29,7, :] = [-0.37223506,  -0.05634593] 
    u[29,8, :] = [-0.39883132,  -0.09493572] 
    u[29,9, :] = [-0.43571319,  -0.14661735] 
    u[29,10, :] = [-0.49207826,  -0.19662117] 
    u[29,11, :] = [-0.48512603,  -0.23367944] 
    u[29,12, :] = [ 0.43312866,  -0.3119552 ] 
    u[29,13, :] = [ 0.37737705,  -0.35544271] 
    u[29,14, :] = [-0.17795916,   0.46765755] 
    u[29,15, :] = [-0.22651188,   0.4330127 ] 
    u[29,16, :] = [-0.30920128,   0.37010484] 
    u[29,17, :] = [-0.36995328,   0.36136387] 
    u[29,18, :] = [ 0.06668533,  -0.5136437 ] 
    u[29,19, :] = [-0.03192772,  -0.50746313] 
    u[29,20, :] = [0.41447278 , 0.33463407] 
    u[29,21, :] = [0.37762346 , 0.31272504] 
    u[29,22, :] = [0.33265327 , 0.29077736] 
    u[29,23, :] = [0.28329358 , 0.26529317] 
    u[29,24, :] = [0.21096864 , 0.25071709] 
    u[29,25, :] = [0.14359132 , 0.20920826] 
    u[29,26, :] = [0.1086702  , 0.17247362] 
    u[29,27, :] = [0.09797245 , 0.1344133 ] 
    u[29,28, :] = [-0.19791287,  -0.03559317] 
    u[29,29, :] = [-0.25193609,  -0.07673825] 
    return u

if __name__ == "__main__": main()

