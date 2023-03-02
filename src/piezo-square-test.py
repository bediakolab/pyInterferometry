
import numpy as np  
import matplotlib.pyplot as plt

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
def alpha(d_0, eps_p): return d_0 * (eps_p - 1) / (4 * np.pi) # same units as d_0, so nm

def in_rs_unitcell(r, a0=1):
    a1 = a0 * [1,0]
    a2 = a0 * [0,1]#[1/2, np.sqrt(3)/2]
    # want r = c1 a1 + c2 a2, c1,c1 in [0,1)
    c2 = r[1]/a0 #(2*r[1])/(np.sqrt(3)*a0)
    c1 = r[0]/a0 #- c2/2
    if not ( c1 >= 0 and c1 < 1 ):
        print('OUTSIDE UNIT CELL FOR ', c1, r)
    if not ( c2 >= 0 and c2 < 1 ):
        print('OUTSIDE UNIT CELL FOR ', c2, r)

def reduce_rs_unitcell(rs_val, a0=1):
    c1, c2 = rs_val[0]/a0, rs_val[1]/a0
    #c2 = (2*rs_val[1])/(np.sqrt(3)*a0)
    #c1 = (rs_val[0]/a0) - c2/2
    while c1 < 0: c1 += 1
    c1 = c1%1
    while c2 < 0: c2 += 1
    c2 = c2%1
    rs_val = [a0*c1, a0*c2] #[a0*c1 + a0*c2/2, a0*np.sqrt(3)*c2/2]
    in_rs_unitcell(rs_val, a0)
    return rs_val

# if lim=1, first star. if lim=4 up to 4th star... dont include g=0
def generate_fake_g(a_0, lim, visualize, include_zero=True, truncate_corners=False):
    # hexagonal unit cell w/ a1 = a[1,0] and a2 = a[1/2, root3/2]
    b1 = ( np.pi * 2 / a_0 ) * np.array([1,0]) #np.array([1, -1/np.sqrt(3)])
    b2 = ( np.pi * 2 / a_0 ) * np.array([0,1]) #np.array([0,  2/np.sqrt(3)])
    gvecs = []
    if visualize: f, ax = plt.subplots()
    for i in np.arange(-lim,lim+1,1):
        for j in np.arange(-lim,lim+1,1):
            if truncate_corners and np.abs(i - j) > lim: continue
            if i==j==0 and not include_zero: continue
            g = i * b1 + j * b2
            gmag = np.dot(g,g) ** 0.5
            gvecs.append(g)
            if visualize: 
                ax.scatter(g[0], g[1], c='k')
                #ax.text(g[0], g[1], "{}".format(gmag))
    if visualize: plt.show()
    return gvecs

def get_tiling_rs_unitcell(spacing, a0=1, visualize=False):
    a1 = a0 * [1,0]
    a2 = a0 * [0,1]
    N = int(1/spacing)-1
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

# if generate u_r from u_g, make sure rs2ms(u_r)=u_g!
def sanitycheck1(tol=1e-8):

    nrings = 9
    spacing = 0.050 # want same number of points in rs and ms
    g = generate_fake_g(a_0=1, lim=nrings, visualize=False)
    x, y = get_tiling_rs_unitcell(spacing, visualize=False)
    print(len(g)**0.5)
    print(x.shape[0])
    assert(x.shape[0] * x.shape[1] == len(g))
    Ng = len(g) # 9x9
    u_ms = np.zeros((Ng, 2))
    for n in range(Ng): u_ms[n,0] = 0.2 

    u_rs = ms_to_rs_u(g, x, y, u_ms) # RS that should be perfectly represented by the above MS
    u_ms_2 = rs_to_ms_u(g, x, y, u_rs) # should be the same as u_ms
    u_rs_2 = ms_to_rs_u(g, x, y, u_ms_2)

    # making sure that get u_ms perfectly from this data
    print(u_ms[:,0])
    print(u_ms_2[:,0])
    exit()

    assert(np.max(np.abs((u_ms[:,0] - u_ms_2[:,0]))) < tol)
    assert(np.max(np.abs((u_rs_2[:,:,0] - u_rs[:,:,0]))) < tol)
    assert(np.max(np.abs((u_rs_2[:,:,1] - u_rs[:,:,1]))) < tol)
    assert(np.max(np.abs((u_ms[:,1] - u_ms_2[:,1]))) < tol)
    print('sanitycheck 1.1 passes')

    for n in range(Ng): 
        u_ms[n,0] = 0.15
        u_ms[n,1] = 0.10

    u_rs = ms_to_rs_u(g, x, y, u_ms) # RS that should be perfectly represented by the above MS
    u_ms_2 = rs_to_ms_u(g, x, y, u_rs) # should be the same as u_ms
    u_rs_2 = ms_to_rs_u(g, x, y, u_ms_2)

    assert(np.max(np.abs((u_ms[:,0] - u_ms_2[:,0]))) < tol)
    assert(np.max(np.abs((u_rs_2[:,:,0] - u_rs[:,:,0]))) < tol)
    assert(np.max(np.abs((u_rs_2[:,:,1] - u_rs[:,:,1]))) < tol)
    assert(np.max(np.abs((u_ms[:,1] - u_ms_2[:,1]))) < tol)
    print('sanitycheck 1.2 passes')

    for n in range(Ng): 
        u_ms[n,0] = 0.15*(1-n)
        u_ms[n,1] = 0.10

    u_rs = ms_to_rs_u(g, x, y, u_ms) # RS that should be perfectly represented by the above MS
    u_ms_2 = rs_to_ms_u(g, x, y, u_rs) # should be the same as u_ms
    u_rs_2 = ms_to_rs_u(g, x, y, u_ms_2)

    assert(np.max(np.abs((u_ms[:,0] - u_ms_2[:,0]))) < tol)
    assert(np.max(np.abs((u_rs_2[:,:,0] - u_rs[:,:,0]))) < tol)
    assert(np.max(np.abs((u_rs_2[:,:,1] - u_rs[:,:,1]))) < tol)
    assert(np.max(np.abs((u_ms[:,1] - u_ms_2[:,1]))) < tol)
    print('sanitycheck 1.3 passes')

    for n in range(Ng): 
        u_ms[n,0] = 0.15*(1-n)
        u_ms[n,1] = 0.45*(1-n)

    u_rs = ms_to_rs_u(g, x, y, u_ms) # RS that should be perfectly represented by the above MS
    u_ms_2 = rs_to_ms_u(g, x, y, u_rs) # should be the same as u_ms
    u_rs_2 = ms_to_rs_u(g, x, y, u_ms_2)

    assert(np.max(np.abs((u_ms[:,0] - u_ms_2[:,0]))) < tol)
    assert(np.max(np.abs((u_rs_2[:,:,0] - u_rs[:,:,0]))) < tol)
    assert(np.max(np.abs((u_rs_2[:,:,1] - u_rs[:,:,1]))) < tol)
    assert(np.max(np.abs((u_ms[:,1] - u_ms_2[:,1]))) < tol)
    print('sanitycheck 1.4 passes')

# if generate u_g from u_r, make sure ms2rs(u_g)=u_r!
def sanitycheck2(tol=1e-8):

    nrings = 10 # changes value for some reason!!
    spacing = 0.05 # changes value for some reason!!
    g = generate_fake_g(a_0=1, lim=nrings, visualize=False)
    x, y = get_tiling_rs_unitcell(spacing, visualize=False)
    Nx, Ny = x.shape[0], x.shape[1]
    u_rs = np.zeros((Nx, Ny, 2))
    
    for i in range(Nx):
        for j in range(Ny): 
            u_rs[i,j,0] = 0.5
            u_rs[i,j,1] = 0.0
    
    u_ms = rs_to_ms_u(g, x, y, u_rs) 
    u_rs_2 = ms_to_rs_u(g, x, y, u_ms) 

    # making sure that get u_ms perfectly from this data
    assert(np.max(np.abs((u_rs[:,:,0] - u_rs_2[:,:,0]))) < tol)
    assert(np.max(np.abs((u_rs[:,:,1] - u_rs_2[:,:,1]))) < tol)
    print('sanitycheck 2.1 passes')

    for i in range(Nx):
        for j in range(Ny): 
            u_rs[i,j,0] = -0.15
            u_rs[i,j,1] = 0.04
    
    u_ms = rs_to_ms_u(g, x, y, u_rs) 
    u_rs_2 = ms_to_rs_u(g, x, y, u_ms) 

    # making sure that get u_ms perfectly from this data
    assert(np.max(np.abs((u_rs[:,:,0] - u_rs_2[:,:,0]))) < tol)
    assert(np.max(np.abs((u_rs[:,:,1] - u_rs_2[:,:,1]))) < tol)
    print('sanitycheck 2.2 passes')

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
    return np.gradient(f, axis=1) * 1/a0 * 1/spacing

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
    g = generate_fake_g(a_0=a_0, lim=20, visualize=False)
    sanitycheck1()
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

# u_t_rs[i,j,0] = sum_n u_t[n,0] cos{ g[n] dot (x[i,j],y[i,j]) } 
# u_t_rs[i,j,1] = sum_n u_t[n,1] cos{ g[n] dot (x[i,j],y[i,j]) } 
# u_b_rs[i,j,0] = sum_n u_b[n,0] cos{ g[n] dot (x[i,j],y[i,j]) } 
# u_b_rs[i,j,1] = sum_n u_b[n,1] cos{ g[n] dot (x[i,j],y[i,j]) } 
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
    return u_ms #/Nr

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


if __name__ == "__main__": main()

