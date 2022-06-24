
# constants 
d_0 = 0.65                       # interlayer distance minimum (expanded about) in nm for MoS2 - average sep over all stacking configs present
d = d_0                          # interlayer distance, nm. taking as minumum above. 
e_11_t  = 2.9e-10                # in C-1 m, for MoS2
e_11_b  = -e_11_t                # true for AP
alpha_t = alpha(d_0, eps_p=16.3) # in-plane 2d polarizability for MoS2
alpha_b = alpha_t                # same in-plane 2d polarizability


# import g and u
# NOTE: need reference frame with positive direction of y-axis being 
# in-plane projection of a M-X vector in bottom layer!
# they expanded up to "80th reciprocal space star"


# inline equation before S9
def alpha(d_0, eps_p):
	return d_0 * (eps_p - 1) / (4 * np.pi)

# equation 11 in SI for top layer
def varphi_t(n):
	g_n = np.abs(g[n])
	a = 4 * np.pi * sinh(g_n * d)
	b_t = ( np.exp(g[n] * d)  + g_n * alpha_t * a)
	b_b = ( np.exp(g[n] * d)  + g_n * alpha_b * a)
	num_t = a * (rho_t(n) + rho_b(n) * b_t)
	denom = g_n * (b_t * b_b - 1)
	varphi_t = num_t / denom
	return varphi_t

# equation 11 in SI for bottom layer
def varphi_b(n):
	g_n = np.abs(g[n])
	a = 4 * np.pi * sinh(g_n * d)
	b_t = ( np.exp(g[n] * d)  + g_n * alpha_t * a)
	b_b = ( np.exp(g[n] * d)  + g_n * alpha_b * a)
	num_b = a * (rho_b(n) + rho_t(n) * b_b)
	denom = g_n * (b_t * b_b - 1)
	varphi_b = num_b / denom
	return varphi_b

# equation 12 in SI for top layer
def rho_t(n):
	gnx, gny = g[n][0], g[n][1]
	u_t_nx, u_t_ny = u_t[n][0], u_t[n][1]
	a = 2 * gnx * gny 
	b = gnx*gnx - gny*gny
	rho_t = e_11_t * (a*u_t_nx + b*u_t_ny)
	return rho_t

# equation 12 in SI for bottom layer
def rho_b(n):
	gnx, gny = g[n][0], g[n][1]
	u_t_nx, u_t_ny = u_t[n][0], u_t[n][1]
	a = 2 * gnx * gny 
	b = gnx*gnx - gny*gny
	rho_b = e_11_b * (a*u_b_nx + b*u_b_ny)
	return rho_b

# inline equation after eq12 in SI for top layer
def rho_t_tot(r):
    Re_val = 0
    Im_val = 0
	for n in range(len(g)):
	    g_n = np.abs(g[n])
		a = 4 * np.pi * g_n * g_n * alpha_t * varphi_t(n)
		Re_val += (rho_t(n) - a) * np.cos( np.dot(g[n], r) )
		Im_val += (rho_t(n) - a) * np.sin( np.dot(g[n], r) )
	assert( np.abs(Im_val) < 1e-8 )
	return Re_val

# inline equation after eq12 in SI for bottom layer
def rho_b_tot(r):
    Re_val = 0
    Im_val = 0
	for n in range(len(g)):
	    g_n = np.abs(g[n])
		a = 4 * np.pi * g_n * g_n * alpha_b * varphi_b(n)
		Re_val += (rho_b(n) - a) * np.cos( np.dot(g[n], r) )
		Im_val += (rho_b(n) - a) * np.sin( np.dot(g[n], r) )
	assert( np.abs(Im_val) < 1e-8 )	
	return Re_val

# test, u for top layer
def u_t(r):
    Re_val = np.array([0, 0])
    Im_val = np.array([0, 0])
	for n in range(len(g)):
		Re_val += u_t[n] * np.cos( np.dot(g[n], r) )
		Im_val += u_t[n] * np.sin( np.dot(g[n], r) )
	assert( np.abs(Im_val[0]) < 1e-8 )
	assert( np.abs(Im_val[1]) < 1e-8 )

# test, u for bottom layer
def u_b(r):
    Re_val = np.array([0, 0])
    Im_val = np.array([0, 0])
	for n in range(len(g)):
		Re_val += u_b[n] * np.cos( np.dot(g[n], r) )
		Im_val += u_b[n] * np.sin( np.dot(g[n], r) )
	assert( np.abs(Im_val[0]) < 1e-8 )
	assert( np.abs(Im_val[1]) < 1e-8 )


