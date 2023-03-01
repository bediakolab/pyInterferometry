
from basis_utils import cartesian_to_latticevec, cartesian_to_rzcartesian
from diskset import DiskSet
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import polar

def get_u_XY_grid(Transformation, theta_layer1=0, N=10, plotflag=False):
	# grid in XY basis
	if plotflag: f, (ax, ax2) = plt.subplots(1,2)
	X, Y, Ux, Uy = np.zeros((N+1,N+1)), np.zeros((N+1,N+1)), np.zeros((N+1,N+1)), np.zeros((N+1,N+1))
	j = 0
	for x in np.arange(-N/2, N/2 + 1):
		i = 0
		for y in np.arange(-N/2, N/2 + 1):
			sx_sy = np.matmul(Rmat(theta_layer1), np.array([x,y])) # rotate the x,y coordinates away from scan direction by phi
			sx, sy = sx_sy[0,0], sx_sy[0,1] 
			r0 = np.array([sx,sy])
			r1 = np.matmul(Transformation, r0)
			r1 = np.array([r1[0,0],r1[0,1]])
			u = r1[:] - r0[:] 
			if plotflag: ax.scatter(r0[0], r0[1], color='k')
			if plotflag: ax.scatter(r1[0], r1[1], color='b')
			if plotflag: ax.arrow(r0[0],  r0[1], u[0], u[1], color='r')
			Ux[i,j] = u[0]
			Uy[i,j] = u[1]
			X[i,j]  = x 
			Y[i,j]  = y 
			i += 1
		j += 1
	if plotflag: ax2.quiver(X, Y, Ux, Uy)	
	if plotflag: plt.show()
	return X, Y, Ux, Uy

def get_u_LV_grid(Transformation, theta_layer1=0, N=10, plotflag=False, a0=1):

	if plotflag: f, (ax, ax2) = plt.subplots(1,2)
	X, Y, Ux, Uy = np.zeros((N+1,N+1)), np.zeros((N+1,N+1)), np.zeros((N+1,N+1)), np.zeros((N+1,N+1))
	j = 0
	R = Rmat(theta_layer1)
	for c1 in np.arange(-N/2, N/2 + 1):
		i = 0
		for c2 in np.arange(-N/2, N/2 + 1):
			r0 = np.matmul(R, r_0(c1,c2,a0))
			r0 = np.array([r0[0,0],r0[0,1]])
			r1 = np.matmul(Transformation, r0)
			r1 = np.array([r1[0,0],r1[0,1]])
			#r1 = np.matmul(R, r1)
			#r1 = np.array([r1[0,0],r1[0,1]])
			u = r1[:] - r0[:] 
			if plotflag: ax.scatter(r0[0], r0[1], color='k')
			if plotflag: ax.scatter(r1[0], r1[1], color='b')
			if plotflag: ax.arrow(r0[0],  r0[1], u[0], u[1], color='r')
			Ux[i,j] = u[0]
			Uy[i,j] = u[1]
			X[i,j] = r0[0]
			Y[i,j] = r0[1]
			i += 1
		j += 1
	if plotflag: ax2.quiver(X, Y, Ux, Uy)	
	if plotflag: plt.show()
	return X, Y, Ux, Uy

def plot_LV_grid(theta_layer1, theta_layer2, N=10, plotflag=False, a0=1):

	if plotflag: f, ax = plt.subplots()
	X, Y, Ux, Uy = np.zeros((N+1,N+1)), np.zeros((N+1,N+1)), np.zeros((N+1,N+1)), np.zeros((N+1,N+1))
	j = 0
	R  = Rmat(theta_layer1)
	R2 = Rmat(theta_layer2)
	for c1 in np.arange(-N/2, N/2 + 1):
		i = 0
		for c2 in np.arange(-N/2, N/2 + 1):
			r0 = r_0(c1,c2,a0)
			r0 = np.array([r0[0],r0[1]])
			r1 = np.matmul(R, r0)
			r1 = np.array([r1[0,0],r1[0,1]])
			r2 = np.matmul(R2, r0)
			r2 = np.array([r2[0,0],r2[0,1]])
			ax.scatter(r0[0], r0[1], color='k', s=1)
			ax.scatter(r1[0], r1[1], color='k', s=1)
			ax.scatter(r2[0], r2[1], color='k', s=1)
			i += 1
		j += 1
	plt.show() 

def strain_u_LV_grid(Transformation, theta_layer1=0, N=10, plotflag=False):

	_, _, Ux, Uy = get_u_LV_grid(Transformation, theta_layer1, N, plotflag)

	dux_da1 = np.gradient(Ux * 0.5, axis=1) 
	dux_da2 = np.gradient(Ux * 0.5, axis=0) 
	duy_da1 = np.gradient(Uy * 0.5, axis=1) 
	duy_da2 = np.gradient(Uy * 0.5, axis=0) 
	da1_dx = 1/a0
	da1_dy = -1/(a0*np.sqrt(3))
	da2_dx = 0
	da2_dy = 2/(a0*np.sqrt(3))
	dux_dx = dux_da1 * da1_dx + dux_da2 * da2_dx # need to correct these assume xy parallel with one layer
	dux_dy = dux_da1 * da1_dy + dux_da2 * da2_dy 
	duy_dx = duy_da1 * da1_dx + duy_da2 * da2_dx 
	duy_dy = duy_da1 * da1_dy + duy_da2 * da2_dy 

	gamma_max = np.zeros((N+1,N+1))
	for i in range(N+1):
		for j in range(N+1):
			offd = (dux_dy[i,j]+duy_dx[i,j])/2
			evals, evecs = np.linalg.eig(np.matrix([[dux_dx[i,j], offd], [offd, duy_dy[i,j]]]))
			gamma_max[i,j] = np.max(evals) - np.min(evals)

	theta_tot = (dux_dy - duy_dx) 
	dilation = dux_dx + duy_dy	  
	shear = dux_dy + duy_dx

	if False:
		# without reconstruction, dilation is 0.5 * (1+delta) * (2 - epsilon*(pr-1)) * cos(theta_m) - 1
		print(0.5 * (1+delta) * (2 - epsilon*(pr-1)) * np.cos(theta_m) - 1)
		print('calc: ', dilation[2,2])

		# without reconstruction, theta_tot is 0.5 * (1+delta) * (epsilon*(pr-1) - 2) * sin(theta_m)
		print(0.5 * (1+delta) * (2 - epsilon*(pr-1)) * np.sin(theta_m))
		print('calc: ', theta_tot[2,2])

		# without reconstruction, shear is 0.5 * (1+delta) * (epsilon*(pr+1)) * sin(theta_m - 2*theta_t)
		print(0.5 * (1+delta) * (-epsilon*(pr+1)) * np.sin(theta_m - 2*theta_s))

	if False:
		f, ax = plt.subplots(2,2)
		ax = ax.flatten()
		ax[0].imshow(theta_tot)
		ax[0].set_title('theta')
		ax[1].imshow(dilation)
		ax[1].set_title('dilation')
		ax[2].imshow(shear)
		ax[2].set_title('shear')
		ax[3].imshow(gamma_max)
		ax[3].set_title('gamma_max')
		plt.show()

	return dux_dx, dux_dy, duy_dx, duy_dy

def strain_u_XY_grid(Transformation, theta_layer1=0, N=10, plotflag=False, correct=False):

	_, _, U1, U2 = get_u_XY_grid(Transformation, theta_layer1, N, plotflag)
	
	if correct:
		Ux = np.cos(theta_layer1) * U1 - np.sin(theta_layer1) * U2 # moves to basis defined by first layer orrientation 
		Uy = np.cos(theta_layer1) * U2 + np.sin(theta_layer1) * U1
	else:
		Ux, Uy = U1, U2

	dux_dx = np.gradient(Ux * 0.5, axis=1) 
	dux_dy = np.gradient(Ux * 0.5, axis=0) 
	duy_dx = np.gradient(Uy * 0.5, axis=1) 
	duy_dy = np.gradient(Uy * 0.5, axis=0) 
	gamma_max = np.zeros((N+1,N+1))
	
	for i in range(N+1):
		for j in range(N+1):
			offd = (dux_dy[i,j]+duy_dx[i,j])/2
			evals, evecs = np.linalg.eig(np.matrix([[dux_dx[i,j], offd], [offd, duy_dy[i,j]]]))
			gamma_max[i,j] = np.max(evals) - np.min(evals)

	theta_tot = (dux_dy - duy_dx) 
	dilation = dux_dx + duy_dy	  
	shear = dux_dy + duy_dx

	if False:
		# without reconstruction, dilation is 0.5 * (1+delta) * (2 - epsilon*(pr-1)) * cos(theta_m) - 1
		print(0.5 * (1+delta) * (2 - epsilon*(pr-1)) * np.cos(theta_m) - 1)
		print('calc: ', dilation[2,2])

		# without reconstruction, theta_tot is 0.5 * (1+delta) * (epsilon*(pr-1) - 2) * sin(theta_m)
		print(0.5 * (1+delta) * (2 - epsilon*(pr-1)) * np.sin(theta_m))
		print('calc: ', theta_tot[2,2])

		# without reconstruction, shear is 0.5 * (1+delta) * (epsilon*(pr+1)) * sin(theta_m - 2*theta_t)
		print(0.5 * (1+delta) * (-epsilon*(pr+1)) * np.sin(theta_m - 2*theta_s))

	if False:
		f, ax = plt.subplots(2,2)
		ax = ax.flatten()
		ax[0].imshow(theta_tot)
		ax[0].set_title('theta')
		ax[1].imshow(dilation)
		ax[1].set_title('dilation')
		ax[2].imshow(shear)
		ax[2].set_title('shear')
		ax[3].imshow(gamma_max)
		ax[3].set_title('gamma_max')
		plt.show()

	return dux_dx, dux_dy, duy_dx, duy_dy

def test_rotcor(dux_dx, dux_dy, duy_dx, duy_dy, theta_layer1, theta_layer2):
	
	# in naive reference configuration
	if False:
		print('expect dux_dx: ', round( 0.5*(np.cos(theta_layer2+theta_layer1) - np.cos(theta_layer1)), 4) )
		print('calc dux_dx: ',   round( dux_dx[2,2], 4 ) )
		print('expect dux_dy: ', round( 0.5*(np.sin(theta_layer2+theta_layer1)-np.sin(theta_layer1)), 4) )
		print('calc dux_dy: ',   round( dux_dy[2,2], 4 ) )
		print('expect duy_dx: ', round( -0.5*(np.sin(theta_layer2+theta_layer1)-np.sin(theta_layer1)), 4) )
		print('calc duy_dx: ',   round( duy_dx[2,2], 4 ) )
		print('expect duy_dy: ', round( 0.5*(np.cos(theta_layer2+theta_layer1) - np.cos(theta_layer1)), 4) )
		print('calc duy_dy: ',   round( duy_dy[2,2], 4 ) )
		print('in this reference config,  theta_tot=', round( 180/np.pi * (dux_dy[2,2]-duy_dx[2,2]), 3 ) )
		print('in this reference config,  dilation=',  100 * round( dux_dx[2,2]+duy_dy[2,2], 3 ),'%' )

	# rotate reference frame by -theta_layer1
	# so layer_2 is twisted by theta_layer2 and layer_1 by 0
	if False:
		rotation_correction = -theta_layer1
		cor_dux_dx  = np.cos(rotation_correction) * dux_dx - np.sin(rotation_correction) * dux_dy 
		cor_dux_dy  = np.sin(rotation_correction) * dux_dx + np.cos(rotation_correction) * dux_dy 
		cor_duy_dx  = np.cos(rotation_correction) * duy_dx - np.sin(rotation_correction) * duy_dy 
		cor_duy_dy  = np.sin(rotation_correction) * duy_dx + np.cos(rotation_correction) * duy_dy 
		print('expect corrected dux_dx: ', round( 0.5*(np.cos(theta_layer2)-1), 4) )
		print('calc corrected dux_dx: ',   round( cor_dux_dx[2,2], 4 ) )
		print('expect corrected dux_dy: ', round( 0.5*np.sin(theta_layer2), 4) )
		print('calc corrected dux_dy: ',   round( cor_dux_dy[2,2], 4 ) )
		print('expect corrected duy_dx: ', round( -0.5*np.sin(theta_layer2), 4) )
		print('calc corrected duy_dx: ',   round( cor_duy_dx[2,2], 4 ) )
		print('expect corrected duy_dy: ', round( 0.5*(np.cos(theta_layer2)-1), 4) )
		print('calc corrected duy_dy: ',   round( cor_duy_dy[2,2], 4 ) )
		print('in this reference config,  theta_tot=', round( 180/np.pi * (cor_dux_dy[2,2]-cor_duy_dx[2,2]), 3 ) )
		print('in this reference config,  dilation=',  100 * round( cor_dux_dx[2,2]+cor_duy_dy[2,2], 3 ),'%' )

	# rotate reference frame by - theta_layer1 - theta_layer2/2
	# so layer_2 is twisted by theta_layer2/2 and layer_1 by -theta_layer2/2
	if True:
		rotation_correction = - (theta_layer1 + theta_layer2/2)
		cor_dux_dx  = np.cos(rotation_correction) * dux_dx - np.sin(rotation_correction) * dux_dy 
		cor_dux_dy  = np.sin(rotation_correction) * dux_dx + np.cos(rotation_correction) * dux_dy 
		cor_duy_dx  = np.cos(rotation_correction) * duy_dx - np.sin(rotation_correction) * duy_dy 
		cor_duy_dy  = np.sin(rotation_correction) * duy_dx + np.cos(rotation_correction) * duy_dy 
		print('expect corrected dux_dx: ', round( 0, 4) )
		print('calc corrected dux_dx: ',   round( cor_dux_dx[2,2], 4 ) )
		print('expect corrected dux_dy: ', round( np.sin(theta_layer2/2), 4) )
		print('calc corrected dux_dy: ',   round( cor_dux_dy[2,2], 4 ) )
		print('expect corrected duy_dx: ', round( -np.sin(theta_layer2/2), 4) )
		print('calc corrected duy_dx: ',   round( cor_duy_dx[2,2], 4 ) )
		print('expect corrected duy_dy: ', round( 0, 4) )
		print('calc corrected duy_dy: ',   round( cor_duy_dy[2,2], 4 ) )
		print('in this reference config,  theta_tot=', round( 180/np.pi * (cor_dux_dy[2,2]-cor_duy_dx[2,2]), 3 ) )
		print('in this reference config,  dilation=',  100 * round( cor_dux_dx[2,2]+cor_duy_dy[2,2], 3 ), '%' )

# transforms from LV basis to cartesian
def r_0(c1,c2,a0=1):
	#R = Rmat(phi)
	R = Rmat(0)
	sp1 = np.matmul(R, np.array([1,0]))
	sp1 = np.array([sp1[0,0],sp1[0,1]])
	sp2 = np.matmul(R, np.array([1/2,np.sqrt(3)/2]))
	sp2 = np.array([sp2[0,0],sp2[0,1]])
	a1, a2 = a0 * sp1, a0 * sp2
	return c1*a1 + c2*a2

def Rmat(theta): 
	return np.matrix([[np.cos(theta), np.sin(theta)], [- np.sin(theta), np.cos(theta)]])

def heterostrain(epsilon, pr, theta):	
	R, Rinv = Rmat(theta), Rmat(-theta)
	T = np.matrix([[1+epsilon, 0], [0, 1-(pr*epsilon)]])
	return multiply_three_matrices(Rinv, T, R)

def multiply_three_matrices(M1, M2, M3):
	return np.matmul(M1, np.matmul(M2, M3))

def main_tlg_ft_test():

	theta_layer1 = -1.5 * np.pi/180  # twist from xy
	theta_layer2 =  3.0 * np.pi/180  # twist from layer_1, so theta_layer1+theta_layer2 overall from xy
	print('overall sample rotation of ', (theta_layer1 + (theta_layer2+theta_layer1) )*0.5*180/np.pi ) 
	delta = 0.0
	pr = 0.16 # graphene
	epsilon = 0 / 100
	theta_s = 0 #1.0 * np.pi/180
	a0 = 1
	N=20
	plot_LV_grid(theta_layer1, theta_layer2, N, plotflag=True)
	exit()


def main():

	theta_layer1 = -1.5 * np.pi/180  # twist from xy
	theta_layer2 =  3.0 * np.pi/180  # twist from layer_1, so theta_layer1+theta_layer2 overall from xy
	print('overall sample rotation of ', (theta_layer1 + (theta_layer2+theta_layer1) )*0.5*180/np.pi ) 
	delta = 0.0
	pr = 0.16 # graphene
	epsilon = 0 / 100
	theta_s = 0 #1.0 * np.pi/180
	a0 = 1

	heterobilayer_mat = np.matrix([[1+delta, 0], [0, 1+delta]])
	HS_mat = heterostrain(epsilon, pr, theta_s)
	Transformation = multiply_three_matrices(Rmat(theta_layer2), heterobilayer_mat, HS_mat)

	N=50
	X, Y, Ux, Uy = get_u_XY_grid(Transformation, theta_layer1, N, plotflag=False)
	uvecs_cart = np.zeros((Ux.shape[0], Ux.shape[1],2))
	uvecs_cart[:,:,0], uvecs_cart[:,:,1] = Ux[:,:], Uy[:,:]
	uvecs_rz   = cartesian_to_rzcartesian(uvecs_cart, sign_wrap=False)
	uvecs_rzlv = cartesian_to_latticevec(uvecs_rz)
	uvecs_lv   = cartesian_to_latticevec(uvecs_cart)

	print('****** USING THE CART BASIS VECTORS THESE WILL BE GOOD *******')
	Ux, Uy = uvecs_cart[:,:,0], uvecs_cart[:,:,1] 
	test_rotcor(np.gradient(Ux * 0.5, axis=1), np.gradient(Ux * 0.5, axis=0), np.gradient(Uy * 0.5, axis=1), np.gradient(Uy * 0.5, axis=0), theta_layer1, theta_layer2)	

	#print('****** NOW USING THE LV BASIS VECTORS THESE WILL BE WRONG *******')
	#Ux, Uy = uvecs_lv[:,:,0], uvecs_lv[:,:,1] 
	#test_rotcor(np.gradient(Ux * 0.5, axis=1), np.gradient(Ux * 0.5, axis=0), np.gradient(Uy * 0.5, axis=1), np.gradient(Uy * 0.5, axis=0), theta_layer1, theta_layer2)	

	f, ax = plt.subplots(2,2)
	ax[0,0].quiver(X, Y, uvecs_cart[:,:,0], uvecs_cart[:,:,1])	
	ax[0,1].quiver(X, Y, uvecs_rz[:,:,0],   uvecs_rz[:,:,1])	
	ax[1,0].quiver(X, Y, uvecs_lv[:,:,0],   uvecs_lv[:,:,1])	
	ax[1,1].quiver(X, Y, uvecs_rzlv[:,:,0], uvecs_rzlv[:,:,1])	
	plt.show()

	# making fake interferometry data from these u
	f, ax = plt.subplots(6,4)
	ax = ax.reshape((12,2))
	nx, ny = N, N
	ds = DiskSet(12, N, N)
	gvecs = [ [1,0], [0,1], [1,1], [1,-1], [-1,1], [-1,-1], [2,-1], [-2,1], [1,-2], [-1,2], [-1,0], [0,-1]]
	g1 = np.array([ 0, 2/np.sqrt(3)])
	g2 = np.array([-1, 1/np.sqrt(3)])
	gvecs = np.array(gvecs)
	ndisks=12
	for disk in range(ndisks):
		I = np.zeros((nx, ny))
		for i in range(nx):
			for j in range(ny):
				g = gvecs[disk][0] * g1 + gvecs[disk][1] * g2
				ds.set_useflag(disk, True)
				ds.set_x( disk, g[1] )
				ds.set_y( disk, g[0] )
				gdotu = np.dot(g, np.array([uvecs_cart[i,j,1], uvecs_cart[i,j,0]]))
				I[i,j] = np.cos(np.pi * gdotu)**2 
		ax[disk, 1].imshow(I)
		ax[disk, 1].set_title('{}{} from cart'.format(gvecs[disk][0], gvecs[disk][1]))
		ds.set_df(disk, I)
	plt.show()	

	ds.get_rotatation(sanity_plot=True, savepath='../data/test_g.png', printing=True)

	savepath = '../data/diskset.pkl'
	with open(savepath, 'wb') as f: 
		pickle.dump( ds, f )
	exit()

	savepath = '../data/fakeuvecs.pkl'
	with open(savepath, 'wb') as f: 
		pickle.dump( [uvecs_rz], f )
	exit()

	N = 10
	dux_dx, dux_dy, duy_dx, duy_dy = strain_u_XY_grid(Transformation, theta_layer1, N, plotflag = True)
	test_rotcor(dux_dx, dux_dy, duy_dx, duy_dy, theta_layer1, theta_layer2)

main_tlg_ft_test()
