

#
# ENO-like strain scheme
# picks the "smoothest" stencil to use and then figures out derivative
# candidate stencils derived accounting for fact we have cell averages 
# Q: not sure if matters that beamwidth might be > or < than stepsize...?
# strain computed at the center of each cell. 
# Q: how to compute averages then???

# derivatives for two point stencils of cell averaged quants u = [u0, u1...]
# h is the step size 
# k is the index of u that we want the derivative at (evaluated at center off the cell)
# returns the 'smoothest' derivative using 2 point stencils
def deriv_2pt(u, h):

	# SI of Jiang et al, J Comp Phys, 126 (1996). See equation 3.1
	# simple for two point stencils
	def smoothness(j): return (u[j+1] - u[j]) ** 2

	# derivative, j controls stencil (forward, backward, etc)
	# working with a vector of cell averages not pointwise values so we in general can't
	# use standard finite differences. 
	# find the polynomial a+bx that satisfies:
	# u_j = 1/h * integral from x_j-h/2 to x_j+h/2 of a+bx 
	# u_j+1 = 1/h * integral from x_j+h/2 to x_j+3h/2 of a+bx 
	# this gives you a set of linear equations to solve. 
	# derivative is then just b here
	# works out to standard finite difference formulas in this case. 
	def deriv(j): return (1/h) * (u[j+1] - u[j])

	# j parametrizes the scheme. use points j+1 and j to determine derivative at x_k
	# when k=j, forward difference. when k=j+1, backward difference.
	stencils = [k, k-1] # possible stencils (values of j), here forward and back only.
	SIs = [smoothness(j) for j in stencils]
	best_stencil = stencils[np.argmin(SIs)]
	best_derivative = deriv(best_stencil)
	return best_derivative	

# derivatives for two point stencils of cell averaged quants u = [u0, u1...]
# h is the step size 
# k is the index of u that we want the derivative at (evaluated at center off the cell)
# returns the 'smoothest' derivative using 3 point stencils
def deriv_3pt(u, h):

	# SI of Jiang et al, J Comp Phys, 126 (1996). See equation 3.1
	# NOT ENTIRELY SURE I COMPUTED THIS CORRECTLY WARNING
	def smoothness(j): 
		v1 = (-0.5*u[j+2] + 2*u[j+1] - 1.5*u[j])
		v2 = ( 0.5*u[j+2]   - u[j+1] + 0.5*u[j])
		return v1**2 + (13/12)*v2**2 + 4*v2*(v1*(k-j) + v2*(k-j)**2)

	# derivative, j controls stencil (forward, backward, etc)
	# working with a vector of cell averages not pointwise values so we in general can't
	# use standard finite differences. 
	# find the polynomial a+bx+cx^2 that satisfies:
	# u_j = 1/h * integral from x_j-h/2 to x_j+h/2 of a+bx+c^2
	# u_j+1 = 1/h * integral from x_j+h/2 to x_j+3h/2 of a+bx+cx^2 
	# u_j+2 = 1/h * integral from x_j+3h/2 to x_j+5h/2 of a+bx+cx^2 
	# this gives you a set of linear equations to solve. 
	# derivative at x_k is then just b+2cx_k here:
	#   (k=j+1 -> centered stencil)
	#   (k=j   -> forward  stencil)
	#   (k=j+2 -> backward stencil)
	def deriv(j): # evaluated at midpoint
		coef1 = k - j - 1/2
		coef2 = 2 + 2*j - 2*k
		coef3 = k - j - 3/2
		return (1/h) * (coef1*u[j+2] + coef2*u[j+1] + coef3*u[j])

	# from integrating the interp polynomial a+bx+cx^2 from x_k - h/2 to x_k + h/2 and dividing by h
	def cellavg_deriv(j):
		# same????

	stencils = [k, k-1, k-2] # possible stencils (values of j), here forward, centered, and back only.
	SIs = [smoothness(j) for j in stencils]
	best_stencil = stencils[np.argmin(SIs)]
	best_derivative = deriv(best_stencil)
	return best_derivative	


