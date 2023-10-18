import py4DSTEM
if py4DSTEM.__version__ != '0.11.5':
	print('WARNING: you are using py4DSTEM version {}'.format(py4DSTEM.__version__))
	print('please use py4DSTEM version 0.11.5')
	print("type 'pip install py4DSTEM==0.11.5' in the virtual environment you're using")

def plot_single_pix_diffraction(f, scan_shape0, scan_shape1):
	data = py4DSTEM.io.read(f, data_id="datacube_0")
	data.set_scan_shape(scan_shape0, scan_shape1)
	x,y = 125,125
	avg_dp = data.data[y,x,:,:]
	f, ax = plt.subplots()
	ax.imshow(avg_dp, cmap='plasma')
	plt.show()

# moved, faster if not in storage...
#f ='/media/pc4dstem/072064ad-58f9-4428-972c-44ac46190847/4dstem-data/tlg/20220906/TZKG-A5/13_250x250_0p5nm_180rot_full_4x_0p0133s_CL800_10umC2_mono8p69_80kV_1p71alpha/dp.h5'
f = 'dp.h5'
plot_single_pix_diffraction(f, 250, 250)