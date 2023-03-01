
import matplotlib.pyplot as plt
import numpy as np



def draw_cicle(shape,diamiter):
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0
    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)

def filter_circle(TFcircleIN,fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
    temp[TFcircleIN] = fft_img_channel[TFcircleIN]
    return(temp)

def imshow_fft(absfft):
    magnitude_spectrum = 20*np.log(absfft)
    return(ax.imshow(magnitude_spectrum,cmap="gray"))

def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco,(1,2,0))
    return(img_reco)

def main(img):

	shape = img.shape[:2]

	fft_img = np.zeros_like(img,dtype=complex)
	for ichannel in range(fft_img.shape[2]):
	    fft_img[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(img[:,:,ichannel]))

	fft_img_filtered_IN = []
	for ichannel in range(fft_img.shape[2]):
	    fft_img_channel  = fft_img[:,:,ichannel]
	    temp = filter_circle(draw_cicle(shape=img.shape[:2],diamiter=50),fft_img_channel)
	    fft_img_filtered_IN.append(temp)
	    
	fft_img_filtered_IN = np.array(fft_img_filtered_IN)
	fft_img_filtered_IN = np.transpose(fft_img_filtered_IN,(1,2,0))
	abs_fft_img              = np.abs(fft_img)
	abs_fft_img_filtered_IN  = np.abs(fft_img_filtered_IN)

	fig = plt.figure(figsize=(25,18))
	ax  = fig.add_subplot(1,3,1)
	ax.imshow(np.abs(inv_FFT_all_channel(fft_img)))
	ax.set_title("original image")
	ax  = fig.add_subplot(1,3,2)
	ax.imshow(np.abs(inv_FFT_all_channel(fft_img_filtered_IN)))
	ax.set_title("low pass filter image")
	plt.show()

