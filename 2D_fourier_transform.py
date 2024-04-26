import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunkit_image.enhance as enhance
import cv2
import os
import sunpy
from scipy import fftpack, signal

import CoronaImageProcess_utils
from CoronaImageProcess_utils import build_atrous_coef, a_trous_wavelet_2D, radial_slit, interp_to_slit

save_or_not = 0

# preparation
'''HelioViewer JP2 data download url: https://helioviewer.org/jp2/'''

# data path 
jp2_path = 'E:/Research/Data/HelioViewer/LASCO_C2/20211007/'
save_path = 'E:/Research/Work/tianwen_IPS/LASCO_obs/20211007/'

# Select slit
beg_point = (512, 512)
end_point = (412, 309)
min_y = 0
x_slit, y_slit = radial_slit(beg_point, end_point, min_y, step=0.1)
slit_image = []
for root, dirs, files in os.walk(jp2_path):
    for file in files:
        if file.endswith('.jp2'):
            jp2_name = file            

            # Read LASCO C2 jp2 images processed by Helioviewer.org
            jp2_file = jp2_path + jp2_name
            jp2_image = cv2.imread(jp2_file, cv2.IMREAD_UNCHANGED)

            # Save jp2 images as jpg images
            jpg_file = jp2_file[:-4] + '.jpg'
            cv2.imwrite(jpg_file, jp2_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            jpg_image = cv2.imread(jpg_file, cv2.IMREAD_UNCHANGED)
            jpg_image = jpg_image.astype(float, copy=False)
            
            # Apply A-Trous Wavelet
            a_trous_median_filter = signal.medfilt2d(jpg_image, kernel_size=5)
            output_w = a_trous_wavelet_2D(a_trous_median_filter, level_num=4, method='B_spline')
            a_trous_image = np.sum(output_w[:,:,1:-1], axis=2)
            
            input_image = a_trous_image

            # 2D Fourier Transform
            f = np.fft.fft2(input_image)
            fshift = np.fft.fftshift(f)
            magni_spectrum = 20 * np.log(np.abs(fshift))
            phase_spectrum = np.angle(fshift)

            # High pass filter
            rows, cols = input_image.shape
            crow, ccol = rows//2, cols//2
            mask = np.ones((rows, cols), np.uint8)
            num_mask = 2#5
            mask[crow-num_mask:crow+num_mask, ccol-num_mask:ccol+num_mask] = 0
            fshift = fshift * mask

            # Inverse Fourier Transform
            f_ishift = np.fft.ifftshift(fshift)
            recon_image = np.fft.ifft2(f_ishift)
            recon_image = np.real(recon_image)

            # plot figures
            plt.figure(figsize=(12, 6))

            plt.subplot(1,2,1)
            plt.imshow(magni_spectrum, cmap='soholasco2')
            plt.colorbar(location='bottom')
            plt.title('Magnitude-Spectrum-' + jp2_name[:20])

            plt.subplot(1,2,2)
            plt.imshow(phase_spectrum, cmap='soholasco2')
            plt.colorbar(location='bottom')
            plt.title('Phase-Spectrum-' + jp2_name[:20])
            
            if save_or_not == 1:
                # plt.savefig(save_path + 'a_trous_2D_fourier/' + jp2_name[:20] + '_spectrum.jpg', format='jpg')
                plt.close()
            
            plt.figure(figsize=(8,6))
            plt.imshow(recon_image, cmap='soholasco2', vmin=-3, vmax=3)
            plt.colorbar()
            plt.title('High-Pass-Image-' + jp2_name[:20])
            
            if save_or_not == 1:
                # plt.savefig(save_path + 'a_trous_2D_fourier/' + jp2_name[:20] + '_high_pass.jpg', format='jpg')
                plt.close()
            
            # Extract slit pixels
            slit_pixels = interp_to_slit(recon_image, x_slit, y_slit)
            slit_pixels = slit_pixels.reshape(-1,1)
            slit_image.append(slit_pixels)
            
            # Plot reconstruct images with slit
            plt.figure(figsize=(8,6))
            plt.imshow(recon_image, cmap='soholasco2', vmin=-3, vmax=3)
            plt.plot(x_slit, y_slit, c='k')
            plt.colorbar()
            plt.title('Reconstructed-' + jp2_name[:20])

            if save_or_not == 1:
                plt.savefig(save_path + 'a_trous_2D_fourier_slit/' + str(end_point) + '/' + jp2_name[:20] + '_a_trous_slit.jpg', format='jpg')
                plt.close()
            else:
                plt.show()
            
slit_image_array = np.column_stack(slit_image)
slit_image_array = np.flipud(slit_image_array)

plt.figure(figsize=(12,8))
plt.pcolormesh(slit_image_array,cmap='soholasco2',vmin=-0.75,vmax=0.75)
plt.colorbar()
plt.title('Slit Observation at ' + str(end_point))

if save_or_not == 1:
    plt.savefig(save_path + 'a_trous_2D_fourier_slit/' + 'slit_' + str(end_point) + '.jpg', format='jpg')

plt.show()