import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunkit_image.enhance as enhance
import cv2
import os
import sunpy
from scipy import fftpack, signal

import CoronaImageProcess_utils
from CoronaImageProcess_utils import build_atrous_coef, a_trous_wavelet_2D, radial_slit, interp_to_slit, insert_nan_columns

save_or_not = 1

# preparation
'''HelioViewer JP2 data download url: https://helioviewer.org/jp2/'''

# Select slit
beg_point = (512, 512)
end_point = (126, 294)
min_x = 0
min_y = 0
x_slit, y_slit = radial_slit(beg_point, end_point, min_x, min_y, step=0.5)

# data path 
jp2_path = 'E:/Research/Data/HelioViewer/LASCO_C2/20211004/'
save_path = 'E:/Research/Work/tianwen_IPS/LASCO_obs/20211004/a_trous_slit/' + str(end_point) + '_diff' + '/'

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
            medfilt_image = signal.medfilt2d(jpg_image, kernel_size=5)
            
            # Apply A-Trous Wavelet
            output_w = a_trous_wavelet_2D(medfilt_image, level_num=4, method='B_spline')
            a_trous_image = np.sum(output_w[:,:,1:-1], axis=2)
            
            # if save_or_not == 0:
            #     plt.figure()
            #     plt.subplot(2,2,1); plt.imshow(output_w[:,:,0], cmap='soholasco2', vmin=-3, vmax=3); plt.colorbar(); plt.title('Component 1')
            #     plt.subplot(2,2,2); plt.imshow(output_w[:,:,1], cmap='soholasco2', vmin=-3, vmax=3); plt.colorbar(); plt.title('Component 2')
            #     plt.subplot(2,2,3); plt.imshow(output_w[:,:,2], cmap='soholasco2', vmin=-3, vmax=3); plt.colorbar(); plt.title('Component 3')
            #     plt.subplot(2,2,4); plt.imshow(output_w[:,:,-1], cmap='soholasco2'); plt.colorbar(); plt.title('Component 4')
            #     plt.show()
            
            # Apply Multiscale Guassian Normalization
            mgn_image = enhance.mgn(jpg_image,sigma=[1.25, 2.5, 5, 10, 20],weights=[0.907,0.976,1,1,1], k=0.8, gamma=1, h=0.9)
            
            # Extract slit pixels
            image4slit = a_trous_image # image for slit observation
            # vmin, vmax = 0, 250 # for raw image
            vmin, vmax = -3, 3 # for a-trous wavelet image
            # vmin, vmax = 0, 0.8 # for mgn image
            slit_pixels = interp_to_slit(image4slit, x_slit, y_slit)
            slit_pixels = slit_pixels.reshape(-1,1)
            slit_image.append(slit_pixels)
            
            # Plot reconstruct images with slit
            plt.figure(figsize=(8,6))
            plt.imshow(image4slit, cmap='soholasco2', vmin=vmin, vmax=vmax)
            plt.plot(x_slit, y_slit, c='k')
            plt.colorbar()
            plt.title('Slit-' + jp2_name[:20])

            if save_or_not == 1:
                plt.savefig(save_path + jp2_name[:20] + '_a_trous_slit.jpg', format='jpg')
                plt.close()
            else:
                plt.show()
            
slit_image = np.column_stack(slit_image)
slit_image = np.flipud(slit_image)

# running difference
slit_image_with_zero = np.insert(slit_image, 0, 0, axis=1)
slit_image_diff = np.diff(slit_image_with_zero, axis=1)
slit_image = slit_image_diff

# insertions = [(5, 2), (13, 1), (36, 1), (41, 1), (60, 2), (68, 1), (93, 1), (95, 2), (103, 2)] # for 20211007
insertions = [(5, 2), (13, 1), (42, 1), (61, 2), (69, 2), (96, 2), (104, 2)] # for 20211004
slit_image_inserted = insert_nan_columns(slit_image, insertions)
x_slit_image_inserted = np.arange(slit_image_inserted.shape[1])
y_slit_image_inserted = np.arange(slit_image_inserted.shape[0])
y_slit_image_inserted = y_slit_image_inserted / 368 * (2 * 700) # [Mm]

plt.figure(figsize=(12,8))
plt.pcolormesh(x_slit_image_inserted, y_slit_image_inserted, slit_image_inserted,cmap='soholasco2',vmin=-1, vmax=1)
plt.xticks(np.arange(0, 120, 10), [f"{hour:02d}:00" for hour in range(0,23,2)])
plt.xlabel('Time [HH:MM]')
plt.ylabel('Solar Distance [Mm]')
plt.colorbar()
plt.title('Slit Observation at ' + str(end_point))

if save_or_not == 1:
    plt.savefig(save_path + 'slit_obs.jpg', format='jpg')

image4enhance = slit_image

# 2D Fourier Transform
f = np.fft.fft2(image4enhance)
fshift = np.fft.fftshift(f)
magni_spectrum = 20 * np.log(np.abs(fshift))
phase_spectrum = np.angle(fshift)

# plot figures
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.pcolormesh(magni_spectrum, cmap='soholasco2')
plt.colorbar(location='bottom')
plt.title('Magnitude-Spectrum of Slit Observation at ' + str(end_point))

plt.subplot(1,2,2)
plt.pcolormesh(phase_spectrum, cmap='soholasco2')
plt.colorbar(location='bottom')
plt.title('Phase-Spectrum of Slit Observation at ' + str(end_point))

if save_or_not == 1:
    plt.savefig(save_path + 'slit_spectrum.jpg', format='jpg')

# High pass filter
rows, cols = image4enhance.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols), np.uint8)
maskrow, maskcol = 5, 5
mask[crow-maskrow:crow+maskrow, ccol-maskcol:ccol+maskcol] = 0
fshift_mask = fshift * mask

# Inverse Fourier Transform
f_ishift = np.fft.ifftshift(fshift_mask)
slit_image_2D_fourier = np.fft.ifft2(f_ishift)
slit_image_2D_fourier = np.real(slit_image_2D_fourier)
slit_image_2D_fourier_inserted = insert_nan_columns(slit_image_2D_fourier, insertions)
image4plot = slit_image_2D_fourier_inserted

plt.figure(figsize=(12,6))
plt.pcolormesh(x_slit_image_inserted, y_slit_image_inserted, image4plot, cmap='soholasco2', vmin=vmin, vmax=vmax)
plt.xticks(np.arange(0, 120, 10), [f"{hour:02d}:00" for hour in range(0,23,2)])
plt.xlabel('Time [HH:MM]')
plt.ylabel('Solar Distance [Mm]')
plt.colorbar()
plt.title('Slit Observation with 2D-fourier at ' + str(end_point))

if save_or_not == 1:
    plt.savefig(save_path + 'slit_obs_2D_fourier.jpg', format='jpg')

output_w = a_trous_wavelet_2D(image4enhance, level_num=4, method='B_spline')
slit_image_a_trous = np.sum(output_w[:,:,1:-1], axis=2)
slit_image_a_trous_inserted = insert_nan_columns(slit_image_a_trous, insertions)
image4plot = slit_image_a_trous_inserted

plt.figure(figsize=(12,6))
plt.pcolormesh(x_slit_image_inserted, y_slit_image_inserted, image4plot, cmap='soholasco2', vmin=-1.5, vmax=1.5)
plt.xticks(np.arange(0, 120, 10), [f"{hour:02d}:00" for hour in range(0,23,2)])
plt.xlabel('Time [HH:MM]')
plt.ylabel('Solar Distance [Mm]')
plt.colorbar()
plt.title('Slit Observation with a-trous-wavelet at ' + str(end_point))

if save_or_not == 1:
    plt.savefig(save_path + 'slit_obs_a_trous.jpg', format='jpg')

plt.show()

db