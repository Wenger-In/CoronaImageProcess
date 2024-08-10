import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunkit_image.enhance as enhance
import cv2
import os
import sunpy
from scipy import fftpack, signal

import CoronaImageProcess_utils
from CoronaImageProcess_utils import build_atrous_coef, a_trous_wavelet_2D

save_or_not = 1

# preparation
'''HelioViewer JP2 data download url: https://helioviewer.org/jp2/'''

# data path 
jp2_path = 'E:/Research/Data/HelioViewer/LASCO_C2/20211004/'
save_path = 'E:/Research/Work/tianwen_IPS/LASCO_obs/20211004/'

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

            # Plot jpg images
            plt.figure(figsize=(8,6))
            plt.imshow(jpg_image,cmap='soholasco2')
            plt.colorbar()
            plt.title('Helioviewer-' + jp2_name[:20])

            if save_or_not == 1:
                plt.savefig(save_path + 'raw/' + jp2_name[:20] + '_raw.jpg', format='jpg')
                plt.close()
            
            ########## Apply Mulitscale Guassian Normalization only ##########
            mgn_image = enhance.mgn(jpg_image,sigma=[1.25, 2.5, 5, 10, 20],weights=[0.907,0.976,1,1,1], k=0.8, gamma=1, h=0.9)
            
            # Plot MGN images
            plt.figure(figsize=(8,6))
            plt.imshow(mgn_image,cmap='soholasco2',vmin=0,vmax=0.8) # for C2
            plt.colorbar()
            plt.title('MGN-' + jp2_name[:20])
            
            if save_or_not == 1:
                plt.savefig(save_path + 'mgn/' + jp2_name[:20] + '_mgn.jpg', format='jpg')
                plt.close()

            ########## Apply A-Trous Wavelet only ##########
            a_trous_median_filter = signal.medfilt2d(jpg_image, kernel_size=5)
            output_w = a_trous_wavelet_2D(a_trous_median_filter, level_num=4, method='B_spline')
            a_trous_image = np.sum(output_w[:,:,1:-1], axis=2)

            # Plot a-trous images
            plt.figure(figsize=(8,6))
            plt.imshow(a_trous_image,cmap='soholasco2',vmin=-3,vmax=3)
            plt.colorbar()
            plt.title('A-Trous-Wavelet-' + jp2_name[:20])

            if save_or_not == 1:
                plt.savefig(save_path + 'a_trous/' + jp2_name[:20] + '_a_trous.jpg', format='jpg')
                plt.close()
            
            ########## Apply A-Trous Wavelet upon MGN ##########
            mgn_a_trous_median_filter = signal.medfilt2d(mgn_image, kernel_size=5)
            output_w = a_trous_wavelet_2D(mgn_a_trous_median_filter, level_num=4, method='B_spline')
            mgn_a_trous_image = np.sum(output_w[:,:,:-1], axis=2)

            # Plot mgn-a-trous images
            plt.figure(figsize=(8,6))
            plt.imshow(mgn_a_trous_image,cmap='soholasco2',vmin=-0.04,vmax=0.05)
            plt.colorbar()
            plt.title('A-Trous-Wavelet-upon-MGN-' + jp2_name[:20])

            if save_or_not == 1:
                plt.savefig(save_path + 'mgn_a_trous/' + jp2_name[:20] + '_mgn_a_trous.jpg', format='jpg')
                plt.close()
            
            ######### Apply MGN upon A-Trous Wavelet ##########
            a_trous_mgn_image = enhance.mgn(a_trous_image,sigma=[1.25, 2.5, 5, 10, 20],weights=[0.907,0.976,1,1,1], k=0.8, gamma=1, h=0.9)
                        
            # Plot a-trous-mgn images
            plt.figure(figsize=(8,6))
            plt.imshow(a_trous_mgn_image,cmap='soholasco2',vmin=-0.2,vmax=0.2)
            plt.colorbar()
            plt.title('MGN-upon-A-Trous-Wavelet-' + jp2_name[:20])

            if save_or_not == 1:
                plt.savefig(save_path + 'a_trous_mgn/' + jp2_name[:20] + '_a_trous_mgn.jpg', format='jpg')
                plt.close()
            else:
                plt.show()
            
