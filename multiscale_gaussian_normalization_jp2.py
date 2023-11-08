import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunkit_image.enhance as enhance
import cv2
import os

save_or_not = 0

jp2_path = 'E:/Research/Data/HelioViewer/LASCO_C2/20210117/'
# jp2_path = 'C:/Users/rzhuo/sunpy/data/'
save_path = 'E:/Research/Work/[else]/corona_image_process/'

# Read LASCO C2 raw fits to get colormaps
raw_file = 'C:/Users/rzhuo/sunpy/data/20210117/lasco_c2/22799618.fts'
raw_map = sunpy.map.Map(raw_file)
cmap = raw_map.cmap

# iterate through .jp2 files
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
            plt.figure(figsize=(12,8))
            plt.imshow(jpg_image,cmap=cmap)
            plt.colorbar()
            plt.title('Helioviewer@' + jp2_name[:21])
            
            plt.close()

            if save_or_not == 1:
                plt.savefig(save_path + 'raw/' + jp2_name[:21] + 'raw.jpg', format='jpg')

            # Apply Mulitscale Guassian Normalization
            mgn_image = enhance.mgn(jpg_image,sigma=[1.25, 2.5, 5, 10, 20],weights=[0.907,0.976,1,1,1], k=0.7, gamma=1, h=0.9)

            plt.figure(figsize=(12,8))
            plt.imshow(mgn_image,cmap=cmap,vmin=0,vmax=0.8)
            plt.colorbar()
            plt.title('MGN@' + jp2_name[:21])

            if save_or_not == 1:
                plt.savefig(save_path + 'mgn/' + jp2_name[:21] + 'mgn.jpg', format='jpg')

            # plt.show()
            plt.close()