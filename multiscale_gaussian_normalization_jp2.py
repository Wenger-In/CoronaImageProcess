import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunkit_image.enhance as enhance
import cv2

###########################################################################
# Read LASCO C2 raw fits to get colormaps
raw_file = 'C:/Users/rzhuo/sunpy/data/20210117/lasco_c2/22799618.fts'
raw_map = sunpy.map.Map(raw_file)
cmap = raw_map.cmap

# Read LASCO C2 jp2 images processed by Helioviewer.org
jp2_path = 'E:/Research/Data/HelioViewer/LASCO_C2/20210117/'
jp2_file = jp2_path + '2021_01_17__00_00_07_509__SOHO_LASCO_C2_white-light.jp2'
jp2_image = cv2.imread(jp2_file, cv2.IMREAD_UNCHANGED)

# Save jp2 images as jpg images
jpg_file = jp2_file[:-4] + '.jpg'
cv2.imwrite(jpg_file, jp2_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
jpg_image = cv2.imread(jpg_file, cv2.IMREAD_UNCHANGED)
jpg_image = jpg_image.astype(float, copy=False)

###########################################################################
# Plot jpg images
plt.figure()
plt.imshow(jpg_image,cmap=cmap)
plt.colorbar()
plt.title('Helioviewer')

# Apply Mulitscale Guassian Normalization
mgn_image = enhance.mgn(jpg_image,sigma=[1.25, 2.5, 5, 10, 20],weights=[0.907,0.976,1,1,1], k=0.7, gamma=1, h=0.9)

plt.figure()
plt.imshow(mgn_image,cmap=cmap,vmin=0,vmax=0.8)
plt.colorbar()
plt.title('MGN')

plt.show()