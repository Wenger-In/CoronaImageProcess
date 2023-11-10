import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunkit_image.enhance as enhance
import cv2
import os

save_or_not = 0

jp2_path = 'E:/Research/Data/HelioViewer/LASCO_C2/20210117/'
save_path = 'E:/Research/Work/[else]/corona_image_process/'

# Read LASCO C2 raw fits to get colormaps
raw_file = 'C:/Users/rzhuo/sunpy/data/20210117/lasco_c2/22799618.fts'
raw_map = sunpy.map.Map(raw_file)
cmap = raw_map.cmap

# Select observation slit
center_x = 512
center_y = 512
radius = 400
dtheta = 0.005
x_slit = []
y_slit = []

for theta in np.arange(np.pi/3, 3*np.pi/4, dtheta):
    x = int(center_x - radius * np.sin(theta))
    y = int(center_y - radius * np.cos(theta))
    if not x_slit or not (x == x_slit[-1] and y == y_slit[-1]):
        x_slit.append(x)
        y_slit.append(y)

# Iterate through .jp2 files
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

            # Plot jpg images
            plt.figure(figsize=(12,8))
            plt.imshow(jpg_image,cmap=cmap)
            plt.colorbar()
            plt.title('Helioviewer@' + jp2_name[:21])
            
            plt.close()

            if save_or_not == 1:
                plt.savefig(save_path + 'raw/' + jp2_name[:20] + 'raw.jpg', format='jpg')

            # Apply Mulitscale Guassian Normalization
            mgn_image = enhance.mgn(jpg_image,sigma=[1.25, 2.5, 5, 10, 20],weights=[0.907,0.976,1,1,1], k=0.8, gamma=1, h=0.9)
            
            # Extract slit observations
            slit_pixels = [mgn_image[y, x] for x, y in zip(x_slit, y_slit)]
            slit_pixels = np.array(slit_pixels)
            slit_pixels = slit_pixels.reshape(-1,1)
            slit_pixels = np.flipud(slit_pixels)
            slit_image.append(slit_pixels)

            # Plot MGN images with selected slit
            plt.figure(figsize=(12,8))
            plt.imshow(mgn_image,cmap=cmap,vmin=0,vmax=0.8)
            plt.colorbar()
            plt.scatter(x_slit,y_slit,c='k')
            plt.title('MGN@' + jp2_name[:20])

            if save_or_not == 1:
                plt.savefig(save_path + 'mgn/' + jp2_name[:20] + 'mgn.jpg', format='jpg')

            # plt.show()
            plt.close()

slit_image_array = np.column_stack(slit_image)

plt.figure(figsize=(12,8))
plt.pcolor(slit_image_array,cmap=cmap,vmin=0,vmax=0.8)
plt.colorbar()
plt.title('slit observation')

plt.show()