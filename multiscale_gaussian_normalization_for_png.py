import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunkit_image.enhance as enhance
import cv2
import os

save_or_not = 0

png_path = 'E:/Research/Data/STEREO/COR2/'
save_path = 'E:/Research/Work/[else]/corona_image_process/STEREO/'

# Select observation slit
center_x = 512
center_y = 512
radius = 300
dtheta = 0.005
x_slit = []
y_slit = []

for theta in np.arange(0, 2*np.pi, dtheta):
    x = int(center_x - radius * np.sin(theta))
    y = int(center_y - radius * np.cos(theta))
    if not x_slit or not (x == x_slit[-1] and y == y_slit[-1]):
        x_slit.append(x)
        y_slit.append(y)

# Iterate through .png files
slit_image = []
for root, dirs, files in os.walk(png_path):
    for file in files:
        if file.endswith('.png'):
            png_name = file
            # Read LASCO C2 png images processed by Helioviewer.org
            png_file = png_path + png_name
            png_image = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
            png_image = cv2.cvtColor(png_image, cv2.COLOR_BGR2GRAY)
            png_image = png_image.astype(float, copy=False)

            # Plot png images
            plt.figure(figsize=(12,8))
            plt.imshow(png_image,cmap='hot')
            plt.colorbar()
            plt.title('Raw@' + png_name[:15])

            if save_or_not == 1:
                plt.savefig(save_path + 'raw/' + png_name[:15] + 'raw.jpg', format='jpg')
            
            plt.close()

            # Apply Mulitscale Guassian Normalization
            mgn_image = enhance.mgn(png_image,sigma=[1.25, 2.5, 5, 10, 20],weights=[0.907,0.976,1,1,1], k=0.7, gamma=1, h=0.8)
            
            # Extract slit observations
            slit_pixels = [mgn_image[y, x] for x, y in zip(x_slit, y_slit)]
            slit_pixels = np.array(slit_pixels)
            slit_pixels = slit_pixels.reshape(-1,1)
            slit_pixels = np.flipud(slit_pixels)
            slit_image.append(slit_pixels)

            # Plot MGN images with selected slit
            plt.figure(figsize=(12,8))
            plt.imshow(mgn_image,cmap='hot',vmin=0,vmax=0.8)
            plt.colorbar()
            plt.title('MGN@' + png_name[:20])

            if save_or_not == 1:
                plt.savefig(save_path + 'mgn/' + png_name[:20] + 'mgn.jpg', format='jpg')

            plt.scatter(x_slit,y_slit,c='k')
            # plt.show()
            plt.close()

slit_image_array = np.column_stack(slit_image)

plt.figure(figsize=(12,8))
plt.pcolor(slit_image_array,cmap='seismic',vmin=0,vmax=0.8)
plt.colorbar()
xlabel('Frame')
plt.title('slit observation')

# Split when there is no LASCO C2 data
split_1, split_2 = 15, 27

split_array = np.hsplit(slit_image_array, [split_1, split_2])
nan_array_1 = np.full((1257, 5), np.nan) # 24
nan_array_2 = np.full((1257, 5), np.nan) # 34

# Insert nan arrays
slit_image_array_1 = np.hstack([split_array[0], nan_array_1])
slit_image_array_2 = split_array[1]
slit_image_array_3 = np.hstack([nan_array_2, split_array[2]])
slit_image_merged = np.hstack([slit_image_array_1, slit_image_array_2, slit_image_array_3])

plt.figure(figsize=(12,8))
plt.pcolor(slit_image_merged,cmap='seismic',vmin=0.25,vmax=0.5)
plt.colorbar()
plt.title('slit observation merged')

plt.show()