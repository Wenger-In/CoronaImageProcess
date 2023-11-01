import numpy as np
import matplotlib.pyplot as plt
import sunpy.data.sample
import sunpy.map

import sunkit_image.enhance as enhance

import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map.maputils import all_coordinates_from_map
from sunpy.net import Fido
from sunpy.net import attrs as a

import matplotlib.cm as cm
import matplotlib.colors as mcolors

###########################################################################
download = 0
instrument = 'lasco_c2'
path = 'C:/Users/rzhuo/sunpy/data/20210117/'+instrument+'/'

if download == 1:
    result = Fido.search(a.Time('2021/01/17 00:00', '2021/01/17 00:30'),
                         a.Instrument.eui,
                         a.Level(2))
    image_file = Fido.fetch(result,path=path)
else:
    if instrument == 'lasco_c2':
        image_file = path + '22799655.fts'
    elif instrument == 'eui_fsi':
        image_file = path + 'solo_L2_eui-fsi174-image_20210117T000000168_V05.fits'
    elif instrument == 'aia':
        path = 'C:/Users/rzhuo/sunpy/data/example/'
        image_file = path + 'aia_lev1_171a_2011_05_04t00_00_00_34z_image_lev1.fits'
image_map = sunpy.map.Map(image_file)

# The original image is plotted to showcase the difference.
fig = plt.figure()
ax0 = fig.add_subplot(projection=image_map)
image_map.plot()
plt.colorbar()
ax0.set_title('Raw')

###########################################################################
# Applying Multi-scale Gaussian Normalization on a solar image.

if instrument == 'lasco_c2':
    mgn_image = enhance.mgn(image_map.data.astype(float, copy=False),
                            sigma=[1.25, 2.5, 5, 10, 20, 40],weights=[0.907,0.976,1,1,1], k=0.8, gamma=1, h=0.9)
    vmin, vmax = -0.1, 1.1
elif instrument == 'eui_fsi':
    mgn_image = enhance.mgn(image_map.data.astype(float, copy=False), 
                            sigma=[1.25,2.5,5,10], weights=[0.907,0.976,1,1], k=0.7, gamma=2, h=0.9)
    vmin, vmax = 0.07, 1
elif instrument == 'aia':
    mgn_image = enhance.mgn(image_map.data.astype(float, copy=False), 
                            sigma=[1.25, 2.5, 5, 10, 20, 40],weights=[0.907,0.976,1,1,1], k=0.7, gamma=3.2, h=0.7)
    vmin, vmax = -0.01, 1
mgn_image = sunpy.map.Map(mgn_image, image_map.meta)

fig = plt.figure()
ax1 = fig.add_subplot(projection=mgn_image)
mgn_image.plot()
plt.colorbar()
ax1.set_title('MGN')

###########################################################################
# Then we add a mask on MGN result.

if instrument == 'lasco_c2':
    pixel_coords = all_coordinates_from_map(mgn_image)
    solar_center = SkyCoord(0*u.deg, 0*u.deg, frame=mgn_image.coordinate_frame)
    pixel_radii = np.sqrt((pixel_coords.Tx-solar_center.Tx)**2 +
                        (pixel_coords.Ty-solar_center.Ty)**2)
    mask_inner = pixel_radii < mgn_image.rsun_obs*2.4
    mask_outer = pixel_radii > mgn_image.rsun_obs*6
    final_mask = mask_inner + mask_outer

    masked_image = sunpy.map.Map(mgn_image.data, mgn_image.meta, mask=final_mask)

    fig = plt.figure()
    ax2 = fig.add_subplot(projection=masked_image)
    masked_image.plot(axes=ax2)
    masked_image.draw_limb()
    plt.colorbar()
    ax2.set_title('Masked MGN')

plt.show()