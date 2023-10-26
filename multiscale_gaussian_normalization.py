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

###########################################################################
# SunPy sample data contains a number of suitable images, which we will use here.

# result = Fido.search(a.Time('2011/05/04 00:00', '2011/05/04 00:05'),
#                      a.Instrument.aia)
path = 'C:/Users/rzhuo/sunpy/data/example/'
# lasco_file = Fido.fetch(result,path=path)
lasco_file = path + 'aia_lev1_171a_2011_05_04t00_00_00_34z_image_lev1.fits'
# lasco_file = path + 'solo_L1_eui-fsi174-image_20210117T000000168_V05.fits'
# lasco_file = path + '22799655.fts'
lasco_map = sunpy.map.Map(lasco_file)

# The original image is plotted to showcase the difference.
fig = plt.figure()
ax0 = fig.add_subplot(projection=lasco_map)
lasco_map.plot()
ax0.set_title('Original')

# plt.show()

###########################################################################
# Applying Multi-scale Gaussian Normalization on a solar image.

mgn_lasco = enhance.mgn(lasco_map.data.astype(float, copy=False), sigma=[1.25, 2.5, 5, 10, 20, 40], weights=[0.907,0.976,1,1,1], k=0.8, gamma=2.5, h=0.7) # aia 171
# mgn_lasco = enhance.mgn(lasco_map.data.astype(float, copy=False), sigma=[5,6,7,8,9,10,12,20], k=0.7, gamma=0.5, h=0.5) # lasco c2
# mgn_lasco = enhance.mgn(lasco_map.data.astype(float, copy=False), sigma=[0.625,1.25,2.5,5,10],weights=[0.599,0.907,0.976,1,1], k=0.7, gamma=0.5, h=0.5) # eui fsi
mgn_lasco = sunpy.map.Map(mgn_lasco, lasco_map.meta)

fig = plt.figure()
ax1 = fig.add_subplot(projection=mgn_lasco)
mgn_lasco.plot()
ax1.set_title('MGN')

###########################################################################
# Then we add a mask on MGN result.

# pixel_coords = all_coordinates_from_map(mgn_lasco)
# solar_center = SkyCoord(0*u.deg, 0*u.deg, frame=mgn_lasco.coordinate_frame)
# pixel_radii = np.sqrt((pixel_coords.Tx-solar_center.Tx)**2 +
#                       (pixel_coords.Ty-solar_center.Ty)**2)
# mask_inner = pixel_radii < mgn_lasco.rsun_obs*2.4
# mask_outer = pixel_radii > mgn_lasco.rsun_obs*6
# final_mask = mask_inner + mask_outer

# masked_lasco = sunpy.map.Map(mgn_lasco.data, mgn_lasco.meta, mask=final_mask)

# fig = plt.figure()
# ax2 = fig.add_subplot(projection=masked_lasco)
# masked_lasco.plot(axes=ax2)
# masked_lasco.draw_limb()
# ax2.set_title("Masked MGN LASCO C2")

plt.show()
