import requests
import sunpy.map
import matplotlib.pyplot as plt
from sunpy.net import Fido, attrs as a

# 定义URL和文件名
_url = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/suvi-l2-ci195/2023/03/26/"
_file = "dr_suvi-l2-ci195_g18_s20230326T000000Z_e20230326T000400Z_v1-0-2.fits"

# 本地数据文件夹
local_data_dir = 'E:/Research/Data/GOES/'

# 下载FITS文件 到 本地数据文件夹
response = requests.get(_url+_file) # Fido.fetch(a.file(_url + _file), path='./')
open(local_data_dir + _file, "wb").write(response.content)
download_filename = local_data_dir + _file

# 读取FITS文件并创建SunPy Map
goes_map = sunpy.map.Map(download_filename)

# 绘制太阳图像
fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(projection=goes_map)
goes_map.plot(axes=ax1)
plt.colorbar()

# 显示图像
plt.show()