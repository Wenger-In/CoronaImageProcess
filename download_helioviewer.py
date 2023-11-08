from sunpy.io import read_file
from sunpy.map import Map
from sunpy.net import helioviewer
from sunpy.net import Fido
from sunpy.net import attrs as a
import datetime as dt
import matplotlib.pyplot as plt

date = dt.datetime(2011,1,14,12,0)

fig = plt.figure()

# Helioviewer image
hv = helioviewer.HelioviewerClient()
file = hv.download_jp2(date, observatory='SOHO', instrument='LASCO', detector='C2')
data, header = read_file(file)[0]

print(header['CROTA2']) # -> -173.555

map = Map(data, header)

ax1 = fig.add_subplot(1,2,1,projection=map)
map.plot(axes=ax1)

# Vso image
result = Fido.search(a.Time(date, date + dt.timedelta(minutes=10)), a.Instrument.lasco, a.Detector.c3)
downloaded_files = Fido.fetch(result)

data, header = read_file(downloaded_files[0])[0]
print(header['CROTA2']) # -> -173.554
map2 = sunpy.map.Map(data, header)
ax2 = fig.add_subplot(1,2,2,projection=map2)
map2.plot(axes=ax2)

plt.show()