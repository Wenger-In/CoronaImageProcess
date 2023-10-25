import cv2
import numpy as np
import pywt
from pywt import dwt2, idwt2

# 读取图像
image = cv2.imread('C:/Users/rzhuo/Desktop/images_512_aheadXcor2/20210116_000915_n4c2A.jpg', 0)

# 对img进行haar小波变换：
cA,(cH,cV,cD)=dwt2(image,'haar')
 
# 小波变换之后，低频分量对应的图像：
cv2.imwrite('lena.png',np.uint8(cA/np.max(cA)*255))
# 小波变换之后，水平方向高频分量对应的图像：
cv2.imwrite('lena_h.png',np.uint8(cH/np.max(cH)*255))
# 小波变换之后，垂直平方向高频分量对应的图像：
cv2.imwrite('lena_v.png',np.uint8(cV/np.max(cV)*255))
# 小波变换之后，对角线方向高频分量对应的图像：
cv2.imwrite('lena_d.png',np.uint8(cD/np.max(cD)*255))
 
# 根据小波系数重构回去的图像
rimg = idwt2((cA,(cH,cV,cD)), 'haar')
cv2.imwrite('rimg.png',np.uint8(rimg))