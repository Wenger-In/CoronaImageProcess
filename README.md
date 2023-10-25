# CoronaImageProcess

This is a repository about Corona Image Process. It contains: 

-- multiscale_gaussian_normalization.py

Based on 【金山文档】 (Morgan)(SoP-2014) Multi-Scale Gaussian Normalization for Solar Image Processing https://kdocs.cn/l/cjTKcX8CWohu. The fundamental code is provided by sunkit-image/multiscale_gaussian_normalization.html.

One should note that variables 'g_i' in Morgan(2014) are all set to be '1' in source code enhance.mgn(). If the Guassian Core used is small than 3, one should feed 'g_i' into enhance.mgn(), according the list or calculating method provided in Morgan(2014).
