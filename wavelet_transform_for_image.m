clc; clear;

I = imread('C:\Users\rzhuo\Desktop\images_512_aheadXcor2\20210116_000915_n4c2A.jpg');

figure(1);
subplot(2,2,1);
imshow(I,[]);
title('原图');
fprintf('原图尺寸为:%f\n',size(I));
fprintf('\n');

% 2维离散小波变换: 小波基函数为haar
[cA,cH,cV,cD] = dwt2(I,'haar');

subplot(2,2,2);
imshow(cA,[]);
title('近似矩阵');
fprintf('近似矩阵尺寸为:%f\n',size(cA));
fprintf('\n');

f_hb = [cA cH;cV cD];
subplot(2,2,3);
imshow(f_hb,[]);
title('4个图合并展示');

% 逆变换的结果: 再得回原图像
I_n = idwt2(cA,cH,cV,cD,'haar');
subplot(2,2,4);
imshow(I_n,[]);
title('逆回原图结果');
fprintf('逆回原图尺寸为:%f\n',size(I_n));

figure(2)
subplot(2,2,1);
imshow(cA,[]);
title('近似矩阵');
subplot(2,2,2);
imshow(cH,[]);
title('水平近似矩阵');
subplot(2,2,3);
imshow(cV,[]);
title('垂向近似矩阵');
subplot(2,2,4);
imshow(cD,[]);
title('对角近似矩阵');