import numpy as np
from scipy import fftpack,signal
from scipy.interpolate import interp2d, griddata

def build_atrous_coef(j, method):
    if method == 'B_spline':
        if j == 0:
            atrous_coef = np.matrix([1/16.0, 1/4.0, 3/8.0, 1/4.0, 1/16.0])
            return np.array(np.matmul(atrous_eff.T,atrous_eff))
        elif j >= 1:
            len_atrous_eff = 5 + 2**(j+1)
            atrous_coef = np.zeros(len_atrous_eff)
            atrous_coef[0] = 1/16.0
            atrous_coef[int(2**(j-1)+1)] = 1/4.0
            atrous_coef[int((len_atrous_eff-1)/2)] = 3/8.0
            atrous_coef[int(3*2**(j-1)+3)] = 1/4.0
            atrous_coef[-1] = 1/16.0
            atrous_coef = np.matrix(atrous_coef)
            return np.array(np.matmul(atrous_coef.T, atrous_coef))
        else:
            print("Please input a positive integer")
    elif method == 'linear':
        if j == 0:
            atrous_coef = np.matrix([0.25,0.5,0.25])
            return np.array(np.matmul(atrous_eff.T,atrous_eff))
        elif j >= 1:
            len_atrous_eff = 3 + 2**j
            atrous_coef = np.zeros(len_atrous_eff)
            atrous_coef[0] = 0.25
            atrous_coef[int((len_atrous_eff-1)/2)] = 0.5
            atrous_coef[-1] = 0.25
            atrous_coef = np.matrix(atrous_coef)
            return np.array(np.matmul(atrous_coef.T, atrous_coef))
        else:
            print("Please input a positive integer")
    else:
        print("choose a method: 'B_spline' or 'linear'")
        
def a_trous_wavelet_2D(input_data, level_num, method='B_spline'):
    if method == 'B_spline':
        if level_num == 1:
            len_atrous_coef = 5
        elif level_num >= 2:
            len_atrous_coef = 5 + 2**(level_num)
        else:
            print("Please input a positive integer")
    elif method == 'linear':
        if level_num == 1:
            len_atrous_coef = 3
        elif level_num >= 2:
            len_atrous_coef = 3 + 2**(level_num-1)
        else:
            print("Please input a positive integer")
    else:
        print("choose a method: 'B_spline' or 'linear'")
        return 0
        
    data_shape = np.shape(input_data)

    if np.min(data_shape) > len_atrous_coef:
        output_c = np.empty(shape=(data_shape[0],data_shape[1],level_num))
        wavelet_coef = np.empty(shape=(data_shape[0],data_shape[1],level_num))
        output_c[:,:,0] = np.copy(input_data)
        for i in range(1,level_num):
            output_c[:,:,i] = signal.convolve(input_data, build_atrous_coef(i, method), mode = "same")
        wavelet_coef[:,:,:-1] = -np.diff(output_c, axis=2)
        wavelet_coef[:,:,-1] = np.copy(output_c[:,:,-1])
        return wavelet_coef
    else:
        print('Please reduce the number of level')

def interp_to_slit(image, x_slit, y_slit):
    # Construct Cartesian Coordinate
    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    X, Y = np.meshgrid(x, y)

    # # Interpolate with griddata
    # interp_array = griddata((X.flatten(), Y.flatten()), image.flatten(), (x_slit, y_slit), method='linear')
    # slit_pixels = interp_array.reshape(x_slit.shape)
    
    # Interpolate with interp2d
    interpfun = interp2d(x, y, image, kind='linear')
    interp_matrix = interpfun(x_slit, y_slit)
    interp_array = np.diagonal(interp_matrix)
    slit_pixels = interp_array.reshape(x_slit.shape)
    
    return slit_pixels

def radial_slit(beg_point, end_point, min_x, min_y, step=0.1):
    beg_x, beg_y = beg_point
    end_x, end_y = end_point
    direct = np.array([end_x - beg_x, end_y - beg_y])
    distance = np.linalg.norm(direct)
    e_direct = direct / distance
    x_slit, y_slit = [beg_x], [beg_y]
    while True:
        new_x = x_slit[-1] + step * e_direct[0]
        new_y = y_slit[-1] + step * e_direct[1]
        if new_x < min_x or new_y < min_y:
            break
        x_slit.append(new_x)
        y_slit.append(new_y)
    return np.array(x_slit), np.array(y_slit)

def insert_nan_columns(image, insertions):
    for col, num_insertions in insertions:
        for _ in range(num_insertions):
            image = np.insert(image, col, np.nan, axis=1)
            col += 1
    return image