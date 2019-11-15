# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import sys

def py_im2double(img):
    grayscaled = rgb2gray(img)
    return grayscaled
#     original_img = grayscaled.astype('float')
#     img_min, img_max = np.min(original_img.ravel()), np.max(original_img.ravel())
#     return (original_img - img_min) / (img_max - img_min)

def laplacian_of_gaussian_filter(sigma):
    kernel_size = np.round(4*sigma)
    if kernel_size % 2 == 0:
        kernel_size+=1
    half_size=np.floor(kernel_size/2)
    x, y = np.meshgrid(np.arange(-half_size, half_size+1), np.arange(-half_size, half_size+1))
    
    exp_term=np.exp(-(x**2+y**2) / (2*sigma**2))
    exp_term[exp_term < sys.float_info.epsilon * exp_term.max()] = 0
    if exp_term.sum() != 0:
        exp_term = exp_term/exp_term.sum() 
    else: 
        exp_term
    kernel = -((x**2 + y**2 - (2*sigma**2)) / sigma**2) * exp_term 
    kernel=kernel-kernel.mean()
    return kernel

def create_scale_space(gray_image,sigma_scale_factor,initial_sigma,level):
    h,w=np.shape(gray_image)
    scale_space = np.zeros((h,w,level),np.float32)
    sigma = [0]*(level+1)
    sigma[0] = initial_sigma
    for i in range(0,level):
        print('Convolving with sigma={}'.format(sigma[i]))
        kernel=laplacian_of_gaussian_filter(sigma[i])
        convolved_image=convolve(gray_image,kernel)
        scale_space[:,:,i] = np.square(convolved_image)
        sigma[i+1]=sigma[i]*sigma_scale_factor
    return scale_space, sigma

def max_scaled_spaces(scale_spaces, i, j, l1, l2):
    h, w = scale_spaces[:, :, 0].shape
    search_over = [(0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0)]
    flag = True
    for _i, _j in search_over:
        if (i+_i in range(h)) and (j+_j in range(w)):
            if scale_spaces[i+_i, j+_j, l2] >= scale_spaces[i, j, l1]:
                flag = False    
    return flag

def non_max_suppression_blobs(scale_spaces, scaling, sigma, level_, cutoff=0.003):
    scale_spaces_max = scale_spaces.copy()
    h, w = scale_spaces_max[:, :, 0].shape
    kernel = [int(np.ceil(s)) for s in sigma]
    blob_location = []
    for le in range(0, level_):
        curr = kernel[le]
        print ("Iterating for Level={}".format(le))
        scale_spaces_max[-curr:, -curr:, le] = 0
        scale_spaces_max[:curr, :curr, le] = 0
        for i in range(curr+1, (h - curr - 1)):
            for j in range(curr+1, (w - curr - 1)):
                if scale_spaces[i, j, le] < cutoff:
                    continue
                curr_flag = max_scaled_spaces(scale_spaces, i, j, le, le)
                lower_flag = (le > 0) and scale_spaces[i, j, le-1] < scale_spaces[i, j, le] and max_scaled_spaces(scale_spaces, i, j, le, le-1)
                upper_flag = (le < level_-1) and scale_spaces[i, j, le+1] < scale_spaces[i, j, le]  and max_scaled_spaces(scale_spaces, i, j, le, le+1) 
                if curr_flag and lower_flag and upper_flag: #and lower_flag
                    blob_location.append([i, j, le, scale_spaces[i, j, le]])
                    scale_spaces_max[i, j, le] = 1   
    blobs = np.zeros((len(blob_location), 5))
    i = 0
    for item in blob_location:
        x, y = item[0], item[1]
        radius = sigma[item[2]]
        score = item[3]
        blobs[i] = [y, x, radius, -1, score] #(x, y, radius, angle, score)
        i+=1
    return blobs

def detectBlobs(im, param={}):
    #Default Params
    default_cutoff, default_sigma, default_level, default_scaling = 0.002, 1.2, 14, np.sqrt(1.8)
    cutoff = param.get("cutoff") if param.get("cutoff") is not None else default_cutoff
    sigma_seed = param.get("sigma_seed") if param.get("sigma_seed") is not None else default_sigma
    level = param.get("level") if param.get("level") is not None else default_level
    scaling = param.get("scaling") if param.get("scaling") is not None else default_scaling
    
    processed_im = py_im2double(rgb2gray(im))
    scale_space, sigma = create_scale_space(processed_im, scaling, sigma_seed, level)
    blobs = non_max_suppression_blobs(scale_space, scaling, sigma, level, cutoff)
    return blobs