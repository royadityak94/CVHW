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
    original_img = img.astype('float')
    img_min, img_max = np.min(original_img.ravel()), np.max(original_img.ravel())
    return (original_img - img_min) / (img_max - img_min)

def fspecial_log(p2, std):
    siz = int((p2-1)/2)
    x = y = np.linspace(-siz, siz, 2*siz+1)
    x, y = np.meshgrid(x, y)
    arg = -(x**2 + y**2) / (2*std**2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h/h.sum() if h.sum() != 0 else h
    h1 = h*(x**2 + y**2 - 2*std**2) / (std**4)
    return h1 - h1.mean()


def laplacian_of_gaussian_filter(sigma):
    kernel_size = np.round(3*sigma)#np.round(6*sigma)
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
        #print('Convolving with sigma={}'.format(sigma[i]))
        kernel=laplacian_of_gaussian_filter(sigma[i])
        convolved_image=convolve(gray_image,kernel)
        #cv2.imshow("LoG Convolved Image with sigma={}".format(sigma[i]),convolved_image)
        scale_space[:,:,i] = np.square(convolved_image)
        sigma[i+1]=sigma[i]*sigma_scale_factor
    return scale_space, sigma


def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 5 array with blob in each row in (x, y, radius, angle, score)
    #
    # Dummy - returns a blob at the center of the image
    processed_im = py_im2double(rgb2gray(im))
    initial_sigma = 1.4 #1.6
    k = np.sqrt(2) #initial_scale
    sigma_scale_factor = np.sqrt(1.7)
    n_iterations = 15
    level = 15 #10-15
    threshold_factor = .003
    h, w = processed_im.shape
    scale_space = np.zeros((h, w, level))
    scale_space, sigma = create_scale_space(processed_im,sigma_scale_factor, initial_sigma,level)
    max_scale_space = np.copy(scale_space)
    mask = [0] * (level)
    index = [(1, 0), (-1, 0), (0, 1), (0, -1), 
             (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for i in range(0, level):
        mask[i]=int(np.ceil(np.sqrt(2)*sigma[i]))
    size = np.shape(scale_space[:,:,0])

    def check(l):
        return all(scale_space[i + dx, j + dy, l] < scale_space[i, j, k] 
           for dx, dy in index 
           if  0<= i + dx < size[0] and 0<= j + dy <size[1])

    blob_location =[]
    for k in range(0,level):
        max_scale_space[:mask[k],:mask[k],k] = 0
        max_scale_space[-mask[k]:,-mask[k]:,k] = 0
        for i in range(mask[k]+1,size[0]-mask[k]-1):
            for j in range(mask[k]+1,size[1]-mask[k]-1):
                if scale_space[i, j, k] < threshold_factor:
                    continue
                c_max = check(k)
                l_max = u_max = True
                if k - 1 >= 0:
                    l_max = check(k - 1) and \
                    scale_space[i, j, k - 1] < scale_space[i, j, k]
                if k + 1 < level:
                    u_max = check(k + 1) and \
                    scale_space[i, j, k + 1] < scale_space[i, j, k]
                if c_max and l_max and u_max:
                    max_scale_space[i, j, k] = 1
                    blob_location.append((i, j, k, scale_space[i, j, k]))

    blobs = np.zeros((len(blob_location), 5))
    i = 0
    for center in blob_location:
        x, y = center[0], center[1]
        radius = int(np.ceil(np.sqrt(2)*sigma[center[2]])) 
        score = center[3]
        blobs[i] = [y, x, radius, -1, score] #(x, y, radius, angle, score)
        i+=1
    return np.array(blobs)
