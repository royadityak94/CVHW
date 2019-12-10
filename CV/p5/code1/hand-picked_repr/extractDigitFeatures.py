import numpy as np
import utils
from utils import loadmat
from joblib import Parallel, delayed
import multiprocessing

# EXTRACTDIGITFEATURES extracts features from digit images
#   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
#   images X of the provided FEATURETYPE. The images are assumed to the of
#   size [W H 1 N] where the first two dimensions are the width and height.
#   The output is of size [D N] where D is the size of each feature and N
#   is the number of images. 

def pixelFeatures(x):
    return feature_normalization(x.reshape(-1), 'L2-Norm')

def feature_normalization(patch, type, epsilon=1e-5):
    patch = patch.astype('longdouble')
    if type == 'Sqrt':
        return np.sqrt(patch)
    elif type == 'L2-Norm':
        return patch / np.sqrt(np.sum(patch ** 2) + epsilon)
    else:
        return -1
    
def extract_relevant_window(arr, patch_size, index_i, index_j):
    curr_arr = np.zeros((patch_size, patch_size))
    for i in range(patch_size):
        for j in range(patch_size):
            curr_arr[i][j] = arr[index_j*patch_size + i][index_i*patch_size + j]     
    return curr_arr

def hogFeatures(x):
    # Applying Non Linear Mapping
    img = np.sqrt(x)

    # Computing the channel gradient
    r_grad, c_grad = np.empty(img.shape).astype('longdouble'), np.empty(img.shape).astype('longdouble')
    
    r_grad[1:-1,] = img[2:, :] - img[:-2, :] 
    c_grad[:, 1:-1] = img[:, 2:] - img[:, :-2]
    c_grad[:, 0], c_grad[:, -1] = 0, 0
    r_grad[0, :], r_grad[-1, 0] = 0, 0

    img_magnitude = np.sqrt(np.power(r_grad, 2) + np.power(c_grad, 2))
    img_theta = np.rad2deg(np.arctan(c_grad/(r_grad+0.00000001))) % 180
    orientation_bins = 8
    patch_size = 4
    tot_r, tot_c = img.shape
    hog = np.zeros((int(tot_r/patch_size), int(tot_c/patch_size), orientation_bins))
    for j in range(int(tot_c/patch_size)):
        for i in range(int(tot_r/patch_size)):
            # Extract the Current Patch and weight
            curr_patch = extract_relevant_window(img_theta, patch_size, i, j)
            curr_weight = extract_relevant_window(img_magnitude, patch_size, i, j)
            # Applying Histogram calculations
            hog[j][i] = np.histogram(np.ndarray.flatten(curr_patch), weights=np.ndarray.flatten(curr_weight), 
                                     bins=np.linspace(0, 180, num=(orientation_bins+1)))[0]        
    hog_norm = feature_normalization(hog, 'L2-Norm')
    return hog_norm.ravel()

def compute_from_patch(patch):
    if patch.shape[0] != 3 and patch.shape[0] != patch.shape[1]:
        raise ValueError('Patch Size Mismatch')
    patch_sub = patch - np.ravel(patch)[4]
    patch_sub[patch_sub>0] = 1
    patch_sub[patch_sub<=0] = 0
    flattened_patch = np.delete(np.ravel(patch_sub), 4)
    return flattened_patch.dot(2**np.arange(flattened_patch.size)[::1])

def lbpFeatures(x):
    patch_size = 3
    img = x
    final_img = np.empty((img.shape[0]-(patch_size-1), img.shape[1]-(patch_size-1))).astype('longdouble')
    for r in range(0, final_img.shape[0]):
        for c in range(0, final_img.shape[1]):
            final_img[r][c] = compute_from_patch(img[r:r+patch_size, c:c+patch_size])
    return feature_normalization(np.histogram(np.ndarray.flatten(final_img), bins=np.linspace(1, 255, num=257))[0], 'L2-Norm')

def extractDigitFeatures(x, featureType):
    N = x.shape[2]
    if featureType == 'pixel':
        features = np.empty((784, N)).astype('longdouble')
        features = np.array(Parallel(n_jobs=multiprocessing.cpu_count()) 
                           (delayed(pixelFeatures)(x[:, :, X_idx]) for X_idx in range(N)))
    elif featureType == 'hog':
        features = np.empty((392, N)).astype('longdouble')
        features = np.array(Parallel(n_jobs=2) 
                           (delayed(hogFeatures)(x[:, :, X_idx]) for X_idx in range(N)))
    elif featureType == 'lbp':
        features = np.empty((256, N)).astype('longdouble')
        features = np.array(Parallel(n_jobs=multiprocessing.cpu_count()) 
                           (delayed(lbpFeatures)(x[:, :, X_idx]) for X_idx in range(N)))
        
    features = feature_normalization(features, 'Sqrt')
    return features.T