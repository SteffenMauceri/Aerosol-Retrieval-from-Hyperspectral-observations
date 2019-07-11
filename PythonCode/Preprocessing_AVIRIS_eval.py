# Normalizes and converts multiple AVIRIS flights for Neural Network to later compare with MODIS, CAMS, AERONET
#
# input:    csv files that contain the AVIRIS radiances
# output:   numpy array of normalized AVIRIS radiances
#
# Steffen Mauceri, June 2018

import numpy as np

# read 21 concaternated AVIRIS-NG observations for AOT retrieval and comparison to AERONET, MODIS, CAMS
input=np.genfromtxt('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/data/AVIRIS_eval_5_refl.csv', delimiter=',')

# number of wavelength bands in input file
wl_bands = 319

# load normalization from training data
data = np.load('data/normalization_mix_5_refl.npz')
input_mean = data['input_mean']
input_std = data['input_std']

# normalize AVIRIS radiances to zero mean unit variance
input = (input - input_mean)/input_std

#save normalized AVIRIS radiance as numpy array
np.savez('/Users/stma4117/Studium/LASP/Hyper/NN/Python/data/Aviris_eval_5_refl', input=input)