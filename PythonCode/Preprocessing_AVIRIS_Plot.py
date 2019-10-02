# Normalizes and converts individual AVIRIS flight scene for Neural Network
#
# input:    csv file that contains the AVIRIS radiances
# output:   numpy array of normalized AVIRIS radiances
#
# Steffen Mauceri, Mar 2018

import numpy as np

# number of wavelength bands in input file
wl_bands = 319

# read in csv file
input = np.genfromtxt('...data/AVIRIS_20160204_refl.csv', delimiter=',')

# load normalization from training data
data = np.load('data/normalization_mix_5_refl.npz')
input_mean = data['input_mean']
input_std = data['input_std']

# normalize AVIRIS radiances to zero mean unit variance
input = (input - input_mean)/input_std

#save normalized AVIRIS radiance as numpy array
np.savez('...data/AVIRIS_20160204_refl', AVIRIS=input)
