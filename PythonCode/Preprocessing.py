# Normalizes and converts Neural Network input and outputs from csv to numpy array
#
# input:    csv files that contain the neural network inputs and outputs and surface types
# output:   numpy array of neural network inputs and outputs and surface types
#
# Steffen Mauceri, Feb 2018

import numpy as np

# do we want to load an existing normalization (mean/std of training set)
load_normalization_IO= False


# read in csv files and convert to numpy arrays
input=np.genfromtxt('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/data/input_mix_5_05_interp_noise1_refl.csv', delimiter=',')
output=np.genfromtxt('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/data/output_mix_5_05_interp_noise1_refl.csv', delimiter=',')
type =np.genfromtxt('/Users/stma4117/Studium/LASP/Hyper/NN/PythonOverflow/data/Type_mix_5_05_interp_noise1_refl.csv', delimiter=',')

# number of wavelength bands in input file
wl_bands = 319

#normalize inputs/outpus for training to zero mean and unit variance
if load_normalization_IO:
    data = np.load('data/normalization_mix_5_refl.npz')
    input_mean = data['input_mean']
    input_std = data['input_std']
    output_std = data['output_std']
    output_mean = data['output_mean']
else:
    input_mean = np.mean(input, axis=0)        #calculate mean of Features
    input_std = np.std(input, axis=0)         #calculate standard-distribution of Features
    output_mean = np.mean(output, axis=0)  # calculate mean of Features
    output_std = np.std(output, axis=0)  # calculate standard-distribution of Features
    # save normalization to reverse after prediction
    np.savez('/Users/stma4117/Studium/LASP/Hyper/NN/Python/data/normalization_mix_5_refl',
             input_std=input_std, output_std=output_std, input_mean=input_mean, output_mean=output_mean)

input = (input - input_mean)/input_std    #normalize Features
output = (output - output_mean)/output_std    #normalize Features

#save neural network inputs/outputs
np.savez('/Users/stma4117/Studium/LASP/Hyper/NN/Python/data/input_output_mix_5_refl',
         input=input, output=output, type = type)

