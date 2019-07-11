# Functions to reverse normalization of Neural Network output and export numpy arrays to .mat for Matlab
#
# input:    Name of network
# output:   .mat files for Matlab
#
# Steffen Mauceri, June 2018

import numpy as np
import scipy.io

# export predictions for radiative transfer calculations
def graph(name):
    #import aerosol optical thickness (AOT) prediction
    data = np.load('data/prediction'+ name +'.npz')
    prediction = data['prediction_np']
    target = data['target_np']

    #reverse normalization
    data = np.load('data/normalization_mix_5_refl.npz') # load normalization
    target_raw_std = data['output_std'] #load standard deviation
    target_raw_mean = data['output_mean'] # load mean
    # targets contain combined AOT in first column that is currently not used
    target = (target*target_raw_std[1:4])+target_raw_mean[1:4]
    prediction = (prediction*target_raw_std[1:4])+target_raw_mean[1:4]

    # print standard deviation for AOT for prediction-target
    print('Std for aerosols')
    for i in range(0,3):
        print(str(np.std(np.abs(prediction[:,i]-target[:,i]))))

    scipy.io.savemat('../trained/prediction'+ name +'.mat', dict(prediction=prediction, target=target))

# export predictions for AVIRIS retrievals
def export_AVIRIS(name):
    #import aerosol optical thickness (AOT) prediction
    data = np.load('../PythonOverflow/data/prediction_AVIRIS'+ name +'.npz')
    prediction = data['prediction_np']

    # reverse normalization
    data = np.load('data/normalization_mix_5_refl.npz') # load normalization
    target_raw_std = data['output_std']                 # load standard deviation
    target_raw_mean = data['output_mean']               # load mean
    prediction = (prediction*target_raw_std[1:4])+target_raw_mean[1:4]

    scipy.io.savemat('../PythonOverflow/trained/prediction_AVIRIS'+ name +'.mat', dict(prediction=prediction))


# export predictions for sensitivity analysis
def export_sensitivity(name):
#import our prediction
    data = np.load('../PythonOverflow/data/prediction_sensitivity'+ name +'.npz')
    prediction = data['prediction_np']
    target = data['target_np']
    
    # reverse normalization
    data = np.load('data/normalization_mix_5_refl.npz')
    target_raw_std = data['output_std']
    target_raw_mean = data['output_mean']

    target = (target*target_raw_std[1:4])+target_raw_mean[1:4]

    target_raw_mean = np.reshape(target_raw_mean[1:4], (1,3,1))
    target_raw_std = np.reshape(target_raw_std[1:4], (1,3,1))

    prediction = (prediction*np.tile(target_raw_std, (len(prediction),1,322)))+np.tile(target_raw_mean, (len(prediction),1,322))

    scipy.io.savemat('../PythonOverflow/trained/prediction_sensitivity'+ name +'.mat', dict(prediction=prediction, target=target))



#name = '5.1_05_128x32_noise1_reg5000_2200_ind_relu_refl'
#graph(name)
#export_AVIRIS(name)
#export_missing(name)
