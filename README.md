# Aerosol-Retrieval-from-Hyperspectral-observations
Code for the paper: "Neural Network for Aerosol Retrieval from Hyperspectral Imagery" available at: https://www.atmos-meas-tech-discuss.net/amt-2019-228/. If you use part of this code in your published work, please cite this paper.

Radiative Transfer calculations are first prepared in Matlab to train the Neural Network: 
  1. run Interpolate_RadTrans_calc.m
  2. run Add_Surface.m
  3. run make_training_set.m
  
The actual training of the Neural Network is performed in Python / Tensorflow. After training on the radiative transfer calculations the Neural Network can be used to predict aerosol optical thickness from hyperspectral observations from AVIRIS-NG
  1. run Preprocessing.py
  2. run NN_for_AerosolRetrieval.py
  3. run NN_for_verification.py
  

  
  
  note: the full training data can be made available upon request. Shortened datasets are provided.
