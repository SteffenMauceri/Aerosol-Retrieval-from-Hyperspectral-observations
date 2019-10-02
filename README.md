# Aerosol-Retrieval-from-Hyperspectral-observations
Code for the paper: "Neural Network for Aerosol Retrieval from Hyperspectral Imagery" available at: https://www.atmos-meas-tech-discuss.net/amt-2019-228/. If you use part of this code in your published work, please cite this paper.

Radiative Transfer calculations are first prepared in Matlab to train the Neural Network: 
  1. run MODTInterpolator.m
  2. run RadianceGenerator4.m
  3. run make_input_output_mix.m
  
Hyperspectral observations from AVIRIS-NG are prepared to make predictions of aerosol optical thickness once the model is trained: 
  1. run make_input_AVIRIS.m
  
The actual training of the Neural Network is performed in Python / Tensorflow. After training on the radiative transfer calculations the Neural Network can be used to predict aerosol optical thickness from hyperspectral observations from AVIRIS-NG
  1. run Preprocessing.py
  2. run Preprocessing_AVIRIS_Plot.py
  3. run Preprocessing_AVIRIS_eval.py
  4. run NN_5.1_for_AerosolRetrieval.py
  5. run NN_3.0_for_verification.py
  
 To analyze the results run the follwoing programs in Matlab in no particular order
  1. run compare_AVIRIS_CAMS.m (comparison to CAMS aerosol model)
  2. run compare_AVIRIS_MODIS.m (comparison to MODIS aerosol retrieval)
  3. run PlotAVIRIS.m (plot spatially resolved aerosol retrieval for AVIRIS scene)
  
  
  
  note: training data can be made available upon request
