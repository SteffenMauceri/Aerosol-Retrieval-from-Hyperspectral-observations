# Neural Network for aerosol optical thickness (AOT) retrieval from hyperspectral observations
# Main program for neural network definition, training and predictions
#
# input: normalized (zero mean unit variance) hyperspectral radiance adjusted for Sun-Earth distance
# (radiance * (1 - 0.01672*cos(0.9856*(day_of_year - 4))^2) and SZA (radiance / cos(SZA))
#
# output:   AOT for brown carbon, dust and sulfate @ 550nm.
#           Is automatically exported to a .mat file to read into MATLAB with functions from export_5_ind.py
#
# note:
# - run Preprocessing.py with load_normalization_IO = False to generate input file for training
# - run Preprocessing_AVIRIS_Plot.py to generate input file for AOT retrieval of one AVIRIS scene for visualization on a per pixel basis
# - run Preprocessing_AVIRIS_eval.py to generate input file for AOT retrieval for 21 AVIRIS scenes for comparison to other aerosol retrievals
#
# Code for the paper: "Neural Network for Aerosol Retrieval from Hyperspectral Imagery"
# available at: https://www.atmos-meas-tech-discuss.net/amt-2019-228/
# Steffen Mauceri, Feb. 2018

import numpy as np
import random
import tensorflow as tf
import time

# import functions for export of neural network predictions
from export_2_Matlab import graph
from export_2_Matlab import export_AVIRIS
from export_2_Matlab import export_sensitivity


def NN(ID):
    logs_path = 'tmp/logs/' + name + str(ID)

    # Network
    n_inputs = 319+3    # number of inputs= radiance at 319 wavelength bands + SZA, Ground Elevation, Distance Sensor-Surface
    n_outputs = 3       # number of outputs= AOT for brown carbon, dust and sulfate

    # number of neurons for the first 4 layers
    n_hidden_1 = n_neurons1
    n_hidden_2 = n_hidden_1
    n_hidden_3 = n_hidden_1
    n_hidden_4 = n_hidden_1
    n_hidden_4x = n_neurons2

    # tf Graph input
    X = tf.placeholder('float', [None, n_inputs])
    Y = tf.placeholder('float', [None, n_outputs])

    # Define layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1], stddev=0.07)), #std: (2/n_inputs)^0.5
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.12)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.12)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=0.12)),
        'h41': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4x], stddev=0.12)),
        'h42': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4x], stddev=0.12)),
        'h43': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4x], stddev=0.12)),
        'h41out': tf.Variable(tf.random_normal([n_hidden_4x, 1], stddev=0.25)),
        'h42out': tf.Variable(tf.random_normal([n_hidden_4x, 1], stddev=0.25)),
        'h43out': tf.Variable(tf.random_normal([n_hidden_4x, 1], stddev=0.25))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'b41': tf.Variable(tf.random_normal([n_hidden_4x])),
        'b42': tf.Variable(tf.random_normal([n_hidden_4x])),
        'b43': tf.Variable(tf.random_normal([n_hidden_4x])),
        'b41out': tf.Variable(tf.random_normal([1])),
        'b42out': tf.Variable(tf.random_normal([1])),
        'b43out': tf.Variable(tf.random_normal([1]))
    }


    def MLP(x): # Neural Network Architecture for AOT retrieval
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))

        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
        layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))

        layer_41 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h41']), biases['b41']))
        layer_41out = tf.add(tf.matmul(layer_41, weights['h41out']), biases['b41out'])

        layer_42 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h42']), biases['b42']))
        layer_42out = tf.add(tf.matmul(layer_42, weights['h42out']), biases['b42out'])

        layer_43 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h43']), biases['b43']))
        layer_43out = tf.add(tf.matmul(layer_43, weights['h43out']), biases['b43out'])

        out_layer = tf.concat([layer_41out,layer_42out,layer_43out], 1)
        return out_layer

    logits = MLP(X)

    # define cost to optimize for
    regularization = ( tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h3'])
                      + tf.nn.l2_loss(weights['h4']) +tf.nn.l2_loss(weights['h41']) +tf.nn.l2_loss(weights['h42'])
                      + tf.nn.l2_loss(weights['h43']) + tf.nn.l2_loss(weights['h41out'])
                      + tf.nn.l2_loss(weights['h42out'])
                      + tf.nn.l2_loss(weights['h43out']))* reg

    if weighted_IO:
        mean_square_error = tf.math.reduce_mean(tf.concat([tf.math.square(logits[:,1:3] - Y[:,1:3])*0.75, 1.5*tf.math.square(logits[:,0:1] - Y[:,0:1])], 1))
    else:
        mean_square_error = tf.losses.mean_squared_error(logits, Y)

    MSE = mean_square_error


    loss_op = mean_square_error + regularization  # loss operation we seek to minimize

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1) # optimizer algorithm that minimizes loss_op
    train_op = optimizer.minimize(loss_op)

    saver = tf.train.Saver() #placeholder for our save files
    init = tf.global_variables_initializer()# Initializing the variables for training

    # Initialize a tracking variable for loss
    minima = 100  # initialize a minima to something big, loss should be continuously decreasing
    last_MSE = minima

    # ...........................start training...........................
    with tf.Session() as sess:
        break_ = 0
        sess.run(init)
        writer = tf.summary.FileWriter(logs_path, sess.graph)
        writer.close()

        if Load_IO:
            saver.restore(sess, '../trained/'+ name_learn)  # restore our trained model weights and biases
            print('model restored')

        # Training cycle
        for epoch in range(training_epochs):  # training_epochs = number of total epoch runs

            n_batches = int(len(features_training_[:, 1]) / batch_size)  # number of batches

            features_training = features_training_[rand,:] #scramble order of training examples
            targets_training = targets_training_[rand, :]


            for i in range(n_batches):
                sess.run([train_op, loss_op], #run training step
                         feed_dict={X: features_training[i * batch_size:i * batch_size + batch_size, :],
                                    Y: targets_training[i * batch_size:i * batch_size + batch_size, :]})

            if epoch % display_step == 0:
                print('\n')
                print('epoch ' + str(epoch))

                # calculate Mean Square Error (MSE) on training and validation set
                MSE_test = sess.run(MSE, feed_dict={X: features_test, Y: targets_test})
                MSE_training = sess.run(MSE, feed_dict={X: features_training[0:10000,:],
                                                        Y: targets_training[0:10000,:]})

                print(MSE_training)
                print(MSE_test)

                delta_MSE = last_MSE - MSE_training #check whether our MSE on the training-set is going down. If it doesn't -> stop training
                if delta_MSE < 0.00001 and delta_MSE > 0:
                    print('No more learning')
                    break_ = break_+1
                    if break_ > 3:
                        break
                last_MSE = MSE_training

                if minima * 1.01 < MSE_test: #check whether our MSE on the validation-set is going up. If that happens repeadatly: stop training since we are overfitting
                    break_ = break_ + 1
                    print('Over Fitting')
                    if break_ > 3:
                        break

                if minima > MSE_test:
                    minima = MSE_test

        # if we ran out of epochs, could not minimize the MSE on the training-set anymore or started overfitting the model the training is finished
        print('finished training')

        if training_epochs > 1: #make sure that we were actually training the model and not just making predictions.
            # save model weights and biases
            saver.save(sess, '../trained/' + name)
            print('saved')

        prediction = sess.run([logits], feed_dict={X: features_test}) # calculate AOT prediction with trained model
        prediction_np = np.asarray(prediction[0], dtype=np.float64) # convert to numpy array
        target_np = np.asanyarray(targets_test, dtype=np.float64)

        np.savez('data/prediction' + name + '.npz', prediction_np=prediction_np,
                 target_np=target_np)  # save predictions and ground truth

        print(str(ID) + ' saved ' + name)
        print('reg' + str(reg))
        graph(name) #reverse normalization and export to a .mat file for Matlab

        #AVIRIS
        if Aviris_IO:
            prediction = sess.run([logits], feed_dict={X: Aviris_eval_in}) # calculate AOT prediction with trained model
            prediction_np = np.asarray(prediction[0], dtype=np.float64)# convert to numpy array
            np.savez('... data/prediction_AVIRIS' + name + '.npz', prediction_np=prediction_np)
            export_AVIRIS(name)#reverse normalization and exort to a .mat file for Matlab

        #sensitivity analysis
        if Sensitivity_IO:
            prediction_sensitivity = np.zeros((len(features_test), 3, n_inputs), dtype=np.float64)
            start= time.time() #measure time for execution
            for i in range(0,n_inputs):
                features_sensitivity = np.copy(features_test) #make fresh copy of radiances
                features_sensitivity[:, i] = features_sensitivity[:, i] * 0.99#0.99 # peturb radiance at one wavelength

                pred_i = sess.run([logits], feed_dict={X: features_sensitivity}) #perform prediction with peturbed inputs
                prediction_sensitivity[:, :, i] = np.asarray(pred_i[0], dtype=np.float64)

            duration = time.time() - start
            print(duration) #print time it took for 314*10000 retrievals

            prediction_np = np.asarray(prediction_sensitivity, dtype=np.float64)
            target_np = np.asanyarray(targets_test, dtype=np.float64)
            np.savez('data/prediction_sensitivity' + name + '.npz',
                     prediction_np=prediction_np, target_np=target_np) #save prediction
            print(str(ID) + ' saved ' + name)
            export_sensitivity(name) #reverse normalization and exort to a .mat file for Matlab

# START Set Parameters ..........................................................................
# name format: version_smoothing_neurons1_neurons2_noise_regularization_epochs_NNarchitecture_activation_normalization
name = 'My_NN' #name of network that we are training
name_learn = 'Pretrained_NN' # name of network we are loading if (Load_IO == True)

Load_IO = False          #do we want to load a pretrained network
Aviris_IO = False        #do we want to predict AOT for AVIRIS-NG observations
AVIRIS_eval_IO = False  #do we want to predict AOT for all AVIRIS-NG observations.
Sensitivity_IO = False  #do we want to perform a sensitivity analysis
weighted_IO = False     #do we want to weight error on carbon twice as much as dust and sulfate

reg = 1 / 5000          # L2 regularization factor
n_neurons1 = 128        # number of neurons for first four layers
n_neurons2 = 32         # number of neurons for last layer / 3

training_epochs = 200     # how often to we maximally iterate over our samples. Set to 0 for no training, just prediction
batch_size = 8        # how many examples to we use at once

# parameter for print/update
display_step = 40
# END Set Parameters ..........................................................................

print(name)
data = np.load('data/input_output.npz') #Training data
targets = data['output']
targets = targets[:,1:4]
features = data['input']

if Aviris_IO:
    if AVIRIS_eval_IO:
        data = np.load('... data/Aviris_eval_5_refl.npz')
        Aviris_eval_in = data['input']
    else:
        data = np.load('... data/AVIRIS_20160110_refl.npz')  # AVIRIS-NG observations we want to predict AOT for
        Aviris_eval_in = data['AVIRIS']

np.random.seed(12345) #initialize random number generator for repeatability

randTest = np.int32(np.random.rand(10) * (len(features[:-1, 1])))
all = np.arange(len(features[:-1, 1]))
mask = np.ones(len(features[:-1, 1]), dtype=bool)
mask[randTest] = False
notTest = all[mask]

#choose validation set
features_test = features[randTest,:]
targets_test=targets[randTest,:]

#choose training set
features_training_=features[notTest,:]
targets_training_=targets[notTest,:]

print(randTest[0:8])# sanity check
print(notTest[0:8])

rand = np.arange(0,len(features_training_))
random.shuffle(rand)

# start the Neural Network
NN(1)