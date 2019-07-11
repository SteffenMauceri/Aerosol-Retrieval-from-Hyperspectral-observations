# Neural Network / AutoEncoder to verify whether sample comes form training distribution
# Is used to mask out individual pixels in retrievals from Neural Netork for aerosol retrieval "NN_5.1_mixed_ind_relu.py"
#
# input: normalized (zero mean unit variance) hyperspectral radiance adjusted for Sun-Earth distance
# (radiance * (1 - 0.01672*cos(0.9856*(day_of_year - 4))^2) and SZA (radiance / cos(SZA))
#
# output:   same as input
#
# note:
# - run Preprocessing.py with load_normalization_IO = True to generate input file for training
# - best results for few epochs: <10
#
# See paper for details: "Neural Network for Aerosol Retrieval from Hyperspectral Imagery"
# available at: https://www.atmos-meas-tech-discuss.net/amt-2019-228/
# Steffen Mauceri, Sept. 2018

import numpy as np
import random
import tensorflow as tf
import scipy.io

import scipy.io

def NN_ver(ID):
    logs_path = 'tmp/logs/' + name + str(ID)

    # Network
    n_inputs = 319+3  # number of inputs= radiance at 319 wavelength bands + SZA, Ground Elevation, Distance Sensor-Surface
    n_outputs = n_inputs  # number of outputs equals number of inputs

    # number of neurons for neural network layers
    n_hidden_1 = n_neurons1
    n_hidden_2 = n_neurons2 #
    n_hidden_3 = n_neurons1

    # tf Graph input
    X = tf.placeholder('float', [None, n_inputs])

    # Define layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1], stddev=0.07)), #std: (2/n_inputs)^0.5
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.06)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.25)),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_outputs], stddev=0.06))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_outputs]))
    }

    def MLP(x): # Neural Network Architecture for verification whether sample comes form training distribution
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))         #Compression lyer
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))   #Decompression
        out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
        return out_layer
    logits = MLP(X)

    # define cost to optimize for
    regularization = (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])+tf.nn.l2_loss(weights['h3'])+tf.nn.l2_loss(weights['out']))* reg
    mean_square_error = (tf.losses.mean_squared_error(logits, X))
    MSE = mean_square_error
    loss_op = mean_square_error + regularization   # loss operation we seek to minimize

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1) # optimizer algorithm that minimizes loss_op
    train_op = optimizer.minimize(loss_op)

    # placeholder for our save files
    saver = tf.train.Saver()
    # Initializing the variables
    init = tf.global_variables_initializer()

    # ...........................start training...........................

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logs_path, sess.graph)
        writer.close()

        if Load_IO:
            saver.restore(sess, '../trained/' + name_learn)  # restore our trained model weights and biases
            print('model restored')

        # Training cycle
        for epoch in range(training_epochs):  # training_epochs = number of total epoch runs

            n_batches = int(len(features_training_[:, 1]) / batch_size)  # number of batches

            features_training = features_training_[rand,:] #scramble order of training examples

            for i in range(n_batches):
                sess.run([train_op, loss_op], #run training step
                         feed_dict={X: features_training[i * batch_size:i * batch_size + batch_size, :]})

            if epoch % display_step == 0:

                # calculate Mean Square Error (MSE) for trainind and vaildation set
                MSE_test = sess.run(MSE, feed_dict={X: features_test})
                MSE_training = sess.run(MSE, feed_dict={X: features_training[:55000,:]})

                print(MSE_training)
                print(MSE_test)
                print('epoch ' + str(epoch))

        # if we trained for a set number of epochs the model training is finished
        print('finished training')

        if training_epochs >1:
            # save model weights and biases
            saver.save(sess, '../trained/' + name)
            print('NN saved')

        # AVIRIS
        if Aviris_IO:
            prediction = sess.run([logits], feed_dict={X: Aviris_eval_in})
            error = prediction - Aviris_eval_in
            verification = np.asarray(error[0], dtype=np.float64)
            # export as MATLAB file without reversing normalization
            scipy.io.savemat('../PythonOverflow/trained/prediction_AVIRIS' + name + '.mat', dict(verification=verification))
            print('AVIRIS saved')


# START Set Parameters ..........................................................................
# name: version_smoothing_neurons1_neurons2_noise_identifier_regularization_epochs
name='5.1_05_512x32_noise1_verification_reg5000_500_110'
name_learn = '5.1_05_512x32_noise1_verification_reg5000_500'

Load_IO = True #do we want to load a pretrained network
Aviris_IO = True #do we want to predict AOT for AVIRIS-NG observations
AVIRIS_eval_IO = False #do we want to predict AOT for all AVIRIS-NG observations.

reg = 1 / 5000 # L2 regularization factor
n_neurons1 = 512 # number of neurons for first four layers
n_neurons2 = 32 # number of neurons for last layer / 3

training_epochs = 0  # how often to we iterate over our samples
batch_size = 128  # how many examples to we use at once

# parameter for print/update
display_step = 20
# END Set Parameters ..........................................................................

print(name)
data = np.load('data/input_output_mix_5_refl.npz') #Training data
features = data['input']

if Aviris_IO:
    if AVIRIS_eval_IO:
        data = np.load('data/Aviris_eval_5_refl.npz')
        Aviris_eval_in = data['input']
    else:
        data = np.load('../PythonOverflow/data/AVIRIS_20160110_refl.npz')  # AVIRIS-NG observations we want to predict AOT for
        Aviris_eval_in = data['AVIRIS']

np.random.seed(12345)  #initialize random number generator for repeatability

randTest = np.int32(np.random.rand(10000) * (len(features[:-1, 1]))) #validation set
all = np.arange(len(features[:-1, 1]))
mask = np.ones(len(features[:-1, 1]), dtype=bool)
mask[randTest] = False
notTest = all[mask]

#choose validation set
features_test = features[randTest,:]

#choose training set
features_training_=features[notTest,:]

print(randTest[0:8])
print(notTest[0:8])

rand = np.arange(0,len(features_training_))
random.shuffle(rand)

# start the Neural Network
NN_ver(1)