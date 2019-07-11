# Trains multiple Neural Networks for the retrieval of aerosol optical thickness (AOT) from hyperspectral observations
# similar to NN_5.1_mixed_ind_relu. However, for different amounts of AVIRIS-NG equivalent noise and sampling resolution
#
# input: normalized (zero mean unit variance) hyperspectral radiance adjusted for Sun-Earth distance
# (radiance * (1 - 0.01672*cos(0.9856*(day_of_year - 4))^2) and SZA (radiance / cos(SZA))
#
# output: AOT for brown carbon, dust and sulfate @ 550nm.
#
# note:
# - runtime is approximately 100h on NVIDIA GTX 1060 GPU
# - run Preprocessing.py with load_normalization_IO = True to generate input file for training
#
# Code for the paper: "Neural Network for Aerosol Retrieval from Hyperspectral Imagery"
# available at: https://www.atmos-meas-tech-discuss.net/amt-2019-228/
# Steffen Mauceri, Feb. 2018

import numpy as np
import random
import tensorflow as tf

from export_5_ind import graph

# amount of AVIRIS-NG equivalent noise. (1 = AVIRIS-NG noise) see paper for details
for n in [0, 1, 3, 9]: #0, 1, 3, 9
    # number of equally spaced wavelenegth bands used for retrieval of AOT
    for s in [319, 107, 36, 12, 4]:  # 319, 107, 36, 12, 4

        tf.reset_default_graph() # reset tensorflow graph before each training

        def NN(ID):
            logs_path = 'tmp/logs/' + name2 + str(ID)

            # Network
            n_inputs = 319+3  # number of inputs= radiance at 319 wavelength bands + SZA, Ground Elevation, Distance Sensor-Surface
            n_outputs = 3  # number of outputs= AOT for brown carbon, dust and sulfate

            n_hidden_1 = np.int32(128)
            n_hidden_2 = n_hidden_1
            n_hidden_3 = n_hidden_1
            n_hidden_4 = n_hidden_1
            n_hidden_4x = 32

            reg = 1 / 5000

            # tf Graph input
            X = tf.placeholder('float', [None, n_inputs])
            Y = tf.placeholder('float', [None, n_outputs])

            # Store layers weight & bias
            weights = {
                'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1], stddev=0.07)),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.07)),
                'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.07)),
                'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=0.07)),
                'h41': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4x], stddev=0.07)),
                'h42': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4x], stddev=0.07)),
                'h43': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4x], stddev=0.07)),
                'h41out': tf.Variable(tf.random_normal([n_hidden_4x, 1], stddev=0.2)),
                'h42out': tf.Variable(tf.random_normal([n_hidden_4x, 1], stddev=0.2)),
                'h43out': tf.Variable(tf.random_normal([n_hidden_4x, 1], stddev=0.2))
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

            def MLP(x):
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
            regularization = (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h3'])
                              + tf.nn.l2_loss(weights['h4']) +tf.nn.l2_loss(weights['h41']) +tf.nn.l2_loss(weights['h42'])
                              + tf.nn.l2_loss(weights['h43']) + tf.nn.l2_loss(weights['h41out'])
                              + tf.nn.l2_loss(weights['h42out'])
                              + tf.nn.l2_loss(weights['h43out']))* reg

            mean_square_error = tf.losses.mean_squared_error(logits, Y)
            MSE = mean_square_error
            loss_op = mean_square_error + regularization  # add up both losses

            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1)
            train_op = optimizer.minimize(loss_op)

            saver = tf.train.Saver() #placeholder for our save files
            # Initializing the variables
            init = tf.global_variables_initializer()

            minima = 100  # make it large
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

                    features_training = features_training_[rand,:]
                    targets_training = targets_training_[rand, :]

                    for i in range(n_batches):
                        sess.run([train_op, loss_op],#run training step
                                 feed_dict={X: features_training[i * batch_size:i * batch_size + batch_size, :],
                                            Y: targets_training[i * batch_size:i * batch_size + batch_size, :]})

                    if epoch % display_step == 0:

                        # calculate Mean Square Error (MSE) on training and validation set
                        MSE_test = sess.run(MSE, feed_dict={X: features_test, Y: targets_test})
                        MSE_training = sess.run(MSE, feed_dict={X: features_training[0:10000,:],
                                                                Y: targets_training[0:10000,:]})

                        print('\n')
                        print('epoch ' + str(epoch))
                        print(MSE_training)
                        print(MSE_test)

                        delta_MSE = last_MSE - MSE_training
                        if delta_MSE < 0.00001 and delta_MSE > 0:
                            print('No more learning')
                            break_ = break_+1
                            if break_ > 3:
                                break
                        last_MSE = MSE_training

                        if minima * 1.01 < MSE_test:
                            break_ = break_ + 1
                            print('Over Fitting')
                            if break_ > 3:
                                break

                        if minima > MSE_test:
                            minima = MSE_test

                # if we ran out of epochs, could not minimize the MSE on the training-set anymore or started overfitting the model the training is finished
                print('finished training')

                if training_epochs > 1:
                    # save model weights and biases
                    saver.save(sess, '../trained/' + name2)
                    print('saved')

                # calculate AOT prediction with trained model
                prediction = sess.run([logits], feed_dict={X: features_test})
                prediction_np = np.asarray(prediction[0], dtype=np.float64) # convert to numpy array
                target_np = np.asanyarray(targets_test, dtype=np.float64)
                np.savez('data/prediction' + name2 + '.npz', prediction_np=prediction_np, target_np=target_np)
                print(str(ID) + ' saved ' + name2)
                print('reg' + str(reg))
                graph(name2) #reverse normalization and export to a .mat file for Matlab

        #..........................................................................
        NoiseCase = n # AVIRIS-NG equivalent nosie multiplier
        SamplingCase = s # Number of wavelengths available for AOT retrieval
        Load_IO = False #do we want to load a pretrained network

        name2='5.1_05_128x32_noise_' + str(NoiseCase) + '_sampling_' + str(SamplingCase) + '_reg5000_ind_relu'
        name_learn = '5.1_128x32_noise_' + str(NoiseCase) + '_sampling_' + str(SamplingCase) + '_reg5000_ind_relu'
        print(name2)

        # Choose the training dataset that differ by the amount of AVIRIS-NG equivalent noise
        if NoiseCase == 0:
            data = np.load('data/input_output_mix_5_05_refl_noise0.npz')
        elif NoiseCase == 1:
            data = np.load('data/input_output_mix_5_05_refl.npz')
        elif NoiseCase == 3:
            data = np.load('data/input_output_mix_5_05_refl_noise3.npz')
        elif NoiseCase == 9:
            data = np.load('data/input_output_mix_5_05_refl_noise9.npz')

        targets = data['output']
        targets = targets[:,1:4]
        features = data['input']

        # Remove wavelengths for SamplingCase
        if SamplingCase ==107:
            cut = np.concatenate((np.arange(0, 319, 3), np.arange(307, 322)))
        elif SamplingCase ==36:
            cut = np.concatenate((np.arange(0, 319, 9), np.arange(307, 322)))
        elif SamplingCase == 12:
            cut = np.concatenate((np.arange(0, 319, 27), np.arange(307, 322)))
        elif SamplingCase == 4:
            cut = np.concatenate((np.arange(0, 319, 81), np.arange(307, 322)))

        if SamplingCase < 319:
            features = np.concatenate((features[:,cut], np.zeros((len(features), 322 - len(cut)))),1)


        np.random.seed(12345) #initialize random number generator for repeadability

        randTest = np.int32(np.random.rand(10000) * (len(features[:-1, 1]))) #validation set
        all = np.arange(len(features[:-1, 1]))
        mask = np.ones(len(features[:-1, 1]), dtype=bool)
        mask[randTest] = False
        notTest = all[mask]

        # choose validation set
        features_test = features[randTest,:]
        targets_test=targets[randTest,:]

        # choose training set
        features_training_=features[notTest,:]
        targets_training_=targets[notTest,:]

        rand = np.arange(0,len(features_training_))
        random.shuffle(rand)

        training_epochs = 10000  # how often to maximally we iterate over our samples
        batch_size = 128  # how many examples to we use at once

        # parameter for print/update
        display_step = 40

        # start the Neural Network with unique ID
        NN(s+n)