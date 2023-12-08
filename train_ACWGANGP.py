# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.initializers import RandomNormal
from functools import partial
import tensorflow as tf
import keras.backend as K

import matplotlib.pyplot as plt
import h5py
import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class ACWGANGP():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.num_classes = 5
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.9)

        # Build the generator and critic
        self.generator = self.build_generator()
        #self.generator.load_weights('generator_weights.hdf5' ## load pretrained wieghts if you have them
        self.critic = self.build_critic()
        #self.critic.load_weights('critic_weights.hdf5') ## load pretrained wieghts if you have them

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)
        real_label = Input(shape=(1,))

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        z_label = Input(shape=(1,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator([z_disc, z_label])

        # Discriminator determines validity of the real and fake images
        fake, f_label = self.critic(fake_img)
        valid, v_label = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated, i_label = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, real_label, z_disc, z_label],
                            outputs=[valid, fake, validity_interpolated, v_label, f_label])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss, 'sparse_categorical_crossentropy',
                                              'sparse_categorical_crossentropy'],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10, 1, 1])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        z_gen_label = Input(shape=(1,))
        # Generate images based of noise
        img = self.generator([z_gen,z_gen_label])
        # Discriminator determines validity
        valid, gen_label = self.critic(img)
        # Defines generator model
        self.generator_model = Model([z_gen, z_gen_label], [valid, gen_label])
        self.generator_model.compile(loss=[self.wasserstein_loss, 'sparse_categorical_crossentropy'], optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(8*8*16, activation='relu', input_dim=120 ))
        model.add(Reshape((8,8,16)))
        model.add(Dropout(rate=0.25))
        
        model.add(UpSampling2D(interpolation='nearest')) # 16*16
        model.add(Conv2D(128, kernel_size=3, padding='same')) # shape remains same
        model.add(LeakyReLU())
        
        model.add(UpSampling2D(interpolation='nearest')) # 32*32
        model.add(Conv2D(64, kernel_size=3,  padding='same')) # shape remains same
        model.add(LeakyReLU())
    
        
        model.add(UpSampling2D(interpolation='nearest')) # 64*64
        model.add(Conv2D(32, kernel_size=3, padding='same')) # shape remains same
        model.add(LeakyReLU())

        
        model.add(UpSampling2D(interpolation='nearest')) # 128*128
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 20)(label))

        model_input = Concatenate()([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_critic(self):

        model = Sequential()

        # Input 128 x 128 x 1
        model.add(Conv2D(32,kernel_size=3, strides=2, input_shape = self.img_shape,  padding='same')) #64 x 64 x 32
        model.add(LeakyReLU())    

        model.add(Conv2D(64,kernel_size=3,strides=2, input_shape = self.img_shape,  padding='same')) #32 x 32 x 64
        model.add(LeakyReLU())    
        
        model.add(Conv2D(128, kernel_size=3, strides=2,  padding='same')) #16 x 16 x 128
        model.add(LeakyReLU())
        
        model.add(Conv2D(16, kernel_size=3, strides=2, padding='same')) #8 x 8 x 16
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dropout(rate=0.25))

        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1)(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size, sample_interval=50):

        # load dataset
        f = h5py.File("./dataset/UHCS128_Images.h5",'r')
        X_train = f['UHCS_dataset']
        X_train = np.array(X_train)
        X_train = np.concatenate((X_train,np.fliplr(X_train)),axis=0)

        f = h5py.File('./dataset/UHCS128_labels.h5','r')
        labels = f['Conditioning_vars']
        labels = np.array(labels) ## cooling, temperature, time
        y_train = labels[:,0]
        y_train = np.concatenate((y_train, y_train), axis=0)

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        cr_hist, cf_hist, ci_hist, cr_classify_hist, cf_classify_hist, g_GAN_hist, g_classify_hist = list(), list(), list(), list(), list(), list(), list()

        for epoch in range(epochs):
            cr_dummy, cf_dummy, ci_dummy, cr_classify_dummy, cf_classify_dummy = list(), list(), list(), list(), list() # collect critic score
            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                sampled_labels = y_train[idx]
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, sampled_labels, noise, sampled_labels],[valid, fake, dummy, sampled_labels, sampled_labels])
                cr_dummy.append(d_loss[1])
                cf_dummy.append(d_loss[2])
                ci_dummy.append(d_loss[3])
                cr_classify_dummy.append(d_loss[4])
                cf_classify_dummy.append(d_loss[5])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
            g_GAN_hist.append(g_loss[1])
            g_classify_hist.append(g_loss[2])

            # average losses for critic
            cr_hist.append(np.average(cr_dummy))
            cf_hist.append(np.average(cf_dummy))
            ci_hist.append(np.average(ci_dummy))
            cr_classify_hist.append(np.average(cr_classify_dummy))
            cf_classify_hist.append(np.average(cf_classify_dummy))
            # Plot the progress
            print ("%d [Dr loss: %f] [Df loss: %f] [Dgp loss: %f] [Dr Closs: %f] [Df Closs: %f] [G loss: %f] [G Closs: %f]" % (epoch, d_loss[1], d_loss[2], d_loss[3], d_loss[4], d_loss[5], g_loss[1], g_loss[2] ))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, cr_hist, cf_hist, ci_hist,cr_classify_hist, cf_classify_hist, g_GAN_hist, g_classify_hist)
                self.save_model()

    def sample_images(self, epoch, cr_hist, cf_hist, ci_hist,cr_classify_hist, cf_classify_hist, g_GAN_hist, g_classify_hist):
        r, c = 3, 5
        noise = np.random.normal(0,1, (1, self.latent_dim))
        for i in range(r):
            n = np.random.normal(0,1, (1, self.latent_dim))
            n_repeat = np.repeat(n,c,axis=0)
            noise = np.vstack((noise,n_repeat))
        noise = noise[1:,:]
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 127.5 * gen_imgs + 127.5
        # plot images
        plt.figure(figsize=(10, 7))
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/UHCS_wgangp_%d.png" % epoch, bbox_inches = 'tight')
        plt.close()

        # plot WGAN history
        plt.plot(cr_hist, label='crit_real')
        plt.plot(cf_hist, label='crit_fake')
        plt.plot(g_GAN_hist, label='gen')
        plt.legend()
        plt.savefig('images/Loss_plot.png')
        plt.close()

        # save loss
        h5f = h5py.File('Loss.h5', 'w')
        h5f.create_dataset('Critic_real', data=np.array(cr_hist))
        h5f.create_dataset('Critic_fake', data=np.array(cf_hist))
        h5f.create_dataset('Critic_intp', data=np.array(ci_hist))
        h5f.create_dataset('Critic_real_classify', data=np.array(cr_classify_hist))
        h5f.create_dataset('Critic_fake_classify', data=np.array(cf_classify_hist))
        h5f.create_dataset('Gan_score', data=np.array(g_GAN_hist))
        h5f.create_dataset('Gan_classify', data=np.array(g_classify_hist))
        h5f.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "%s.json" % model_name
            weights_path = "%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.critic, "critic")

with tf.device('/gpu:3'):
    acwgangp = ACWGANGP()
    nEpoch = int(491*1.5e4)
    acwwgangp.train(epochs=nEpoch, batch_size=64, sample_interval=1000)