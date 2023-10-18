# School work - Recurrent neural networks 3/3
# Simo Sjögren

import os
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def main():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    noise_factor = 0.2

    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape)
    test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape)
    train_images_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
    test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)

    input_layer = Input(shape=(28, 28, 1))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    autoencoder.summary()

    # train_labels = keras.utils.to_categorical(train_labels)
    # test_labels = keras.utils.to_categorical(test_labels)

    autoencoder.fit(train_images, train_images, epochs=1, batch_size=64, shuffle=True)

    # Plotting the images
    plt.subplot(1,3,1)
    plt.imshow(train_images[40])
    plt.subplot(1,3,2)
    plt.imshow(autoencoder.predict(train_images[40:41])[0])
    plt.subplot(1,3,3)
    plt.imshow(autoencoder.predict(train_images[40:41])[0])
    plt.show()

    # Picture is plotted using 

    a = autoencoder.evaluate(test_images, test_labels)

    print("Accuracy is:", str(round(max(a), 2) * 100), '%')

    # bOTH with 10 epochs and batchsize 64.
    ## Using clean images and clean images evaluation we get 81.5% accuracy.
    ## Using noisy train images and noisy images evaluation we get 81.5% accuracy.

    ## So seems that actually using the autoencoder structure, it completely eliminates the noise element.

main()