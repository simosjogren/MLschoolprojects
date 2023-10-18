# School work - Recurrent neural networks 2/3
# Simo Sjögren

import os
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

    print(train_images.shape)
    print(train_images_noisy.shape)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=(28,28,1)))

    model.add(tf.keras.layers.Conv2D(10, (3,3), strides=(2,2), input_shape=(28,28,1,1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))

    model.add(tf.keras.layers.Conv2D(10, (3,3), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    # Y_train_onehot = tf.keras.utils.to_categorical(Y_train, 2)
    # Training with noisy images.
    model.fit(train_images_noisy, train_labels, epochs=20, batch_size=32)

    # Y_test_onehot = tf.keras.utils.to_categorical(Y_test, 2)
    a = model.evaluate(test_images_noisy, test_labels)

    print("Accuracy is:", str(round(max(a), 2) * 100), '%')

    # Accuracies using CNN model with clean images only.
    # Classification accuracy for clean images is: 94.05%
    # Classification accuracy for noisy images is:c 91.91%
main()