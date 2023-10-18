# School work - Significance of optimizer in Neural Network
# Simo Sjögren

import imageio as iio
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def imagereader(path):
    # Assuming that there is only one hiearchical level of subdirs in path.
    # This dataloader is for the german sign recognitino only.
    CLASS_DICT = {
    "class1" : 0,
    "class2" : 1
    }
    X_images = []
    Y_images = []
    print("Starting to load images...")
    for folder_name in os.listdir(path):
        current_path = path + '/' + folder_name
        for current_image in os.listdir(current_path):
            X_images.append(iio.imread(current_path + '/' + current_image))
            Y_images.append(CLASS_DICT[folder_name])
            # print(current_image, " : ", CLASS_DICT[folder_name])
    print("Image loading done. Total amount of images: " + str(len(X_images)))
    return X_images, Y_images


def main():
    path = "C:/Users/simos/Desktop/MRPR2/ex2/GTSRB_subset_2"
    X_images, Y_images = imagereader(path)

    # Time to turn data into NumPy format.
    X_images = np.array(X_images)
    Y_images = np.array(Y_images)

    X_train, X_test, Y_train, Y_test = train_test_split(X_images, Y_images, test_size=0.2,shuffle=True, random_state=131)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=(64,64,3)))
    model.add(tf.keras.layers.Rescaling(scale = 1./255))

    model.add(tf.keras.layers.Conv2D(10, (3,3), strides=(2,2), input_shape=(64,64,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))

    model.add(tf.keras.layers.Conv2D(10, (3,3), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    
    '''
    With SGD-optimizer we get only ~70%, but when we switch into adam, we get 98-100% accuracy

    SGD_opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    '''
    model.compile(optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    # Y_train_onehot = tf.keras.utils.to_categorical(Y_train, 2)
    model.fit(X_train, Y_train, epochs=20, batch_size=32)

    # Y_test_onehot = tf.keras.utils.to_categorical(Y_test, 2)
    a = model.evaluate(X_test, Y_test)

    print("Accuracy is:", str(round(max(a), 2) * 100), '%')

if __name__ == "__main__":
    main()