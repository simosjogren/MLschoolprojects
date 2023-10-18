# School work - Course's first neural network build
# Simo Sjögren

import imageio as iio
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def imagereader(path):
    # Assuming that there is only one hiearchical level of subdirs in path.
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


def traintest_index_splitter(train_size, total_image_amount):
    # train_size in normalized form.
    if train_size > 1:
        return None
    index_split = round(total_image_amount*train_size)
    print("Testing data will start from index: " + str(index_split))
    return index_split


def main():
    path = "C:/Users/simos/Desktop/MRPR2/ex2/GTSRB_subset_2"
    X_images, Y_images = imagereader(path)

    # Time to turn data into NumPy format.
    X_images = np.array(X_images)
    Y_images = np.array(Y_images)

    
    # Lets normalize the values in picture.
    X_images = X_images 

    X_train, X_test, Y_train, Y_test = train_test_split(X_images, Y_images, test_size=0.2,shuffle=True, random_state=131)

    # Neural network structure is clearly simple conv sequential
    model = tf.keras.models.Sequential()

    # Input layer as flattened
    model.add(tf.keras.layers.Flatten(input_shape=(64,64,3)))
    model.add(tf.keras.layers.Rescaling(scale = 1./255))
    # First layer
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='relu'))
    # Output layer
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    # Lets use Gradient Descent as optimizer and MSE and loss function.
    model.compile(optimizer='SGD',
        loss=tf.keras.losses.BinaryCrossentropy(), metrics = ["accuracy"])

    # Time to train data
    model.fit(X_train, Y_train, epochs=10)

    ##Y_test_onehot_p = model.predict(X_test)
    ##Y_test_p = np.argmax(Y_test_onehot_p, axis=1)
    ##test_acc = 1-np.count_nonzero(Y_test-Y_test_p)/len(Y_test)
    ##test_acc = round((test_acc * 100), 3)
    ##print("Prediction accuracy: ", str(test_acc), "%")

    # All the commented above can actually be shortened and done faster by the next command.
    a = model.evaluate(X_test, Y_test)
    print("Accuracy is:", str(round(max(a), 2) * 100), '%')

if __name__ == "__main__":
    main()