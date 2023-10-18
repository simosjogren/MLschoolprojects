# School work - Advanced Q-learning 2/2
# Simo Sjögren

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import random, pickle

import tensorflow as tf
from tensorflow import keras
from keras import layers

def sample_data(qtable, num_samples):
    X = []
    y = []
    for i in range(num_samples):
        state = random.randint(0, num_states - 1)
        action = random.randint(0, num_actions - 1)
        X.append(np.eye(num_states)[state])
        y.append(qtable[state, action])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Load the Q-table
with open('qtable.pkl', 'rb') as f:
    qtable = pickle.load(f)

# Set up the neural network
num_states = qtable.shape[0]
num_actions = qtable.shape[1]

inputs = keras.Input(shape=(num_states,))
x = layers.Dense(500, activation="relu")(inputs)
x = layers.Dense(500, activation="relu")(x)
outputs = layers.Dense(num_actions, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="taxi_model")
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())

num_samples = num_states * num_actions
X, y = sample_data(qtable, num_samples)

history = model.fit(X, y, batch_size=32, epochs=100, validation_split=0.1)

env = gym.make('Taxi-v3')

total_rewards = 0
state = env.reset()
done = False
while not done:
    # Get  Q-values from the network
    q_values = model.predict(np.eye(num_states)[state][np.newaxis])[0]
    
    action = np.argmax(q_values)
    
    # Take the action
    new_state, reward, done, info = env.step(action)
    
    # Update the state and total rewards
    state = new_state
    total_rewards += reward

print("Total reward:", total_rewards)