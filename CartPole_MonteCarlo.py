# Author: Oliver Zhang
# Last Modified: 2/22/18
#
# Goal: Learn how to play the CartPole Game. I'm using the implementation of
# CartPole by OpenAI:
# https://gym.openai.com/docs/
# 
# Overall Design:
# ===============
# I use Monte Carlo Learning to train a model which predicts the value of an 
# action given a state. The state/observation contains the position and velocity
# of the cart as well as the angle and angular velocity of the pole. At any
# given time, there are two actions: accelerating to the left or the right.
# 
# For learning reinforcement learning, I suggest David Silver's youtube lectures
# https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT
#
# 
#

import gym
import keras
from keras.layers import BatchNormalization, Dense, Activation
from keras.optimizers import Adam
import random
import h5py
import time
import numpy as np

# After Every SAVE_FREQUENCY episodes, we save the weights of the model in path.
SAVE_FREQUENCY = 100

# The number of total episodes to run.
TOTAL_EPISODES = 10000

# The size of each layer in the model.
LAYER_SIZE = 30

# How many features recieved from the environment. This is always four unless
# you change the game played from CartPole to something else.
INPUT_SIZE = 4

# Here, REWARD_DECAY is how much we care about the delayed reward compared to
# the immediate reward. REWARD_DECAY = 1 means we care about all reward the
# same, REWARD_DECAY = 0 means we don't care at all about the later rewards.
REWARD_DECAY = 1

class CartPoleAI:
    def __init__(self, learning_rate, display_img, debugging, path):
        self.display_img = display_img
        self.debugging = debugging
        self.path = path

        # self.env is the implementation of the CartPole game. Code:
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        self.env = gym.make('CartPole-v0')

        # Epsilon sometimes randomizing the player's actions. Helps with
        # exploration of more possibilities.
        self.epsilon = 2

        self.create_model()


    # This Function Creates a Keras Model with three sections of:
    # a Batch Norm Layer, a Dense layer, and an Activation.
    # There a fourth section with no activation because the output
    # isn't limited in a 0-1 range.
    def create_model(self):
        self.model = keras.models.Sequential()
        self.model.add(BatchNormalization(input_shape = (INPUT_SIZE + 2,)))
        self.model.add(Dense(LAYER_SIZE))
        self.model.add(Activation("relu"))

        self.model.add(BatchNormalization())
        self.model.add(Dense(LAYER_SIZE)) 
        self.model.add(Activation("relu"))

        self.model.add(BatchNormalization())
        self.model.add(Dense(LAYER_SIZE)) 
        self.model.add(Activation("relu"))

        self.model.add(BatchNormalization())
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer = Adam(learning_rate))

    # Policy is how the model picks an action for a given situation and weights.
    # This is not training.

    # We introduce a bit of variation to encourage the model to try different
    # paths. This is called an epsilon greedy policy. Here's a good resource
    # for it:
    # https://jamesmccaffrey.wordpress.com/2017/11/30/the-epsilon-greedy-algorithm/

    # The model inputs the observation with either [1, 0] or [0, 1]
    # appended to its end. It will then output the predicted value of either
    # move.
    def policy(self, observation):
        left_move = np.array([np.concatenate([observation, np.array([1.0, 0.0])])])
        right_move = np.array([np.concatenate([observation, np.array([0.0, 1.0])])])

        value_left = self.model.predict(left_move)[0]
        value_right = self.model.predict(right_move)[0]

        variation = random.random()
        
        if(value_left[0] > value_right[0]):
            if(variation < 1/self.epsilon):
                self.epsilon += 1/100
                return 1
            else:
                return 0
        else:
            if(variation < 1/self.epsilon):
                self.epsilon += 1/100
                return 0
            else:
                return 1

    # This is called at the end of the episode. It uses Monte Carlo to train
    # the model. The model inputs the observation-action pair and outputs a
    # predicted value.
    def train_model(self, state_array, action_array, verbose):
        # This contains the final values of each state and action pair in the
        # episode. The model uses this to predict the values more accurately.
        answers = []

        current_reward = 1
     
        for i in range(len(state_array)):
            answers = [current_reward] + answers
            current_reward *= REWARD_DECAY
            current_reward += 1

        # Preprocessing the state array. We append the actions taken to every
        # state in the state array. This is how we get state-action pairs to
        # feed into the model.

        one_hot = {0: [1, 0], 1: [0, 1]}

        inputs = []

        for i in range(len(state_array)):
            inputs.append(np.append(state_array[i], [one_hot[action_array[i]]]))

        inputs = np.array(inputs)

        if(self.debugging):
            print("Inputs of Model: Observations and the taken Action")
            print(inputs)
            print("")
            print("Targets of Model: Rewards of each Observation-Action pair")

        self.model.fit(x = inputs, y = np.array(answers), verbose = verbose)

    # Saves the model's weights.
    def save(self, s):
        self.model.save_weights(s)

    # Loads the weights of a previous model.
    def load(self, s):
        self.model.load_weights(s)

    # Runs through LOOP_ITERATION episodes. After each episode, it trains on
    # the episode.
    def main(self, path_for_weights = ""):
        if(len(path_for_weights) != 0):
            self.load(path_for_weights)

        for i_episode in range(TOTAL_EPISODES):
            observation = self.env.reset()
            
            state_array = [observation]
            action_array = []

            for t in range(200):
                if(self.display_img):
                    self.env.render()

                # Chose an action and take it
                action = self.policy(observation)

                observation, reward, done, info = self.env.step(action)

                state_array.append(observation)
                action_array.append(action)
                
                # Check if done. We're only training once we finish the entire
                # episode.
                     
                if done:
                    print("Episode finished after {} timesteps".format(t+1))

                    # The last observation is removed because the model didn't
                    # take an action then.
                    state_array.pop()

                    self.train_model(state_array, action_array, 0)
                    
                    break

            # After Every SAVE_FREQUENCY episodes, we save the weights of the
            # model in path.
            if(i_episode % SAVE_FREQUENCY == 0):	
                self.save(self.path + "/CartPole_MonteCarloW%d" % (i_episode))

learning_rate = 0.00007
display_img = False
debugging = False
path = "/home/oliver/Desktop"

x = CartPoleAI(learning_rate, display_img, debugging, path)
x.main()
