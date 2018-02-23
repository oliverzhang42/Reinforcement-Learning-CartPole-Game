# Reinforcement-Learning-CartPole-Game
An implementation of an algorithm that plays OpenAI's CartPole game. The algorithm is based in reinforcement learning, using the Monte Carlo Algorithm.

Author: Oliver Zhang
Last Modified: 2/22/18

Goal: Learn how to play the CartPole Game. I'm using the implementation of
CartPole by OpenAI:
https://gym.openai.com/docs/
 
# Overall Design:
I use Monte Carlo Learning to train a model which predicts the value of an 
action given a state. The state/observation contains the position and velocity
of the cart as well as the angle and angular velocity of the pole. At any
given time, there are two actions: accelerating to the left or the right.
 
For learning reinforcement learning, I suggest David Silver's youtube lectures
https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT

# How to Run:
Copy the file CartPole_MonteCarlo.py to your computer.
Modify path variable to point to a folder for saving weights.
Run it on python3.
