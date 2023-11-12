# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import SarsaAgent
from gym.wrappers.monitoring.video_recorder import VideoRecorder
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0", render_mode="rgb_array")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
#env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED) 

##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

####### START CODING HERE #######

video = VideoRecorder(env, path="D:/video", enabled=True)

# initialize Q table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# set hyperparameters
learning_rate = 0.1
gamma = 0.9
epsilon = 0.5
epsilon_decay = 0.99

# create agent
agent = SarsaAgent(all_actions, learning_rate, gamma, Q)

max_steps_per_episode = 500

# start training
for episode in range(400):
    episode_reward = 0
    s = env.reset()  # 重置环境 #(36, {'prob': 1})
    s = s[0]
    for iter in range(max_steps_per_episode):
        a = agent.choose_action(s, epsilon)
        s_, r, isdone, info, prob = env.step(a)
        episode_reward += r
        # Decay epsilon

        a_ = agent.choose_action(s_, epsilon)
        agent.learn(s, a, r, s_, a_, isdone)
        s = s_
        if isdone:
            break
    epsilon *= epsilon_decay
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', epsilon)

s = env.reset()
s = s[0]
while True:
    a = agent.choose_action(s,1e-500)
    s_, r, isdone, info, prob = env.step(a)
    env.render()
    s = s_
    if isdone:
        break

env.close()

####### END CODING HERE #######

