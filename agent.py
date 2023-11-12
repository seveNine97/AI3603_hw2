import math, os, time, sys
import numpy as np
import gym

class SarsaAgent(object):
    def __init__(self, all_actions, learning_rate, gamma, Q):
        """
        initialize the agent.
        learning_rate: The learning rate determines to what extent the newly acquired information
        overrides the old information.
        gamma: The discount factor determines the importance of future rewards.
        epsilon: The epsilon value determines the balance between exploration and exploitation.
        epsilon_decay: The rate at which epsilon decays over time.
        """
        self.all_actions = all_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = Q

    def choose_action(self, observation, epsilon):
        """
        Choose action with an epsilon-greedy algorithm.
        With probability epsilon, choose a random action; otherwise, choose the action with the highest Q-value.
        """
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.all_actions)
        else:
            # Choose the action with the highest Q-value
            action = np.argmax(self.Q[observation])
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Learn from experience using the Sarsa algorithm.
        Update the Q-value based on the observed state, action, reward, next state, and next action.
        """
        td_target = reward + self.gamma * self.Q[next_state][next_action] * (1 - done)
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error


    def your_function(self, params):
        """You can add other functions as you wish."""
        return None


class QLearningAgent(object):
    def __init__(self, all_actions, learning_rate, gamma, epsilon, epsilon_decay, Q):
        """
        initialize the agent.
        learning_rate: The learning rate determines to what extent the newly acquired information overrides the old information.
        gamma: The discount factor determines the importance of future rewards.
        epsilon: The epsilon value determines the balance between exploration and exploitation.
        epsilon_decay: The rate at which epsilon decays over time.
        """
        self.all_actions = all_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = Q

    def choose_action(self, observation):
        """
        Choose action with an epsilon-greedy algorithm.
        With probability epsilon, choose a random action; otherwise, choose the action with the highest Q-value.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            # Choose the action with the highest Q-value
            action = np.argmax(self.Q[observation])
        return action

    def learn(self, state, action, reward, next_state, done):
        """
        Learn from experience using the Q-learning algorithm.
        Update the Q-value based on the observed state, action, reward, and next state.
        """
        td_target = reward + self.gamma * np.max(self.Q[next_state]) * (1 - done)
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error

        # Decay epsilon
        if self.epsilon > 0:
            self.epsilon *= self.epsilon_decay

    def your_function(self, params):
        """You can add other functions as you wish."""
        return None
