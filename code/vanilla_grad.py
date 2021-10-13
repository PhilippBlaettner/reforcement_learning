import torch
from torch._C import _parse_source_def
import torch.nn as nn
import torch.nn.functional as F
import gym
import random
from collections import deque
import numpy as np


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.action_size = action_size

    def forward(self, x):
        h = F.relu(self.fc1(x))
        o = F.softmax(self.fc2(h))
        return o

    def get_action(self,x, eps):
        action_prob = self.forward(x)
        if  random.random < eps:
            # fix this
             action = torch.randint(0,self.action_size)
        else:
            action = torch.argmax(action_prob)
        return action

class Trainer:
    """ A Trainer class run a simple 
    """
    def __init__(self, agent, learning_rate, epoch_size, batch_size, sample_size, nr_epochs, mem_cap):
        self.agent = agent
        self.env = env = gym.make('CartPole-v0')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.sample_size = sample_size
        self.nr_epochs = nr_epochs

        memory = deque(maxlen=mem_cap)


    def _collect_episode(env, agent, max_length, render=False):
        observation = env.reset()

        observations = []
        actions = []
        rewards = []

        for i in range(max_length):
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            if done:
                break
            if render:
                env.render()
        return [observations, actions, rewards]

    def train(self):

        sampled_av_reward = []
        losses = []

        for n in range(self.nr_epochs):
            with torch.no_grad:
                current_av_reward = self._collect_data()
                sampled_av_reward.append(current_av_reward)
            print("The current sampled return is: %s".format(current_av_reward))

            current_loss = self.train_epoch()
            losses.append(current_loss)
            print("The current training loss is: %s".format(current_av_reward))
        return sampled_av_reward, losses


    def test(self, render=False):
        pass

    def train_epoch(self):
        

        return 0

    def train_batch(self):
        pass

    def _sample_batch(self):
        episodes = zip(*random.sample(self.memory, self.batch_size))

        observations, actons, rewards = list(episodes)

    def _estimate_return(self, rewards):
        return rewards

    
    def _collect_data(self):
        for i in range(self.sample_size):
            episode = self._collect_episode(self.env, self.agent, self.max_length)
            self.memory.append(episode)



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation = env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    
    env.close()