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
        super(SimpleNet, self).__init__()
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
    def __init__(self, agent,optimizer, learning_rate, epoch_size, batch_size, sample_size, nr_epochs, mem_cap, discount_rate, max_length):
        self.agent = agent
        self.optimizer = optimizer(self.agent.parameters(),learning_rate)
        self.env = env = gym.make('CartPole-v0')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.sample_size = sample_size
        self.nr_epochs = nr_epochs
        self.discount_rate = discount_rate
        self.max_length = max_length

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
        av_loss = 0
        for i in range(self.epoch_size):
            batch = self._sample_batch()
            self.optimizer.zero_grad()
            av_loss += self.train_batch(batch)

        return av_loss/self.epoch_size/self.batch_size

    def train_batch(self, batch):
        ob_batch, actions_batch, discounted_returns = batch
        action_prob = self.agent(ob_batch)
        loss = torch.sum(discounted_returns*torch.log(action_prob))
        loss.backward()
        self.optimizer_step()

        

    def _sample_batch(self):
        episodes = zip(*random.sample(self.memory, self.batch_size))

        observations, actions, rewards = list(episodes)
        ob_batch = torch.cat(observations)
        actions_batch = torch.cat(actions)
        discounted_returns = self._estimate_return(rewards, self.discount_rate)
        rewards_batch = torch.cat(discounted_returns)
        return ob_batch, actions_batch, discounted_returns

    def _estimate_return(self, rewards, gamma):
        t_steps = np.arange(rewards.size)
        r = rewards * gamma**t_steps
        r = r[::-1].cumsum()[::-1] / gamma**t_steps
        return r

    
    def _collect_data(self):
        for i in range(self.sample_size):
            episode = self._collect_episode(self.env, self.agent, self.max_length)
            self.memory.append(episode)



if __name__ == '__main__':
    net = SimpleNet(4,16,2)
    adam = torch.optim.Adam
    learning_rate = 0.01
    epoch_size = 10
    batch_size = 16
    sample_size = 100
    nr_epochs = 10
    mem_cap = 2000
    discount_rate = 0.99
    max_length = 200
    Trainer(net, adam, learning_rate, epoch_size, batch_size, sample_size, nr_epochs, mem_cap, discount_rate, max_length)
    