import vizdoom as vzd
import random

import os
from time import sleep, time
from collections import deque
from tqdm import trange
import skimage.transform

import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorCritic(nn.Module):
  def __init__(self, out_dims, lr=0.003):
    super(ActorCritic, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(8), nn.ReLU(),

      nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(8), nn.ReLU(),

      nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
      nn.BatchNorm2d(8), nn.ReLU(),

      nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
      nn.BatchNorm2d(16), nn.ReLU(),
    )

    self.fc_actor = nn.Sequential(
      nn.Linear(96, 64),       nn.ReLU(),
      nn.Linear(64, out_dims), nn.Softmax(dim=-1)
    )

    self.fc_critic = nn.Sequential(
      nn.Linear(96, 64),       nn.ReLU(),
      nn.Linear(64, 1)
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.to('cpu')

  def actor(self, state):
    z = self.conv(state)
    z = z.view(-1, 192)
    z = z[:, :96]  # input for the net to calculate the state value
    dist = self.fc_actor(z)
    dist = Categorical(dist)
    return dist

  def critic(self, state):
    z = self.conv(state)
    z = z.view(-1, 192)
    z = z[:, :96]  # input for the net to calculate the state value
    value = self.fc_critic(z)
    return value

class PPOAgent:
  def __init__(self, out_dims): 
    super(PPOAgent, self).__init__()

    self.lr    = 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 4
    self.eps_clip = 0.2
    self.bsize = 5 # batch_size

    self.memory = []
    self.AC = ActorCritic(out_dims)

    #self.optimizer = optim.Adam(self.parameters(), lr=lr)
    #self.to('cpu')
    
  def get_action(self, obs):
    obs = np.expand_dims(obs, axis=0)
    obs = torch.from_numpy(obs).float().to('cpu')
    value  = self.AC.critic(obs)
    dist   = self.AC.actor(obs)
    action = dist.sample()
    probs  = torch.squeeze(dist.log_prob(action)).item()
    action = torch.squeeze(action).item()
    value  = torch.squeeze(value).item()
    return action, probs, value

  def store(self, state, action, reward, probs, vals, done):
    self.memory.append( (state, action, reward, probs, vals, done) )

  def train(self):
    for _ in range( self.epoch):
      states  = np.array([x[0] for x in self.memory])
      actions = np.array([x[1] for x in self.memory])
      rewards = np.array([x[2] for x in self.memory])
      probs   = np.array([x[3] for x in self.memory])
      values  = np.array([x[4] for x in self.memory])
      dones   = np.array([1-int(x[5]) for x in self.memory])

      ####### Advantage using gamma returns
      nvalues = np.concatenate([values[1:] ,[values[-1]]])
      delta = rewards + self.gamma*nvalues*dones - values
      advantage, adv = [], 0
      for d in delta[::-1]:
        adv = self.gamma * self.lamda * adv + d
        advantage.append(adv)
      advantage.reverse()

      advantage = torch.tensor(advantage).to('cpu')
      values    = torch.tensor(values).to('cpu')

      # create mini batches
      num = len( states ) 
      batch_start = np.arange(0, num, self.bsize)

      indices = np.arange( num, dtype=np.int64 )
      np.random.shuffle( indices )
      batches = [indices[i:i+self.bsize] for i in batch_start]

      for batch in batches:
        _states  = torch.tensor(states[batch], dtype=torch.float).to('cpu')
        old_probs   = torch.tensor(probs[batch], dtype=torch.float).to('cpu')
        _actions = torch.tensor(actions[batch], dtype=torch.float).to('cpu')

        # Evaluating old actions and values
        dist = self.AC.actor(_states)
        crit = self.AC.critic(_states)
        crit = torch.squeeze(crit)

        # Finding the ratio (pi_theta / pi_theta__old)
        new_probs = dist.log_prob(_actions)
        ratio = new_probs.exp() / old_probs.exp()

        # Finding Surrogate Loss
        surr1 = ratio * advantage[batch]
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
        returns = advantage[batch] + values[batch]

        # final loss of clipped objective PPO: loss = actor_loss + 0.5*critic_loss 
        loss = -torch.min(surr1, surr2).mean() + 0.5*((returns-crit)**2).mean()

        # take gradient step
        self.AC.optimizer.zero_grad()
        loss.mean().backward()
        self.AC.optimizer.step()

    self.memory = []
    pass


print("Initializing doom...")
game = vzd.DoomGame()
game.load_config("ViZDoom/scenarios/simpler_basic.cfg")
#game.load_config("ViZDoom/scenarios/defend_the_center.cfg")
game.set_window_visible(False)
game.set_mode(vzd.Mode.PLAYER)
game.set_screen_format(vzd.ScreenFormat.GRAY8)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.init()
print("Doom initialized.")


def preprocess(img):
  """Down samples image to resolution"""
  img = skimage.transform.resize(img, resolution)
  img = img.astype(np.float32)
  img = np.expand_dims(img, axis=0)
  return img


# parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

shoot = [0, 0, 1]
left  = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

num_epochs      =  20
steps_per_epoch = 2000
batch_size      = 64

#start_time = time()

import itertools as it
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]


agent = PPOAgent(len(actions))


for epoch in range(num_epochs):
  game.new_episode()
  scores = []
  num_steps = 0
  print("\nEpoch #" + str(epoch + 1))
  for _ in trange(steps_per_epoch, leave=False):
    state  = preprocess(game.get_state().screen_buffer)
    action, probs, value = agent.get_action(state)

    reward = game.make_action(actions[action], frame_repeat)

    done = game.is_episode_finished()
    agent.store(state, action, reward, probs, value, done)

    num_steps += 1
    #print( reward )
    #print( game.get_total_reward() )

    if done: 
      #print( game.get_total_reward() )
      scores.append(game.get_total_reward())
      game.new_episode()

    if num_steps % batch_size == 0: 
      agent.train()


  #agent.update_target_net()
  scores = np.array(scores)
  print("Results: mean: %.1f +/- %.1f," % (scores.mean(), scores.std()), "min: %.1f," % scores.min(), "max: %.1f," % scores.max())

game.close()
