import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical

import vizdoom as vzd
from collections import deque

import numpy as np
import random                # Handling random number generation
import time                  # Handling time calculation

import matplotlib.pyplot as plt
from utils import *

class ActorCritic(nn.Module):
  def __init__(self, in_dims, out_dims, lr=0.003):
    super(ActorCritic, self).__init__()
    self.in_dims  = in_dims
    self.out_dims = out_dims

    self.conv = nn.Sequential(
      nn.Conv2d(in_dims[0], 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU()
    )
        
    self.fc_actor = nn.Sequential(
      nn.Linear(self.conv_size(), 512),  nn.ReLU(),
      nn.Linear(512, out_dims),          nn.Softmax(dim=1)
    )

    self.fc_critic = nn.Sequential(
      nn.Linear(self.conv_size(), 512),  nn.ReLU(),
      nn.Linear(512, 1)     
    )
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    value = self.fc_critic(x)
    probs = self.fc_actor(x)
    return probs, value
  
  def conv_size(self):
    return self.conv(autograd.Variable(torch.zeros(1, *self.in_dims))).view(1, -1).size(1)


class PPO:
  def __init__(self, in_dims, out_dims):
    self.lr    = 0.0001
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 5
    self.eps_clip = 0.2
    self.bsize = 32 # batch_size

    self.memory = []
    self.ACnet = ActorCritic(in_dims, out_dims)

  def get_action(self, state):
    state = torch.from_numpy(state).unsqueeze(0).to('cpu')
    probs, value = self.ACnet.forward(state)
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob, value

  def train(self):
    for _ in range( self.epoch):
      states  = np.array([x[0] for x in self.memory])
      actions = np.array([x[1] for x in self.memory])
      rewards = np.array([x[2] for x in self.memory])
      l_probs = np.array([x[3] for x in self.memory])
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


      _states  = torch.tensor(states , dtype=torch.float).to('cpu')
      ol_probs = torch.tensor(l_probs, dtype=torch.float).to('cpu')
      _actions = torch.tensor(actions, dtype=torch.float).to('cpu')

      _, nl_probs, crit = self.get_action(_states)

      ratio = nl_probs.exp() / ol_probs.exp()

      # Finding Surrogate Loss
      surr1 = ratio * advantage
      surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
      returns = advantage + values

      # final loss of clipped objective PPO: loss = actor_loss + 0.5*critic_loss 
      loss = -torch.min(surr1, surr2).mean() + 0.5*((returns-crit)**2).mean()

      # take gradient step
      self.ACnet.optimizer.zero_grad()
      loss.mean().backward()
      self.ACnet.optimizer.step()

    del self.memory[:]
    pass


print("Initializing doom...")
game = vzd.DoomGame()
game.load_config("scenarios/defend_the_center.cfg")            # Load configurations
game.set_doom_scenario_path("scenarios/defend_the_center.wad") # Load scenarios

#game.load_config("ViZDoom/scenarios/health_gathering.cfg")
#game.set_doom_scenario_path("scenarios/health_gathering.wad") 

game.set_window_visible(False)                                 # env.render() == False 
print("Doom initialized.")

# Training environment configs
action_space = np.identity(3,dtype=int).tolist()
INPUT_SHAPE  = (4, 84, 84)
ACTION_SIZE  = len(action_space)

update_every = 1000   # how often to update the network
num_episodes = 1000   # number of training episodes

agent = PPO(INPUT_SHAPE, ACTION_SIZE)


scores = []
scores_window = deque(maxlen=20)

print("Training agent")
game.init()
for epi in range(1, num_episodes+1):
  game.new_episode()
  score = 0
  step_cnt = 0
  state = stack_frames(None, game.get_state().screen_buffer.transpose(1, 2, 0), True) 
  while True:
    action, log_probs, value = agent.get_action(state)
    reward = game.make_action(action_space[action])
    done = game.is_episode_finished()
    agent.memory.append( (state, action, reward, log_probs, value, done) )

    step_cnt +=1
    score += reward

    if step_cnt % update_every == 0: agent.train()
    if done: break
    # get next state
    state = stack_frames(state, game.get_state().screen_buffer.transpose(1, 2, 0), False)
  scores.append(score)

  print('Episode {} Episode score: {:.2f} Running average score: {:.2f}'.format(epi, score, np.mean(scores[-100:])))

  scores_window.append(score)       # save most recent score
  scores.append(score)              # save most recent score

game.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('foo.png')
plt.show()


