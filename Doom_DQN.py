import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 
import torch.optim as optim

import vizdoom as vzd
from collections import deque

import numpy as np
import random                # Handling random number generation
import time                  # Handling time calculation

import matplotlib.pyplot as plt
from utils import *

class Network(nn.Module):
  def __init__(self, in_dims, out_dims, lr):
    super(Network, self).__init__()
    self.in_dims = in_dims
    self.out_dims = out_dims

    self.features = nn.Sequential(
      nn.Conv2d(in_dims[0], 32, kernel_size=8, stride=4), nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),         nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),         nn.ReLU()
    )
    
    self.fc = nn.Sequential(
      nn.Linear(self.feature_size(), 512),  nn.ReLU(),
      nn.Linear(512, out_dims)
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

  def feature_size(self):
    return self.features(autograd.Variable(torch.zeros(1, *self.in_dims))).view(1, -1).size(1)

class DQN:
  def __init__(self, in_dims, out_dims, MaxSize):
    self.lr = 1e-3 
    self.gamma   = 0.99
    self.epsilon = 1.0
    self.eps_min = 0.05
    self.eps_dec = 5e-4

    self.out_dims = out_dims
    
    # neural networks
    self.policy = Network(in_dims, out_dims, self.lr).to('cpu') # policy network
    self.target = Network(in_dims, out_dims, self.lr).to('cpu') # target network
    self.target.eval() # since no learning is performed on the target net

    # memory
    self.idx  = 0         # postion in memory stack
    self.size = MaxSize   # max size of memory stack
    self.states  = np.zeros((MaxSize, *in_dims), dtype=np.float32)
    self.nstates = np.zeros((MaxSize, *in_dims), dtype=np.float32)
    self.actions = np.zeros( MaxSize, dtype=np.int32)
    self.rewards = np.zeros( MaxSize, dtype=np.float32)
    self.dones   = np.zeros(MaxSize, dtype=bool)

  def store(self, state, action, reward, nstate, done):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = state
    self.actions[idx] = action
    self.rewards[idx] = reward
    self.nstates[idx] = nstate
    self.dones[idx] = done

  def sample(self, batch_size, device='cpu'):
    csize = min(self.idx, self.size)    # current size of memory stack
    batch = np.random.choice(csize, batch_size) # sample batch
    
    states  = torch.from_numpy( self.states[batch]  )
    rewards = torch.from_numpy( self.rewards[batch] )
    nstates = torch.from_numpy( self.nstates[batch] )
    actions = self.actions[batch] 
    dones   = self.dones[batch]   
    return states, actions, rewards, nstates, dones

  def get_action(self, state):
    if np.random.random() > self.epsilon:
      state   = torch.tensor([state], dtype=torch.float32).to('cpu') # make sure state is pytorch tensor
      actions = self.policy.forward(state)  # collect actions values
      action  = torch.argmax( actions ).item()  # return action with highest Q
    else:
      action = np.random.randint( self.out_dims, size=1)[0] 
    return action

  def update_target_net(self):
    self.target.load_state_dict( self.policy.state_dict() )

  def update_epsilon(self):
    self.epsilon = max(self.eps_min, self.epsilon*self.eps_dec)
  
  def train(aelf, batch_size=64):
    if len(self.memory) < batch_size: return # if buffer not full don't learn 
    else:
      states, actions, rewards, nstates, terminals = self.memory.sample(batch_size)

      indices = np.arange(self.batch_size)

      q_pred = self.policy.forward(states)[indices, actions]
      q_next = self.target.forward(nstates).max(dim=1)[0]

      q_next[dones] = 0.0  # set all terminal states' value to zero 
      q_target = rewards + self.gamma*q_next

      loss = self.policy.loss(q_target, q_pred).to('cpu')
      self.policy.optimizer.zero_grad()   # clearing tempory gradients 
      loss.backward()                      # determine gradients
      self.policy.optimizer.step()         # update weights

      #self.update_epsilon()    # update epsilon value after each episode 
      #self.update_target_net() # update target network after each episode

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

agent = DQN(INPUT_SHAPE, ACTION_SIZE, MaxSize=5000)


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
    action = agent.get_action(state)
    reward = game.make_action(action_space[action])
    done = game.is_episode_finished()

    step_cnt +=1
    score += reward
    if step_cnt % update_every == 0: agent.train()
    if done: 
      agent.store(state, action, reward, state, done)
      break
    else:
      next_state = stack_frames(state, game.get_state().screen_buffer.transpose(1, 2, 0), False)
      agent.store(state, action, reward, next_state, done)
      state = next_state

  agent.update_epsilon()    # update epsilon value after each episode 
  agent.update_target_net() # update target network after each episode
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

