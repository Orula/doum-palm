import vizdoom as vzd
import random
import numpy as np

import os
from time import sleep, time
from collections import deque
from tqdm import trange
import skimage.transform


print("Initializing doom...")
game = vzd.DoomGame()
#game.load_config("ViZDoom/scenarios/basic.cfg")
game.load_config("ViZDoom/scenarios/defend_the_center.cfg")

#game.set_window_visible(False)
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



#parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

shoot = [0, 0, 1]
left  = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

num_epochs      =  5
steps_per_epoch = 2000

start_time = time()

for epoch in range(num_epochs):
  game.new_episode()
  scores = []
  global_step = 0
  print("\nEpoch #" + str(epoch + 1))
  for _ in trange(steps_per_epoch, leave=False):
    state  = preprocess(game.get_state().screen_buffer)
    #action = agent.get_action(state)
    action = random.choice([0,1,2])

    reward = game.make_action(actions[action], frame_repeat)
    done = game.is_episode_finished()

    if done: break

    next_state = preprocess(game.get_state().screen_buffer)
    #agent.append_memory(state, action, reward, next_state, done)

    #if global_step > agent.batch_size: agent.train()

  scores.append(game.get_total_reward())
  game.new_episode()

  global_step += 1

  #agent.update_target_net()
  scores = np.array(scores)
  print("Results: mean: %.1f +/- %.1f," % (scores.mean(), scores.std()), "min: %.1f," % scores.min(), "max: %.1f," % scores.max())

game.close()
