# Doom bots

## Doum-Palm

Hyphaene thebaica, with common names doum palm (Ar: دوم) and gingerbread tree (also mistakenly **doom palm**) - wikipedia

This project is my attempt at implementing a reinforcement agent to play the original Doom video game, using the vizdoom toolkit.

<p align="center">
	<img src="Images/Figure_1.png" width="400"/>
</p>

## Usage
You can train the model by executing the following command:
```bash
python Doom.py
```
Currenty i am working on this project on my i5-4300U CPU with 8GB RAM and as such I am struggling to run proper tests. I will get to proper test results soon.

## Game rules: doom_defend_center
*The purpose of this scenario is to teach the agent that killing the monsters is GOOD and when monsters kill you is BAD. In addition, wasting amunition is not very good either. Agent is rewarded only for killing monsters so he has to figure out the rest for himself. Map is a large circle. Player is spawned in the exact center. 5 melee-only, monsters are spawned along the wall. Monsters are killed after a single shot. After dying each monster is respawned after some time.Episode ends when the player dies.* - https://deepanshut041.github.io/Reinforcement-Learning/cgames/03_doom_defend_center/

## Bot algorithm
The agent is based on the clip varient of PPO (Proximal Policy Optimization). But i am struggle with the sparse reward problem and as such I might switch to a Q-learning based agent 

# Libraries Used
- vizdoom
- PyTorch
- numpy
- opencv-python
- matplotlib
