from torch import nn
import torch 
import gym 
from collections import deque 
import itertools
import numpy as np 
import random
import sys
import random
import os
from datetime import datetime
from pathlib import Path
from gym_unity.envs import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment
from train_simple_DQN import mlp_network
# define path to config file
configuration_file = Path("..", "..", "configs", "tests", "optic_test_oneGoal.yaml")

NUM_STEPS = 10000
EPSILON = 0.02

#make the animal-ai env basedon config file
aai_env = AnimalAIEnvironment(
    seed = 123,
    worker_id=random.randint(0, 6550),
    file_name="../../env/AnimalAI",
    arenas_configurations=configuration_file,
    play=False,
    base_port=5000,
    inference=True,
    useCamera=False,
    resolution=1,
    useRayCasts=True,
    raysPerSide = 2,
    rayMaxDegrees = 60,
)
# wrap it into a gym enviorn
env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=True, flatten_branched=True)
# laod model from file
model = torch.load(Path("models", "mlp_test", "test_1000000.pth"))

obs = env.reset()

for _ in range(NUM_STEPS):
    # seelct random action with chance epsilon
    if random.random() <= EPSILON:
        action = env.action_space.sample()
    # else make an intelligent ation from network 
    else: 
        action = model.act(obs) 

    obs, rew, done, info = env.step(action)

    env.render() 

    if done:
        env.reset()

env.close()