from stable_baselines3 import DQN, A2C, PPO
import torch as th

import sys
import random
import os
from datetime import datetime
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from gym_unity.envs import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment
# from stable_baselines3 import make_vec_env

# %load_ext tensorboard
# import tensorflow as tf
# import datetime
# def train_agent_single_config(configuration_file):
configuration_file = "..\\..\\configs\\tests\\optic_test.yaml"

aai_env = AnimalAIEnvironment(
    seed = 123,
    worker_id=random.randint(0, 6550),
    file_name="../../env/AnimalAI",
    arenas_configurations=configuration_file,
    play=False,
    base_port=5000,
    inference=False,
    useCamera=False,
    resolution=1,
    useRayCasts=True,
    raysPerSide = 3,
    rayMaxDegrees = 90,
)

# env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=False)
# def make_env():
#     def _thunk():
#         env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=True)
#         return env
#     return _thunk
# env = DummyVecEnv([make_env()])
env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
# env = make_vec_env(env, n_envs = 4)
now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
runname = "2layerWalls_run_{}".format(now)

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[32, 32])
model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=("./dqn_tensorboard/" + runname), buffer_size=1000000, optimize_memory_usage=True)
# runname = "testrun"
# model_no = "1650000"
# model = DQN.load(F"results/{runname}/model_{model_no}")
# model.set_env(env)
# model = A2C("MlpPolicy", env, verbose=1,  tensorboard_log=("./dqn_tensorboard/" + runname))
# env.reset()

no_saves = 10
no_steps = 500000
reset_num_timesteps = True
for i in range(no_saves):
    model.learn(no_steps, reset_num_timesteps=reset_num_timesteps)
    model.save("results/" + runname + "/model_" + str( (i+1)*no_steps ))
    reset_num_timesteps = False
env.close()

# Loads a random competition configuration unless a link to a config is given as an argument.
# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         configuration_file = sys.argv[1]
#     else:   
#         competition_folder = "../configs/competition/"
#         configuration_files = os.listdir(competition_folder)
#         configuration_random = random.randint(0, len(configuration_files))
#         configuration_file = (
#             competition_folder + configuration_files[configuration_random]
#         )
#     train_agent_single_config(configuration_file=configuration_file)
