import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import DQN
from gym_unity.envs import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment

import sys
import random
import os

if len(sys.argv) > 1:
    configuration_file = sys.argv[1]
else:   
    configuration_file = "../../configs/tests/optic_test_oneGoal.yaml"

aai_env = AnimalAIEnvironment( 
    inference=True, #Set true when watching the agent
    # seed = 123,
    worker_id=random.randint(0, 65500),
    file_name="../../env/AnimalAI",
    arenas_configurations=configuration_file,
    useCamera=False,
    resolution=36,
    useRayCasts=True,
    raysPerSide = 4,
    rayMaxDegrees = 60
)

env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=False, flatten_branched=True)
env.num_envs  = 1

video_folder = 'results\\videos\\'
video_length = 100

env = DummyVecEnv([lambda: UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=False, flatten_branched=True)]) 

obs = env.reset()


# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="random-agent")

runname = "testrun" #Assume you have your model saved in results/runname/
model_no = "1650000"
model = DQN.load(F"results/{runname}/model_{model_no}")
env.reset()

for _ in range(video_length + 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
# Save the video
env.close()