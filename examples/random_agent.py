from stable_baselines3 import DQN

from gym_unity.envs import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment

import sys
import random
import os

def rand_agent_single_config(configuration_file):

    # AnimalAI settings must be the same as used for training except with inference = true
    # best to load from a common config file
    configuration_file = "../configs/tests/optic_test_noWalls.yaml"
    aai_env = AnimalAIEnvironment( 
        inference=True, #Set true when watching the agent
        # seed = 123,
        worker_id=random.randint(0, 65500),
        file_name="../env/AnimalAI",
        arenas_configurations=configuration_file,
        useCamera=True,
        resolution=36,
        useRayCasts=False,
    )

    env = UnityToGymWrapper(aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=True)
    runname = "testrun" #Assume you have your model saved in results/runname/
    model_no = "70000"

    model = DQN.load(F"results/{runname}/model_{model_no}")
    obs = env.reset()
    while True:
        action = random.randint(0, 8)
        obs, reward, done, info = env.step(action)
        # print(obs)
        env.render()
        if done:
            obs=env.reset()
    env.close()

if __name__ == "__main__":
    print("This is an example script that shows how you might load and watch a trained agent.")
    print("You will need to edit it for your needs.")

    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]
    else:   
        competition_folder = "configs/competition/"
        configuration_files = os.listdir(competition_folder)
        configuration_random = random.randint(0, len(configuration_files))
        configuration_file = (
            competition_folder + configuration_files[configuration_random]
        )
    rand_agent_single_config(configuration_file=configuration_file)
