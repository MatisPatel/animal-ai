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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# define path to config file
configuration_file = Path("..", "..", "configs", "tests", "optic_test_oneGoal.yaml")


# Hyper Parameters taken from DQN paper (atari)
GAMMA = 0.99 # discount rate for computing temporal diff target
BATCH_SIZE = 32 # how many transitions to sample from replay buffer
BUFFER_SIZE = 500000 # how many transitions to store in replay buffer
MIN_REPLAY_SIZE = 1000 # no of trans before training
EPSILON_START = 1.0 # starting epsiolon (random actions)
EPSILON_END = 0.05 # ending rate of random actions
EPSILON_DECAY = 50000 # decay period over which epsilon will decay from start to end values
TARGET_UPDATE_FREQ=1000 # after how many transitions our target network gets updated by the online network
LEARNING_RATE = 1e-4
# number of steps to run model training for 
NUM_STEPS = 10000
DOUBLEQ = True

# define network feed forward mlp takes two hiddenlayer args that define size of layer 1 and 2 default is 32
class mlp_network(nn.Module):
    def __init__(self, env, hiddenDim_1, hiddenDim_2, isDoubleQ=True):
        super().__init__()
        self.hiddenDim_1 = hiddenDim_1
        self.hiddenDim_2 = hiddenDim_2
        self.isDoubleQ = isDoubleQ
    
        in_features = int(np.prod(env.observation_space[0].shape))
        # in_features = 2

        self.net = nn.Sequential(
            nn.Linear(in_features, hiddenDim_1),
            nn.Tanh(),
            nn.Linear(hiddenDim_2, env.action_space.n)
        )
    
    def forward(self, x):
        return self.net(x) 
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs[0], dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0)) 

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item() 
        
        return action

if __name__ == "__main__":
    # env = gym.make('CartPole-v0')
    # test1 = mlp_network(env, 16, 16)
    # print(test1)
    # test2 = mlp_network(env)
    # print(test2)

    #make the animal-ai env basedon config file
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
        raysPerSide = 2,
        rayMaxDegrees = 60,
    )
    # wrap it into a gym enviorn
    env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=True, flatten_branched=True)
    # make a small 10/10 network
    # model = mlp_network(env, 10, 10).to(device)

    # make buffer dequeue
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    # make buffer to track progress
    rew_buffer = deque([0.0], maxlen=100)

    # store reward for most recent episode
    episode_reward = 0.0
    hiddenDim_1 = 32 
    hiddenDim_2 = 32
    # define online and target nets
    online_net = mlp_network(env, hiddenDim_1, hiddenDim_2)
    target_net = mlp_network(env, hiddenDim_1, hiddenDim_2) 
    # set target net params same as online net
    target_net.load_state_dict(online_net.state_dict())

    # create optimizer 
    optimiser = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)


    # initialise replay buffer 
    obs = env.reset()
    # loop over the number needed
    for _ in range(MIN_REPLAY_SIZE):
        # randomly sample action space
        action = env.action_space.sample() 
        # apply random action to env and get back new state 
        # new obs of state, reward gained, if the env is done, info dict
        new_obs, rew, done, info = env.step(action)
        # generate this memeory of the expereince
        transition = (obs, action, rew, done, new_obs)
        # add to replay buffer 
        replay_buffer.append(transition) 
        # set obs to the new obs 
        obs = new_obs

        # if env is finished reset
        if done:
            obs = env.reset() 

    # print(replay_buffer)
    # Main Training Loop       
    obs = env.reset()

    for step in range(NUM_STEPS):
        # step=1
        # decay epsilon by taking the correct linear inerped value
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END] )

        # seelct random action with chance epsilon
        if random.random() <= epsilon:
            action = env.action_space.sample()
        # else make an intelligent ation from network 
        else: 
            action = online_net.act(obs) 

        new_obs, rew, done, info = env.step(action)
        # generate this memeory of the expereince
        transition = (obs, action, rew, done, new_obs)
        # add to replay buffer 
        replay_buffer.append(transition) 
        # set obs to the new obs 
        obs = new_obs

        # store episode reward
        episode_reward += rew

        # if env is finished reset
        if done:
            obs = env.reset() 

            rew_buffer.append(episode_reward)
            episode_reward = 0.0 

        # start gradient update step 
        # get a random sample of transitions from replay buffer
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        # unpack all the elements of transitions into seperate lists
        observations, actions, rewards, dones, new_observations = np.asarray(transitions).T

        observations = np.asarray([t[0][0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4][0] for t in transitions])

        observations_t = torch.as_tensor(observations, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32)

        with torch.no_grad():
            if online_net.isDoubleQ:
                # use online net to get q values and the action best chosen
                target_online_q_values = online_net(new_observations_t)
                target_online_best_q_idx = target_online_q_values.argmax(dim=1, keepdim=True)
                # but get q values for that target from the target net using online net idxs
                targets_target_q_values = target_net(new_observations_t)
                targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=target_online_best_q_idx)
                targets = rewards_t + GAMMA * (1-dones_t) * targets_selected_q_values
            else:
                # compute targets for loss function
                # calculate q values for each observation 
                target_q_values = target_net(new_observations_t)
                # calculate the max q value or the action the target net thinks is best
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                # caclulate targets 
                targets = rewards_t + GAMMA * (1-dones_t) * max_target_q_values

        # compute loss 
        # q vlaues from the online net 
        q_values = online_net(observations_t)
        # q values of actions actually taken by online net
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t) 
        # calc loss 
        loss = nn.functional.smooth_l1_loss(action_q_values, targets) 


        #gradient descent setep 
        optimiser.zero_grad() 
        # compute gradients
        loss.backward() 
        # clip gradients -1 to 1
        for param in online_net.parameters():
            param.grad.data.clamp_(-1,1)
        #update network
        optimiser.step()

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # logging 
        if step % 5000 == 0:
            print() 
            print('Step', step)
            print('epsilon', epsilon)
            print('Rew', np.mean(rew_buffer))

        if step % 100000 == 0:
            torch.save(online_net, Path("models", "mlp_test", "test_{}.pth".format(step)))

    env.close()