import numpy as np
from opengym_wrapper import EnvironmentGym
from environment import Environment
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from gymnasium.wrappers.time_limit import TimeLimit
from read_track_data import TrackDataReader
from Rewards import *
from custom_callbacks import callbackset
from BaselineModels import GetBaseCar_v1_0 as GetBaseCar, GetBaseTrack_v1_0 as GetBaseTrack
import os

# Straight-line overload
s = np.linspace(0, 1000, 1000)
k = np.zeros(s.shape)
trackdata = {'s': s, 'k': k}

reward_fun = path_following

# Track data
track : TrackDefinition = GetBaseTrack(trackdata)

# Car Model
car : BicycleModel = GetBaseCar()
control_freq = 10.0
odemodel : Environment = Environment(vehicleModel=car, track=track, 
                                    fixed_update=control_freq)
env = EnvironmentGym(model=odemodel, reward_fun=reward_fun, pdf_interval=1000000)

agent = SAC.load('/home/rboyles/CapstoneProject/CapstoneProject/best_models/Bo_1C/best_model.zip', env=env)

obs, _ = env.reset()
done = False
score = 0
env.save_path = './model_compare/live'
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action) # np.array([1, 0.0])
    score += reward
    print('Score : %.2f \t Step Reward = %.4f \t vx = %.2f' % (score, reward, obs[6]))
#print('Score : %.2f \t Step Reward = %.4f \t vx = %.2f' % (score, reward, obs[6]))
env.render(mode=None)

obs, _ = env.reset()
env.steps_count = 0
done = False
score = 0
env.save_path = './model_compare/best'
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(np.array([1, 0.0])) # 
    score += reward
    print('Score : %.2f \t Step Reward = %.4f \t vx = %.2f' % (score, reward, obs[6]))
#print('Score : %.2f \t Step Reward = %.4f \t vx = %.2f' % (score, reward, obs[6]))
env.render(mode=None)
