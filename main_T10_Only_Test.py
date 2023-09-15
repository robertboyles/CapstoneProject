
import numpy as np
from opengym_wrapper import EnvironmentGym
from environment import Environment
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from gym.wrappers.time_limit import TimeLimit
from read_track_data import TrackDataReader
from Rewards import *
from custom_callbacks import callbackset
from BaselineModels import GetBaseCar, GetBaseTrack
import os

# Task Definitions
save_name = "testing"

nepisodes = 5000000

echo_freq = 5

tensorboard_path = './tfb/' + save_name

log_save_path = './logs/' + save_name
log_params_freq = 10000

eval_save_path = './evaluations/' + save_name
eval_steps_freq = 2000


best_save_path = './best_models/' + save_name

learning_path = './plots/' + save_name
pdf_interval = 3000


trackdata = TrackDataReader.readCSV("./data/old_T10.csv")
reward_fun = path_following

# Track data
track : TrackDefinition = GetBaseTrack(trackdata)

# Car Model
car : BicycleModel = GetBaseCar()

# Environment Model
control_freq = 10.0
odemodel : Environment = Environment(vehicleModel=car, track=track, 
                                    fixed_update=control_freq)
env = EnvironmentGym(model=odemodel, reward_fun=reward_fun, pdf_interval=pdf_interval, save_path=learning_path)
env = TimeLimit(env, max_episode_steps=4000)

# Agent and learn
model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_path, 
            batch_size=1024, gamma=0.99)
callbacks = callbackset(log_save_path, save_name, log_params_freq, control_freq, eval_save_path, best_save_path, env, eval_steps_freq)
model.learn(total_timesteps=nepisodes, log_interval=echo_freq, callback=callbacks)
model.save(save_name)
