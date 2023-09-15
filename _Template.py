import sys
import os

# add parent directory 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

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
from BaselineModels import GetBaseCar_v1_0 as GetBaseCar, GetBaseTrack_v1_0 as GetBaseTrack
import os

raise NotImplementedError

save_name = "TEMPLATE"

trackdata = TrackDataReader.readCSV("./data/old_T10.csv")
bStraightline_overload = False
reward_fun = path_following

nepisodes = 1000000
log_params_freq = 10000
eval_steps_freq = 2500
pdf_interval = 2000

echo_freq = 5
tensorboard_path = './tfb/' + save_name
log_save_path = './logs/' + save_name
eval_save_path = './evaluations/' + save_name
best_save_path = './best_models/' + save_name
learning_path = './plots/' + save_name

if bStraightline_overload:
    # Straight-line overload
    s = np.linspace(0, 1000, 1000)
    k = np.zeros(s.shape)
    trackdata = {'s': s, 'k': k}

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
