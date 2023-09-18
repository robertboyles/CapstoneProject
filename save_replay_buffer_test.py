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
from gymnasium.wrappers.time_limit import TimeLimit
from read_track_data import TrackDataReader
from Rewards import *
from custom_callbacks import callbackset
from BaselineModels import GetBaseCar_v1_0 as GetBaseCar, GetBaseTrack_v1_0 as GetBaseTrack
import os

save_name = "Replay_Buffer_Save_Test_Parent"
save_ERB = True # 200MB each @ 1e6 buffer size

resume_from_termination, reset_steps_count = True, False
stage_name = '10000_more'

trackdata = TrackDataReader.readCSV("./data/old_T10.csv")
bStraightline_overload = False
reward_fun = path_following
batch_size, gamma, learning_starts = 1024, 0.99, 100

nepisodes = 10000
log_params_freq = 1000000
eval_steps_freq = 1000000
pdf_interval = 1000

if resume_from_termination:
    parent_name = save_name
    save_name = save_name + stage_name

echo_freq = 5
tensorboard_path = './tfb/' + save_name
log_save_path = './logs/' + save_name
eval_save_path = './evaluations/' + save_name
best_save_path = './best_models/' + save_name
learning_path = './plots/' + save_name
terminal_path = './terminal_models'

isExist = os.path.exists(terminal_path)
if not isExist:
    os.makedirs(terminal_path)

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

# Agent
if resume_from_termination:
    model : SAC = SAC.load(os.path.join(terminal_path, parent_name), env=env)
    if os.path.exists(os.path.join(terminal_path, parent_name + '_ERB.pkl')):
        model.load_replay_buffer(os.path.join(terminal_path, parent_name + '_ERB.pkl'))
    tb_log_name = stage_name
else:
    model = SAC(MlpPolicy, env)
    tb_log_name = 'parent'

# Learn
model.batch_size = batch_size
model.gamma = gamma
model.tensorboard_log = tensorboard_path
model.verbose = 1
model.learning_starts = learning_starts
model.env.reset()

callbacks = callbackset(log_save_path, save_name, log_params_freq, control_freq, eval_save_path, best_save_path, env, eval_steps_freq)
model.learn(total_timesteps=nepisodes, log_interval=echo_freq, 
            callback=callbacks, reset_num_timesteps=reset_steps_count, 
            tb_log_name=tb_log_name)

# Save termnial
model.save(os.path.join(terminal_path, save_name))
if save_ERB:
    model.save_replay_buffer(os.path.join(terminal_path, save_name + '_ERB'))
