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
from TerminationFunctions import *
from custom_callbacks import callbackset
from BaselineModels import GetBaseCar_v1_0 as GetBaseCar, GetBaseTrack_v1_0 as GetBaseTrack
import os
from Rewards import _default_reward_weights

parent_name = "Baseline_FT3_Tn2M_Extension"
save_ERB = True # 200MB each @ 1e6 buffer size

terminal_reward = 0.0
_W_reward = _default_reward_weights()

policy_kwargs = None
#policy_kwargs = dict(net_arch=dict(qf=[1024, 1024], pi=[1024, 1024]))

resume_from_termination, load_ERB, reset_steps_count = True, True, False
stage_name = '_2M_Extension'

trackdata = TrackDataReader.readCSV("./data/first_corners.csv")
bStraightline_overload = False
reward_fun = path_finding
termination_fun = FixedTimeTermination(
    tlimit0=20.0,
    tlimit_min=10.0,
    tlimit_max=30.0,
    deltaOnSuccess=-3,
    deltaOnExceed=10)
#FixedTimeTermination(tlimit0=10.0, deltaOnExceed=+10.0, deltaOnSuccess=-3.0)
batch_size, gamma, learning_starts = 1024, 0.99, 100
seed = 0
ERB_Size = 1000000

nepisodes = 2000000
log_params_freq = 20000
eval_steps_freq = 5000
pdf_interval = 4000

save_name = parent_name
if resume_from_termination:
    save_name = save_name + stage_name

echo_freq = 5
tensorboard_path = './tfb/' + parent_name
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
env = EnvironmentGym(model=odemodel, reward_fun=reward_fun, pdf_interval=pdf_interval, save_path=learning_path, 
                     termination_fun=termination_fun, reward_weights=_W_reward,
                     terminal_reward_value=terminal_reward)

# Agent
if resume_from_termination:
    model : SAC = SAC.load(os.path.join(terminal_path, parent_name), env=env) # Will inherit original ERB size
    if os.path.exists(os.path.join(terminal_path, parent_name + '_ERB.pkl')) and load_ERB:
        model.load_replay_buffer(os.path.join(terminal_path, parent_name + '_ERB.pkl'))
    tb_log_name = stage_name
else:
    model : SAC = SAC(MlpPolicy, env, seed=seed, buffer_size=ERB_Size, policy_kwargs=policy_kwargs)
    tb_log_name = 'parent'

# Learn
model.batch_size = batch_size
model.gamma = gamma
model.tensorboard_log = tensorboard_path
model.verbose = 1
model.learning_starts = learning_starts
model.env.reset()

callbacks = callbackset(log_save_path, save_name, log_params_freq, control_freq, eval_save_path, best_save_path, model.env, eval_steps_freq)
model.learn(total_timesteps=nepisodes, log_interval=echo_freq, 
            callback=callbacks, reset_num_timesteps=reset_steps_count, 
            tb_log_name=tb_log_name)

# Save termnial
model.save(os.path.join(terminal_path, save_name))
if save_ERB:
    model.save_replay_buffer(os.path.join(terminal_path, save_name + '_ERB'))
