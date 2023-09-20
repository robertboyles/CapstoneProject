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
import time
import matplotlib.pyplot as plt

import os

trackdata = TrackDataReader.readCSV("./data/first_corners.csv")
bStraightline_overload = False
reward_fun = path_finding

# Track data
track : TrackDefinition = GetBaseTrack(trackdata)

# Car Model
car : BicycleModel = GetBaseCar()
control_freq = 10.0
odemodel : Environment = Environment(vehicleModel=car, track=track, 
                                    fixed_update=control_freq)
env = EnvironmentGym(model=odemodel, reward_fun=reward_fun, pdf_interval=1000000)

agent = SAC.load('/home/rboyles/CapstoneProject/CapstoneProject/best_models/dynamic_2/best_model.zip', env=env)

obs, _ = env.reset()
done = False
score = 0
env.save_path = './model_compare/best'
start_time = time.time()
reward_history = []
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action) # np.array([1, 0.0])
    done = done | truncated
    score += reward
    reward_history.append(reward)
    print('Score : %.2f \t Step Reward = %.4f \t vx = %.2f' % (score, reward, obs[6]))
#print('Score : %.2f \t Step Reward = %.4f \t vx = %.2f' % (score, reward, obs[6]))
t1 = env.model.GetTime()
s1 = env.model.GetStateTrajectory('s')
dsdt1 = env.model.GetOutputTrajectory('dsdt')
sf = env.model.sfinal
tfinal = np.interp(sf, s1, t1)
print('Reward = ' + score.__str__() + ' Laptime = ' + tfinal.__str__() + ' Wall time = ' + (time.time() - start_time).__str__())
env.render(mode=None)

r1 = np.array(reward_history)


agent = SAC.load('/home/rboyles/CapstoneProject/CapstoneProject/terminal_models/dynamic_2.zip', env=env)

obs, _ = env.reset()
env.steps_count = 0
done = False
score = 0
env.save_path = './model_compare/final'
start_time = time.time()
reward_history = []
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action) 
    done = done | truncated
    score += reward
    reward_history.append(reward)
    print('Score : %.2f \t Step Reward = %.4f \t vx = %.2f' % (score, reward, obs[6]))
#print('Score : %.2f \t Step Reward = %.4f \t vx = %.2f' % (score, reward, obs[6]))
t2 = env.model.GetTime()
s2 = env.model.GetStateTrajectory('s')
dsdt2 = env.model.GetOutputTrajectory('dsdt')
sf = env.model.sfinal
tfinal = np.interp(sf, s2, t2)
print('Reward = ' + score.__str__() + ' Laptime = ' + tfinal.__str__() + ' Wall time = ' + (time.time() - start_time).__str__())
env.render(mode=None)

r2 = np.array(reward_history)

r1 = np.insert(r1, 0, 0)
r2 = np.insert(r2, 0, 0)

if t1[-1] > t2[-1]:
    # sample 2 on to 1
    t2_interp = np.interp(s1, s2, t2)
    r2_interp = np.interp(s1, s2, r2)
    dsdt2_interp = np.interp(s1, s2, dsdt2)
    s2, t2, r2, dsdt2 = s1, t2_interp, r2_interp, dsdt2_interp
else:
    t1_interp = np.interp(s2, s1, t1)
    r1_interp = np.interp(s2, s1, r1)
    dsdt1_interp = np.interp(s2, s1, dsdt1)
    s1, t1, r1, dsdt1 = s2, t1_interp, r1_interp, dsdt1_interp

plt.figure()
plt.subplot(4,1,1)
plt.plot(s1,r1)
plt.plot(s2,r2)
plt.subplot(4,1,2)
plt.plot(s1,dsdt1)
plt.plot(s2,dsdt2)
plt.subplot(4,1,3) 
plt.plot(s1, r2 - r1)
plt.subplot(4,1,4) 
plt.plot(s1, t2 - t1)
plt.savefig('reward_overlay.pdf')