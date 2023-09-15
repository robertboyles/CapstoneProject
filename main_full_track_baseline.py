import gym
import numpy as np
import time
from Wheel import Wheel, Rill as Tyre
from opengym_wrapper import EnvironmentGym
from environment import Environment
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
from stable_baselines3.sac.policies import MlpPolicy ,SACPolicy
from stable_baselines3.td3.policies import MlpPolicy as MlpPolicy_td3
from stable_baselines3 import SAC, TD3
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.type_aliases import Schedule
from read_track_data import TrackDataReader

loadcheckpoint, from_log = False, False
save_name = "sac_pathfinding_rewards_easyCar"
save_name = "./logs/" + save_name if from_log else save_name
nepisodes = 5000000

# Environment setup
s = np.linspace(0, 1000, 10000)
k = np.ones(s.shape) * 1e-1

trackdata = TrackDataReader().load_example_data()
    
s = trackdata['s']
k = trackdata['k']

track : TrackDefinition = TrackDefinition(s, k, width=10.0, k_error_scale=1.5)
track.X0 = np.array([0, 0, 0, 0, 0, 0])

chassis_params = {
            'mass': 500,
            'Izz' : 2000,
            'lf' : 1.5,
            'lr' : 1.5,
            'steering_ratio': -12.0, # aHandWheel/aSteer
            'max_aHandWheel': 200 * np.pi/180.0
        }

tyref_params = {
            'FzN':4000,     'Fz2N':8000, 
            'dF0xN':200000, 'dF0x2N':210000,
            'dF0yN':80000,  'dF0y2N':90000,
            'sMxN':0.11,     'sMx2N':0.2,
            'sMyN':0.24,     'sMy2N':0.45,
            'FMxN':8700,   'FMx2N':10000,
            'FMyN':7500,   'FMy2N':10000,
            'xComb' :0.1,   'yComb':0.1}

tyrer_params = {
            'FzN':4000,     'Fz2N':8000, 
            'dF0xN':200000, 'dF0x2N':210000,
            'dF0yN':90000,  'dF0y2N':100000,
            'sMxN':0.11,     'sMx2N':0.2,
            'sMyN':0.24,     'sMy2N':0.45,
            'FMxN':10000,   'FMx2N':10000,
            'FMyN':9000,   'FMy2N':10000,
            'xComb' :0.1,   'yComb':0.1}


tyref = Tyre(
    parameters=tyref_params, rRolling=0.3
)

tyrer = Tyre(
    parameters=tyrer_params, rRolling=0.3
)
    
wheelf = Wheel(Izz=1.5, tyre=tyref)
wheelr = Wheel(Izz=1.5, tyre=tyrer)

car : BicycleModel = BicycleModel(parameters=chassis_params, wheelf_overload=wheelf, wheelr_overload=wheelr)
# car : BicycleModel = BicycleModel()
car.powertrain.MDrive_ref = 200.0
car.X0 = np.array([30,0,0,30/0.3,30/0.3,0,0.0001,0,0,0.2,0])

carmodel : Environment = Environment(vehicleModel=car, track=track, 
                                    fixed_update=10.0)

env = EnvironmentGym(model=carmodel)
env = TimeLimit(env, max_episode_steps=4000)

if not loadcheckpoint:
    #policy_kwargs = dict(net_arch=[256, 256])
    model = SAC(MlpPolicy, env, verbose=1, tensorboard_log='./tfb/' + save_name, batch_size=1024)
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/",
    name_prefix=save_name
    )

    model.learn(total_timesteps=nepisodes, log_interval=5, callback=checkpoint_callback)
    model.save(save_name)
else:
    model = SAC.load(save_name)
    obs = env.reset()
    dones = False
    score = 0.0
    start_time = time.time()
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        score += rewards
    env.render()
    t = env.model.GetTime()
    s = env.model.GetStateTrajectory('s')
    sf = env.model.sfinal
    tfinal = np.interp(sf, s, t)
    print('Reward = ' + score.__str__() + ' Laptime = ' + tfinal.__str__() + ' Wall time = ' + (time.time() - start_time).__str__())