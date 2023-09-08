import gym
import numpy as np
from Wheel import Wheel, Rill as Tyre
from opengym_wrapper_copy import EnvironmentGym
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

loadcheckpoint, from_log = True, False
save_name = "sarc_easy_car"
save_name = "./logs/" + save_name if from_log else save_name
nepisodes = 300000

# Environment setup
s = np.linspace(0, 1000, 10000)
k = np.ones(s.shape) * 1e-1

trackdata = TrackDataReader().load_example_data()
    
s = trackdata['s']
k = trackdata['k']

track : TrackDefinition = TrackDefinition(s, k, width=2.0)
track.X0 = np.array([0, 0, 0, 0, 0, 0])

chassis_params = {
            'mass': 500,
            'Izz' : 2000,
            'lf' : 1.5,
            'lr' : 1.5,
            'steering_ratio': -12.0 # aHandWheel/aSteer
        }

tyre_params = {
            'FzN':4000,     'Fz2N':8000, 
            'dF0xN':200000, 'dF0x2N':210000,
            'dF0yN':70000,  'dF0y2N':90000,
            'sMxN':0.15,     'sMx2N':0.2,
            'sMyN':0.4,     'sMy2N':0.45,
            'FMxN':20000,   'FMx2N':30000,
            'FMyN':20000,   'FMy2N':30000,
            'xComb' :0.01,   'yComb':0.01}

tyrecommon = Tyre(
    parameters=tyre_params, rRolling=0.3
)
    
wheelf = Wheel(Izz=1.5, tyre=tyrecommon)
wheelr = Wheel(Izz=1.5, tyre=tyrecommon)

car : BicycleModel = BicycleModel(parameters=chassis_params, wheelf_overload=wheelf, wheelr_overload=wheelr)
car.powertrain.MDrive_ref = 200.0
car.X0 = np.array([30,0,0,30/0.3,30/0.3,0,0.0001,0,0,0.2,0])

carmodel : Environment = Environment(vehicleModel=car, track=track, 
                                    fixed_update=15.0)

env = EnvironmentGym(model=carmodel)
env = TimeLimit(env, max_episode_steps=4000)



if not loadcheckpoint:
    policy_kwargs = dict(net_arch=[512, 512])
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
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
    env.render()