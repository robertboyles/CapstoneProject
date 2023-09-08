import gym
import numpy as np

#from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3 import SAC, TD3

env = gym.make('Pendulum-v0')
print('step')
model = TD3(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
#model.save("sac_pendulum")
#print('step')
#del model # remove to demonstrate saving and loading

#model = SAC.load("sac_pendulum")

#obs = env.reset()
#while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()