from SAC.CriticNetwork_copy import Agent
from opengym_wrapper_copy import EnvironmentGym
from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
from environment import Environment
import os
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
from read_track_data import TrackDataReader
import matplotlib.pyplot as plt

trackdata = TrackDataReader().load_example_data()
    
s = trackdata['s']
k = trackdata['k']

# straight line
s = np.linspace(0, 1000, 10000)
k = np.zeros(s.shape)

track : TrackDefinition = TrackDefinition(s, k, width=1.0)
track.X0 = np.array([0, 0, 0, 0, 0, 0])

car : BicycleModel = BicycleModel()
car.X0 = np.array([1,0,0,1/0.3,1/0.3,0,0.0001,0,0])

model : Environment = Environment(vehicleModel=car, track=track, 
                                    fixed_update=20.0)

env = EnvironmentGym(model=model)
agent = Agent(input_dims=env.observation_space.shape[0], env=env,
              n_actions=env.action_space.shape[0],max_size=1000)
n_games = 250000

agent.load_model()
# agent.actor.load_weights(os.path.join('tmp/sac_initial_fail', 'actor'+'_sac'))
# agent.critic1.load_weights(os.path.join('tmp/sac_initial_fail', 'critic1'+'_sac'))
# agent.critic2.load_weights(os.path.join('tmp/sac_initial_fail', 'critic2'+'_sac'))
# agent.critic1_target.load_weights(os.path.join('tmp/sac_initial_fail', 'critic1_target'+'_sac'))
# agent.critic2_target.load_weights(os.path.join('tmp/sac_initial_fail', 'critic2_target'+'_sac'))
replay = True
#replay = False

best_score = env.reward_range[0]
score_history = []
max_step = 5000
for i in range(n_games):
    observation, _ = env.reset()
    done = False
    score = 0
    nsteps = 0
    learning = False
    while not done and nsteps < max_step:
        action = agent.choose_action(
            observation, deterministic=replay)
     
        observation_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
        
        agent.remember(observation, action, reward, observation_, done)
        
        #if nsteps % 1 == 0:
        #    learning = agent.learn()
            
        observation = observation_
        nsteps += 1
        
    score_history.append(score)
    avg_score = np.mean(score_history[-20:])
    env.render()
    
    if avg_score >= best_score:
        best_score = avg_score
        #agent.save_models()
        #env.render()
        
    print('episode ', i, ' score %.1f' % score, 'avg_score %.1f' % avg_score, ' nSteps ', nsteps, 'Learning : ', learning)
    
plt.figure()
plt.plot(score_history)
plt.show()
