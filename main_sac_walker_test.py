import gymnasium as gym
import numpy as np
from SAC.CriticNetwork_copy import Agent
import matplotlib.pyplot as plt

env = env = gym.make("Pendulum-v1")# "BipedalWalker-v3", render_mode="human")#"Pendulum-v1", render_mode="human") # 'MountainCarContinuous-v0'
agent = Agent(input_dims=env.observation_space.shape[0], env=env,
              n_actions=env.action_space.shape[0],max_size=10000000)
n_games = 250

#agent.load_model()
#replay = True
replay = False

best_score = env.reward_range[0]
score_history = []
max_step = 200
for i in range(n_games):
    observation, _ = env.reset()
    done = False
    score = 0
    nsteps = 0
    while not done and nsteps < max_step:
        action = agent.choose_action(
            observation, deterministic=replay)
     
        observation_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
        agent.remember(observation, action, reward, observation_, done)

        if nsteps % 1 == 0:
            agent.learn()

        observation = observation_
        nsteps += 1
        
    score_history.append(score)
    avg_score = np.mean(score_history[-20:])
    env.render()
    
    if avg_score >= best_score:
        best_score = avg_score
        agent.save_models()
        #env.render()
        
    print('episode ', i, ' score %.1f' % score, 'avg_score %.1f' % avg_score, ' nSteps ', nsteps)
    
plt.figure()
plt.plot(score_history)
plt.show()
