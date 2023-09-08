from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
from environment import Environment
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
from read_track_data import TrackDataReader
import matplotlib.pyplot as plt

class EnvironmentGym(gym.Env):
    scale_progress_term, progress_period = 1.0, 1/5
    scale_boundary_term, length_track_width_term = -20.0, 6.0
    sf = 4567.0
    timelimit = 400.0
    
    def __init__(self, model:Environment) -> None:
        # rBrakeThrottle, aHandWheel
        self.action_space = gym.spaces.Box(
            np.array([-1, -200.0 * (np.pi/180)]).astype(np.float32),
            np.array([+1, +200.0 * (np.pi/180)]).astype(np.float32))
        
        # yError, yawError
        self.observation_space = gym.spaces.Box(
            np.array([-1000.0, -2*np.pi]).astype(np.float32),
            np.array([+1000.0, +2*np.pi]).astype(np.float32)
        )
        
        self.render_mode = None
        self.model : Environment = model
        self.previous_slap = 0.0
    
    def _reward(self
    ) -> float:
        new_slap, _ = self.model.GetStateValue('s')
        progress = self.scale_progress_term * (
            new_slap - self.previous_slap) * (
                self.model.h / self.progress_period)
        self.previous_slap = new_slap
        
        yError, _ = self.model.GetStateValue('ey')
        if yError > 0.5 * self.length_track_width_term:
            boundary = self.scale_boundary_term * yError
        else:
            boundary = 0.0
        
        return progress + boundary
    
    def _state(self) -> np.array:
        var = self.model.GetStateValue
        out = self.model.GetOutputValue
        return np.array(
            [
              var('ey')[0], var('ephi')[0]  
            ], dtype=np.float32
        )
        
    def render(self) -> None:
        return
    
    def step(self, action: np.array
             ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        
        self.model.step(action[0], action[1])
        
        s, _ = self.model.GetStateValue('s')
        ey, _ = self.model.GetStateValue('ey')
        vx, _ = self.model.GetStateValue('vx')
        vy, _ = self.model.GetStateValue('vy')
        
        vmag = np.sqrt((vx * vx) + (vy * vy))
        
        terminated = s > self.sf
        truncated = (ey > (10.0 * self.length_track_width_term * 0.5) or 
                     self.model.t > self.timelimit) or vmag < 10
        
        return self._state(), self._reward(), terminated, truncated, {}
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[Any, dict[str, Any]]:
        self.model.initialise()
        return self._state(), {}
    
    def close(self):
        return

if __name__ == "__main__":

    trackdata = TrackDataReader().load_example_data()
    
    s = trackdata['s']
    k = trackdata['k']
    
    car : BicycleModel = BicycleModel()
    car.X0 = np.array([55,0,0,55/0.3,55/0.3,0,0.02])

    track : TrackDefinition = TrackDefinition(s, k)

    model : Environment = Environment(vehicleModel=car, track=track, 
                                      fixed_update=2)
    
    env = EnvironmentGym(model=model)
    
    score = 0.0
    cnt = 1
    while True:
        action = env.action_space.sample()
        state_, reward, terminated, truncated, _ = env.step(action)
        score += reward
        vmag = np.sqrt(
            (env.model.GetStateValue('vx')[0] * env.model.GetStateValue('vx')[0]) + 
            (env.model.GetStateValue('vy')[0] * env.model.GetStateValue('vy')[0]))
        print(vmag)
        
        if vmag > 100 or np.isnan(vmag):
            print('oops')
   
        print("rBrakeThrottle = %f \t aHandWheel = %f \t Step = %i \t Score = %f" % (action[0]*100, action[1]*180/(np.pi), cnt, score))
        cnt += 1
        
        if terminated or truncated:
            break
    
    env.reset()
    
    score = 0.0
    cnt = 1
    while True:
        action = env.action_space.sample()
        state_, reward, terminated, truncated, _ = env.step(action)
        score += reward
        vmag = np.sqrt(
            (env.model.GetStateValue('vx')[0] * env.model.GetStateValue('vx')[0]) + 
            (env.model.GetStateValue('vy')[0] * env.model.GetStateValue('vy')[0]) + 1e-3)
        print(vmag)
        
        if vmag > 100 or np.isnan(vmag):
            print('oops')
            
        print("rBrakeThrottle = %f \t aHandWheel = %f \t Step = %i \t Score = %f" % (action[0]*100, action[1]*180/(np.pi), cnt, score))
        cnt += 1
        
        if terminated or truncated:
            print("Terminated = %i \t Truncated = %i" % (terminated, truncated))
            break
    