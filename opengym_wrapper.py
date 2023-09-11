from typing import Any, SupportsFloat
import gym as gym
import numpy as np
from environment import Environment
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
from read_track_data import TrackDataReader
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


class EnvironmentGym(gym.Env):
    scale_progress_term, progress_period = 10.0, 1/5
    scale_boundary_term = -20.0
    pdf_interval = 2000
        
    def __init__(self, model:Environment) -> None:
        # drBrakeThrottle, daHandWheel
        self.action_space = gym.spaces.Box(
            np.array([-1, -1]).astype(np.float32),
            np.array([+1, +1]).astype(np.float32))
        
        nk = 10
        curve_max = np.ones([nk]) * 0.1
        curve_min = np.ones([nk]) * -0.1
        self.observation_space = gym.spaces.Box(
            np.array([-1, -1, -model.sfinal, 0, -1, -200*np.pi/180, 10,  -100, -15.0, -2*np.pi, -100, -60, -70*np.pi/180 , 
                      -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
                       0, 0, 0, 0, 0]).astype(np.float32),
            np.array([1, 1, model.sfinal*0.1, 1, 1,  200*np.pi/180, 100, 100, 15.0,  2*np.pi, 50, 60, 70*np.pi/180, 
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      100, 100, 100, 100, 100]).astype(np.float32)
        )
        
        self.render_mode = None
        self.model : Environment = model

        self.print_next_terminal = False
        self.steps_count = 0
    
    def _reward(self, action
    ) -> float:
        new_slap, _ = self.model.GetStateValue('s')
        progress = self.scale_progress_term * (
            new_slap - self.previous_slap)
        self.previous_slap = new_slap
        
        yError, _ = np.abs(self.model.GetStateValue('ey'))
        width, _ = self.model.GetOutputValue('width')
        
        dsdt, _ = self.model.GetOutputValue('dsdt')
        kappaf, _ = self.model.GetStateValue('kappaF')
        kappar, _ = self.model.GetStateValue('kappaR')
        rBrakeThrottle, _ = self.model.GetStateValue('rBrakeThrottle')
        aHandwheel, _ = self.model.GetStateValue('aHandWheel')
        
        drBrakeThrottle, daHandWheel = self.__scale_actions__(action)

        lout_of_bounds = np.abs(yError) - (width/2)
        bout_of_bounds = lout_of_bounds > 0.0

        max_dsdt = 100
        boundary = -(0.5 + (dsdt * (np.tanh(lout_of_bounds)) / (2*max_dsdt))) * yError * yError * yError * yError
        to_slow_pen = 100 * np.tanh(0.4 * dsdt + 2) - 99.1445
        to_slow_deprecated = 10 * np.clip(dsdt, -100, 0)
        slip_pen_fun = lambda slip : -100 * slip**2 - 9 * slip + 0.1
        combined_pen = np.abs(rBrakeThrottle) * np.abs(aHandwheel / self.model.car.max_ahandwheel)
         
        reward = (progress * (1 - bout_of_bounds) + boundary
                  + slip_pen_fun(kappaf) / 500
                  + slip_pen_fun(kappar) / 500
                  + to_slow_pen 
                  - combined_pen
                  - 1e-2 * drBrakeThrottle * drBrakeThrottle 
                  - 1e-2 * daHandWheel * daHandWheel)
        reward = reward / 100.0 # help critic loss remain within a sensible range

        if new_slap >= self.model.sfinal:
            reward += 50

        return reward
    
    def _state(self) -> np.array:
        var = self.model.GetStateValue
        out = self.model.GetOutputValue
        
        # raw = np.array(
        #     [
        #       var('vx')[0], var('vy')[0], out('gLong')[0], out('gLat')[0],
        #       var('ey')[0], var('ephi')[0],
        #       out('range0')[0], out('range1')[0], out('range2')[0], out('range3')[0], out('range4')[0],
        #       self.model.track.K(var('s')[0] + 5), self.model.track.K(var('s')[0] + 20)  
        #     ], dtype=np.float32
        # )
        
        nk = 10
        time_horizon = 10.0
        s_horizon = out('dsdt')[0] * time_horizon
        ds = s_horizon / nk
        
        s_curves = var('s')[0] + range(nk) * ds
        k_future = self.model.track.K(s_curves)
        
        if np.abs(var('ey')[0]) > (out('width')[0] / 2):
            out_of_bounds = 1
        else:
            out_of_bounds = 0
        
        lidar = self.model.track.rangefinder.getDistances(
            var('ephi')[0], var('ey')[0], var('s')[0]
        )

        raw = np.concatenate(
            [np.array(
            [
              var('kappaF')[0], var('kappaR')[0], var('s')[0] - self.model.sfinal, out_of_bounds, var('rBrakeThrottle')[0], var('aHandWheel')[0],
              var('vx')[0], var('vy')[0],
              var('ey')[0], var('ephi')[0],
              out('ax')[0], out('ay')[0],
              var('nYaw')[0]
            ], dtype=np.float32
        ), k_future, lidar])
        
        obs_high = self.observation_space.high
        obs_low = self.observation_space.low
        
        scaled = (2 * ((raw - obs_low) / (obs_high - obs_low))) - 1.0
        
        return scaled
    
    def __scale_actions__(self, actions):
        a0, a1 = self.model.scale_actions(actions[0], actions[1])
        return np.array([a0, a1])
        
    def render(self, mode) -> None:
        x, y, xl, yl, xr, yr = self.model.track.plot_track(show=False)
        
        plt.close()
        plt.ioff()
        plt.figure()
        plt.subplot(2,3,1)
        plt.plot(self.model.GetTime(), self.model.GetActionTrajectory('drBrakeThrottle'))
        plt.title('drBrakeThrottle')
        plt.subplot(2,3,2)
        plt.plot(self.model.GetTime(), self.model.GetActionTrajectory('daHandWheel') * 180/np.pi)
        plt.title('daHandwheel')
        plt.subplot(2,3,4)
        plt.plot(self.model.GetTime(), self.model.GetStateTrajectory('rBrakeThrottle'))
        plt.title('rBrakeThrottle')
        plt.subplot(2,3,5)
        plt.plot(self.model.GetTime(), self.model.GetStateTrajectory('aHandWheel') * 180/np.pi)
        plt.title('aHandWheel')
        plt.subplot(2,3,6)
        plt.plot(self.model.GetTime(), self.model.GetStateTrajectory('ey'))
        plt.title('ey')
        plt.subplot(2,3,3)
        plt.plot(self.model.GetTime(), np.sqrt(
                 self.model.GetStateTrajectory('vx') * self.model.GetStateTrajectory('vx') +
                 self.model.GetStateTrajectory('vy') * self.model.GetStateTrajectory('vy')))
        plt.title('dsdt')
        plt.savefig('./plots/' + self.steps_count.__str__() + '_SAC_Controls.pdf')

        plt.close()
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.model.GetStateTrajectory('x_global'), self.model.GetStateTrajectory('y_global'))
        plt.title('Spatial Trajectory Taken')
        plt.subplot(2,1,2)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('dsdt') * 3.6)
        plt.title('Velocity Profile')
        plt.xlabel('sLap [m]')
        plt.ylabel('vPath [kph]')
        plt.savefig('./plots/' + self.steps_count.__str__() + '_SAC_PathTaken.pdf')

        plt.close()
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(self.model.GetOutputTrajectory('alphaF'), self.model.GetOutputTrajectory('wheel_f_Fy'), marker='*', linestyle="None")
        plt.title('Front Lat.')
        plt.subplot(2,2,2)
        plt.plot(self.model.GetOutputTrajectory('alphaR'), self.model.GetOutputTrajectory('wheel_r_Fy'), marker='*', linestyle="None")
        plt.title('Rear Lat.')
        plt.subplot(2,2,3)
        plt.plot(self.model.GetOutputTrajectory('kappaF'), self.model.GetOutputTrajectory('wheel_f_Fx'), marker='*', linestyle="None")
        plt.title('Front Long.')
        plt.subplot(2,2,4)
        plt.plot(self.model.GetOutputTrajectory('kappaR'), self.model.GetOutputTrajectory('wheel_r_Fx'), marker='*', linestyle="None")
        plt.title('Rear Long.')
        plt.savefig('./plots/' + self.steps_count.__str__() + '_SAC_TyreLoads.pdf')

        plt.close()
        plt.figure()
        plt.plot(self.model.GetStateTrajectory('x_global'), self.model.GetStateTrajectory('y_global'),
             linewidth=0.1)
        plt.plot(xl, yl, color='red', linewidth=0.1)
        plt.plot(xr, yr, color='red', linewidth=0.1)
        plt.axis('square')
        plt.savefig('./plots/' + self.steps_count.__str__()  + '_SAC_RacingLine.pdf')
        return
    
    def step(self, action: np.array
             ):
        self.steps_count += 1

        self.model.step(action[0], action[1])
        
        s, _ = self.model.GetStateValue('s')
        ey, _ = self.model.GetStateValue('ey')        
        dsdt, _ = self.model.GetOutputValue('dsdt')
        
        terminated = s > self.model.sfinal
        truncated = (np.abs(ey) > 5.0 or # (10.0 * width * 0.5)  ## This fixed value of 5 helps keep ey*ey*ey*ey in rewards low, otherwise wider tracks cause convergence issues
                     self.model.t > self.model.t_limit) or dsdt < -10.0
        
        if truncated:
            info_dict = {"is_success": False, "TimeLimit.truncted": True}
        elif terminated:
            info_dict = {"is_success": True, "TimeLimit.truncted": False}
        else:
            info_dict = {"is_success": False, "TimeLimit.truncted": False}
        
        if self.steps_count % self.pdf_interval == 0:
            self.print_next_terminal = True
            
        if (terminated or truncated) and self.print_next_terminal:
            self.render(None)
            self.print_next_terminal = False

        return self._state(), self._reward(action), terminated or truncated, info_dict
    
    def reset(self, *, seed= None, options= None
              ):
        self.model.initialise()
        self.previous_slap = 0.0
        return self._state()
    
    def close(self):
        return

if __name__ == "__main__":

    trackdata = TrackDataReader().load_example_data()
    
    s = trackdata['s']
    k = trackdata['k']
    v = trackdata['v']
    
    car : BicycleModel = BicycleModel()
    car.X0 = np.array([
        v[0],0,0,
        v[0]/car.wheelf.tyre.rRolling,
        v[0]/car.wheelr.tyre.rRolling,
        0,0.02])

    track : TrackDefinition = TrackDefinition(s, k, width=2, sf=None)

    model : Environment = Environment(vehicleModel=car, track=track, 
                                      fixed_update=20.0)
    
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
    