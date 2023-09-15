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
from Rewards import initial_working
import pickle
import os


class EnvironmentGym(gym.Env):
    scale_progress_term, progress_period = 20.0, 1/5
    scale_boundary_term = -20.0
        
    def __init__(self, model:Environment, reward_fun=initial_working, save_path='./plots/unnamed', pdf_interval=2000) -> None:
        # drBrakeThrottle, daHandWheel
        self.action_space = gym.spaces.Box(
            np.array([-1, -1]).astype(np.float32),
            np.array([+1, +1]).astype(np.float32))
        
        self.observation_space = gym.spaces.Box(
            np.array([-1, -1, -model.sfinal, 0, -1, -200*np.pi/180, 10,  -100, -15.0, -2*np.pi, -100, -60, -70*np.pi/180 , 
                      -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]).astype(np.float32),
            np.array([1, 1, model.sfinal*0.1, 1, 1,  200*np.pi/180, 100, 100, 15.0,  2*np.pi, 50, 60, 70*np.pi/180, 
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).astype(np.float32)
        )
        
        self.save_path = save_path
        self.pdf_interval = pdf_interval
        self.render_mode = None
        self.model : Environment = model
        self._rewardfun = reward_fun
        self._initialise_indices()

        self.print_next_terminal = False
        self.steps_count = 0
        self.lap_steps = 0.0
    
    def _reward(self, action
    ) -> float:
        return self._rewardfun(self, action)
    
    def _state(self) -> np.array:
        # Get model values
        dsdt = self.model.GetOutputValue_index(
            self.ind_out_dsdt
        )
        s = self.model.GetStateValue_index(
            self.ind_var_s
        )
        ey = self.model.GetStateValue_index(
            self.ind_var_ey
        )
        width = self.model.GetOutputValue_index(
            self.ind_out_width
        )
        kappaf = self.model.GetStateValue_index(
            self.ind_var_kappaf
        )
        kappar = self.model.GetStateValue_index(
            self.ind_var_kappar
        )
        rbrakethrottle = self.model.GetStateValue_index(
            self.ind_var_rbrakethrottle
        )
        ahandwheel = self.model.GetStateValue_index(
            self.ind_var_ahandwheel
        )
        vx = self.model.GetStateValue_index(
            self.ind_var_vx
        )
        vy = self.model.GetStateValue_index(
            self.ind_var_vy
        )
        ephi = self.model.GetStateValue_index(
            self.ind_var_ephi
        )
        ax = self.model.GetOutputValue_index(
            self.ind_out_ax
        )
        ay = self.model.GetOutputValue_index(
            self.ind_out_ay
        )
        nyaw = self.model.GetStateValue_index(
            self.ind_var_nyaw
        )

        nk = 10
        time_horizon = 10.0
        s_horizon = dsdt * time_horizon
        ds = s_horizon / nk
        
        s_curves = s + range(nk) * ds
        k_future = self.model.track.K(s_curves)
        
        if np.abs(ey) > (width / 2):
            out_of_bounds = 1
        else:
            out_of_bounds = 0
        
        #lidar = self.model.track.rangefinder.getDistances(
        #    var('ephi')[0], var('ey')[0], var('s')[0]
        #)

        raw = np.concatenate(
            [np.array(
            [
              kappaf,
              kappar,
              s - self.model.sfinal, 
              out_of_bounds, 
              rbrakethrottle,
              ahandwheel,
              vx, vy,
              ey, ephi,
              ax, ay,
              nyaw
            ], dtype=np.float32
        ), k_future])
        
        obs_high = self.observation_space.high
        obs_low = self.observation_space.low
        
        scaled = (2 * ((raw - obs_low) / (obs_high - obs_low))) - 1.0
        
        return scaled
    
    def __scale_actions__(self, actions):
        a0, a1 = self.model.scale_actions(actions[0], actions[1])
        return np.array([a0, a1])
        
    def render(self, mode) -> None:

        # parent folder based on constructed log location, else default
        # Set the folder name (based on count)

        x, y, xl, yl, xr, yr = self.model.track.plot_track(show=False)

        root_path = os.path.join(self.save_path, self.steps_count.__str__())
        isExist = os.path.exists(root_path)
        if not isExist:
            os.makedirs(root_path)

        FinalTrajectory().save(self.model, root_path)

        plt.close()
        plt.ioff()
        plt.figure()
        plt.subplot(2,3,1)
        plt.plot(self.model.GetTime(), self.model.GetActionTrajectory('drBrakeThrottle'), linewidth=0.1)
        plt.title('drBrakeThrottle')
        plt.subplot(2,3,2)
        plt.plot(self.model.GetTime(), self.model.GetActionTrajectory('daHandWheel') * 180/np.pi, linewidth=0.1)
        plt.title('daHandwheel')
        plt.subplot(2,3,4)
        plt.plot(self.model.GetTime(), self.model.GetStateTrajectory('rBrakeThrottle'), linewidth=0.1)
        plt.title('rBrakeThrottle')
        plt.subplot(2,3,5)
        plt.plot(self.model.GetTime(), self.model.GetStateTrajectory('aHandWheel') * 180/np.pi, linewidth=0.1)
        plt.title('aHandWheel')
        plt.subplot(2,3,6)
        plt.plot(self.model.GetTime(), self.model.GetStateTrajectory('ey'), linewidth=0.1)
        plt.title('ey')
        plt.subplot(2,3,3)
        plt.plot(self.model.GetTime(), np.sqrt(
                 self.model.GetStateTrajectory('vx') * self.model.GetStateTrajectory('vx') +
                 self.model.GetStateTrajectory('vy') * self.model.GetStateTrajectory('vy')), linewidth=0.1)
        plt.title('vMag')
        plt.savefig(os.path.join(root_path, '_SAC_Controls.pdf'))

        plt.close()
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.model.GetStateTrajectory('x_global'), self.model.GetStateTrajectory('y_global'), linewidth=0.1)
        plt.title('Spatial Trajectory Taken')
        plt.subplot(2,1,2)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('dsdt') * 3.6, linewidth=0.1)
        plt.title('Velocity Profile')
        plt.xlabel('sLap [m]')
        plt.ylabel('vPath [kph]')
        plt.savefig(os.path.join(root_path, '_SAC_PathTaken.pdf'))

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
        plt.savefig(os.path.join(root_path, '_SAC_TyreLoads.pdf'))

        plt.close()
        plt.figure()
        plt.plot(self.model.GetStateTrajectory('x_global'), self.model.GetStateTrajectory('y_global'),
             linewidth=0.1)
        plt.plot(xl, yl, color='red', linewidth=0.1)
        plt.plot(xr, yr, color='red', linewidth=0.1)
        plt.axis('square')
        plt.savefig(os.path.join(root_path, '_SAC_RacingLine.pdf'))

        plt.close()
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('beta'))
        plt.title('Side Slip')
        plt.subplot(2,2,2)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('aUndersteer_yaw'), linewidth=0.1)
        plt.title('aUndersteer_yaw')
        plt.subplot(2,2,4)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('alphaF'), linewidth=0.1)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('alphaR'), linewidth=0.1)
        plt.title('Front and Rear Slip Angles')
        plt.savefig(os.path.join(root_path, '_SAC_Stability.pdf'))
        return
    
    def step(self, action: np.array
             ):
        self.steps_count += 1
        self.lap_steps += 1

        self.model.step(action[0], action[1])

        dsdt = self.model.GetOutputValue_index(
            self.ind_out_dsdt
        )
        s = self.model.GetStateValue_index(
            self.ind_var_s
        )
        ey = self.model.GetStateValue_index(
            self.ind_var_ey
        )
                
        terminated = s > self.model.sfinal
        truncated = (np.abs(ey) > 5.0 or # (10.0 * width * 0.5)  ## This fixed value of 5 helps keep ey*ey*ey*ey in rewards low, otherwise wider tracks cause convergence issues
                     self.model.t > self.model.t_limit) or dsdt < -10.0
        
        if truncated:
            info_dict = {"is_success": False, "TimeLimit.truncted": True, "nSteps_time": 0}
        elif terminated:
            info_dict = {"is_success": True, "TimeLimit.truncted": False, "nSteps_time": self.lap_steps}
        else:
            info_dict = {"is_success": False, "TimeLimit.truncted": False, "nSteps_time": 0}
        
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
        self.lap_steps = 0.0
        return self._state()
    
    def close(self):
        return
    
    def _initialise_indices(self):
        _, self.ind_out_dsdt = self.model.GetOutputValue('dsdt')
        _, self.ind_var_s = self.model.GetStateValue('s')
        _, self.ind_var_ey = self.model.GetStateValue('ey')
        _, self.ind_out_width = self.model.GetOutputValue('width')
        _, self.ind_var_kappaf = self.model.GetStateValue('kappaF')
        _, self.ind_var_kappar = self.model.GetStateValue('kappaR')
        _, self.ind_var_rbrakethrottle = self.model.GetStateValue('rBrakeThrottle')
        _, self.ind_var_ahandwheel = self.model.GetStateValue('aHandWheel')
        _, self.ind_var_vx = self.model.GetStateValue('vx')
        _, self.ind_var_vy = self.model.GetStateValue('vy')
        _, self.ind_var_ephi = self.model.GetStateValue('ephi')
        _, self.ind_out_ax = self.model.GetOutputValue('ax')
        _, self.ind_out_ay = self.model.GetOutputValue('ay')
        _, self.ind_var_nyaw = self.model.GetStateValue('nYaw')

class FinalTrajectory():
    def __init__(self) -> None:
        self.time = None
        self.states = None
        self.outputs = None
        self.actions = None
        self.state_names = None
        self.output_names = None
        self.action_names = None
    
    def save(self, model:Environment, save_path):
        self.states, self.state_names = model.GetFullStateArray()
        self.outputs, self.output_names = model.GetFullOutputArray()
        self.actions, self.action_names = model.GetFullActionArray()
        self.time = model.GetTime()
        
        file_name = os.path.join(save_path, 'data' + '.pkl')
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
            print(f'Object successfully saved to "{file_name}"')

    @staticmethod
    def load(save_path):
        file_name = os.path.join(save_path, 'data' + '.pkl')
        with open(file_name, 'rb') as pickle_file:
            data : FinalTrajectory = pickle.load(pickle_file)
        return data
        

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
    