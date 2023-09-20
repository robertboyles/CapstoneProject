from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
from environment import Environment
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
from read_track_data import TrackDataReader
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from Rewards import path_following, _default_reward_weights
import pickle
import os
from TerminationFunctions import FixedDisplacementTermination, TerminationHandling


class EnvironmentGym(gym.Env):
    reward_lower = -1
    reward_upper = 0.3
    minimum_observed_laptime = None
    last_observered_laptime = None

    def __init__(self, model:Environment, 
                 reward_fun=path_following, 
                 termination_fun:TerminationHandling=FixedDisplacementTermination(), 
                 save_path='./plots/unnamed', 
                 pdf_interval=2000, 
                 reward_weights=None,
                 terminal_reward_value=0.0) -> None:
        
        self.terminal_reward = terminal_reward_value
        self.reward_weights = _default_reward_weights() if reward_weights is None else reward_weights
        self.save_path = save_path
        self.pdf_interval = pdf_interval
        self.render_mode = None
        self.model : Environment = model
        self._rewardfun = reward_fun
        self._terminationfun : TerminationHandling = termination_fun
        self._initialise_indices()

        self.print_next_terminal = False
        self.steps_count = 0
        self.lap_steps = 0.0
        self.n_success = 0

        # Initialise MDP vectors
        self.action_space = gym.spaces.Box(
            np.array([-1, -1]).astype(np.float32),
            np.array([+1, +1]).astype(np.float32))
        
        obs_low = []
        obs_high = []
        for _, _, _low, _high in self._state_defs_():
            obs_low.append(_low)
            obs_high.append(_high)

        self.observation_space = gym.spaces.Box(
            np.array(obs_low).astype(np.float32),
            np.array(obs_high).astype(np.float32)
        )      
    
    def _reward(self, scalars_dict) -> float:
        total, values, _ = self._rewardfun(scalars_dict, self.reward_weights, distance_trunc=self._terminationfun.IsDistanceTruncation())
        # print(values)
        #scaled = (2 * ((total - (self.reward_lower)) / (self.reward_upper - (self.reward_lower)))) - 1.0
        return total
    
    def _state(self) -> np.array:
        self.current_dsdt = self.model.GetOutputValue_index(
            self.ind_out_dsdt)
        self.current_s = self.model.GetStateValue_index(
            self.ind_var_s)
        
        raw = []
        for _, _func, _, _ in self._state_defs_():
            raw.append(_func())
        
        raw = np.array(raw, dtype=np.float32)

        obs_high = self.observation_space.high
        obs_low = self.observation_space.low
        
        scaled = (2 * ((raw - obs_low) / (obs_high - obs_low))) - 1.0
        
        return scaled
    
    def _state_defs_(self):
        return [
            ('kappaf', lambda : self.model.GetOutputValue_index(self.ind_out_kappaf), -1, 1),
            ('kappar', lambda : self.model.GetOutputValue_index(self.ind_out_kappar), -1, 1),
            ('out_of_bounds', lambda : self._calculate_out_of_bounds(), 0, 1),
            ('rbrakethrottle', lambda : self.model.GetOutputValue_index(self.ind_out_rbrakethrottle), -1, 1),
            ('ahandwheel', lambda : self.model.GetOutputValue_index(self.ind_out_ahandwheel), 
             -self.model.car.max_ahandwheel, self.model.car.max_ahandwheel),
            ('vx', lambda : self.model.GetStateValue_index(self.ind_var_vx), -10, 100),
            ('vy', lambda : self.model.GetStateValue_index(self.ind_var_vy), -50, 50),
            ('ey', lambda : self.model.GetStateValue_index(self.ind_var_ey), -5, 5),
            ('ephi', lambda : self.model.GetStateValue_index(self.ind_var_ephi), -np.pi/2, np.pi/2),
            ('ax', lambda : self.model.GetOutputValue_index(self.ind_out_ax), -100, 50),
            ('ay', lambda : self.model.GetOutputValue_index(self.ind_out_ay), -80, 80),
            ('nyaw', lambda : self.model.GetStateValue_index(self.ind_var_nyaw), -70 * np.pi/180, 70 * np.pi/180),
            ('k_0', lambda : self._get_future_k_(n=0, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_1', lambda : self._get_future_k_(n=1, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_2', lambda : self._get_future_k_(n=2, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_3', lambda : self._get_future_k_(n=3, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_4', lambda : self._get_future_k_(n=4, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_5', lambda : self._get_future_k_(n=5, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_6', lambda : self._get_future_k_(n=6, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_7', lambda : self._get_future_k_(n=7, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_8', lambda : self._get_future_k_(n=8, nk=10, t_horizon=10.0), -0.1, 0.1),
            ('k_9', lambda : self._get_future_k_(n=9, nk=10, t_horizon=10.0), -0.1, 0.1)
        ]
    
    def _get_future_k_(self, n, nk, t_horizon):
        '''
            nk = 10
            time_horizon = 10.0
            s_horizon = dsdt * time_horizon
            ds = s_horizon / nk
            
            s_curves = s + range(nk) * ds
            k_future = self.model.track.K(s_curves)'''
        
        s_sample = self.current_s + n * np.min([self.current_dsdt * t_horizon / nk, 1.0])
        return self.model.track.K(s_sample)

    def _calculate_out_of_bounds(self):
        ey = self.model.GetStateValue_index(
            self.ind_var_ey)
        width = self.model.GetOutputValue_index(
            self.ind_out_width)
        if np.abs(ey) > (width / 2):
            out_of_bounds = 1
        else:
            out_of_bounds = 0
        return out_of_bounds

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
        plt.plot(self.model.GetOutputTrajectory('alphaF') * 180/np.pi, self.model.GetOutputTrajectory('wheel_f_Fy'), marker='*', linestyle="None")
        plt.title('Front Lat.')
        plt.subplot(2,2,2)
        plt.plot(self.model.GetOutputTrajectory('alphaR') * 180/np.pi, self.model.GetOutputTrajectory('wheel_r_Fy'), marker='*', linestyle="None")
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
        plt.subplot(2,2,3)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('kappaF'), linewidth=0.1)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('kappaR'), linewidth=0.1)
        plt.title('Front and Rear Slip Ratios')
        plt.subplot(2,2,4)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('alphaF') * 180/np.pi, linewidth=0.1)
        plt.plot(self.model.GetStateTrajectory('s'), self.model.GetOutputTrajectory('alphaR') * 180/np.pi, linewidth=0.1)
        plt.title('Front and Rear Slip Angles')
        plt.savefig(os.path.join(root_path, '_SAC_Stability.pdf'))
        return
    
    def step(self, action: np.array
             ):
        self.steps_count += 1
        self.lap_steps += 1

        self.model.step(action[0], action[1])
        steps_reward = self._reward(self._reward_scalars(action))

        dsdt = self.model.GetOutputValue_index(
            self.ind_out_dsdt
        )
        s = self.model.GetStateValue_index(
            self.ind_var_s
        )
        ey = self.model.GetStateValue_index(
            self.ind_var_ey
        )

        lap_time = self._calculate_accurate_laptime_(
            self.model.GetStateTrajectory('s'), 
            self.model.GetTime(), 
            self.model.sfinal)
        
        success, terminated, truncated, info_dict = \
            self._terminationfun.evaluate(dsdt, s, self.model.sfinal, ey, self.lap_steps, lap_time, self.model.t, self.model.t_limit)
        self.n_success += success   

        if s > self.model.sfinal:
            self.minimum_observed_laptime = lap_time    if self.minimum_observed_laptime is None or lap_time <= self.minimum_observed_laptime \
                                                        else self.minimum_observed_laptime
            self.last_observered_laptime = lap_time
            
        if self.steps_count % self.pdf_interval == 0:
            self.print_next_terminal = True
            
        if (terminated or truncated) and self.print_next_terminal:
            self.render(None)
            self.print_next_terminal = False
        
        self.previous_slap = s # update for progress after reward calculated
        self.previous_dsdt = dsdt

        if terminated:
            steps_reward += self.terminal_reward

        return self._state(), steps_reward, terminated, truncated, info_dict
    
    
    def _calculate_accurate_laptime_(self, s, t, sfinal):
        if s[-1] < sfinal:
            return None
        elif s[-1] > sfinal:
            return np.interp(sfinal, s, t)
        else:
            return t[-1]

    def _reward_scalars(self, action):
        
        s1 = self.previous_slap
        s2 = self.model.GetStateValue_index(
                self.ind_var_s)
        yError = self.model.GetStateValue_index(
                self.ind_var_ey)
        width = self.model.GetOutputValue_index(
                self.ind_out_width)
        dsdt = self.model.GetOutputValue_index(
                self.ind_out_dsdt)
        dsdt_prev = self.previous_dsdt
        kappaf = self.model.GetOutputValue_index(
                self.ind_out_kappaf)
        kappar = self.model.GetOutputValue_index(
                self.ind_out_kappar)
        rBrakeThrottle = self.model.GetOutputValue_index(
                self.ind_out_rbrakethrottle)
        aHandwheel = self.model.GetOutputValue_index(
                self.ind_out_ahandwheel)
        max_ahandwheel = self.model.car.max_ahandwheel
        drBrakeThrottle, daHandWheel = self.__scale_actions__(action)
        time = self.model.GetElapsedTime()

        n_succ = self.n_success

        return {
             's1': s1, 's2':s2, 
             'yError':yError, 'width':width, 'dsdt': dsdt, 
             'kappaf':kappaf, 'kappar':kappar, 'rBrakeThrottle':rBrakeThrottle,
             'aHandwheel': aHandwheel, 'max_ahandwheel':max_ahandwheel,
             'drBrakeThrottle': drBrakeThrottle, 'daHandWheel': daHandWheel, 
             'time':time, 'n_succ': n_succ}

    def reset(self, *, seed= None, options= None
              ):
        self.model.initialise()
        self.previous_slap = 0.0
        self.lap_steps = 0.0
        self.previous_dsdt = self.model.GetOutputValue_index(self.ind_out_dsdt)
        return self._state(), {}
    
    def close(self):
        return
    
    def _initialise_indices(self):
        _, self.ind_out_dsdt = self.model.GetOutputValue('dsdt')
        _, self.ind_var_s = self.model.GetStateValue('s')
        _, self.ind_var_ey = self.model.GetStateValue('ey')
        _, self.ind_out_width = self.model.GetOutputValue('width')
        _, self.ind_out_kappaf = self.model.GetOutputValue('kappaF')
        _, self.ind_out_kappar = self.model.GetOutputValue('kappaR')
        _, self.ind_var_rbrakethrottle = self.model.GetStateValue('rBrakeThrottle')
        _, self.ind_var_ahandwheel = self.model.GetStateValue('aHandWheel')
        _, self.ind_out_rbrakethrottle = self.model.GetOutputValue('rBrakeThrottle')
        _, self.ind_out_ahandwheel = self.model.GetOutputValue('aHandWheel')
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
    