from typing import Any
import gym as gym
import numpy as np
#from opengym_wrapper import EnvironmentGym

def initial_working(self, action
    ) -> float:
    new_slap = self.model.GetStateValue_index(
            self.ind_var_s)
    progress = self.scale_progress_term * (
        new_slap - self.previous_slap)
    self.previous_slap = new_slap
    
    yError = self.model.GetStateValue_index(
            self.ind_var_ey)
    abs_yError = np.abs(yError)
    width = self.model.GetOutputValue_index(
            self.ind_out_width)
    
    dsdt = self.model.GetOutputValue_index(
            self.ind_out_dsdt)
    kappaf = self.model.GetStateValue_index(
            self.ind_var_kappaf)
    kappar = self.model.GetStateValue_index(
            self.ind_var_kappar)
    rBrakeThrottle = self.model.GetStateValue_index(
            self.ind_var_rbrakethrottle)
    aHandwheel = self.model.GetStateValue_index(
            self.ind_var_ahandwheel)
    
    drBrakeThrottle, daHandWheel = self.__scale_actions__(action)

    lout_of_bounds = abs_yError - (width/2)
    bout_of_bounds = lout_of_bounds > 0.0

    max_dsdt = 100
    boundary = -(0.5 + (dsdt * (np.tanh(lout_of_bounds)) / (2*max_dsdt))) * yError * yError * yError * yError
    to_slow_pen = 100 * np.tanh(0.4 * dsdt + 2) - 99.1445
    to_slow_deprecated = 10 * np.clip(dsdt, -100, 0)
    slip_pen_fun = lambda slip : -100 * slip**2 - 9 * slip + 0.1
    combined_pen = np.abs(rBrakeThrottle) * np.abs(aHandwheel / self.model.car.max_ahandwheel)
        
    reward = (progress * (1 - bout_of_bounds) 
                + boundary
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

def path_finding(self, action
    ) -> float:
    new_slap = self.model.GetStateValue_index(
            self.ind_var_s)
    progress = self.scale_progress_term * (
        new_slap - self.previous_slap)
    self.previous_slap = new_slap
    
    yError = self.model.GetStateValue_index(
            self.ind_var_ey)
    abs_yError = np.abs(yError)
    width = self.model.GetOutputValue_index(
            self.ind_out_width)
    
    dsdt = self.model.GetOutputValue_index(
            self.ind_out_dsdt)
    kappaf = self.model.GetStateValue_index(
            self.ind_var_kappaf)
    kappar = self.model.GetStateValue_index(
            self.ind_var_kappar)
    rBrakeThrottle = self.model.GetStateValue_index(
            self.ind_var_rbrakethrottle)
    aHandwheel = self.model.GetStateValue_index(
            self.ind_var_ahandwheel)
    
    drBrakeThrottle, daHandWheel = self.__scale_actions__(action)

    overlap = 0.0
    lout_of_bounds = abs_yError - ((width - overlap)/2)
    bout_of_bounds = lout_of_bounds > 0.0

    boundary = -10000 * lout_of_bounds * lout_of_bounds if lout_of_bounds > 0.0 else 0.0
    to_slow_pen = 100 * np.tanh(0.4 * dsdt + 2) - 99.1445
#     slip_pen_fun = lambda slip : -100 * slip**2 - 9 * slip + 0.1
    slip_pen_fun = lambda slip : -100 * (np.abs(slip) - 0.1) * (np.abs(slip) - 0.1) - 9 * (np.abs(slip) - 0.1) + + 0.15 if (np.abs(slip) - 0.1) > 0.0 else 0.0
    combined_pen = np.abs(rBrakeThrottle) * np.abs(aHandwheel / self.model.car.max_ahandwheel)
        
    reward = (progress * (1 - bout_of_bounds) 
                + boundary
                + slip_pen_fun(kappaf) / 500
                + slip_pen_fun(kappar) / 500
                + to_slow_pen 
                - combined_pen
                - 1e-2 * drBrakeThrottle * drBrakeThrottle 
                - 1e-2 * daHandWheel * daHandWheel)
    reward = reward / 100.0 # help critic loss remain within a sensible range

#     if new_slap >= self.model.sfinal:
#         reward += (500 * dsdt)

    return reward

def path_following(self, action
    ) -> float:
    new_slap = self.model.GetStateValue_index(
            self.ind_var_s)
    progress = self.scale_progress_term * (
        new_slap - self.previous_slap)
    self.previous_slap = new_slap
    
    yError = self.model.GetStateValue_index(
            self.ind_var_ey)
    abs_yError = np.abs(yError)
    width = self.model.GetOutputValue_index(
            self.ind_out_width)
    
    dsdt = self.model.GetOutputValue_index(
            self.ind_out_dsdt)
    kappaf = self.model.GetStateValue_index(
            self.ind_var_kappaf)
    kappar = self.model.GetStateValue_index(
            self.ind_var_kappar)
    rBrakeThrottle = self.model.GetStateValue_index(
            self.ind_var_rbrakethrottle)
    aHandwheel = self.model.GetStateValue_index(
            self.ind_var_ahandwheel)
    
    drBrakeThrottle, daHandWheel = self.__scale_actions__(action)

    overlap = 0.0
    lout_of_bounds = abs_yError - ((width - overlap)/2)
    bout_of_bounds = lout_of_bounds > 0.0

    boundary = -(0.5 + (0.25 * np.tanh(lout_of_bounds))) * yError * yError * yError * yError
    to_slow_pen = 100 * np.tanh(0.4 * dsdt + 2) - 99.1445
    # slip_pen_fun = lambda slip : -100 * slip**2 - 9 * slip + 0.1
    slip_pen_fun = lambda slip : -100 * (np.abs(slip) - 0.1) * (np.abs(slip) - 0.1) - 9 * (np.abs(slip) - 0.1) + + 0.15 if (np.abs(slip) - 0.1) > 0.0 else 0.0
    combined_pen = np.abs(rBrakeThrottle) * np.abs(aHandwheel / self.model.car.max_ahandwheel)
        
    reward = (progress * (1 - bout_of_bounds) 
                + boundary
                + slip_pen_fun(kappaf) / 500
                + slip_pen_fun(kappar) / 500
                + to_slow_pen 
                - combined_pen
                - 1e-2 * drBrakeThrottle * drBrakeThrottle 
                - 1e-2 * daHandWheel * daHandWheel)
    reward = reward / 100.0 # help critic loss remain within a sensible range

#     if new_slap >= self.model.sfinal:
#         reward += (500 * dsdt)

    return reward