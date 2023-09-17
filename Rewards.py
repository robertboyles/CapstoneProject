from typing import Any
import gym as gym
import numpy as np

def _default_reward_weights():
     return [0.2, 0.01, 2e-5, 2e-5, 0.01, 0.01, 1e-4, 1e-4]

def path_finding(scalars_dict, reward_weights=_default_reward_weights()) -> float:
    term_values, names = _reward_default(scalars_dict, 0.0, reward_weights)
    return sum(term_values), term_values, names

def path_following(scalars_dict, reward_weights=_default_reward_weights()):
     term_values, names = _reward_default(scalars_dict, 1.0, reward_weights)
     return sum(term_values), term_values, names

def _reward_default(modelState, mu, reward_weights
    ) -> float:
    get = lambda name : modelState[name]
    W = reward_weights
    
    overlap = 0.0
    lout_of_bounds = np.abs(get('yError')) - ((get('width') - overlap)/2)
    bout_of_bounds = lout_of_bounds > 0.0

    return (
           course_progress(W[0], get('s1'), get('s2'), bout_of_bounds),
           boundary(W[1], mu, lout_of_bounds, get('yError')),
           slip_general(W[2], get('kappaf'), 0.1),
           slip_general(W[3], get('kappar'), 0.1),
           too_slow(W[4], get('dsdt')),
           combined(W[5], get('rBrakeThrottle'), get('aHandwheel'), get('max_ahandwheel')),
           throttle_control_regularisation(W[6], get('drBrakeThrottle')),
           steering_control_regularisation(W[7], get('daHandWheel'))), reward_term_names()

def reward_term_names():
     # returned with above
     return (
          'course_progress', 'boundary', 'slip_kappa_f', 'slip_kappa_r',
          'too_slow', 'combined', 'throttle_control_reg', 'steering_control_reg'
     )
  
def course_progress(scale, s1, s2, bout_of_bounds):
        delta_s = s2 -s1
        return (1 - bout_of_bounds) * scale * delta_s

def boundary(scale, mu, lout_of_bounds, yError):
     # barrier parameter mu, as mu -> 0, we approach path finding
     pathfollowing = (-(0.5 + (0.25 * np.tanh(lout_of_bounds))) * yError * yError * yError * yError)
     pathfinding = (-10000 * lout_of_bounds * lout_of_bounds if lout_of_bounds > 0.0 else 0.0)
     compound = mu * pathfollowing + (1 - mu) * pathfinding
     return scale * compound

def slip_general(scale, slip, bound):
     violation = np.abs(slip) - bound
     return scale * (-100 * violation**2 - 9 * violation + 0.15 if violation > 0.0 else 0.0)

def too_slow(scale, dsdt):
     # Just says, you are going in the correct direction! Issue is that this will encourage 
     # prolonging of the episode... should really be offset to never be positive.
     # At vMax (100m/s) -> 0.8555 ... place holder called offset
     offset = 0.0
     return scale * (100 * np.tanh(0.4 * dsdt + 2) - 99.1445 - offset)

def combined(scale, rBrakeThrottle, aHandwheel, max_ahandwheel):
     return scale * -1 * np.abs(rBrakeThrottle) * np.abs(aHandwheel / max_ahandwheel)

def throttle_control_regularisation(scale, drBrakeThrottle):
     return scale * -1 * drBrakeThrottle**2

def steering_control_regularisation(scale, daHandWheel):
     return scale * -1 * daHandWheel**2


# Deprecated
# def initial_working(self, action
#     ) -> float:
#     new_slap = self.model.GetStateValue_index(
#             self.ind_var_s)
#     progress = self.scale_progress_term * (
#         new_slap - self.previous_slap)
#     self.previous_slap = new_slap
    
#     yError = self.model.GetStateValue_index(
#             self.ind_var_ey)
#     abs_yError = np.abs(yError)
#     width = self.model.GetOutputValue_index(
#             self.ind_out_width)
    
#     dsdt = self.model.GetOutputValue_index(
#             self.ind_out_dsdt)
#     kappaf = self.model.GetStateValue_index(
#             self.ind_var_kappaf)
#     kappar = self.model.GetStateValue_index(
#             self.ind_var_kappar)
#     rBrakeThrottle = self.model.GetStateValue_index(
#             self.ind_var_rbrakethrottle)
#     aHandwheel = self.model.GetStateValue_index(
#             self.ind_var_ahandwheel)
    
#     drBrakeThrottle, daHandWheel = self.__scale_actions__(action)

#     lout_of_bounds = abs_yError - (width/2)
#     bout_of_bounds = lout_of_bounds > 0.0

#     max_dsdt = 100
#     boundary = -(0.5 + (dsdt * (np.tanh(lout_of_bounds)) / (2*max_dsdt))) * yError * yError * yError * yError
#     to_slow_pen = 100 * np.tanh(0.4 * dsdt + 2) - 99.1445
#     to_slow_deprecated = 10 * np.clip(dsdt, -100, 0)
#     slip_pen_fun = lambda slip : -100 * slip**2 - 9 * slip + 0.1
#     combined_pen = np.abs(rBrakeThrottle) * np.abs(aHandwheel / self.model.car.max_ahandwheel)
        
#     reward = (progress * (1 - bout_of_bounds) 
#                 + boundary
#                 + slip_pen_fun(kappaf) / 500
#                 + slip_pen_fun(kappar) / 500
#                 + to_slow_pen 
#                 - combined_pen
#                 - 1e-2 * drBrakeThrottle * drBrakeThrottle 
#                 - 1e-2 * daHandWheel * daHandWheel)
#     reward = reward / 100.0 # help critic loss remain within a sensible range

#     if new_slap >= self.model.sfinal:
#         reward += 50

#     return reward