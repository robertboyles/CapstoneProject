from typing import Any
import gymnasium  as gym
import numpy as np

def _default_reward_weights():
     return [5, 0.01, 2e-5, 2e-5, 0.001, 1e-4, 1e-4, 0.1]

def path_finding(scalars_dict, reward_weights=_default_reward_weights(), distance_trunc=True) -> float:
    term_values, names = _reward_default(scalars_dict, 0.0, reward_weights, distance_trunc)
    return sum(term_values), term_values, names

def path_following(scalars_dict, reward_weights=_default_reward_weights(), distance_trunc=True):
     term_values, names = _reward_default(scalars_dict, 1.0, reward_weights, distance_trunc)
     return sum(term_values), term_values, names

def dynamic_reward(scalars_dict, reward_weights=_default_reward_weights(), distance_trunc=True):
     mu = 1.0 - (scalars_dict['n_succ'] / 100.0) if scalars_dict['n_succ'] < 100 else 0.0
     term_values, names = _reward_default(scalars_dict, mu, reward_weights, distance_trunc)
     return sum(term_values), term_values, names

class DynamicReward():
     def __init__(self, nSuccesful_total=100) -> None:
          self.n = nSuccesful_total
     def __call__(self,scalars_dict, reward_weights=_default_reward_weights(), distance_trunc=True) -> Any:
          mu = 1.0 - (scalars_dict['n_succ'] / self.n) if scalars_dict['n_succ'] < self.n else 0.0
          term_values, names = _reward_default(scalars_dict, mu, reward_weights, distance_trunc)
          return sum(term_values), term_values, names
          

def _reward_default(modelState, mu, reward_weights, distance_trunc=True
    ) -> float:
    get = lambda name : modelState[name]
    W = reward_weights
    
    overlap = 0.0
    lout_of_bounds = np.abs(get('yError')) - ((get('width') - overlap)/2)
    bout_of_bounds = lout_of_bounds > 0.0

    return (
           course_progress(W[0], get('s1'), get('s2'), bout_of_bounds, get('dsdt'), get('time'), distance_trunc),
           boundary(W[1], mu, lout_of_bounds, get('yError')),
           slip_general(W[2], get('kappaf'), 0.1),
           slip_general(W[3], get('kappar'), 0.1),
           combined(W[4], get('rBrakeThrottle'), get('aHandwheel'), get('max_ahandwheel')),
           throttle_control_regularisation(W[5], get('drBrakeThrottle')),
           steering_control_regularisation(W[6], get('daHandWheel')),
           too_slow(W[7], get('dsdt'))), reward_term_names()

def reward_term_names():
     # returned with above
     return (
          'course_progress', 'boundary', 'slip_kappa_f', 'slip_kappa_r',
          'combined', 'throttle_control_reg', 'steering_control_reg', 'too_slow'
     )
  
def course_progress(scale, s1, s2, bout_of_bounds, dsdt, time, distance_trunc):
     #    ref_speed = 5 # m/s
     #    T = 1/10.0
     #    minimum_distance = ref_speed * T
     #    sTravelled = s2 - s1 # range -10/10.0 = -1 -> 100/10.0 = 10
     #    excess_distance = sTravelled - minimum_distance # 5/10 = 0.5
     #    alpha = 1.2
     #    value = ((np.exp(alpha * excess_distance)) - np.exp(alpha * 5)) / 1e5

     #    max_, min_ = 22000, -1.5
     #    value = (((value - (min_)) / (max_ - min_)))
     #dsdt_max = 100
     #delta_s = (s2 -s1) * (((dsdt + dsdt_prev) / 2) / dsdt_max)**6
     if distance_trunc:
          if time == 0.0:
               average_speed = 0.0
          else:
               average_speed = s2 / time
          av_speed_sq = average_speed
          max_, min_ = 80.0, 0.0
          value = (av_speed_sq - min_) / (max_ - min_)
     else:
          # Time truncation
          value = np.max([(s2 - s1), 0.0])
          max_, min_ = 10.0, 0.0
          value = 0.8 * (value - min_) / (max_ - min_)
     
     return (1 - bout_of_bounds) * scale * value
     

def boundary(scale, mu, lout_of_bounds, yError):
     # barrier parameter mu, as mu -> 0, we approach path finding
     pathfollowing = (-(0.5 + (0.25 * np.tanh(lout_of_bounds))) * yError * yError * yError * yError)
     pathfinding = (-10000 * lout_of_bounds * lout_of_bounds if lout_of_bounds > 0.0 else 0.0)
     compound = mu * pathfollowing + (1 - mu) * pathfinding
     return scale * compound

def slip_general(scale, slip, bound):
     violation = np.abs(slip) - bound
     value = (-100 * violation**2 - 9 * violation + 0.15 if violation > 0.0 else 0.0)
     return scale * value

def combined(scale, rBrakeThrottle, aHandwheel, max_ahandwheel):
     value = -1 * np.abs(rBrakeThrottle) * np.abs(aHandwheel / max_ahandwheel)
     return scale * value 

def throttle_control_regularisation(scale, drBrakeThrottle):
     value = -1 * drBrakeThrottle**2
     return scale * value

def steering_control_regularisation(scale, daHandWheel):
     value = -1 * daHandWheel**2
     return scale * value

def too_slow(scale, dsdt):
     value = 100*np.tanh(0.4 * dsdt + 2) - 99.1445 - 0.8555
     return scale * value

def smoothMin(x, y, eps):
    return 0.5 * ((x + y) - np.sqrt(((x - y) * (x - y)) + (eps * eps)))

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