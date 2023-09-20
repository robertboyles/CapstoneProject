import numpy as np
from abc import ABCMeta, abstractmethod

class TerminationHandling(metaclass = ABCMeta):
    @abstractmethod
    def evaluate(self, dsdt, s, sfinal, ey, lap_steps, t, t_limit):
        raise NotImplementedError
    @abstractmethod
    def IsDistanceTruncation(self):
        raise NotImplementedError

class FixedDisplacementTermination(TerminationHandling):

    def evaluate(self, dsdt, s, sfinal, ey, lap_steps, t, t_limit):
        success = 0
        truncated_eol = s > sfinal
        truncated_to = (t > t_limit)
        terminated = np.abs(ey) > 5.0 or dsdt < -10.0

        if terminated:
            info_dict = {"is_success": False, "TimeLimit.truncted": False, "nSteps_time": 0}
        elif truncated_eol:
            info_dict = {"is_success": True, "TimeLimit.truncted": True, "nSteps_time": lap_steps}
            success = 1            
        elif truncated_to:
            info_dict = {"is_success": False, "TimeLimit.truncted": True, "nSteps_time": 0}
        else:
            info_dict = {"is_success": False, "TimeLimit.truncted": False, "nSteps_time": 0}

        return success, terminated, truncated_to or truncated_eol, info_dict
    
    def IsDistanceTruncation(self):
        return True

class FixedTimeTermination(TerminationHandling):
    def __init__(self, tlimit0=10.0, deltaOnExceed=+5.0, deltaOnSuccess=-3.0, verbose=True) -> None:
        self.current_limit = tlimit0
        self.on_hit = deltaOnExceed
        self.on_success = deltaOnSuccess
        self.verbose = verbose
    
    def IsDistanceTruncation(self):
        return False
        
    def evaluate(self, dsdt, s, sfinal, ey, lap_steps, t, t_limit):
        success = s > sfinal
        truncated = (t > self.current_limit)
        terminated = np.abs(ey) > 5.0 or dsdt < -10.0

        if terminated:
            info_dict = {"is_success": False, "TimeLimit.truncted": False, "nSteps_time": 0}        
        elif truncated and success:
            info_dict = {"is_success": True, "TimeLimit.truncted": True, "nSteps_time": lap_steps}
            self._on_success_(t)
        elif truncated and not success:
            info_dict = {"is_success": False, "TimeLimit.truncted": True, "nSteps_time": 0}
            self._on_hit_()
        else:
            info_dict = {"is_success": False, "TimeLimit.truncted": False, "nSteps_time": 0}

        return success, terminated, truncated, info_dict
    
    def _on_hit_(self) -> None:
        if self.verbose:
            print('+++ Increasing Time Limit : %.2f +%.2fs +++' % (self.current_limit, self.on_hit))
        self.current_limit += self.on_hit
       
    
    def _on_success_(self, t) -> None:
        if t < (self.current_limit + (2 * self.on_success)):
            if self.verbose:
                print('--- Decreasing Time Limit : %.2f  -%.2fs ---' % (self.current_limit, self.on_success))
            self.current_limit += self.on_success
            
        elif self.verbose:
            print('=== Time Limit Remaining at: %.2fs ===' % (self.current_limit))