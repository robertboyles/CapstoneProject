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

    def evaluate(self, dsdt, s, sfinal, ey, lap_steps, lap_time, t, t_limit):
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
    def __init__(self, tlimit0=10.0, deltaOnExceed=+5.0, deltaOnSuccess=-3.0, tlimit_max=10000.0, tlimit_min=0.0, verbose=True) -> None:
        self.current_limit = tlimit0
        self.on_hit = deltaOnExceed
        self.on_success = deltaOnSuccess
        self.tmin = tlimit_min
        self.tmax = tlimit_max
        self.verbose = verbose
    
    def IsDistanceTruncation(self):
        return False
        
    def evaluate(self, dsdt, s, sfinal, ey, lap_steps, lap_time, t, t_limit):
        success = s > sfinal
        truncated = (t > self.current_limit)
        terminated = np.abs(ey) > 5.0 or dsdt < -10.0

        if terminated:
            info_dict = {"is_success": False, "TimeLimit.truncted": False, "nSteps_time": 0}        
        elif truncated and success:
            # t is going to be the current limit as the model was driven to it, 
            # so we need to use the t at the s_final from laptime to judge a reduction
            # -- should not be none on success!
            info_dict = {"is_success": True, "TimeLimit.truncted": True, "nSteps_time": lap_steps}
            self._on_success_(lap_time)
        elif truncated and not success:
            info_dict = {"is_success": False, "TimeLimit.truncted": True, "nSteps_time": 0}
            self._on_hit_()
        else:
            info_dict = {"is_success": False, "TimeLimit.truncted": False, "nSteps_time": 0}

        return success, terminated, truncated, info_dict
    
    def _on_hit_(self) -> None:
        newlimit = self.current_limit + self.on_hit
        
        if newlimit > self.tmax:
            self.current_limit = self.tmax
            if self.verbose:
                print('+++ Time limit at maximum : %.2fs +++' % (self.current_limit))
        else:
            self.current_limit += self.on_hit
            if self.verbose:
                print('+++ Increasing Time Limit : %.2f +%.2fs +++' % (self.current_limit, self.on_hit))
       
    
    def _on_success_(self, t) -> None:
        
        if t is None:
            return # something went wrong!
        
        if t < (self.current_limit + (2 * self.on_success)): # some distance from to avoid overactive changing
            # Then we need to change
            newlimit = self.current_limit + self.on_success
            if newlimit > self.tmin:
                if self.verbose:
                    print('--- Decreasing Time Limit : %.2f  %.2fs ---' % (self.current_limit, self.on_success))
                self.current_limit += self.on_success
            else:
                self.current_limit = self.tmin
                if self.verbose:
                    print('=== Time Limit at minimum: %.2fs ===' % (self.current_limit))