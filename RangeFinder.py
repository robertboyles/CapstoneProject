import numpy as np
import time as timer
from abc import ABCMeta, abstractmethod

class RangeFinder(metaclass=ABCMeta):
    @abstractmethod
    def getDistances(self, ephi:float, ey:float, s:float) -> np.array:
        raise NotImplementedError

class EulerIntegration(RangeFinder):
    def __init__(self, curvature, width,
                 lray:float=100, step:float=1) -> None:
        self.curvature = curvature
        self.width = width
        self.lray = lray
        self.step = step
        self.tol = 1e-3
        self.ray_angle = np.array([
            +90.0, +45.0, 0.0, -45.0, -90.0
         ], dtype=np.float32) * np.pi / 180 # ISO Vehicle, i.e. element[0] is pointing LHS
    
    def getDistances(self, ephi, ey, s) -> np.array :
        #starttime = timer.time()
        distances = []
        for aray in self.ray_angle:
            _ephi = ephi - aray # minus as I think FS Phi is opposite to vehicle ISO
            _ey = ey
            _s = s

            for l in np.linspace(0, self.lray, int(self.lray/self.step)):
                dx = self._differentialEqs_(_ephi, _ey, _s)
                _ephi += dx[0]*self.step
                _ey += dx[1]*self.step
                _s += dx[2]*self.step
                _l = l
                
                if np.abs(_ey) > self.width(_s):
                    _l += self.step
                    correction = (1/np.abs(dx[1])) * (np.abs(_ey) - self.width(_s))
                    #print(_ey, 1/dx[1], correction, _l)
                    _l -= correction
                    break
            
            distances.append(_l)
        #print(timer.time() - starttime)
        return distances
    
    def _differentialEqs_(self, ephi, ey, s) -> np.array:
        dx = np.zeros((3,))
        
        ds = ((np.cos(ephi)) / 
                 (1 - self.curvature(s) * ey))
        
        dx[0] = -ds * self.curvature(s)
        dx[1] = np.sin(ephi)
        dx[2] = ds
        
        return dx

if __name__ == "__main__":
    s = np.linspace(0, 1000, 4000)
    k = lambda s : 1/10.00
    width = lambda s : 2.0
    
    finder : RangeFinder = EulerIntegration(
        k, width, lray=100, step=1e-1
    )
    
    ephi = 0.0 * np.pi / 180
    ey = 0.0
    s = 0.0
    
    distances = finder.getDistances(ephi, ey, s)
    
    print(distances)
    