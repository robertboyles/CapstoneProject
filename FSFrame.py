import numpy as np
from scipy import interpolate as itp
from RangeFinder import EulerIntegration as RangeFinder
import matplotlib.pyplot as plt
from Model_ABC import ModelABC


class TrackDefinition(ModelABC):
    def __init__(self, s, k, width=2.0, sf=None, t_limit=440.0) -> None:
        self.k = itp.splrep(s, k)
        self.W = lambda s : width
        
        self.sf = sf if sf is not None else s[-1]
        self.t_limit = t_limit
        
        self.initialise_references()
        
        self.rangefinder = RangeFinder(
            self.K, self.W, lray=100.0, step=1e-1
        )
    
    def evaluate(self, x:np.array, u:np.array):
        # initialise outputs
        dx = np.zeros((self.nx,))
        y = np.zeros((self.ny,))
        
        ephi, ey, s, aheading = (
            x[self.ind_ephi], x[self.ind_ey], x[self.ind_s],
            x[self.ind_aheading])
        
        vx, vy, nyaw = u[self.ind_vx], u[self.ind_vy], u[self.ind_nyaw]
        
        curvature = self.K(s)
        
        dx[0] = nyaw - (
            (vx * np.cos(ephi) - vy * np.sin(ephi))/
            (1 - curvature * ey)) * curvature
        dx[1] = vx * np.sin(ephi) + vy * np.cos(ephi)
        dx[2] = (
            (vx * np.cos(ephi) - vy * np.sin(ephi))/
            (1 - curvature * ey))
        
        aHeading_wrapped = (aheading + np.pi) % (2 * np.pi) - np.pi
        c_, s_ = np.cos(aHeading_wrapped), np.sin(aHeading_wrapped)
        R_gb = np.array(((c_, -s_), (s_, c_))) # rotation matrix global from body
        
        vglobal = np.matmul(R_gb, np.array([vx, vy]).T)
        
        dx[3] = vglobal[0]
        dx[4] = vglobal[1]
        dx[5] = nyaw
                
        y[self.ind_dsdt] = dx[2]
        y[self.ind_curvature] = curvature
        y[self.ind_width] = self.W(s)
        
        return dx, y
    
    def initialise_references(self) -> None:
        _, self.ind_ephi = self.GetNamedValue('ephi', self.getStateNames())
        _, self.ind_ey = self.GetNamedValue('ey', self.getStateNames())
        _, self.ind_s = self.GetNamedValue('s', self.getStateNames())
        _, self.ind_x_global = self.GetNamedValue('x_global', self.getStateNames())
        _, self.ind_y_global = self.GetNamedValue('y_global', self.getStateNames())
        _, self.ind_aheading = self.GetNamedValue('aHeading_global', self.getStateNames())
        
        _, self.ind_vx = self.GetNamedValue('vx', self.getInputNames())
        _, self.ind_vy = self.GetNamedValue('vy', self.getInputNames())
        _, self.ind_nyaw = self.GetNamedValue('nYaw', self.getInputNames())
        
        _, self.ind_curvature = self.GetNamedValue('curvature', self.getOutputNames())
        _, self.ind_width = self.GetNamedValue('width', self.getOutputNames())
        _, self.ind_dsdt = self.GetNamedValue('dsdt', self.getOutputNames())
    
    def K(self, s):
        k = itp.splev(s, self.k)
        return k
    
    def getStateNames(self) -> tuple:
        return(
            'ephi', 'ey', 's', 'x_global', 'y_global', 'aHeading_global'
        )
    
    def getInputNames(self) -> tuple:
        return (
            'vx', 'vy', 'nYaw'
        )
        
    def getOutputNames(self) -> tuple:
        return (
            'curvature', 'width',
            'dsdt'
        )
    
    def getDefaultInitialConditions(self) -> np.array:
        return np.array([0, 0, 0, 0, 0, 0])
    
    @staticmethod
    def getDefaultParameters() -> dict:
        return {
        }

if __name__ == "__main__":
    print("Testing Track")
    n = 1000
    f = 0.03
    s0, sf = 0.0, 100.0
    s = np.linspace(s0, sf, n)
    #k = np.sin(2 * np.pi * f * s)
    k = np.ones(s.shape) * (1/10.0)
    
    track = TrackDefinition(s, k)
    print(track.K(np.array([0, 1, 2, 3, 4, 5])))
    x0 = np.array([0, 0, 0, 0, 0, 0])
    
    Vy = 0.0
    radius = 10.0
    nYaw = 2.0
    Vx = nYaw * radius
    
    u = np.array([Vx, Vy, nYaw])
    
    t0 = 0.0
    tf = 3.2
    sf = 20.0
    n = int(np.floor(tf * sf))
    dt = 1/sf
    t = np.zeros(shape=(1,n))
    t_ = t0
    x = np.zeros(shape=(len(x0), n))
    x[:, 0] = x0
    cnt = 0
    while cnt < n - 1:
        print(cnt)
        dx, y = track.evaluate(x[:, cnt], u)
        x[:, cnt+1] = x[:, cnt] + (
            dt * dx.T
            )
        t[0,cnt] = t_
        t_ += dt
        cnt += 1
    
    xg = x[3, :-1]
    yg = x[4, :-1]
    
    plt.subplot(1,4,1)
    plt.plot(xg, yg)
    plt.subplot(1,4,2)
    plt.plot(t.T[:-1], x[5, :-1] * (180 / np.pi))
    plt.subplot(1,4,3)
    plt.plot(t.T[:-1], x[1, :-1], marker="o")
    plt.subplot(1,4,4)
    plt.plot(t.T[:-1], x[2, :-1], marker="o")
    plt.show()
    
    
    
    