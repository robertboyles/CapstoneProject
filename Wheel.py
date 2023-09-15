import numpy as np
import matplotlib.pyplot as plt

class FandMOutput():
    def __init__(self, F, M) -> None:
        self.F = F
        self.M = M

class Tyre():
    def __init__(self, Calpha=-10000, Cs=200000, rRolling=0.3) -> None:
        self.Calpha = Calpha
        self.Cs = Cs
        self.rRolling = rRolling
    
    def GetForceAndMoment(self, alpha, kappa, Fz) -> FandMOutput:
        Fx = np.sign(kappa) * np.min([np.abs(self.Cs * kappa), 1e10])
        Fy = self.Calpha * alpha
        return FandMOutput(np.array([Fx, Fy]).T, np.zeros((3,1)))

class WheelOutputs():
    def __init__(self, F_wheel, M_wheel, dOmega, dkappa, dtanalpha, alpha, kappa) -> None:
        self.F = F_wheel
        self.M = M_wheel
        self.dOmega = dOmega
        self.dkappa = dkappa
        self.dtanalpha = dtanalpha
        self.alpha = alpha
        self.kappa = kappa
        
class Rill():
    def __init__(self, parameters:dict=None, rRolling=0.3) -> None:
        self.params = parameters if parameters is not None else self.GetDefaultParameters()
        self.rRolling = rRolling
    
    @property
    def Cs(self):
        return self.params['dF0xN'] # not really true
    
    @property
    def Calpha(self):
        return self.params['dF0yN'] # not really true
    
    def GetForceAndMoment(self, alpha, kappa, Fz) -> FandMOutput:
        _long = self._long_params(Fz)
        _lat = self._lat_params(Fz)
        fx0 = self._pureforce(_long[0], _long[1], _long[2], kappa)
        fy0 = self._pureforce(_lat[0], _lat[1], _lat[2], alpha)
        fx = fx0 * np.cos(self.params['xComb'] * np.sign(np.arctan(alpha)))
        fy = -fy0 * np.cos(self.params['yComb'] * np.sign(kappa))
        #fx = fx0
        #fy = -fy0
        return FandMOutput(np.array([fx, fy]).T, np.zeros((3,1)))
    
    def _lat_params(self, Fz) -> tuple:
        sMy = self._l(Fz, self.params['FzN'], self.params['sMyN'], self.params['sMy2N'])
        FMy = self._q(Fz, self.params['FzN'], self.params['FMyN'], self.params['FMy2N'])
        dF0y = self._q(Fz, self.params['FzN'], self.params['dF0yN'], self.params['dF0y2N'])
        return dF0y, sMy, FMy
    
    def _long_params(self, Fz) -> tuple:
        sMx = self._l(Fz, self.params['FzN'], self.params['sMxN'], self.params['sMx2N'])
        FMx = self._q(Fz, self.params['FzN'], self.params['FMxN'], self.params['FMx2N'])
        dF0x = self._q(Fz, self.params['FzN'], self.params['dF0xN'], self.params['dF0x2N'])
        return dF0x, sMx, FMx
    
    def _pureforce(self, dF0, sM, FM, s) -> float:
        C3 = np.sqrt(((4.0 * dF0 * dF0) / ((FM * FM))) - (9.0 / (sM * sM)))
        C2 = (2.0 * dF0 * FM * C3) / (1e-4 + (FM * C3 * FM * C3))
        return FM * self._Q(C2 * self._sgm(C3 * s))
    
    def _q(self, Fz, FzN, qN, q2N) -> float:
        return (Fz / FzN) * ((2.0 * qN) - (0.5 * q2N) - (((qN - (0.5 * q2N)) * Fz) / FzN))
    
    def _l(self, Fz, FzN, lN, l2N) -> float:
        return lN + ((l2N - lN) * (Fz / (FzN - 1.0)))
    
    def _Q(self, x) -> float:
        return 0.5 * x * (3.0 - (x * x))
    
    def _sgm(self, x) -> float:
        return x / np.sqrt(x * x + 9)
        
    @staticmethod
    def GetDefaultParameters() -> dict:
        return {
            'FzN':4000,     'Fz2N':8000, 
            'dF0xN':120000, 'dF0x2N':200000,
            'dF0yN':55000,  'dF0y2N':80000,
            'sMxN':0.1,     'sMx2N':0.11,
            'sMyN':0.2,     'sMy2N':0.24,
            'FMxN':4400,    'FMx2N':8700,
            'FMyN':4200,    'FMy2N':7500,
            'xComb' :1.0,   'yComb':1.0}
        
        

class Wheel():
    eps_vx = 1e-9
    
    '''
    Wheel submodels.
    '''
    tyre : Tyre = Tyre(Calpha=-10000, Cs=200000, rRolling=0.3)
    #tyre : Tyre = Rill(rRolling=0.3)
    
    def __init__(self, Izz=10, tyre=None) -> None:
        self.Izz = Izz
        if tyre is not None:
            self.tyre = tyre
            
    def Evaluate(self, omega, kappa, tanalpha, vhub_wheel, MBrake, MThrottle, vehicle_mass, h) -> WheelOutputs:
        vxhub = vhub_wheel[0]
        vyhub = vhub_wheel[1]

        alpha = np.arctan(tanalpha)
        #alpha = np.arctan2(vyhub, self._abs(vxhub, 1e-4)) # using strict iso convention
        #alpha = np.arctan2(vyhub, self.tyre.rRolling * np.abs(omega))
        
        #kappa = self.__asr__(omega, vxhub, vehicle_mass, h)
        
        FMTyre = self.tyre.GetForceAndMoment(alpha, kappa, 4000.0)

        k = 3e-1
        MBraking = MBrake * np.tanh(k * omega)
        #MBraking = MBrake * np.sign(omega)
        domega = (1/self.Izz) * (MThrottle - MBraking - FMTyre.F[0]*self.tyre.rRolling)
        dkappa, dtanalpha = self.__bc__(omega, kappa, tanalpha, vxhub, vyhub)
        return WheelOutputs(FMTyre.F, FMTyre.M, domega, dkappa, dtanalpha, alpha, kappa)
    
    def __asr__(self, omega, vxhub, vehicle_mass, h):
        R = self.tyre.rRolling
        nu = 1.1#0.75
        um_hat = (h/2) * self.tyre.Cs * (((R*R)/self.Izz) + (1/vehicle_mass))
        #print(nu*um_hat)
        #return ((R * omega) - vxhub) / np.max([np.abs(vxhub), nu*um_hat])
        return ((R * omega) - vxhub) / (self._abs(vxhub, 0.1))
        #return -(vxhub - (R * omega)) / (R * np.abs(omega) + 1e-4)
    
    def __bc__(self, omega, slip, tanalpha, vxhub, vyhub):
        R = self.tyre.rRolling
        B = 0.3
        b = 0.3
        sgn = np.sign(vxhub)
        if sgn == 0:
            sgn = 1
        dkappa = -((self._abs(vxhub, 1e-4) - (R*omega*sgn))/B) - (self._abs(vxhub, 1e-1)/B)*slip
        dtanalpha = (vyhub/b) - (self._abs(vxhub, 1e-1)/b)*tanalpha
        
        # if slip > 1.0 and dkappa > 0.0:
        #     dkappa = 0.0
        # elif slip < -1.0 and dkappa < 0.0:
        #     dkappa = 0.0
            
        return dkappa, dtanalpha
    
    @staticmethod
    def _abs(x, eps):
        return np.sqrt((x * x) + (eps + eps)) - eps
    
if __name__ == "__main__":
    print("Testing wheel")
    
    #t = Rill(rRolling=0.3)
    t = Tyre(Calpha=-10000, Cs=200000, rRolling=0.3)
    alpha = np.linspace(-0.4, 0.4, 1000)
    kappa = np.linspace(0, 0, 1000)
    
    Fy = np.zeros(1000)
    Fz = 6000
    for i in range(len(alpha)):
        outs : FandMOutput = t.GetForceAndMoment(alpha=alpha[i], kappa=kappa[i], Fz=Fz)
        Fy[i] = outs.F[1]
        
    plt.figure
    plt.plot(alpha, Fy)
    plt.show()
    
    
    m = 10.0
    sf = 2000.0
    h = 1/sf
    
    wheel = Wheel(Izz=2)
    x0 = [0, 0, 0, 0]
    
    t0 = 0.0
    tf = 10.0
    dt  = h
    n = int(np.floor(sf * tf))
    t = np.zeros((1,n))
    t_ = t0
    
    x = np.zeros((4, n))
    x[:, 0] = x0
    
    cnt = 0
    while cnt < n - 1:
        throttle = 0
        brake = 0        
        if t_ > tf/4 and t_ < tf/2:
            throttle = 120
            brake = 0.0
        elif t_ > tf/2:
            throttle = 0.0
            brake = 50
        
        outs : WheelOutputs = wheel.Evaluate(
            x[2, cnt], x[3, cnt], np.array([x[1, cnt], 0]).T, brake, throttle, m, h
        )
        dv_mass = (1/m) * (outs.F[0])
        dx_mass = x[1,cnt]
        
        x[:, cnt + 1] = x[:, cnt] + (
            dt * np.array([dx_mass, dv_mass, outs.dOmega, outs.dkappa]).T
            )
        x[3, cnt + 1] = np.clip(x[3, cnt + 1], -1, 1)   
        t[0,cnt] = t_
        t_ += dt
        cnt += 1
    
    plt.plot(t.T[:-1], x[1,:-1])
    plt.show()