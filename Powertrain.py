import numpy as np

def smoothMax(x, y, eps):
    return 0.5 * ((x + y) + np.sqrt(((x - y) * (x - y)) + (eps * eps)))

class Powertrain_Outputs():
    def __init__(self, MFinalDrive) -> None:
        self.MFinalDrive = MFinalDrive

class Powertrain():
    def resolveBrakeThrottle(self, rBrakeThrottle, omega_axle) -> Powertrain_Outputs:
        pass

class OptimalPower(Powertrain):
    def __init__(self, PowerMax=700e3, effeciency = 0.9) -> None:
        self.PMax = PowerMax
        self.nu = effeciency
        
    def resolveBrakeThrottle(self, rBrakeThrottle, omega_axle) -> Powertrain_Outputs:
        if rBrakeThrottle >= 0.0:
            rThrottle = rBrakeThrottle
        else:
            rThrottle = 0.0

        PEngine = rThrottle * self.PMax * self.nu
        eps = 1
        min_omega = 10 # 3m/s roughly
        MFinalDrive = PEngine / smoothMax(np.abs(omega_axle), min_omega, eps)
    
        return Powertrain_Outputs(MFinalDrive)
