import numpy as np

class BrakingSystemOutput():
    def __init__(self, MFront, MRear) -> None:
        self.MFront = MFront
        self.MRear = MRear

class BrakingSystem():
    rBB = 0.5
    def resolveBrakeThrottle(self, rBrakeThrottle) -> BrakingSystemOutput:
        pass

class SimpleBrake(BrakingSystem):
    MBrake_ref = 20000
    def __init__(self, rBB=0.5, MBrake_ref=20000) -> None:
        self.rBB = rBB
        self.MBrake_ref = MBrake_ref
    
    def resolveBrakeThrottle(self, rBrakeThrottle) -> BrakingSystemOutput:
        if rBrakeThrottle <= 0:
            rBrake = -rBrakeThrottle
        else:
            rBrake = 0.0
        
        max_brake = self.MBrake_ref * rBrake
        return BrakingSystemOutput(MFront=self.rBB * max_brake, MRear=(1 - self.rBB) * max_brake)
        