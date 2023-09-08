import numpy as np

class Powertrain_Outputs():
    def __init__(self, MDriveshaft) -> None:
        self.MDriveshaft = MDriveshaft

class Powertrain():
    def resolveBrakeThrottle(self, rBrakeThrottle) -> Powertrain_Outputs:
        pass
    
class SimpleDirectDrive(Powertrain):
    MDrive_ref = 1200
    def __init__(self, MDrive_ref=1200) -> None:
        self.MDrive_ref = MDrive_ref
        
    def resolveBrakeThrottle(self, rBrakeThrottle) -> Powertrain_Outputs:
        if rBrakeThrottle >= 0.0:
            rThrottle = rBrakeThrottle
        else:
            rThrottle = 0.0
        
        return Powertrain_Outputs(rThrottle * self.MDrive_ref)