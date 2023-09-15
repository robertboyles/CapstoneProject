import numpy as np

class AeroOutputs():
    def __init__(self, FzF, FzR, Fdrag) -> None:
        self.FzF = FzF
        self.FzR = FzR
        self.Fdrag = Fdrag
        
class Aero():
    def __init__(self, A=1.4, rho=1.2) -> None:
        self.A = A
        self.rho = rho
    def GetAeroForce(self, vx, vy) -> AeroOutputs:
        pass
    
class SimpleAero(Aero):
    def __init__(self, Clf, Clr, Cd) -> None:
        super().__init__()
        self.Clf = Clf
        self.Clr = Clr
        self.Cd = Cd
        
    def GetAeroForce(self, vx, vy):
        vmag = np.sqrt((vx*vx) + (vy*vy))
        FzFront = self.__calcforce__(vmag, self.Clf)
        FzRear = self.__calcforce__(vmag, self.Clr)
        Fdrag = self.__calcforce__(vmag, self.Cd)
        return AeroOutputs(FzFront, FzRear, Fdrag)
        
    def __calcforce__(self, v, Coeff):
        return 0.5 * self.rho * self.A * Coeff * v * v 

if __name__ == "__main__":
    print("Test Aero")
    model = SimpleAero(2, 2, 1)
    vx = 50
    vy = 0
    print(model.GetAeroForce(vx, vy).FzF)