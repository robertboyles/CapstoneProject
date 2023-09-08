import numpy as np
from Aero import SimpleAero
from Wheel import Wheel, WheelOutputs
# from Wheel import Tyre
from Wheel import Rill as Tyre
from BrakingSystem import SimpleBrake, BrakingSystemOutput
from Powertrain import SimpleDirectDrive, Powertrain_Outputs
from Model_ABC import ModelABC

class BicycleModel(ModelABC):        
    '''
    Vehicle sub models
    '''
    aero : SimpleAero = SimpleAero(Clf=2.0, Clr=2.0, Cd=1.0)
    # wheelf : Wheel = Wheel(
    #                         Izz=1.38,
    #                         tyre=Tyre(Calpha=-77000, Cs=200000, rRolling=0.3)
    #                         )
    # wheelr : Wheel = Wheel(
    #                         Izz=1.48,
    #                         tyre=Tyre(Calpha=-77000, Cs=200000, rRolling=0.3)
    #                         )
    
    wheelf : Wheel = Wheel(
                            Izz=1.38,
                            tyre=Tyre(rRolling=0.3)
                            )
    wheelr : Wheel = Wheel(
                            Izz=1.48,
                            tyre=Tyre(rRolling=0.3)
                            )
    
    brakesystem : SimpleBrake = SimpleBrake(rBB=0.5, MBrake_ref=4000) 
    
    powertrain : SimpleDirectDrive = SimpleDirectDrive(600)
    
    def __init__(self, parameters:dict=None, 
                 wheelf_overload=None, wheelr_overload=None) -> None:
        self.parameters = BicycleModel.getDefaultParameters() if parameters is None else parameters
        
        if wheelf_overload is not None:
            self.wheelf = wheelf_overload
        
        if wheelr_overload is not None:
            self.wheelr = wheelr_overload

        self.initialise_references()
    
    def evaluate(self, x:np.array, u:np.array):
        # initialise outputs
        dx = np.zeros((self.nx,))
        y = np.zeros((self.ny,))
        
        # local parameter references
        p = self.parameters
        steering_ratio = p['steering_ratio']
        lf = p['lf']
        lr = p['lr']
        mass = p['mass']
        Izz = p['Izz']
        
        # Assign inputs
        rThrottleBrake = np.clip(x[self.ind_rBrakeThrottle], -1, 1)
        aHandWheel = np.clip(x[self.ind_aHandWheel], 
                             -200*np.pi/180, 200*np.pi/180)
        aSteer = aHandWheel * (1/steering_ratio)
        h = u[self.ind_h]
        
        # Assign state references
        vx, vy, nyaw, nwheelf, nwheelr, kappaF, kappaR, tanalphaF, tanalphaR = (
            x[self.ind_vx], x[self.ind_vy], x[self.ind_nyaw], 
            x[self.ind_nwheelf], x[self.ind_nwheelr], 
            x[self.ind_kappaF], x[self.ind_kappaR], x[self.ind_tanalphaF], x[self.ind_tanalphaR])
        
        # Evalate sub models
        FAero = self.aero.GetAeroForce(vx, vy)
        brakes : BrakingSystemOutput = self.brakesystem.resolveBrakeThrottle(rBrakeThrottle=rThrottleBrake)
        driveshaft : Powertrain_Outputs = self.powertrain.resolveBrakeThrottle(rBrakeThrottle=rThrottleBrake)
        
        c_, s_ = np.cos(aSteer), np.sin(aSteer)
        R_wv = np.array(((c_, s_), (-s_, c_))) # rotation matrix to wheel from vehicle

        axle_f_wr = (np.array([0.0, nyaw]).T * lf)
        axle_r_wr = -(np.array([0.0, nyaw]).T * lr)

        vhub_f = (np.matmul(R_wv, np.array([vx, vy + nyaw * lf]).T)) # I think this wr also needs rotating!
        vhub_r = (np.array([vx, vy - nyaw*lr]).T)

        kappaF = np.clip(kappaF, -1.0, 0.0)
        wheelf_out : WheelOutputs = self.wheelf.Evaluate(
            nwheelf, kappaF, tanalphaF, vhub_f, brakes.MFront, 0.0, mass, h
        )
        
        kappaR = np.clip(kappaR, -1.0, 1.0)
        wheelr_out : WheelOutputs = self.wheelr.Evaluate(
            nwheelr, kappaR, tanalphaR, vhub_r, brakes.MRear, driveshaft.MDriveshaft, mass, h
        )
        
        # Evaluate vehicle model
        Ff = np.matmul(R_wv.T, 2.0*wheelf_out.F)
        Fr = 2.0*wheelr_out.F

        Mf = Ff[1] * lf
        Mr = -Fr[1] * lr

        dx[self.ind_vx] = (1/mass) * (Ff[0] + Fr[0] - FAero.Fdrag) + nyaw*vy
        dx[self.ind_vy] = (1/mass) * (Ff[1] + Fr[1]) - nyaw * vx
        dx[self.ind_nyaw] = (1/Izz) * (Mf + Mr)
        dx[self.ind_nwheelf] = wheelf_out.dOmega
        dx[self.ind_nwheelr] = wheelr_out.dOmega
        dx[self.ind_kappaF] = wheelf_out.dkappa
        dx[self.ind_kappaR] = wheelr_out.dkappa
        dx[self.ind_tanalphaF] = wheelf_out.dtanalpha
        dx[self.ind_tanalphaR] = wheelr_out.dtanalpha
        
        drBrakeThrottleRequest = u[self.ind_drBrakeThrottle]
        daHandWheelRequest = u[self.ind_daHandWheel]
        
        if rThrottleBrake < -0.999 and drBrakeThrottleRequest < 0.0:
            drBrakeThrottle = 0.0
        elif rThrottleBrake > 0.999 and drBrakeThrottleRequest > 0.0:
            drBrakeThrottle = 0.0
        else:
            drBrakeThrottle = drBrakeThrottleRequest
        
        if aHandWheel < -199.999*np.pi/180 and daHandWheelRequest < 0.0:
            daHandWheel = 0.0
        elif aHandWheel > 199.999*np.pi/180 and daHandWheelRequest > 0.0:
            daHandWheel = 0.0
        else:
            daHandWheel = daHandWheelRequest
            
        dx[self.ind_rBrakeThrottle] = drBrakeThrottle
        dx[self.ind_aHandWheel] = daHandWheel
        
        y[self.ind_ax] = dx[0] # not really correct
        y[self.ind_ay] = dx[1] # not really correct
        y[self.ind_fyf] = np.array([Ff[1]])
        y[self.ind_fyr] = np.array([Fr[1]])
        y[self.ind_fxf] = np.array([Ff[0]])
        y[self.ind_fxr] = np.array([Fr[0]])
        y[self.ind_asteer] = np.array([aSteer])
        
        y[self.ind_wheel_f_Fx] = np.array([wheelf_out.F[0]])
        y[self.ind_wheel_r_Fx] = np.array([wheelr_out.F[0]])
        y[self.ind_wheel_f_Fy] = np.array([wheelf_out.F[1]])
        y[self.ind_wheel_r_Fy] = np.array([wheelr_out.F[1]])
        
        y[self.ind_outalphaF] = np.array([wheelf_out.alpha])
        y[self.ind_outalphaR] = np.array([wheelr_out.alpha])
        y[self.ind_outkappaF] = np.array([wheelf_out.kappa])
        y[self.ind_outkappaR] = np.array([wheelr_out.kappa])

        y[self.ind_wheel_f_vx] = np.array([vhub_f[0]])
        y[self.ind_wheel_f_vy] = np.array([vhub_f[1]])
        y[self.ind_wheel_r_vx] = np.array([vhub_r[0]])
        y[self.ind_wheel_r_vy] = np.array([vhub_r[1]])

        y[self.ind_Mf] = np.array([Mf])
        y[self.ind_Mr] = np.array([Mr])

        y[self.ind_axle_f_wr] = np.array([axle_f_wr[1]])
        y[self.ind_axle_r_wr] = np.array([axle_r_wr[1]])
        
                
        return dx, y
    
    def initialise_references(self) -> None:
        _, self.ind_daHandWheel = self.GetNamedValue('daHandWheel', self.getInputNames())
        _, self.ind_drBrakeThrottle = self.GetNamedValue('drBrakeThrottle', self.getInputNames())
        _, self.ind_h = self.GetNamedValue('h', self.getInputNames())
        
        _, self.ind_aHandWheel = self.GetNamedValue('aHandWheel', self.getStateNames())
        _, self.ind_rBrakeThrottle = self.GetNamedValue('rBrakeThrottle', self.getStateNames())
        
        _, self.ind_vx = self.GetNamedValue('vx', self.getStateNames())
        _, self.ind_vy = self.GetNamedValue('vy', self.getStateNames())
        _, self.ind_nyaw = self.GetNamedValue('nYaw', self.getStateNames())
        _, self.ind_nwheelf = self.GetNamedValue('nWheelF', self.getStateNames())
        _, self.ind_nwheelr = self.GetNamedValue('nWheelR', self.getStateNames())
        _, self.ind_kappaF = self.GetNamedValue('kappaF', self.getStateNames())
        _, self.ind_kappaR = self.GetNamedValue('kappaR', self.getStateNames())
        _, self.ind_tanalphaF = self.GetNamedValue('tanalphaF', self.getStateNames())
        _, self.ind_tanalphaR = self.GetNamedValue('tanalphaR', self.getStateNames())
        
        _, self.ind_ax = self.GetNamedValue('ax', self.getOutputNames())
        _, self.ind_ay = self.GetNamedValue('ay', self.getOutputNames())
        _, self.ind_fyf = self.GetNamedValue('Fyf', self.getOutputNames())
        _, self.ind_fyr = self.GetNamedValue('Fyr', self.getOutputNames())
        _, self.ind_fxf = self.GetNamedValue('Fxf', self.getOutputNames())
        _, self.ind_fxr = self.GetNamedValue('Fxr', self.getOutputNames())
        _, self.ind_asteer = self.GetNamedValue('aSteer', self.getOutputNames())

        _, self.ind_wheel_f_Fx = self.GetNamedValue('wheel_f_Fx', self.getOutputNames())
        _, self.ind_wheel_f_Fy = self.GetNamedValue('wheel_f_Fy', self.getOutputNames())
        _, self.ind_wheel_r_Fx = self.GetNamedValue('wheel_r_Fx', self.getOutputNames())
        _, self.ind_wheel_r_Fy = self.GetNamedValue('wheel_r_Fy', self.getOutputNames())

        _, self.ind_wheel_f_vx = self.GetNamedValue('wheel_f_vx', self.getOutputNames())
        _, self.ind_wheel_f_vy = self.GetNamedValue('wheel_f_vy', self.getOutputNames())
        _, self.ind_wheel_r_vx = self.GetNamedValue('wheel_r_vx', self.getOutputNames())
        _, self.ind_wheel_r_vy = self.GetNamedValue('wheel_r_vy', self.getOutputNames())

        _, self.ind_Mf = self.GetNamedValue('Mf', self.getOutputNames())
        _, self.ind_Mr = self.GetNamedValue('Mr', self.getOutputNames())

        _, self.ind_axle_f_wr = self.GetNamedValue('axle_f_wr', self.getOutputNames())
        _, self.ind_axle_r_wr = self.GetNamedValue('axle_r_wr', self.getOutputNames())


        
        _, self.ind_outalphaF = self.GetNamedValue('alphaF', self.getOutputNames())
        _, self.ind_outalphaR = self.GetNamedValue('alphaR', self.getOutputNames())
        _, self.ind_outkappaF = self.GetNamedValue('kappaF', self.getOutputNames())
        _, self.ind_outkappaR = self.GetNamedValue('kappaR', self.getOutputNames())
        
    
    @staticmethod
    def getStateNames() -> tuple:
        return (
            'vx', 'vy', 'nYaw', 
            'nWheelF', 'nWheelR', 
            'kappaF', 'kappaR',
            'tanalphaF', 'tanalphaR',
            'rBrakeThrottle', 'aHandWheel',
        )
    
    @staticmethod
    def getOutputNames() -> tuple:
        return (
            'ax', 'ay', 'Fyf', 'Fyr', 'Fxf', 'Fxr', 'aSteer', 
            'wheel_f_Fx', 'wheel_f_Fy', 'wheel_r_Fx', 'wheel_r_Fy',
            'wheel_f_vx', 'wheel_f_vy', 'wheel_r_vx', 'wheel_r_vy', 
            'alphaF', 'alphaR', 'kappaF', 'kappaR', 'Mf', 'Mr',
            'axle_f_wr', 'axle_r_wr'
        )
    
    @staticmethod
    def getDefaultInitialConditions() -> np.array:
        return np.array([0,0,0,0,0,0,0,0,0,0,0])
    
    @staticmethod
    def getInputNames() -> tuple:
        return (
            'drBrakeThrottle', 'daHandWheel', 'h'
        )
    
    @staticmethod
    def getDefaultParameters() -> dict:
        return {
            'mass': 800,
            'Izz' : 1200,
            'lf' : 1.5,
            'lr' : 1.5,
            'steering_ratio': -12.0 # aHandWheel/aSteer
        }

if __name__ == "__main__":
    print("Testing Vehicle Model")
    model = BicycleModel()
    dx, y = model.evaluate([0,0,0,0,0,0,0,0,0], [0,0.000, 1/20.0])
    print(dx)
    print(y)