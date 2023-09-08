import numpy as np
from environment import Environment
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
from read_track_data import TrackDataReader
import matplotlib.pyplot as plt
import time

class AgentOutput():
    def __init__(self, rBrakeThrottle, aHandWheel) -> None:
        self.rBrakeThrottle = rBrakeThrottle
        self.aHandWheel = aHandWheel

class PD():
    rBrakeThrottle_max = 1
    rBrakeThrottle_min = -1
    
    aHandWheel_max = 200 * (np.pi/180)
    aHandWheel_min = -200 * (np.pi/180)
    
    def __init__(self, 
                 vkp, vkd,
                 ykp, ykd,
                 yawkp, yawkd, 
                 updateRate=1/5) -> None:
        
        self.vkp, self.vkd = vkp, vkd
        self.ykp, self.ykd = ykp, ykd
        self.yawkp, self.yawkd = yawkp, yawkd
        
        self.h = updateRate
        self.rBrakeThrottle = 1.0
        self.aHandWheel = 0.0
        self.vErrorLast = 0.0
        self.yErrorLast = 0.0
        self.yawErrorLast = 0.0
        self.last_t = 0.0
    
    def output(self, vError, yError, yawError, t) -> AgentOutput:
        
        if t - self.last_t >= self.h or t == 0.0:
            dvError = vError - self.vErrorLast
            self.vErrorLast = vError
                        
            drBrakeThrottle = self.vkp * vError + self.vkd * dvError
            
            dyError = yError - self.yErrorLast
            self.yErrorLast = yError
            
            dyawError = yawError - self.yawErrorLast
            self.yawErrorLast = yawError
            
            daHandWheel = self.ykp * yError + dyError * self.ykd + self.yawkp * yawError + dyawError * self.yawkd
            
            self.rBrakeThrottle += drBrakeThrottle
            self.rBrakeThrottle = np.min([self.rBrakeThrottle_max, np.max([self.rBrakeThrottle_min, self.rBrakeThrottle])])
            
            self.aHandWheel += daHandWheel
            self.aHandWheel = np.min([self.aHandWheel_max, np.max([self.aHandWheel_min, self.aHandWheel])])
            
        return AgentOutput(self.rBrakeThrottle, self.aHandWheel)
    
if __name__ == "__main__":
    
    trackdata = TrackDataReader().load_example_data()
    
    s = trackdata['s']
    k = trackdata['k']
    v = trackdata['v'] * 0.8
    sf = s[-1]

    car : BicycleModel = BicycleModel()
    car.X0 = np.array([v[0],0,0,v[0]/0.3,v[0]/0.3,0,0.025])
    car.powertrain.MDrive_ref *= 2.0
    car.brakesystem.MBrake_ref *= 2.0
    car.wheelf.Izz *= 3
    car.wheelr.Izz *= 3
    
    track : TrackDefinition = TrackDefinition(s, k)
    
    env : Environment = Environment(vehicleModel=car, track=track, 
                                    fixed_update=20.0)
    
    controller = PD(
        vkp=-0.01, vkd=-0.2,
        ykp=0.002, ykd=0.1,
        yawkp=0.001, yawkd=10,
        updateRate=1/5.0
    )
    
    _, ind_ephi = env.GetStateValue('ephi')
    _, ind_ey = env.GetStateValue('ey')
    _, ind_vx = env.GetStateValue('vx')
    _, ind_vy = env.GetStateValue('vy')
    slap, ind_s = env.GetStateValue('s')
    
    period_disp = 100
    disp_cnt = 0
    start_time = time.time()
    while slap < sf:
        state = env.X
        ephi = state[ind_ephi]
        ey = state[ind_ey]
        vx = state[ind_vx]
        vy = state[ind_vy]
        
        vmag = np.sqrt((vx * vx) + (vy * vy))
        
        ev = vmag - np.interp(slap, s, v)
        
        t = env.t
        
        pid_output : AgentOutput = controller.output(ev, ey, ephi, t)
        
        if disp_cnt % period_disp == 0:
            print("aHandWheel = %f\trThrottle = %f\t time = %f\t v = %f\t s = %f" % (
                pid_output.aHandWheel*180/np.pi, 
                pid_output.rBrakeThrottle, 
                t, vmag, slap))
        disp_cnt += 1
        
        env.step(
            rBrakeThrottle=pid_output.rBrakeThrottle, 
            aHandWheel=pid_output.aHandWheel)
        
        slap = state[ind_s]
    end_time = time.time()
    print("Timing = %f" % (end_time - start_time))    
    rBrakeThrottle = env.GetActionTrajectory('rBrakeThrottle')
    aHandWheel = env.GetActionTrajectory('aHandWheel')
    t = env.GetTime()
    sLap = env.GetStateTrajectory('s')
    
    if False:
        independentvar = t
    else:
        independentvar = sLap
    
    plt.subplot(2,2,1)
    plt.plot(independentvar, env.GetStateTrajectory('vx'))
    plt.plot(s, v, color='red', linewidth=0.5)
    plt.subplot(2,2,2)
    plt.plot(env.GetStateTrajectory('x_global'), env.GetStateTrajectory('y_global'))
    plt.subplot(2,2,3)
    plt.plot(independentvar, rBrakeThrottle)
    plt.subplot(2,2,4)
    plt.plot(independentvar, aHandWheel * 180/np.pi)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(independentvar, env.GetOutputTrajectory('gLat'))
    plt.subplot(2,2,2)
    plt.plot(independentvar, env.GetOutputTrajectory('gLong'))
    plt.subplot(2,2,3)
    plt.plot(independentvar, env.GetOutputTrajectory('Fyf'))
    plt.subplot(2,2,4)
    plt.plot(independentvar, env.GetOutputTrajectory('Fxr'))
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(independentvar, env.GetStateTrajectory('nYaw'))
    plt.subplot(3,1,2)
    plt.plot(independentvar, env.GetOutputTrajectory('Fyf'))
    plt.subplot(3,1,3)
    plt.plot(independentvar, env.GetOutputTrajectory('wheel_f_Fy'))
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(independentvar, env.GetOutputTrajectory('alphaF') * 180/np.pi)
    plt.subplot(2,2,2)
    plt.plot(independentvar, env.GetOutputTrajectory('kappaF'))
    plt.subplot(2,2,3)
    plt.plot(independentvar, env.GetOutputTrajectory('alphaR') * 180/np.pi)
    plt.subplot(2,2,4)
    plt.plot(independentvar, env.GetOutputTrajectory('kappaR'))
    
    plt.show()