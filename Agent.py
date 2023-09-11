import numpy as np
from environment import Environment
from bicyclemodel import BicycleModel
from Wheel import Wheel, Rill as Tyre
from FSFrame import TrackDefinition
from read_track_data import TrackDataReader
import matplotlib.pyplot as plt
import time

class AgentOutput():
    def __init__(self, drBrakeThrottle, daHandWheel) -> None:
        self.drBrakeThrottle = drBrakeThrottle
        self.daHandWheel = daHandWheel

class PD():
    drBrakeThrottle = 0.0
    daHandWheel = 0.0
        
    def __init__(self, 
                 vkp, vkd,
                 ykp, ykd,
                 yawkp, yawkd, 
                 updateRate=1/5, 
                 drBrakeThrottle_max=4, daHandWheel_max=400*np.pi/180.0) -> None:
        
        self.vkp, self.vkd = vkp, vkd
        self.ykp, self.ykd = ykp, ykd
        self.yawkp, self.yawkd = yawkp, yawkd
        
        self.h = updateRate
        self.vErrorLast = 0.0
        self.yErrorLast = 0.0
        self.yawErrorLast = 0.0
        self.last_t = 0.0
        
        self.drBrakeThrottle_max = drBrakeThrottle_max
        self.drBrakeThrottle_min = -drBrakeThrottle_max
    
        self.daHandWheel_max = daHandWheel_max
        self.daHandWheel_min = -daHandWheel_max
    
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
            
            self.drBrakeThrottle = np.min([self.drBrakeThrottle_max, 
                                     np.max([self.drBrakeThrottle_min, 
                                             drBrakeThrottle])])
            
            self.daHandWheel = np.min([self.daHandWheel_max, 
                                  np.max([self.daHandWheel_min, 
                                          daHandWheel])])
            
        return AgentOutput(self.drBrakeThrottle, self.daHandWheel)
    
if __name__ == "__main__":
    
    trackdata = TrackDataReader().load_example_data()
    
    s = trackdata['s']
    k = trackdata['k']
    v = trackdata['v'] * 0 + 20
    k_offset = 1.5e-4 * 0
    k_error_scale = 1.5
    sf = s[-1]
    
    chassis_params = {
            'mass': 500,
            'Izz' : 2000,
            'lf' : 1.5,
            'lr' : 1.5,
            'steering_ratio': -12.0, # aHandWheel/aSteer
            'max_aHandWheel': 200 * np.pi/180.0
        }

    tyre_params = {
                'FzN':4000,     'Fz2N':8000, 
                'dF0xN':200000, 'dF0x2N':210000,
                'dF0yN':50000,  'dF0y2N':55000,
                'sMxN':0.1,     'sMx2N':0.15,
                'sMyN':0.2,     'sMy2N':0.21,
                'FMxN':10000,   'FMx2N':30000,
                'FMyN':6000,   'FMy2N':7000,
                'xComb' :0.01,   'yComb':0.01}

    tyrecommon = Tyre(
        parameters=tyre_params, rRolling=0.3
    )
        
    wheelf = Wheel(Izz=1.5, tyre=tyrecommon)
    wheelr = Wheel(Izz=1.5, tyre=tyrecommon)

    car : BicycleModel = BicycleModel(parameters=chassis_params, wheelf_overload=wheelf, wheelr_overload=wheelr)
    car.powertrain.MDrive_ref = 200.0
    car.X0 = np.array([30,0,0,30/0.3,30/0.3,0,0.0001,0,0,0.2,0])
        
    track : TrackDefinition = TrackDefinition(s, k, width=5, k_offset=k_offset, k_error_scale=k_error_scale)
    
    x,y,xl,yl,xr,yr = track.plot_track(step=1, show=True)
    
    env : Environment = Environment(vehicleModel=car, track=track, 
                                    fixed_update=10.0)
    
    controller = PD(
        vkp=-0.01, vkd=-0.2,
        ykp=0, ykd=10,
        yawkp=0.01, yawkd=100,
        updateRate=1/10.0,
        drBrakeThrottle_max=4, daHandWheel_max=400 * np.pi/180.0
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
                pid_output.daHandWheel*180/np.pi, 
                pid_output.drBrakeThrottle, 
                t, vmag, slap))
        disp_cnt += 1
        
        drBrakeThrottle_scaled, daHandWheel_scaled = env.scale_actions(
            pid_output.drBrakeThrottle, pid_output.daHandWheel, inverse=True
        )
        
        env.step(
            drBrakeThrottle_scaled=drBrakeThrottle_scaled, 
            daHandWheel_scaled=daHandWheel_scaled)
        
        slap = state[ind_s]
    end_time = time.time()
    print("Timing = %f" % (end_time - start_time))    
    rBrakeThrottle = env.GetStateTrajectory('rBrakeThrottle')
    aHandWheel = env.GetStateTrajectory('aHandWheel')
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
    plt.axis('square')
    plt.subplot(2,2,3)
    plt.plot(independentvar, rBrakeThrottle)
    plt.subplot(2,2,4)
    plt.plot(independentvar, aHandWheel * 180/np.pi)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(independentvar, env.GetOutputTrajectory('ay'))
    plt.title('ay')
    plt.subplot(2,2,2)
    plt.plot(independentvar, env.GetOutputTrajectory('ax'))
    plt.title('ax')
    plt.subplot(2,2,3)
    plt.plot(independentvar, env.GetStateTrajectory('ephi'))
    plt.title('ephi')
    plt.subplot(2,2,4)
    plt.plot(independentvar, env.GetStateTrajectory('ey'))
    plt.title('ey')    
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(env.GetOutputTrajectory('alphaF'), env.GetOutputTrajectory('wheel_f_Fy'), marker='*', linestyle="None")
    plt.title('Front Lat.')
    plt.subplot(2,2,2)
    plt.plot(env.GetOutputTrajectory('alphaR'), env.GetOutputTrajectory('wheel_r_Fy'), marker='*', linestyle="None")
    plt.title('Rear Lat.')
    plt.subplot(2,2,3)
    plt.plot(env.GetOutputTrajectory('kappaF'), env.GetOutputTrajectory('wheel_f_Fx'), marker='*', linestyle="None")
    plt.title('Front Long.')
    plt.subplot(2,2,4)
    plt.plot(env.GetOutputTrajectory('kappaR'), env.GetOutputTrajectory('wheel_r_Fx'), marker='*', linestyle="None")
    plt.title('Rear Long.')
    
    plt.figure()
    plt.plot(env.GetStateTrajectory('x_global'), env.GetStateTrajectory('y_global'),
             linewidth=2.0)
    plt.plot(x, y, color='black', linewidth=0.5)
    plt.plot(xl, yl, color='red', linewidth=0.5)
    plt.plot(xr, yr, color='red', linewidth=0.5)
    plt.axis('square')
    plt.show()