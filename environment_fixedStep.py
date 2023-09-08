import numpy as np
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition
import matplotlib.pyplot as plt
from scipy.integrate import RK45 as integrator
from scipy.integrate import solve_ivp as integrator_fixed

class Environment():
    h0 = 1e-6

    @property
    def h(self) -> float:
        
        if self.fixed_update_rate:
            val = self.sf
        else:
            if self.integrator is not None and self.integrator.t_old is not None:
                val = self.integrator.t - self.integrator.t_old
            else:
                val = 1e-6
        
        return val
    
    def __init__(self, vehicleModel : BicycleModel, track : TrackDefinition, 
                 fixed_update:float|bool=1/20.0) -> None:
        self.car = vehicleModel
        _, self.ind_rBrakeThrottle = self.car.GetNamedValue(
            'rBrakeThrottle', self.car.getInputNames()
        )
        _, self.ind_ahandwheel = self.car.GetNamedValue(
            'aHandWheel', self.car.getInputNames()
        )
        _, self.ind_h = self.car.GetNamedValue(
            'h', self.car.getInputNames()
        )
        _, self.ind_car_vx = self.car.GetNamedValue(
            'vx', self.car.getStateNames()
        )
        _, self.ind_car_vy = self.car.GetNamedValue(
            'vy', self.car.getStateNames()
        )
        _, self.ind_car_nyaw = self.car.GetNamedValue(
            'nYaw', self.car.getStateNames()
        )
        
        self.track = track
        _, self.ind_track_vx_in = self.track.GetNamedValue(
            'vx', self.track.getInputNames()
        )
        _, self.ind_track_vy_in = self.track.GetNamedValue(
            'vy', self.track.getInputNames()
        )
        _, self.ind_track_nyaw_in = self.track.GetNamedValue(
            'nYaw', self.track.getInputNames()
        )
        
        
        self.nx = vehicleModel.nx + track.nx
        self.variableNames : tuple = self.car.getStateNames() + self.track.getStateNames()
        self.outputNames : tuple = self.car.getOutputNames() + self.track.getOutputNames()
        self.actionNames : tuple = ('rBrakeThrottle', 'aHandWheel')
        
        if not fixed_update:
            self.fixed_update_rate = False
        else:
            self.fixed_update_rate = True
            self.sf = fixed_update
        
        self.initialise()
    
    def initialise(self) -> None:
        self.X = []
        self.X = np.append(self.car.X0, self.track.X0)
        
        self.state_history = []
        self.state_history.append(self.X.copy())
        
        self.time = []
        self.t = 0.0
        self.time.append(self.t)
        
        self.action_history, self.current_actions = [], None
        self.output_history, self.lastOutputs = [], None
        
        self.integrator = None
        
        if not self.fixed_update_rate:
            self.integrator = integrator(
                        self.ode_fun, float(0), self.X, float(10000), first_step=self.h0)
    
    def  ode_fun(self, t, y):
            u = self.current_actions if self.current_actions is not None else np.ones(2)

            h = self.h
            
            # Car
            inputs_car = np.zeros((self.car.nu,))
            inputs_car[self.ind_rBrakeThrottle] = u[0]
            inputs_car[self.ind_ahandwheel] = u[1]
            inputs_car[self.ind_h] = h
            
            states_car = y[0:self.car.nx]
            dstates_car, outputs_car = self.car.evaluate(states_car, inputs_car)
            
            # Track
            inputs_track = np.zeros((self.track.nx,))
            inputs_track[self.ind_track_vx_in] = states_car[self.ind_car_vx]
            inputs_track[self.ind_track_vy_in] = states_car[self.ind_car_vy]
            inputs_track[self.ind_track_nyaw_in] = states_car[self.ind_car_nyaw]
            
            states_track = y[self.car.nx:self.nx]
            dstates_track, outputs_track = self.track.evaluate(states_track, inputs_track)
            
            dy = np.zeros((self.nx))
            dy[0:self.car.nx] = np.transpose(dstates_car)
            dy[self.car.nx:self.nx] = np.transpose(dstates_track)
            
            self.lastOutputs = np.concatenate([outputs_car, outputs_track])
            
            return dy
    
    def step(self, rBrakeThrottle, aHandWheel):
        
        self.current_actions = np.array([rBrakeThrottle, aHandWheel])
        
        if self.fixed_update_rate:
            self.integrator = integrator(
                self.ode_fun, float(self.t), self.X, float(self.t + self.h))
            
            while self.integrator.status == 'running':
                self.integrator.step()
        else:
            self.integrator.step()
            
        self.X = self.integrator.y
        self.state_history.append(self.integrator.y)
        self.output_history.append(self.lastOutputs)
        self.action_history.append(self.current_actions)
        self.t = self.integrator.t
        self.time.append(self.t)
        
    
    def GetOutputTrajectory(self, name):
        _temp = self.output_history.copy()
        _temp.append(_temp[-1])
        trajectory = np.array(_temp)
        return trajectory[:, self.outputNames.index(name)]
    
    def GetActionTrajectory(self, name):
        _temp = self.action_history.copy()
        _temp.append(_temp[-1])
        trajectory = np.array(_temp)
        return trajectory[:, self.actionNames.index(name)]
    
    def GetStateValue(self, name):
        index = self.variableNames.index(name)   
        return self.X[index], index
    
    def GetOutputValue(self, name):
        index = self.outputNames.index(name)   
        return self.lastOutputs[index], index
    
    def GetStateTrajectory(self, name):
        _, index = self.GetStateValue(name)
        trajectory = np.array(self.state_history)
        return trajectory[:, index]
    
    def GetTime(self) -> np.array:
        return np.array(self.time)
        

if __name__ == "__main__":
    
    s = np.linspace(0, 10000, 100)
    k = np.zeros(s.shape)
    
    car = BicycleModel()
    track = TrackDefinition(s, k)
    
    env : Environment = Environment(vehicleModel=car, track=track)
    print(env.variableNames)
    print(env.X)
    

    while env.t < 255:
        print("t = %f\t h = %f" % (env.t, env.h))
        if env.t > 100 and env.t < 101:
            aSteer = -5 * (np.pi/180)
        else:
            aSteer = 0.0
            
        if env.t > 105:
            rBT = -0.1
        else:
            rBT = 1.0
        
        env.step(rBT, aSteer)

    
    trajectory = np.array(env.state_history)
    x = trajectory[:, 8]
    y = trajectory[:, 9]
    nWheelF = trajectory[:, 3]
    nWheelR = trajectory[:, 4]
    aheading = trajectory[:, 10]
    vx = trajectory[:, 0]
    t = np.array(env.time)
    
    plt.subplot(1,2,1)
    plt.plot(t, vx)
    plt.subplot(1,2,2)
    plt.plot(x,y)
    plt.show()
    