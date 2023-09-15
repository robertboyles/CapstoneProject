import gym
import numpy as np
from Wheel import Wheel, Rill as Tyre
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition

def GetBaseCar():
    chassis_params = {
                'mass': 500,
                'Izz' : 2000,
                'lf' : 1.5,
                'lr' : 1.5,
                'steering_ratio': -12.0, # aHandWheel/aSteer
                'max_aHandWheel': 200 * np.pi/180.0
            }

    tyref_params = {
                'FzN':4000,     'Fz2N':8000, 
                'dF0xN':200000, 'dF0x2N':210000,
                'dF0yN':80000,  'dF0y2N':90000,
                'sMxN':0.11,     'sMx2N':0.2,
                'sMyN':0.24,     'sMy2N':0.45,
                'FMxN':8700,   'FMx2N':10000,
                'FMyN':7500,   'FMy2N':10000,
                'xComb' :0.1,   'yComb':0.1}

    tyrer_params = {
                'FzN':4000,     'Fz2N':8000, 
                'dF0xN':200000, 'dF0x2N':210000,
                'dF0yN':90000,  'dF0y2N':100000,
                'sMxN':0.11,     'sMx2N':0.2,
                'sMyN':0.24,     'sMy2N':0.45,
                'FMxN':10000,   'FMx2N':10000,
                'FMyN':9000,   'FMy2N':10000,
                'xComb' :0.1,   'yComb':0.1}


    tyref = Tyre(
        parameters=tyref_params, rRolling=0.3
    )

    tyrer = Tyre(
        parameters=tyrer_params, rRolling=0.3
    )
        
    wheelf = Wheel(Izz=1.5, tyre=tyref)
    wheelr = Wheel(Izz=1.5, tyre=tyrer)

    car : BicycleModel = BicycleModel(parameters=chassis_params, wheelf_overload=wheelf, wheelr_overload=wheelr)
    car.powertrain.MDrive_ref = 200.0
    car.X0 = np.array([30,0,0,30/0.3,30/0.3,0,0.0001,0,0,0.2,0])
    return car

def GetBaseTrack(trackdata):
    s, k = trackdata['s'], trackdata['k']
    track : TrackDefinition = TrackDefinition(s, k, width=10.0, k_error_scale=1.5)
    track.X0 = np.array([0, 0, 0, 0, 0, 0])
    return track