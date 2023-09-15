import gym
import numpy as np
from Wheel import Wheel, Rill as Tyre
from bicyclemodel import BicycleModel
from FSFrame import TrackDefinition

def GetBaseCar():
    chassis_params = {
                'mass': 800,
                'Izz' : 1300,
                'lf' : 1.5,
                'lr' : 1.5,
                'steering_ratio': -12.0, # aHandWheel/aSteer
                'max_aHandWheel': 200 * np.pi/180.0
            }

    # tyref_params = {
    #             'FzN':4000,     'Fz2N':8000, 
    #             'dF0xN':200000, 'dF0x2N':210000, # 255000
    #             'dF0yN':80000,  'dF0y2N':90000, # 166500
    #             'sMxN':0.11,     'sMx2N':0.2, # 0.02 or 0.04
    #             'sMyN':0.24,     'sMy2N':0.45, # 6deg
    #             'FMxN':6000,   'FMx2N':10000, # 9500 or 9000
    #             'FMyN':7500,   'FMy2N':10000, # 7000
    #             'xComb' :0.1,   'yComb':0.1}

    # tyrer_params = {
    #             'FzN':4000,     'Fz2N':8000, 
    #             'dF0xN':200000, 'dF0x2N':210000, # 153000
    #             'dF0yN':90000,  'dF0y2N':100000, # 250000
    #             'sMxN':0.11,     'sMx2N':0.2, # 0.1
    #             'sMyN':0.24,     'sMy2N':0.45, # 2.7deg
    #             'FMxN':6000,   'FMx2N':10000, # 7000
    #             'FMyN':9000,   'FMy2N':10000, # 8000
    #             'xComb' :0.1,   'yComb':0.1}

    tyref_params = {
                'FzN':4000,     'Fz2N':8000, 
                'dF0xN':255000, 'dF0x2N':265000, # 255000
                'dF0yN':166500,  'dF0y2N':170500, # 166500
                'sMxN':0.04,     'sMx2N':0.1, # 0.02 or 0.04
                'sMyN': 6 * np.pi/180,     'sMy2N':7 * np.pi/180, # 6deg
                'FMxN':9000,   'FMx2N':9500, # 9500 or 9000
                'FMyN':7000,   'FMy2N':10000, # 7000
                'xComb' :0.1,   'yComb':0.1}

    tyrer_params = {
                'FzN':4000,     'Fz2N':8000, 
                'dF0xN':153000, 'dF0x2N':160000, # 153000
                'dF0yN':250000,  'dF0y2N':260000, # 250000
                'sMxN':0.1,     'sMx2N':0.2, # 0.1
                'sMyN':2.7 * np.pi / 180,     'sMy2N':3 * np.pi / 180, # 2.7deg
                'FMxN':7000,   'FMx2N':10000, # 7000
                'FMyN':8000,   'FMy2N':10000, # 8000
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
    car.X0 = np.array([30,0,0,30/0.3,30/0.3,0,0.0001,0,0,0.2,0])
    return car

def GetBaseTrack(trackdata):
    s, k = trackdata['s'], trackdata['k']
    track : TrackDefinition = TrackDefinition(s, k, width=10.0, k_error_scale=1.5)
    track.X0 = np.array([0, 0, 0, 0, 0, 0])
    return track