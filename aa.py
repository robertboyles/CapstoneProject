import numpy as np
import pickle
from opengym_wrapper import FinalTrajectory

data : FinalTrajectory = FinalTrajectory().load('plots/testing/427')

print(data.time)