import numpy as np
import matplotlib.pyplot as plt

class TrackDataReader():
    @staticmethod
    def readCSV(fullpath) -> None:
        data = np.loadtxt(fullpath, delimiter=',', dtype=float)
        slap = data[0,:]
        k = data[1,:]
        v = data[2,:]
        return {'s': slap, 'k':k, 'v':v}
    
    @staticmethod
    def load_example_data():
        return TrackDataReader.readCSV("data/bcn_track_curvature.csv")
    
if __name__ == "__main__":
    fullfile = "data/bcn_track_curvature.csv"
    data = TrackDataReader.readCSV(fullpath=fullfile)
    
    plt.subplot(2,1,1)
    plt.plot(data['s'], data['v'])
    plt.title('vCar')
    plt.subplot(2,1,2)
    plt.plot(data['s'], data['k'])
    plt.title('curvature')
    plt.show()