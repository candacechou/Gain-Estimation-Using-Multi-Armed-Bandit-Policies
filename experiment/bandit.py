import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class Bandits():
    """
    Define Bandits
    """
    def __init__(self,A,B,K):
        self.B = B
        self.A = A
        self.K = K
        self.__generate_arms()
        self.__find_real_norm()

    def __find_real_norm(self):
        print("Analysing the norms...")
        [arms, norms] = signal.freqz(self.B,self.A,10000000)
        self.real_norm = np.max(norms.real**2 + norms.imag**2)

    def __generate_arms(self):
        print("Generating the arms....")
        [self.arms,self.hnorm] = signal.freqz(self.B,self.A,self.K)
        self.__calculate_norm()
        self.maxk = np.argmax(self.norm)
        self.maxnorm= self.norm[self.maxk]
        
    def __calculate_norm(self):
        self.norm = self.hnorm.real**2 + self.hnorm.imag**2
        
    def plot_real_system(self):
        plt.figure()
        plt.plot(self.arms,self.norm)
        plt.title("Frequency Response of the system G")
        plt.xlim([0,3.15])
        plt.xlabel('Frequency $\omega$ [rad/s]')
        plt.ylabel('|G($e^{j\omega}$|')
        
    def print_hinf_norm(self):
        print("The hinf norm on this system is :", self.maxnorm,'Which is in arm : ',self.maxk)