import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from bandit import Bandits

def System_Gain_Estimation(TS):
    beta   = np.sum(TS.SUM_PX2,axis = 1)
    plt.figure()
    plt.plot(TS.bandit.arms,TS.bandit.norm, label = '|G(e^{j$\omega_k$})|')
    plt.scatter(TS.bandit.arms,beta,color = 'orange',label = r'|$\^G$(e^{j$\omega$})|')
    plt.legend()
    plt.xlabel('Frequency $\omega$ [rad/s]')
    plt.ylabel('|G($e^{j\omega_k}$)|')
    plt.show()

def Plot_mean_square_error(TS):
    plt.figure()
    plt.plot(range(TS.N),TS.MSE, linewidth = 1.2 ,label = 'MSE')
    plt.title('The Mean Square Error of the Estimation in '+ TS.name)
    plt.xlabel('Number of Rounds T')
    plt.ylabel(r'MSE($|\^\beta|^2$)')
    plt.show()