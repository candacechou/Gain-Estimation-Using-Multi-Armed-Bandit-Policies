
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def System_Gain_Estimation_TS(TS):
    beta   = np.sum(TS.SUM_PX2,axis = 1)
    plt.figure()
    plt.plot(TS.bandit.arms,TS.bandit.norm, label = '|G(e^{j$\omega_k$})|')
    plt.scatter(TS.bandit.arms,beta,color = 'orange',label = r'|$\^G$(e^{j$\omega$})|')
    plt.legend()
    plt.xlabel('Frequency $\omega$ [rad/s]')
    plt.ylabel('|G($e^{j\omega_k}$)|')
    plt.show()

def Plot_Chosen_Time(model):
    plt.figure()
    plt.scatter(model.bandit.arms,model.l_na)
    plt.title('The times the algorithm explore the arms')
    plt.xlabel('Frequency $\omega$ [rad/s]')
    plt.ylabel(r'Times $n_a$')
    plt.show()


def System_Gain_Estimation_UCB(model):
    plt.figure()
    plt.plot(model.bandit.arms,model.bandit.norm, label = '|G(e^{i$\omega$})|')
    plt.scatter(model.bandit.arms,model.emp_rewards,color = 'orange',label = r'|$\^G$(e^{i$\omega$})|')
    plt.legend()
    plt.xlabel('Frequency $\omega$ [rad/s]')
    plt.ylabel('Gain')
    plt.show()

def Plot_mean_square_error(model):
    plt.figure()
    plt.semilogy(range(model.N),model.MSE, linewidth = 1.2 ,label = 'MSE')
    plt.title('The Mean Square Error of the Estimation in '+ model.name)
    plt.xlabel('Number of Rounds T')
    plt.ylabel(r'MSE($|\^\beta|^2$)')
    plt.show()