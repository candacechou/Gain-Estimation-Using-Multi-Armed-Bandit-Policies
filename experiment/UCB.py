import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from bandit import Bandits

import glob
import os
import imageio

class UCB_Single_Freq():
    """
    The upper confidence bound algorithm 
    (one in RL course slides)
    N      - Number of experiment
    l_na   - number of times a played up to t
    theta  - empirical reward of a up to t
    bandit - bandit class that store info about this
    """
    def __init__(self,bandit,sigma,N,name,tag):
        self.name      = name
        self.bandit    = bandit
        self.N         = N
        self.l_na      = np.zeros((self.bandit.K,1)) 
        self.sigma     = sigma
        self.regret    = [0]
        self.emp_rewards = np.random.random((self.bandit.K,1)) ### empirical rewards
        self.MSE         = []
        self.MSE1        = []
        self.tag = tag
        self.current_arm = 0
#         self.logregret   = []
    def compute_regret(self,arms):
        
        current_regret   = self.regret[-1] + self.bandit.maxnorm - self.bandit.norm[arms]
        
        self.regret.append(current_regret)
        
    def training(self,SHOWED = True,savefig=False,frame = 100):
        #### clear the previous values // re-initialize the values
        
        self.regret      = [0]
        self.l_na        = np.zeros((self.bandit.K,1)) 
        self.regret      = [0]
        self.MSE         = []
        self.MSE1        = []
        
        mse              = 0
        #self.theta       = np.zeros((bandit.K,2))
        self.emp_rewards = np.random.random((self.bandit.K,1))
        self.SUM_PX      = np.zeros((self.bandit.K,2))
        if savefig:
            if not os.path.exists(os.path.join('./Resultfig',"Result_GIF")):
                os.makedirs(os.path.join('./Resultfig',"Result_GIF"))
            filename = os.path.join('./Resultfig',self.name + "_"+self.tag+".gif")
            writer = imageio.get_writer(filename, format='GIF-PIL', mode='I', fps= 10)
            
        for t in range(self.N):
            ### SELECT ARMS
            if t < self.bandit.K:
                self.current_arm = t
            else:
                self.current_arm = np.argmax(self.emp_rewards +  self.sigma*np.sqrt(2*np.log(t+2)/self.l_na))
            
            self.l_na[self.current_arm] +=1
            
            ### perform the experiment and obtain X
            mean = [self.bandit.hnorm[self.current_arm].real,self.bandit.hnorm[self.current_arm].imag]
            cov  = np.identity(2) * self.sigma**2 / 2
            x_t  = np.random.multivariate_normal(mean,cov,1) ### rewards 
            
            self.SUM_PX[self.current_arm][:] += x_t[0]
            
            ### update the emp_rewards
            self.emp_rewards[self.current_arm]    = (self.SUM_PX[self.current_arm][0]**2 + self.SUM_PX[self.current_arm][1]**2) / self.l_na[self.current_arm]**2  
            
            ### update regret and rewards
            mse  = (np.nanmax(self.emp_rewards) - self.bandit.real_norm)**2
            self.MSE.append(mse)
            mse1  = (self.emp_rewards[np.nanargmax(self.l_na)] - self.bandit.real_norm)**2
            self.MSE1.append(mse1)
            self.compute_regret(self.current_arm)
            if savefig and t < frame:
                files = self.__create_png(t)
                image = imageio.imread(files)
                writer.append_data(image)
                os.remove(files)

        
            
    def __create_png(self,i):
        if not os.path.exists(os.path.join("./Resultfig","Result_png")):
            os.makedirs(os.path.join("./Resultfig","Result_png"))
        filename = os.path.join("./Resultfig","Result_png","empirical_"+str(i)+".png")
        plt.figure()
        plt.plot(self.bandit.arms,self.bandit.norm, label = '|G(e^{j$\omega_k$})|')
        dy  = self.sigma*np.sqrt(2*np.log(i+2)/(self.l_na+1e-10))
        y1  = [float(self.emp_rewards[inn] - dy[inn]) for inn in range(self.bandit.K)]
        y2  = [float(self.emp_rewards[inn] + dy[inn]) for inn in range(self.bandit.K)]
        plt.fill_between(self.bandit.arms, y1, y2,color='gray', alpha=0.2)
        plt.scatter(self.bandit.arms,self.emp_rewards,color = 'orange',label = r'|$\^G$(e^{j$\omega$})|')
        plt.scatter(self.bandit.arms[self.current_arm],self.emp_rewards[self.current_arm],color = 'red',label = r'|$\^G$(e^{j$\omega^*$})|')
        plt.legend()
        plt.xlabel('Frequency $\omega$ [rad/s]')
        plt.ylabel(r'|$\hat{G(j\omega)}$)|')
        plt.title("The empirical reward in "+self.name)
        plt.ylim([-1,2])
        plt.savefig(filename)
        plt.close()
        return filename
        
    
      
  
        
   
       

class UCB_Power_Allocation :
    def __init__(self,bandit,sigma,N,prior,name,beta,tag):
        self.name        = name
        self.bandit      = bandit
        self.N           = N
        self.pho         = prior
        self.prior       = prior
        self.emp_rewards = np.random.random((self.bandit.K,1)) ### empirical rewards
        self.l_na        = np.zeros((self.bandit.K,1)) 
        self.SUM_PX      = np.zeros((self.bandit.K,2))
        self.sigma       = sigma
        self.regret      = [0]
        self.MSE         = []
        self.MSE1        = []
        self.run_list    = [i for i in range(self.bandit.K)]
        self.beta        = beta
        self.tag         = tag
        
    def Calculate_Bounds(self,t):
        
        self.bounds = [self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt((np.log(t+1)/self.l_na[arms])) for arms in range(self.bandit.K)]
        self.bounds = np.array(self.bounds)    
        
    def compute_regret(self,t):
        delta_t         = [self.pho[i] * (self.bandit.maxnorm - self.bandit.norm[i]) for i in range(self.bandit.K)]
        current_regret  = self.regret[-1] + np.sum(delta_t)
        self.regret.append(np.copy(current_regret))
    def reset_prior(self):
        self.prior            = [1/self.bandit.K] * self.bandit.K
        self.prior           /= np.sum(self.prior)
        
    def training(self, SHOWED = True,savefig=False,frame=0):
        #### clear the previous values // re-initialize the values
        self.reset_prior
        self.regret      = [0]
        self.l_na        = np.zeros((self.bandit.K,1))
        self.SUM_PX      = np.zeros((self.bandit.K,2))
        self.emp_rewards = np.zeros((self.bandit.K,1))
        self.pho         = self.prior
        self.bounds      = np.zeros((self.bandit.K,1))
        self.MSE         = []
        self.run_list    = [i for i in range(self.bandit.K)]
        if savefig:
            if not os.path.exists(os.path.join('./Resultfig',"Result_GIF")):
                os.makedirs(os.path.join('./Resultfig',"Result_GIF"))
            filename = os.path.join('./Resultfig',"Result_GIF",self.name + "_Power_Allocation_"+self.tag+".gif")
            writer = imageio.get_writer(filename, format='GIF-PIL', mode='I', fps= 10)
            
        for t in range(self.N):
            ### SELECT ARMS
            for arms in self.run_list :
                
                ### perform the experiment and obtain X
                if self.pho[arms] != 0:
                    mean = [self.bandit.hnorm[arms].real,self.bandit.hnorm[arms].imag]
                    cov  = np.identity(2) * self.sigma**2 / (2*self.pho[arms])
                    x_t  = np.random.multivariate_normal(mean,cov,1) ### rewards 
            
                    ### calculate rewards
                    self.SUM_PX[arms][0] += self.pho[arms] * x_t[0][0]
                    self.SUM_PX[arms][1] += self.pho[arms] * x_t[0][1]
                    self.l_na[arms]      += self.pho[arms]
                
                    ### update the emp_rewards
                    self.emp_rewards[arms]    = (self.SUM_PX[arms][0]**2 + self.SUM_PX[arms][1]**2) / self.l_na[arms]**2
                    
            self.Calculate_Bounds(t)
            if savefig and t < frame:
                files = self.__create_png(t)
                image = imageio.imread(files)
                writer.append_data(image)
                os.remove(files)
            self.Update_Pho(t,SHOWED)
            mse  = (np.nanmax(self.emp_rewards) - self.bandit.real_norm)**2
            self.MSE.append(mse)
            mse  = (self.emp_rewards[np.nanargmax(self.l_na)] - self.bandit.real_norm)**2
            self.MSE1.append(mse)
            ### update regret and reward
                
            self.compute_regret(t)
            
            

    def Update_Pho(self,t,SHOWED = False):
        r             = self.beta*np.sqrt(2*self.sigma**2)*(np.log(t+2))/(2+t)
        z             = 0
        T             = 0
        bar           = list(self.bounds.T[0])
        a             = np.nanmax(bar)
        s             = [i  for i in bar if i != a]
        N             = self.bandit.K - len(s)
        b             = 0
        self.run_list = []
        while (T < r) and (N < self.bandit.K) :
            b         = np.max(s)
            T         = T + N * (a - b)
            if T > r:
                break
            a          = b
            s          = [i  for i in s if i!= b]
            N          = self.bandit.K - len(s)
            # print('b = ',b,'T = ',T,'a = ',a,'N = ',N,'s = ',s)
        if t == 0:
            self.pho = [1/self.bandit.K] *self.bandit.K
        else:
            if N >= self.bandit.K:
                self.pho       = [self.bounds[i] for i in range(self.bandit.K)]
                self.run_list  = np.nonzero(self.pho)[0]
                self.pho      /= np.sum(self.pho)
            else:
                z              = b + (T-r)/N
                self.pho       = [a - z if a > z else 0 for a in self.bounds.T[0] ]
                self.run_list  = np.nonzero(self.pho)[0]
                self.pho      /= np.sum(self.pho)

        
            
        if SHOWED == True:
            print("t = ",t," pho = ",self.pho)
            plt.figure()
            plt.scatter(range(self.bandit.K),self.bounds,color = 'blue')
            plt.plot(range(self.bandit.K),self.bounds,color = 'blue')
            plt.bar(range(self.bandit.K),self.pho,color = 'r')
            plt.show()
            input("Press enter to continue...")
        
    def __create_png(self,i):
        if not os.path.exists(os.path.join("./Resultfig","Result_png")):
            os.makedirs(os.path.join("./Resultfig","Result_png"))
        filename = os.path.join("./Resultfig","Result_png","empirical_"+str(i)+".png")
        plt.figure()
        plt.plot(self.bandit.arms,self.bandit.norm, label = '|G(e^{j$\omega_k$})|')
        dy  = self.sigma*np.sqrt(2*np.log(i+2)/(self.l_na+1e-10))
        y1  = [float(self.emp_rewards[inn] - dy[inn]) for inn in range(self.bandit.K)]
        y2  = [float(self.emp_rewards[inn] + dy[inn]) for inn in range(self.bandit.K)]
        plt.fill_between(self.bandit.arms, y1, y2,color='gray', alpha=0.2)
        for current_arm,i_pho in enumerate(self.pho):
            
            if float(i_pho) > 0.0:
                plt.scatter(self.bandit.arms[current_arm],self.emp_rewards[current_arm],color = 'red',label = r'|$\^G$(e^{j$\omega^*$})|')
            else:
                plt.scatter(self.bandit.arms,self.emp_rewards,color = 'orange',label = r'|$\^G$(e^{j$\omega$})|')
        

        plt.xlabel('Frequency $\omega$ [rad/s]')
        plt.ylabel(r'|$\hat{G(j\omega)}$)|')
        plt.title("The empirical reward in "+self.name)
        plt.ylim([-1,2])
        plt.savefig(filename)
        plt.close()
        return filename
