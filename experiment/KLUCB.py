import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from bandit import Bandits
import glob
import os
import imageio

class KLUCB_Single_Freq():
    def __init__(self,bandit,sigma,N,name,c,tag):

        '''
        THE KL-UCB algorithms
        N    - Number of experiment
        l_na - number of times a played up to t
        bandit - bandit class that store info about this
        func   - function of the KL radius

        '''
        if name not in ["KLUCB","KLUCB-Plus","KLUCB-H","KLUCB-H-Plus"]:
            return ValueError(f"No such Type:{name}")
        self.name             = name
        self.bandit           = bandit
        self.N                = N
        self.l_na             = np.zeros((self.bandit.K,1))
        self.sigma            = sigma
        self.regret           = [0]
        self.emp_rewards      = np.zeros((self.bandit.K , 1))
        self.MSE              = []
        self.MSE1             = []
        self.Bound            = np.zeros((self.bandit.K,1))
        self.c                = c
        self.tag = tag
        self.current_arm = 0
        
    def compute_regret(self,arms):

        current_regret = self.regret[-1] + self.bandit.maxnorm - self.bandit.norm[arms]
        self.regret.append(current_regret)
    
    def rate_loglogt(self,t,arms):

        if self.c * np.log(np.log(t+1))+ np.log(t+1) < 0:
            return 1/self.bandit.K
        else:
            a = (np.log(t+1) + self.c * np.log(np.log(t+1)))/(self.l_na[arms])
            return a

    def Calculate_Bounds(self,t):
        
        if self.name == "KLUCB":
            
            """
            KLUCB
            """
            
            self.Bound = np.array([self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt(self.rate_loglogt(t,arms)) for arms in range(self.bandit.K)])
            
        elif self.name == "KLUCB-Plus":
            
            """
            KLUCB-Plus
            """
            
            self.Bound = np.array([self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt(self.rate_loglogt(t,arms)) for arms in range(self.bandit.K)])
            
        elif self.name == "KLUCB-H":
            
            """
            KLUCB-H
            """
            
            self.Bound = np.array([self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt(self.rate_loglogt(self.N,arms)) for arms in range(self.bandit.K)])
            
        elif self.name == "KLUCB-H-Plus":
            
            """
            KLUCB-H-Plus
            """
            
            self.Bound    = np.array([self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt(self.rate_loglogt(self.N/self.l_na[arms],arms)) for arms in range(self.bandit.K)])
        else:
            
            return ValueError(f"No such Type:{self.name}")
        
    def training(self,SHOWED = True,savefig=False,frame = 100):

        #### reinitialize the values
        self.regret          = [0]
        self.l_na            = np.zeros((self.bandit.K,1))
        self.emp_rewards     = np.zeros((self.bandit.K,1))
        self.MSE             = []
        mse                  = 0
        self.SUM_PX      = np.zeros((self.bandit.K,2))
        self.Bound           = np.zeros((self.bandit.K,1))
        
        if savefig:
            
            if not os.path.exists(os.path.join('./Resultfig',"Result_GIF")):
                os.makedirs(os.path.join('./Resultfig',"Result_GIF"))
                
            filename = os.path.join('./Resultfig',"Result_GIF",self.name + "_Single_Frequency_"+self.tag+".gif")
            writer = imageio.get_writer(filename, format='GIF-PIL', mode='I', fps= 10)

        for t in range(self.N):
            ###  arm selection
            if t < self.bandit.K:
                self.current_arm = t
            else:
                self.Calculate_Bounds(t)
                self.current_arm       = int(np.nanargmax(self.Bound))
                
            self.l_na[self.current_arm] += 1

            ### perform the experiment and obtain X
            mean = [self.bandit.hnorm[self.current_arm].real, self.bandit.hnorm[self.current_arm].imag]
            cov  = np.identity(2) * self.sigma**2 / 2
            x_t  = np.random.multivariate_normal(mean,cov,1) ### Rewards
            
            ### calculate Rewards
            self.SUM_PX[self.current_arm][:] += x_t[0]
            
            ### update the emp_rewards
            self.emp_rewards[self.current_arm]    = (self.SUM_PX[self.current_arm][0]**2 + self.SUM_PX[self.current_arm][1]**2) / self.l_na[self.current_arm]**2   
            
            mse                    = (np.nanmax(self.emp_rewards) - self.bandit.real_norm)**2
            self.MSE.append(mse)
            mse                    = (self.emp_rewards[np.nanargmax(self.l_na)] - self.bandit.real_norm)**2
            self.MSE1.append(mse)
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
        
        y1  = [float(2*self.emp_rewards[inn]-self.Bound[inn]) for inn in range(self.bandit.K)]
        y2  = [float(self.Bound[inn]) for inn in range(self.bandit.K)]
        
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

            

'-------------------------------------------------------------------------------------------------------'
class KLUCB_Power_Allocation():
    def __init__(self,bandit,sigma,N,name,prior,c,tag="1",beta = 1):
        if name not in ["KLUCB","KLUCB-Plus","KLUCB-H","KLUCB-H-Plus"]:
            return ValueError(f"No such Type:{name}")

        self.bandit        = bandit
        self.sigma         = sigma
        self.N             = N
        self.name          = name
        self.l_na          = np.zeros((self.bandit.K,1))
        self.regret        = [0]
        self.emp_rewards   = np.zeros((self.bandit.K,1))
        self.pho           = prior
        self.prior         = prior
        self.SUM_PX        = np.zeros((self.bandit.K,2))
        self.Bound         = np.zeros((self.bandit.K,1))
        self.MSE           = []
        self.MSE1          = []
        self.c             = c
        self.beta          = beta
        self.tag           = tag
        
    def compute_regret(self,t):
        
        delta_t   = [self.pho[i] * (self.bandit.maxnorm - self.bandit.norm[i]) for i in range(self.bandit.K)]
        current_regret = self.regret[-1] + np.sum(delta_t)
        self.regret.append(np.copy(current_regret))

    def Calculate_Bounds(self,t):
        
        if self.name == "KLUCB":
            
            """
            KLUCB
            """
            
            self.Bound = np.array([self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt(self.rate_loglogt(t,arms)) for arms in range(self.bandit.K)])
            
        elif self.name == "KLUCB-Plus":
            
            """
            KLUCB-Plus
            """
            
            self.Bound = np.array([self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt(self.rate_loglogt(t,arms)) for arms in range(self.bandit.K)])
            
        elif self.name == "KLUCB-H":
            
            """
            KLUCB-H
            """
            
            self.Bound    = np.array([self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt(self.rate_loglogt(self.N,arms)) for arms in range(self.bandit.K)])
            
        elif self.name == "KLUCB-H-Plus":
            
            """
            KLUCB-H-Plus
            """
            
            self.Bound = np.array([self.emp_rewards[arms] + np.sqrt(2*self.sigma**2)*np.sqrt(self.rate_loglogt(self.N/self.l_na[arms],arms)) for arms in range(self.bandit.K)])
            
        else:
            
            return ValueError(f"No such Type:{self.name}")
           
    def Update_Pho(self,t,SHOWED= False):
        
        r             =  self.beta* abs(np.sqrt(2)*self.sigma * (self.c * np.log(np.log(t+2)) + np.log(t+2))/(2+t))
        z             = 0
        T             = 0
        bar           = list(self.Bound.T[0])
        a             = np.max(bar)
        s             = [i  for i in bar if i != a]
        N             = self.bandit.K - len(s)
        
        while (T < r) and (N < self.bandit.K) :
            b = np.max(s)
            T = T + N * (a - b)
            if T > r:
                break
            a = b
            s = [i  for i in s if i!= b]
            N = self.bandit.K - len(s)
            
        if t == 0:
            self.pho = [1/self.bandit.K] *self.bandit.K
            
        else:
            
            if N >= self.bandit.K:
               self.pho       = [self.Bound[i] for i in range(self.bandit.K)]
               self.run_list  = np.nonzero(self.pho)[0]
               self.pho      /= np.sum(self.pho)

            else:
                z = b + (T-r)/N
                # print('z:',z)
                self.pho       = [a - z if a > z else 0 for a in self.Bound.T[0] ]
                self.run_list  = np.nonzero(self.pho)[0]
                self.pho      /= np.sum(self.pho)

        
            
        if SHOWED == True:
            print("t = ",t," pho = ",self.pho)
            plt.figure()
            plt.scatter(range(self.bandit.K),self.Bound,color = 'blue')
            plt.plot(range(self.bandit.K),self.Bound,color = 'blue')
            plt.bar(range(self.bandit.K),self.pho,color = 'r')
            plt.show()
            input("Press enter to continue...")
    
    def rate_loglogt(self,t,arms):
        
        if self.c * np.log(np.log(t+1))+ np.log(t+1) < 0:
            
            return 1/self.bandit.K
        else:
            
            a = (self.c * np.log(np.log(t+1))+ np.log(t+1))/(2*self.l_na[arms])
            
            return a
        
    def reset_prior(self):
        
        self.prior            = [1/self.bandit.K] * self.bandit.K
        self.prior           /= np.sum(self.prior)
        
    def training(self,SHOWED = True,savefig=False,frame=0):
        
        ### initialization
        self.reset_prior
        self.regret      = [0]
        self.emp_rewards = np.zeros((self.bandit.K,1))
        self.pho         = self.prior
        self.SUM_PX      = np.zeros((self.bandit.K,2))
        self.l_na        = np.zeros((self.bandit.K,1))
        self.Bound       = np.zeros((self.bandit.K,1))
        self.MSE         = []
        mse              = 0
        self.run_list    = [i for i in range(self.bandit.K)]
        
        if savefig:
            if not os.path.exists(os.path.join('./Resultfig',"Result_GIF")):
                os.makedirs(os.path.join('./Resultfig',"Result_GIF"))
            filename = os.path.join('./Resultfig',"Result_GIF",self.name + "_Power_Allocation_"+self.tag+".gif")
            writer = imageio.get_writer(filename, format='GIF-PIL', mode='I', fps= 10)

        for t in range(self.N):
            for arms in self.run_list:
                if self.pho[arms] !=0:
                    mean = [self.bandit.hnorm[arms].real,self.bandit.hnorm[arms].imag]
                    cov  = np.identity(2) * self.sigma**2 / (2 * self.pho[arms])
                    x_t  = np.random.multivariate_normal(mean,cov,1) ### rewards 
                    ### calculate rewards
                    self.SUM_PX[arms][:] += self.pho[arms] * x_t[0]
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

    def __create_png(self,i):
        
        if not os.path.exists(os.path.join("./Resultfig","Result_png")):
            os.makedirs(os.path.join("./Resultfig","Result_png"))
            
        filename = os.path.join("./Resultfig","Result_png","empirical_"+str(i)+".png")
        plt.figure()
        plt.plot(self.bandit.arms,self.bandit.norm, label = '|G(e^{j$\omega_k$})|')
        dy  = self.Bound
        y1  = [float(2*self.emp_rewards[inn]-self.Bound[inn]) for inn in range(self.bandit.K)]
        y2  = [float(self.emp_rewards[inn] + self.Bound[inn]) for inn in range(self.bandit.K)]
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