import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from bandit import Bandits
import glob
import os
import imageio
class ThompsonSampling_single_freq():
    """
    The thompson sampling in single freq
    M      - number of sampling when updateing the posterior
    Ru     - variance of prior
    sigma  - variance of measurement noise
    N      - number of experiment

    bandit - bandit class that store info about this
    """
    def __init__(self,bandit,N,sigma,Ru,M,prior,name,tag):
        self.name       = name
        self.bandit     = bandit
        self.N          = N         #### Number of experiment
        self.sigma      = sigma     #### variance of measurement noise
        self.Ru         = Ru        #### variance of prior
        self.M          = M         #### number of samping when updating the posterior
        self.m          = np.random.multivariate_normal([0,0],np.identity(2)*Ru**2,(self.bandit.K,1)).reshape((self.bandit.K,2))
        self.var        = np.random.random((self.bandit.K,2))
        self.pho        = prior     #### the weight of arm
        self.prior      = prior     #### the prior
        self.MSE        = []        #### the mean square error
        self.MSE1       = []
        self.regret     = [0]       #### regret
        self.SUM_PX2    = np.zeros((self.bandit.K,1))
        self.l_na       = np.zeros((self.bandit.K,1))
        self.tag        = tag
        self.norm       = np.zeros((self.bandit.K,1))
        
    def reset_prior(self):
        self.prior = [np.random.random() for i in range(self.bandit.K)]
        self.prior /= np.sum(self.prior)
        
    def compute_regret(self,t,arms):
        #### regret computation
        current_regret   = self.regret[-1] + self.bandit.maxnorm - self.bandit.norm[arms]
        self.regret.append(current_regret)
                                                                       
    def training(self, SHOWED = True,savefig=False,frame = 100):
        ### clear the historical posterior
        self.reset_prior()
        self.pho        = self.prior
        self.regret     = [0]
        self.m          = np.random.multivariate_normal([0,0],np.identity(2)*self.Ru**2,(self.bandit.K,1)).reshape((self.bandit.K,2))
        self.var        = np.zeros((self.bandit.K,1))+ self.Ru
        self.MSE        = []
        self.MSE1       = []
        self.norm       = np.zeros((self.bandit.K,1))
        self.l_na       = np.zeros((self.bandit.K,1))
        ### declare arms
        self.SUM_PX2    = np.zeros((self.bandit.K,1))
        SUM_PX          = np.zeros((self.bandit.K,2)) ### The summation of Pk * Xk from 0 to t
        s_t             = 0                           ### the norm of the t
        mse             = 0
        if savefig:
            if not os.path.exists(os.path.join('./Resultfig',"Result_GIF")):
                os.makedirs(os.path.join('./Resultfig',"Result_GIF"))
            filename = os.path.join('./Resultfig',"Result_GIF",self.name + "_Single_Frequency_"+self.tag+".gif")
            writer = imageio.get_writer(filename, format='GIF-PIL', mode='I', fps= 10)

        for t in range(self.N):
        
        ### we must make sure all the arms are selects in first K round  
            self.current_arm  = int(np.random.choice(self.bandit.K,1,p = self.pho))
                  
        ### perform the experiment and obtain X
            mean              = [self.bandit.hnorm[self.current_arm].real,self.bandit.hnorm[self.current_arm].imag]
            cov               = np.identity(2) * self.sigma**2 /2
            x_t               = np.random.multivariate_normal(mean,cov,1)
            self.norm[self.current_arm]      = float((x_t[0][0]**2 + x_t[0][1]**2)**0.5)
       
            SUM_PX[self.current_arm][0]      += x_t[0][0]
            SUM_PX[self.current_arm][1]      += x_t[0][1]
            
            self.l_na[self.current_arm]          += 1
            self.m[self.current_arm][0]       = self.Ru**2 * SUM_PX[self.current_arm][0]/(self.sigma**2 +  (self.Ru**2) * self.l_na[self.current_arm])
            self.m[self.current_arm][1]       = self.Ru**2 * SUM_PX[self.current_arm][1]/(self.sigma**2 +  (self.Ru**2) * self.l_na[self.current_arm])
            self.var[self.current_arm]        = self.Ru**2 /(1+ (self.Ru**2/self.sigma**2) * self.l_na[self.current_arm])
            self.Update_Posterior(self.current_arm,t)

            self.SUM_PX2[self.current_arm]    = (SUM_PX[self.current_arm][0]**2 + SUM_PX[self.current_arm][1]**2)/ self.l_na[self.current_arm]**2
            mse              = (np.nanmax(self.norm)- self.bandit.real_norm)**2
            self.MSE.append(mse)
            s_t              = np.nanargmax(self.SUM_PX2) 
            mse1             = (self.SUM_PX2[s_t] - self.bandit.real_norm)**2
            self.MSE1.append(mse1)
            self.compute_regret(t,self.current_arm) 
            if savefig and t < frame:
                files = self.__create_png(t)
                image = imageio.imread(files)
                writer.append_data(image)
                os.remove(files)

    def permute_columns(self,x):
        ix_i = np.random.sample(x.shape).argsort(axis=0)
        ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
        return x[ix_i, ix_j]


    def __create_png(self,i):
        if not os.path.exists(os.path.join("./Resultfig","Result_png")):
            os.makedirs(os.path.join("./Resultfig","Result_png"))
        filename = os.path.join("./Resultfig","Result_png","empirical_"+str(i)+".png")
        plt.figure()
        plt.plot(self.bandit.arms,self.bandit.norm, label = '|G(e^{j$\omega_k$})|')
        dy  = 2 * self.var**0.5
        y1  = [float(self.norm[inn] - dy[inn]) for inn in range(self.bandit.K)]
        y2  = [float(self.norm[inn] + dy[inn]) for inn in range(self.bandit.K)]
        plt.fill_between(self.bandit.arms, y1, y2,color='gray', alpha=0.2)
        plt.scatter(self.bandit.arms,self.norm,color = 'orange',label = r'|$\^G$(e^{j$\omega$})|')
        plt.scatter(self.bandit.arms[self.current_arm],self.norm[self.current_arm],color = 'red',label = r'|$\^G$(e^{j$\omega^*$})|')
        plt.legend()
        plt.xlabel('Frequency $\omega$ [rad/s]')
        plt.ylabel(r'|$\hat{G(j\omega)}$)|')
        plt.title("The estimated norm in "+self.name)
        plt.ylim([-1,2])
        plt.savefig(filename)
        plt.close()
        return filename
    
    def Update_Posterior(self,current_arms,t):
        self.pho = [0] * self.bandit.K
        if t == 0:
            for arms in range(self.bandit.K):
                mean      = [self.m[arms][0],self.m[arms][1]]
                cov       = np.identity(2) * self.var[arms]
                x,y       = np.random.multivariate_normal(mean,cov,self.M).T
                norm      = x**2 + y**2
                norm      = norm.reshape(self.M,1)
                if arms == 0 :
                    self.norm_list = norm
                else:
                    self.norm_list = np.hstack((self.norm_list,norm))     

        else:

            mean      = [self.m[current_arms][0],self.m[current_arms][1]]
            cov       = np.identity(2) * self.var[current_arms]
            x,y       = np.random.multivariate_normal(mean,cov,self.M).T
            norm      = x**2 + y**2
            norm      = norm.reshape(self.M,1)
            
            self.permute_columns(self.norm_list)
            
            self.norm_list[:,current_arms] = norm.ravel()
        ### accumulate the max norm on sampling
        smax   = np.argmax(self.norm_list,axis = 1)    
        #### normalize the pho
        self.pho  = [smax.tolist().count(i) / self.M for i in range(self.bandit.K)]
        



""" ----------------------------------------------------------------------------------- """   

class ThompsonSampling_Power_Allocation():
    
    def __init__(self,bandit,N,sigma,Ru,M,prior,name,tag):
        self.name       = name
        self.bandit     = bandit
        self.N          = N
        self.sigma      = sigma
        self.Ru         = Ru
        self.M          = M
        self.K          = bandit.K
        self.m          = np.random.multivariate_normal([0,0],np.identity(2)*Ru**2,(self.K,1)).reshape((self.K,2))
        self.var        = np.random.random((bandit.K,2))
        self.pho        = [] ### the prior of arm
        self.prior      = prior
        self.MSE        = []
        self.MSE1       = []
        self.regret     = [0]
        self.norm       = np.zeros((self.bandit.K,1))
        self.SUM_PX2    = np.zeros((self.bandit.K,1))
        self.run_list   = [i for i in range(self.bandit.K)]
        self.l_na       = np.zeros((self.bandit.K,1))
        self.tag        = tag
        
    def compute_regret(self,t):
        delta_t         = [self.pho[i] * (self.bandit.maxnorm - self.bandit.norm[i]) for i in range(self.K) if i != self.bandit.maxnorm]
        current_regret  = self.regret[t] + np.sum(delta_t)
        self.regret.append(np.copy(current_regret))
        
    def reset_prior(self):
        self.prior = np.random.random((self.K,1))
        self.prior /= np.sum(self.prior)
        
        
    def training(self, SHOWED = True,savefig=False,frame = 100):
        ### clear the historical posterior
        self.reset_prior()
        self.run_list    = [i for i in range(self.bandit.K)]
        self.pho        = self.prior
        self.regret     = [0]
        self.MSE        = []
        self.m          = np.random.multivariate_normal([0,0],np.identity(2)*self.Ru**2,(self.K,1)).reshape((self.K,2))
        self.var        = np.random.random((self.bandit.K,1))
        ### declare arms
        SUM_PX          = np.zeros((self.bandit.K,2)) ### The summation of Pk * Xk from 0 to t
        self.l_na       = np.zeros((self.bandit.K,1)) ### The summation of Pk from 0 to t
        self.SUM_PX2    = np.zeros((self.bandit.K,1))
        s_t             = 0      ### the norm of the t
        mse             = 0
        self.norm       = np.zeros((self.bandit.K,1))
        
        if savefig:
            if not os.path.exists(os.path.join('./Resultfig',"Result_GIF")):
                os.makedirs(os.path.join('./Resultfig',"Result_GIF"))
            filename = os.path.join('./Resultfig',"Result_GIF",self.name + "_Single_Frequency_"+self.tag+".gif")
            writer = imageio.get_writer(filename, format='GIF-PIL', mode='I', fps= 10)

        for t in range(self.N):
            for arms in self.run_list:
                cov                  = np.identity(2) * self.sigma**2 /(2 * self.pho[arms])
                mean                 = [self.bandit.hnorm[arms].real,self.bandit.hnorm[arms].imag] 
                x_t                  = np.random.multivariate_normal(mean,cov,1)
#                
        ### update the posterior of mk for the frequency
                SUM_PX[arms][0]     += self.pho[arms] * x_t[0][0]
                SUM_PX[arms][1]     += self.pho[arms] * x_t[0][1]
        
                self.l_na[arms]     += self.pho[arms]
                self.SUM_PX2[arms]   = (SUM_PX[arms][0]**2 + SUM_PX[arms][1]**2)/ self.l_na[arms]**2
                
                self.m[arms][0]      = self.Ru**2 * SUM_PX[arms][0]/(self.sigma**2 + (self.Ru**2) * self.l_na[arms])
                self.m[arms][1]      = self.Ru**2 * SUM_PX[arms][1]/(self.sigma**2 + (self.Ru**2) * self.l_na[arms])
                self.var[arms]       = self.Ru**2 /(1+ (self.Ru**2/self.sigma**2) * self.l_na[arms])
                self.SUM_PX2[arms]   = (SUM_PX[arms][0]**2 + SUM_PX[arms][1]**2)/ self.l_na[arms]**2
                self.norm[arms]      = (self.m[arms][0]**2 + self.m[arms][1]**2)**0.5
            self.Update_Posterior(t)

        ### calculate the norm
            mse              = (np.nanmax(self.norm)- self.bandit.real_norm)**2
            self.MSE.append(mse)
            s_t              = np.nanargmax(self.SUM_PX2) 
            mse1             = (self.SUM_PX2[s_t] - self.bandit.real_norm)**2
            self.MSE1.append(mse1)
            self.compute_regret(t)
            if savefig and t < frame:
                files = self.__create_png(t)
                image = imageio.imread(files)
                writer.append_data(image)
                os.remove(files)
        
            

    def permute_columns(self,x):
        ix_i = np.random.sample(x.shape).argsort(axis=0)
        ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
        return x[ix_i, ix_j]


    def Update_Posterior(self,t):

        if t == 0:
            for arms in range(self.bandit.K):
            
                mean      = [self.m[arms][0],self.m[arms][1]]
                cov       = np.identity(2) * self.var[arms]
                x,y       = np.random.multivariate_normal(mean,cov,self.M).T
                norm      = x**2 + y**2
                norm      = norm.reshape(self.M,1)
            
                if arms == 0 :
                    self.norm_list = norm
                else:
                    self.norm_list = np.hstack((self.norm_list,norm))      
        else:
            
            for arms in self.run_list:
                self.permute_columns(self.norm_list)
                mean      = [self.m[arms][0],self.m[arms][1]]
                cov       = np.identity(2) * self.var[arms]
                x,y       = np.random.multivariate_normal(mean,cov,self.M).T
                norm      = x**2 + y**2
                norm      = norm.reshape(self.M,1)
                self.norm_list[:,arms] = norm.ravel()
                      
        ### accumulate the max norm on sampling
        smax   = np.argmax(self.norm_list,axis = 1)    
        #### normalize the pho
        self.pho  = [smax.tolist().count(i) / self.M for i in range(self.K)]
        self.pho  /= np.sum(self.pho)
        self.run_list = np.nonzero(self.pho)[0]
       
        
    def __create_png(self,i):
        if not os.path.exists(os.path.join("./Resultfig","Result_png")):
            os.makedirs(os.path.join("./Resultfig","Result_png"))
        filename = os.path.join("./Resultfig","Result_png","empirical_"+str(i)+".png")
        plt.figure()
        plt.plot(self.bandit.arms,self.bandit.norm, label = '|G(e^{j$\omega_k$})|')
        dy  = 2 * self.var**0.5
        y1  = [float(self.norm[inn] - dy[inn]) for inn in range(self.bandit.K)]
        y2  = [float(self.norm[inn] + dy[inn]) for inn in range(self.bandit.K)]
        plt.fill_between(self.bandit.arms, y1, y2,color='gray', alpha=0.2)
        plt.scatter(self.bandit.arms,self.norm,color = 'orange',label = r'|$\^G$(e^{j$\omega$})|')
        
        plt.legend()
        plt.xlabel('Frequency $\omega$ [rad/s]')
        plt.ylabel(r'|$\hat{G(j\omega)}$)|')
        plt.title("The estimated norm in "+self.name)
        plt.ylim([-1,2])
        plt.savefig(filename)
        plt.close()
        return filename