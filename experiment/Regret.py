import numpy as np
import matplotlib.pyplot as plt


class Model_Regret():
    """
    This class is to calculation the model regret
    """

    def __init__(self, model, N):
        """
        Model : model
        N     : number of time taking average
        """
        self.model    = model
        self.Num      = N
        self.regret   = None
        self.var      = None
        self.mean     = None
        self.new_mean = None
        self.MSE      = None
        self.MSE1     = None
        
    def Reset_Regret(self, Nu, N):
        self.regret   = None
        self.var      = None
        self.mean     = None
        self.new_mean = None
        self.N        = Nu_N
        self.MSE      = None
        self.MSE1     = None
        
    def Compute_Regret(self,showed = False,savefig = False,frame = 0):

        print("Start to train the model!")
        print("=============================")
        for i in range(self.Num):
            if i < self.Num-1 or not savefig:
                self.model.training(showed,savefig = False)
            else:
                self.model.training(showed,savefig = True,frame=frame)
            print(f"The {i+1}th model with the overall Regret: {self.model.regret[-1]}")
          
            if i == 0:
                self.regret   = np.array(self.model.regret).reshape((self.model.N + 1, 1))
                self.MSE      = self.model.MSE
                self.MSE1     = self.model.MSE1
                self.mean     = self.regret
                self.new_mean = self.mean
                self.var      = np.zeros((self.model.N + 1, 1))
            else:
                xn               = np.array(self.model.regret).reshape((self.model.N + 1, 1))
                self.regret      = np.hstack((self.regret, xn))
                self.MSE         = [ (self.MSE[j]  + self.model.MSE[j]) for j in range(self.model.N) ]
                self.MSE1        = [ (self.MSE1[j]  + self.model.MSE1[j]) for j in range(self.model.N) ]
                self.mean        = (self.mean * i + xn)/(i+1)
                self.var         = self.var + (xn - self.new_mean)*(xn - self.mean)
                self.new_mean    = self.mean
                
        self.MSE  = [self.MSE[n] / self.Num for n in range(self.model.N)]
        self.MSE1 = [self.MSE1[n] / self.Num for n in range(self.model.N)]
        self.var  = np.sqrt(self.var/self.Num)
#         
    def Plot_MSE(self):
        plt.figure()
        plt.semilogy(range(self.model.N),self.MSE)
        plt.xlabel('Number of Rounds T')
        plt.ylabel('MSE')
        plt.xlim([1,self.model.N])
        plt.title('The Mean Square Error on Norm Estimation')
        plt.show()   

    def Plot_Regret(self ,Skip = [],T = False):
        if len(Skip) > 1 :
            print('too many outlier!!!')
        elif len(Skip) == 1:
            xn = self.regret[:,Skip[0]].reshape((self.model.N +1,1))
            self.new_mean = self.mean
            self.new_var = self.var
            self.new_mean = self.mean - xn/self.Num
            self.new_var  = self.var + (self.mean - self.new_mean)**2 + ((self.new_mean**2) -(xn - self.new_mean)**2)/self.Num
        else :
            self.new_mean = self.mean
            self.new_var  = self.var
        plt.figure()
        y1 = [float(self.new_mean[i] - 3 * np.sqrt(self.new_var[i]**2)) for i in range(self.model.N + 1)]
        y2 = [float(self.new_mean[i] + 3 * np.sqrt(self.new_var[i]**2)) for i in range(self.model.N + 1)]

        if T == False: 
            plt.plot(range(self.model.N+1), self.new_mean, label='R(T)')
            plt.fill_between(range(self.model.N + 1), np.array(y1), np.array(y2), color='gray', alpha=0.2)
            plt.title('The Regret in ' + self.model.name)
            plt.xlabel('Number of Rounds T')
            plt.xlim([1,self.model.N])
            plt.ylabel('R(T)')
        else:
            delta = [self.model.sigma**2 /(self.model.bandit.maxnorm - self.model.bandit.norm[arms]) for arms in range(self.model.bandit.K) if arms != self.model.bandit.maxk]
            lower_bound = np.sum(delta) * np.log(range(self.model.N+1))
            plt.semilogx(range(self.model.N+1),self.new_mean, label='R(T)')
            plt.semilogx(range(self.model.N+1),lower_bound,label='Lower bound')
            plt.fill_between(range(self.model.N + 1), np.array(y1), np.array(y2), color='gray', alpha=0.2)
            plt.title('The Regret in ' + self.model.name)
            plt.legend()
            plt.xlabel('Number of Rounds log(T)')
            plt.ylabel('R(T)')
            plt.xlim([1,self.model.N])
            plt.show()
        
                
