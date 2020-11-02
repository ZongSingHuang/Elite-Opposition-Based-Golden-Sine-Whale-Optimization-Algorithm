# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:45:56 2020

@author: ZongSing_NB

Main reference:
http://www.ejournal.org.cn/EN/10.3969/j.issn.0372-2112.2019.10.020
"""

import numpy as np
import matplotlib.pyplot as plt

class EGolden_SWOA():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500, 
                 b=1, x_max=1, x_min=0, a_max=2, a_min=0, l_max=1, l_min=-1, a2_max=-1, a2_min=-2):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter
        self.x_max = x_max
        self.x_min = x_min
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        
        self._iter = 1
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        self.X = np.random.uniform(low=self.x_min, high=self.x_max, size=[self.num_particle, self.num_dim])
     
        score = self.fit_func(self.X)
        self.gBest_score = score.min().copy()
        self.gBest_X = self.X[score.argmin()].copy()
        self.gBest_curve[0] = self.gBest_score.copy()
        
    def opt(self):
        tao = (np.sqrt(5)-1)/2
        x1 = -np.pi+(1-tao)
        x2 = -np.pi+tao*2*np.pi
        
        while(self._iter<self.max_iter):
            a = self.a_max - (self.a_max-self.a_min)*(self._iter/self.max_iter)
            self.obl()
            
            for i in range(self.num_particle):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                A = 2*a*r1 - a
                C = 2*r2
                
                if np.abs(A)>=1:
                    X_rand = self.X[np.random.randint(low=0, high=self.num_particle, size=self.num_dim), :]
                    X_rand = np.diag(X_rand).copy()
                    D = np.abs(C*X_rand - self.X[i, :])
                    self.X[i, :] = X_rand - A*D # (4)
                else:
                    if p<0.5:
                        D = np.abs(C*self.gBest_X - self.X[i, :])
                        self.X[i, :] = self.gBest_X - A*D # (1)
                    else:
                        R1 = 2*np.pi*np.random.uniform()
                        R2 = np.pi*np.random.uniform()
                        self.X[i, :] = self.X[i, :]*np.abs(np.sin(R1)) + \
                                       R2*np.sin(R1)*np.abs(x1*self.gBest_X-x2*self.X[i, :]) # (9)
            
            self.X = np.clip(self.X, self.x_min, self.x_max)
                   
            score = self.fit_func(self.X)
            if np.min(score) < self.gBest_score:
                self.gBest_X = self.X[score.argmin()].copy()
                self.gBest_score = score.min().copy()
                
            self.gBest_curve[self._iter] = self.gBest_score.copy()
            self._iter = self._iter + 1
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()

    def obl(self):
        k = np.random.uniform()
        alpha = self.X.min(axis=0)
        beta = self.X.max(axis=0)
        new_X = k*(alpha+beta)-self.X

        idx_too_low = new_X < self.x_min
        idx_too_high = new_X > self.x_max
        
        rand_X = np.random.uniform(low=alpha, high=beta, size=[self.num_particle, self.num_dim])
        new_X[idx_too_high] = rand_X[idx_too_high].copy()
        new_X[idx_too_low] = rand_X[idx_too_low].copy()
        
        self.X = np.concatenate((new_X, self.X), axis=0)
        score = self.fit_func(self.X)
        top_k = score.argsort()[:self.num_particle]
        score = score[top_k].copy()
        self.X = self.X[top_k].copy()

