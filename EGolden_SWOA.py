# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:45:56 2020

@author: ZongSing_NB

Main reference:http://www.ejournal.org.cn/EN/abstract/abstract11643.shtml#
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
        self.bound_max = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_max[np.newaxis, :])
        self.bound_min = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_min[np.newaxis, :])
        
        self._iter = 1
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        self.X = np.random.uniform(size=[self.num_particle, self.num_dim])*(self.x_max-self.x_min) + self.x_min

        new_X = self.obl()
        self.X = np.concatenate((new_X, self.X), axis=0)        
        score = self.fit_func(self.X)
        top_k = score.argsort()[:self.num_particle]
        score = score[top_k].copy()
        self.X = self.X[top_k].copy()
        self.gBest_score = score.min().copy()
        self.gBest_X = self.X[score.argmin()].copy()
        self.gBest_curve[0] = self.gBest_score.copy()
        
    def opt(self):
        while(self._iter<self.max_iter):
            tao = (np.sqrt(5)-1)/2
            x1 = -np.pi+(1-tao)
            x2 = -np.pi+tao*2*np.pi
            a = self.a_max - (self.a_max-self.a_min)*(self._iter/self.max_iter)
            np.sqrt
            for i in range(self.num_particle):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                R1 = 2*np.pi*np.random.uniform()
                R2 = np.pi*np.random.uniform()
                A = 2*a*r1 - a
                C = 2*r2
                l = np.random.uniform()*(self.l_max-self.l_min) + self.l_min
                
                if p>=0.5:
                    self.X[i, :] = self.X[i, :]*np.abs(np.sin(R1)) + \
                                   R2*np.sin(R1)*np.abs(x1*self.gBest_X-x2*self.X[i, :])                   
                else:
                    D = np.abs(C*self.gBest_X - self.X[i, :])
                    self.X[i, :] = self.gBest_X - A*D
            
            self.X[self.bound_max < self.X] = self.bound_max[self.bound_max < self.X]
            self.X[self.bound_min > self.X] = self.bound_min[self.bound_min > self.X]
            
            new_X = self.obl()
            self.X = np.concatenate((new_X, self.X), axis=0)        
            score = self.fit_func(self.X)
            top_k = score.argsort()[:self.num_particle]
            score = score[top_k].copy()
            self.X = self.X[top_k].copy()
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

        idx_too_high = self.bound_max < new_X
        idx_too_low = self.bound_min > new_X
        
        rand_X = np.random.uniform(size=[self.num_particle, self.num_dim])*(self.x_max-self.x_min) + self.x_min
        new_X[idx_too_high] = rand_X[idx_too_high].copy()
        new_X[idx_too_low] = rand_X[idx_too_low].copy()
        
        return new_X
            