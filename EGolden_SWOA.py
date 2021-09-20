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
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 b=1, a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub
        self.lb = lb
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        tao = (np.sqrt(5)-1)/2
        x1 = -np.pi+(1-tao)
        x2 = -np.pi+tao*2*np.pi
        
        # 迭代
        for g in range(self.G):
            # OBL
            self.X, F = self.OBL()
            
            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 更新
            a = self.a_max - (self.a_max-self.a_min)*(g/self.G)
            
            for i in range(self.P):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                A = 2*a*r1 - a
                C = 2*r2
                
                if np.abs(A)>=1:
                    X_rand = self.X[np.random.randint(low=0, high=self.P, size=self.D), :]
                    X_rand = np.diag(X_rand).copy()
                    D = np.abs(C*X_rand - self.X[i, :])
                    self.X[i, :] = X_rand - A*D # (4)
                else:
                    if p<0.5:
                        D = np.abs(C*self.gbest_X - self.X[i, :])
                        self.X[i, :] = self.gbest_X - A*D # (1)
                    else:
                        r3 = 2*np.pi*np.random.uniform()
                        r4 = np.pi*np.random.uniform()
                        self.X[i, :] = self.X[i, :]*np.abs(np.sin(r3)) + \
                                       r4*np.sin(r3)*np.abs(x1*self.gbest_X-x2*self.X[i, :]) # (9)
            
            # 邊界處理
            self.X = np.clip(self.X, self.lb, self.ub)
                   
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.loss_curve[-1], 3))+']')
        plt.plot(self.loss_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()

    def OBL(self):
        # 產生反向解
        k = np.random.uniform()
        alpha = self.X.min(axis=0)
        beta = self.X.max(axis=0)
        obl_X = k*(alpha+beta) - self.X # (5)
        
        # 對反向解進行邊界處理
        rand_X = np.random.uniform(low=alpha, high=beta, size=[self.P, self.D]) # (6)
        mask = np.logical_or(obl_X>self.ub, obl_X<self.lb)
        obl_X[mask] = rand_X[mask].copy()
        
        # 取得新解
        concat_X = np.vstack([obl_X, self.X])
        F = self.fitness(concat_X)
        top_idx = F.argsort()[:self.P]
        top_F = F[top_idx].copy()
        top_X = concat_X[top_idx].copy()
        
        return top_X, top_F

