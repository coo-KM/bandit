import math
import numpy as np
from numpy.random import binomial
from numpy.random import beta
import matplotlib.pyplot as plt
import pandas as pd
import csv

class Arm:
    def __init__(self, p):
        self.cnt = 0
        self.clk = 0
        self._p = p

    def click(self):
        self.cnt += 1
        if binomial(n=1,p=self._p):
            self.clk += 1


def Reglet_ts(arms,T):
    Rgt = []
    for i in range(1, T + 1):
        rand_beta = [beta(a=arm.clk+1,b=arm.cnt-arm.clk+1) for arm in arms]
        max_i = rand_beta.index(max(rand_beta))
        arms[max_i].click()

        max_mu = 0
        for i in range(len(arms)):
            max_mu = max(max_mu, arms[i].clk / (arms[i].cnt+1))
        rgt = 0
        for i in range(len(arms)):
            rgt += (max_mu - (arms[i].clk / (arms[i].cnt+1)))*arms[i].cnt
        Rgt.append(rgt)

    return Rgt

if __name__=="__main__":
    reglet = []
    time = np.arange(1,10 ** 4+1)
    for i in range(100):
        arms=[(Arm(0.1))]
        for j in range(3):
            arms.append(Arm(0.05))
            arms.append(Arm(0.02))
            arms.append(Arm(0.01))

        reglet.append(Reglet_ts(arms=arms, T=10**4))
    reglet_ts_average = np.zeros(10**4)
    reglet_ts_average = np.average(reglet,axis=0)

    fig, ax = plt.subplots()

    ax.set_xscale("log")
    ax.plot(time, reglet_ts_average,label='ts_sampling')
    ax.set_xlabel('time',fontsize=16)
    ax.set_ylabel('reglet',fontsize=16)
    plt.show()