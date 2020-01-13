# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:45:52 2020

@author: Dawen Wu
"""

import numpy as np
from matplotlib import pyplot as plt

def softmax(x):
     e_x = np.exp(x)
     return e_x / e_x.sum()
 
    
def u(player, s_0, s_1):
    return payoff[player, s_0, s_1]

def regret_matching(distribution, payoff, log_intervals, original_RM = None):
    """
    The algorithm introduced in the paper
    Input:
        distribution: the initial probability distribution
        payoff: the payoff matrix
    Output:
        record: The empirical distribution.
    """
    record = []
    alpha_list = []
    L = [False]*10
    
    regret_matrix_0 = np.zeros((3,3))
    regret_matrix_1 = np.zeros((4,4))
    regret_matrix = np.array([regret_matrix_0, regret_matrix_1])
    
    for t in range(1,T+1):
        
        s_0, s_1 = np.random.choice(3, p=distribution[0]), np.random.choice(4, p=distribution[1])
        s = np.array([s_0, s_1])
        record.append(s)
        
        if np.sum(L[-10:]) == 10: using_negative_regret = True
        else : using_negative_regret = False
        for player in player_set:
            u_i = u(player, s_0, s_1)
            
            if original_RM == True:
                using_negative_regret = False
            distribution[player] = updating_deriving(player, u_i, s, t, regret_matrix, using_negative_regret)
            
        L.append(eq3dot1_check(regret_matrix, distribution, t))
        if t % log_intervals == 0:
            alpha_list.append(correlated_equilibrium_checker(record))

    return record, alpha_list, L


def updating_deriving(player, u, s, t, regret_matrix, using_negative_regret):
    j = s[player]
    j_opponent = s[1] if player==0 else s[0]
    distribution = np.zeros_like(action_sets[player])
    #regret-matrix updating
    if player == 0:
        regret_matrix[player][s[player],:] += payoff[player, :, j_opponent] - u
    else :
        regret_matrix[player][s[player],:] += payoff[player, j_opponent, :] - u
    #distribution deriving 
    if using_negative_regret: 
        r = regret_matrix[player][s[player],:]
        distribution = softmax(r)
        return distribution
    else: 
        r = np.clip(regret_matrix[player][s[player],:], 0, None)
        distribution = (1/mu)*(1/t)*r
        distribution[j] = 1 - np.sum(distribution)
        return distribution


def eq3dot1_check(regret_matrix, distribution, t):
    for player in range(2):
        R = np.clip(regret_matrix[player],0,None)/t
        q = distribution[player]
        for j in action_sets[player]:
            if (np.dot(q, R[:,j]) == q[j]*(R[j,:].sum())) ==False:
                a = np.dot(q, R[:,j])
                b = q[j]*(R[j,:].sum())
                return np.square(a - b)
    return True


def record2distribution(record):
    """
    Transform the record to the record_distribution
    """
    unique, counts = np.unique(record, return_counts = True, axis = 0)
    counts = counts/counts.sum()
    record_distribution = dict()
    for i in range(len(unique)):
        record_distribution[tuple(unique[i])] = counts[i]
    for strategy in strategy_set:
        if strategy not in record_distribution.keys():record_distribution[strategy] = 0
    
    return record_distribution


def correlated_equilibrium_checker(record):
    """
    An easy implimentation of equation (1) to check whether a distribution is correlated_equilibrium.
    Input:
        record_distribution: A probability distribution over the startegy set S.
    Output:
        alpha: A float number means that the empiriacal distribution is a correlated alpha-equilibrium.
    """
    record_distribution = record2distribution(record)
    
    lis = []
    for player in player_set:
        for j in action_sets[player]:
            for k in action_sets[player]:
                s = 0
                for strategy in strategy_set:
                    if strategy[player] == j:
                        if player == 0: s_0, s_1= k, strategy[1]
                        else : s_0, s_1= strategy[0], k
                        s += record_distribution[strategy]*(u(player, s_0, s_1) - u(player,*strategy ))
                lis.append(s)
    return np.max(lis)


if __name__ == '__main__':
    # Parameter setting
    T = 1000      #iterations number
    log_intervals = 20
    player_set = [0,1]    
    mu = 2*1000*(4-1)*1.3      # The most important hyperparam similar to the learning rate
    strategy_set = [(i,j) for i in range(3) for j in range(4)]      # The total strategy set S
    action_sets = [[0,1,2],[0,1,2,3]]       # The action set of each player
    using_negative_regret = True 
    
    player_0_payoff = np.random.uniform(-1000,1000,(3,4))     # Generate the payoff-matrix using 
    player_1_payoff = np.random.uniform(-1000,1000,(3,4))     # the uniform distribution ranging from -1000 to 1000
    payoff = np.array([player_0_payoff, player_1_payoff])
    
    initial_distribution_0 = np.array([0.3,0.3,0.4])     #p_0^0
    initial_distribution_1 = np.array([0.3,0.3,0.3,0.1])   #p_0^1
    distribution = np.array([initial_distribution_0,initial_distribution_1])
    
    #if using_negative_regret:
    record_fast, alpha_list_fast, L_fast = regret_matching(distribution.copy(), payoff, log_intervals, original_RM = False)
    #else:
    record, alpha_list, L = regret_matching(distribution.copy(), payoff, log_intervals, original_RM = True)
    
    
    plt.plot(np.arange(0, T ,log_intervals), alpha_list, label = 'Original')
    plt.plot(np.arange(0, T ,log_intervals), alpha_list_fast, label = 'Faster version')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('alpha')
    