import os
import sys
import time

import function_board as fb
import function_tool as ft
import function_get_aiming_grid

import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)

#import ipdb

#%%
def solve_dp_noturn_demo1(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore):

    ## aiming_grid
    num_aiming_location = aiming_grid.shape[0]
    tempvalue = np.zeros(num_aiming_location)
    
    #possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502)
    optimal_value[0] = 0
    #optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1    
        
    ## solve successively for s = 2,...,501
    for score_state in range(2,502):
        ## compute each aiming location
        for target_i in range(num_aiming_location):
        
            ## transit to less score state    
            ## s1 = min(score_state-2, 60)
            ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s1]*v[score_state-s1]
            score_max = min(score_state-2, 60)
            score_max_plus1 = score_max + 1 
            # 这里的逻辑是，期望次数=1+投到i不爆的概率*在状态i下期望次数+投到爆的概率*（爆了所以回归原来，还是期望次数）
            # Ex_throws = 1 + np.dot(prob*optimal_val) + prob_busted * Ex_throws
            # Ex_throws*(1-prob_busted) = 1 + np.dot(prob*optimal_val) = num_tothrow
            # ∴ prob_notbust = num_tothrow / Ex_throws
            # tempvalue[target_i] = num_tothrow / prob_notbust
            num_tothrow = 1.0 + np.dot(prob_grid_normalscore[target_i,1:score_max_plus1], optimal_value[score_state-1:score_state-score_max-1:-1])
            prob_notbust = prob_grid_normalscore[target_i,1:score_max_plus1].sum()
            
            ## transit to finishing
            if (score_state == fb.score_DB): ## hit double bull
                prob_notbust += prob_grid_bullscore[target_i,1]
            elif (score_state <= 40 and score_state%2==0): ## hit double
                doublescore_index = (score_state//2) - 1
                prob_notbust += prob_grid_doublescore[target_i,doublescore_index]
            else: ## not able to finish
                pass
            
            ## expected number of throw for aiming target_i
            prob_notbust = np.maximum(prob_notbust, 0)
            tempvalue[target_i] = num_tothrow / prob_notbust
                            
        ## searching
        optimal_value[score_state] = np.min(tempvalue)
        optimal_action_index[score_state] = np.argmin(tempvalue)

    return [optimal_value, optimal_action_index]


#%%
def solve_dp_noturn_demo2(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore):

    ## aiming_grid
    num_aiming_location = aiming_grid.shape[0]
    
    #possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502)
    optimal_value[0] = 0
    #optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1    
    
    ## solve successively for s = 2,...,501    
    for score_state in range(2,502):            
        ## use matrix operation to search all aiming locations
        
        ## transit to less score state    
        ## s1 = min(score_state-2, 60)
        ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s1]*v[score_state-s1]
        score_max = min(score_state-2, 60)
        score_max_plus1 = score_max + 1 
        
        num_tothrow = 1.0 + np.dot(prob_grid_normalscore[:,1:score_max_plus1], optimal_value[score_state-1:score_state-score_max-1:-1])
        prob_notbust = prob_grid_normalscore[:,1:score_max_plus1].sum(axis=1)
        
        ## transit to finishing
        if (score_state == fb.score_DB): ## hit double bull
            prob_notbust += prob_grid_bullscore[:,1]
        elif (score_state <= 40 and score_state%2==0): ## hit double
            doublescore_index = (score_state//2) - 1
            prob_notbust += prob_grid_doublescore[:,doublescore_index]
        else: ## not able to finish
            pass
        
        ## expected number of throw for all 341*341 aiming locations
        prob_notbust = np.maximum(prob_notbust, 0)
        tempvalue = num_tothrow / prob_notbust
                            
        ## searching
        optimal_value[score_state] = np.min(tempvalue)
        optimal_action_index[score_state] = np.argmin(tempvalue)

    return [optimal_value, optimal_action_index]


#%%
def solve_dp_noturn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore=None, prob_grid_bullscore=None, prob_grid_doublescore_dic=None):

    ## aiming_grid
    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore_1tosmax_dic = {}
    prob_normalscore_1tosmaxsum_dic = {}
    for score_max in range(0,61):
        score_max_plus1 = score_max + 1 
        prob_normalscore_1tosmax_dic[score_max] = np.array(prob_grid_normalscore[:,1:score_max_plus1])
        prob_normalscore_1tosmaxsum_dic[score_max] = prob_normalscore_1tosmax_dic[score_max].sum(axis=1)
    if prob_grid_doublescore_dic is None:
        prob_doublescore_dic = {}
        for doublescore_index in range(20):
            doublescore = 2*(doublescore_index+1)
            prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:,doublescore_index])
    else:
        prob_doublescore_dic = prob_grid_doublescore_dic
    prob_DB = np.array(prob_grid_bullscore[:,1])

    ##
    #possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502)
    #optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1
    
    ## solve successively for s = 2,...,501    
    for score_state in range(2,502):            
        ## use matrix operation to search all aiming locations
        
        ## transit to less score state    
        ## s1 = min(score_state-2, 60)
        ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s1]*v[score_state-s1]
        score_max = min(score_state-2, 60)
        score_max_plus1 = score_max + 1 
        ## transit to next state
        num_tothrow = 1.0 + prob_normalscore_1tosmax_dic[score_max].dot(optimal_value[score_state-1:score_state-score_max-1:-1])
        ## probability of transition to state other than s itself
        prob_otherstate = prob_normalscore_1tosmaxsum_dic[score_max]
        
        ## transit to the end of game
        if (score_state == fb.score_DB): ## hit double bull
            prob_otherstate += prob_DB
        elif (score_state <= 40 and score_state%2==0): ## hit double
            prob_otherstate += prob_doublescore_dic[score_state]
        else: ## game does not end
            pass
        
        ## expected number of throw for all aiming locations
        prob_otherstate = np.maximum(prob_otherstate, 0)
        num_tothrow = num_tothrow / prob_otherstate
                            
        ## searching
        optimal_value[score_state] = num_tothrow.min()
        optimal_action_index[score_state] = num_tothrow.argmin()

    return [optimal_value, optimal_action_index]



#%%
def solve_dp_noturn_valueiteration(aiming_grid, prob_grid_normalscore, prob_grid_doublescore=None, prob_grid_bullscore=None, prob_grid_doublescore_dic=None):

    ## aiming_grid
    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore_1tosmax_dic = {}
    prob_normalscore_1tosmaxsum_dic = {}
    for score_max in range(0,61):
        score_max_plus1 = score_max + 1 
        prob_normalscore_1tosmax_dic[score_max] = np.array(prob_grid_normalscore[:,1:score_max_plus1])
        prob_normalscore_1tosmaxsum_dic[score_max] = prob_normalscore_1tosmax_dic[score_max].sum(axis=1)
    if prob_grid_doublescore_dic is None:
        prob_doublescore_dic = {}
        for doublescore_index in range(20):
            doublescore = 2*(doublescore_index+1)
            prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:,doublescore_index])
    else:
        prob_doublescore_dic = prob_grid_doublescore_dic
    prob_DB = np.array(prob_grid_bullscore[:,1])

    ##
    #possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502)
    #optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1
    
    iter_limit = 1000
    iter_error = 0.0001    
    
    ## solve successively for s = 2,...,501    
    for score_state in range(2,502):            
        ## use matrix operation to search all aiming locations        
        
        ## transit to less score state    
        ## s1 = min(score_state-2, 60)
        ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s1]*v[score_state-s1]
        score_max = min(score_state-2, 60)
        score_max_plus1 = score_max + 1 
        ## transit to next state
        num_tothrow = 1.0 + prob_normalscore_1tosmax_dic[score_max].dot(optimal_value[score_state-1:score_state-score_max-1:-1])

        ## probability of transition to s itself
        prob_bust = 1 - prob_normalscore_1tosmaxsum_dic[score_max]
        
        ## transit to the end of game
        if (score_state == fb.score_DB): ## hit double bull
            prob_bust -= prob_DB
        elif (score_state <= 40 and score_state%2==0): ## hit double
            prob_bust -= prob_doublescore_dic[score_state]
        else: ## game does not end
            pass
        
        value = 0.0
        for iter_index in range(iter_limit):
            num_tothrow_new = num_tothrow + prob_bust*value            
            value_new = np.min(num_tothrow_new)
            if np.abs(value_new - value) < iter_error:
                break
            value = value_new

        ## searching
        optimal_value[score_state] = value_new
        optimal_action_index[score_state] = num_tothrow_new.argmin()

    return [optimal_value, optimal_action_index]