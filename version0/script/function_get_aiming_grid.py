import os
import sys
import math
import numpy as np
import time

import function_board as fb
import function_tool as ft


np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)


#%%

R = fb.R
grid_num = fb.grid_num

## 2-dimension probability grid
def load_aiming_grid(playername_filename, data_parameter_dir=fb.data_parameter_dir, grid_version=fb.grid_version, count_bull=True):
    if playername_filename.startswith('player'):
        filename = data_parameter_dir+'/grid_{}/{}_gaussin_prob_grid_{}.pkl'.format(grid_version, playername_filename, grid_version)
    else:    
        filename = playername_filename    

    result_dic = ft.load_pickle(filename, printflag=True)
    aiming_grid = result_dic['aiming_grid']
    prob_grid_normalscore = result_dic['prob_grid_normalscore'] 
    prob_grid_singlescore = result_dic['prob_grid_singlescore']
    prob_grid_doublescore = result_dic['prob_grid_doublescore']
    prob_grid_triplescore = result_dic['prob_grid_triplescore']
    prob_grid_bullscore = result_dic['prob_grid_bullscore']
    
    ## default seetig count bull score
    if count_bull:
        #print('bull score in counted prob_grid_normalscore')
        prob_grid_normalscore[:,fb.score_SB] += prob_grid_bullscore[:,0]
        prob_grid_normalscore[:,fb.score_DB] += prob_grid_bullscore[:,1]
    else:
        print('bull score in NOT counted in prob_grid_normalscore')
        
    return [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore]

## 3-dimension probability grid
def load_prob_grid(playername_filename, data_parameter_dir=fb.data_parameter_dir, grid_version='full'):
    if playername_filename.startswith('player'):
        filename = data_parameter_dir+'/grid_{}/{}_gaussin_prob_grid.pkl'.format(grid_version, playername_filename)
    else:    
        filename = playername_filename        
    
    prob_grid_dict = ft.load_pickle(filename, printflag=True)    
    prob_grid_singlescore = prob_grid_dict['prob_grid_singlescore']
    prob_grid_doublescore = prob_grid_dict['prob_grid_doublescore']
    prob_grid_triplescore = prob_grid_dict['prob_grid_triplescore']
    prob_grid_bullscore = prob_grid_dict['prob_grid_bullscore']
    
    ## normalscore vector is of siz3 61 and does not include bull score information !!
    prob_grid_normalscore = np.zeros((grid_num, grid_num, 61))
    for temp_s in range(1,61):
        if temp_s <= 20:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_singlescore[:,:,temp_s-1]
        if temp_s%2 == 0 and temp_s <= 40:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_normalscore[:,:,temp_s] + prob_grid_doublescore[:,:,temp_s//2-1]
        if temp_s%3 == 0:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_normalscore[:,:,temp_s] + prob_grid_triplescore[:,:,temp_s//3-1]
    ## prob of hitting zero
    prob_grid_normalscore[:,:,0] =  np.maximum(0, 1-prob_grid_normalscore[:,:,1:].sum(axis=2)-prob_grid_bullscore.sum(axis=2))
    
    return [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore]

def get_aiming_grid_v2(playername_filename, data_parameter_dir=fb.data_parameter_dir, grid_version='full'):
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = load_prob_grid(playername_filename, data_parameter_dir=data_parameter_dir, grid_version=grid_version)
    
    temp_num = 2000
    aiming_grid = np.zeros((temp_num,2), dtype=np.int32)
        
    ###################################################
    ## [-170,170] to [0,340]

    ## center of double bulleye is the first one 
    temp_index = 0
    aiming_grid[temp_index,0] = R
    aiming_grid[temp_index,1] = R
    
    ## square around the single bull eye. 4mm*4mm grid
    for temp_x in range(-16,16+1,4):
        for temp_y in range(-16,16+1,4):
            if (temp_x==0 and temp_y==0):
                continue
            else:
                temp_index += 1
                aiming_grid[temp_index] = [temp_x+R, temp_y+R]

    ## inside single area and outside single area
    #rgrid = [37,58,79, 120,135,150]
    rgrid = [58,135]
    theta_num = 60
    for theta in range(theta_num):
        theta = np.pi*(2.0*theta/theta_num)
        for temp_r in rgrid:
            temp_index += 1 
            temp_x = int(np.round(np.cos(theta)*temp_r))
            temp_y = int(np.round(np.sin(theta)*temp_r))
            aiming_grid[temp_index] = [temp_x+R, temp_y+R]
    
    ## double and triple area
    rgrid = [100, 103, 106,  163, 166, 169]
    theta_num = 120
    for theta in range(theta_num):
        theta = np.pi*(2.0*theta/theta_num)
        for temp_r in rgrid:
            temp_index += 1 
            temp_x = int(np.round(np.cos(theta)*temp_r))
            temp_y = int(np.round(np.sin(theta)*temp_r))
            aiming_grid[temp_index] = [temp_x+R, temp_y+R]
    ###################################################
            
    ## points with the largest hitting probability for each score region
    ## single area
    for temp_s in range(fb.singlescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_singlescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    ## double area
    for temp_s in range(fb.doublescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_doublescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]    
    ## triple area
    for temp_s in range(fb.triplescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_triplescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]    
    ## bull    
    for temp_s in range(fb.bullscorelist_len):
        temp_index += 1
        temp_s_argmax = np.argmax(prob_grid_bullscore[:,:,temp_s])
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    
    ## the point with the largest expected score
    temp_index += 1
    e_score = prob_grid_normalscore.dot(np.arange(61)) + prob_grid_bullscore.dot(np.array([fb.score_SB, fb.score_DB]))
    max_e_score = np.max(e_score)
    temp_argmax = np.argmax(e_score)
    temp_x = temp_argmax/grid_num
    temp_y = temp_argmax%grid_num
    aiming_grid[temp_index] = [temp_x, temp_y]        
    print('max_e_score={}, max_e_score_index={}'.format(max_e_score, aiming_grid[temp_index]))

    ##[0, 340]
    aiming_grid_num = temp_index + 1
    aiming_grid = aiming_grid[:aiming_grid_num,:]
    aiming_grid = np.maximum(aiming_grid, 0)
    aiming_grid = np.minimum(aiming_grid, grid_num-1)

    ## return probability
    prob_grid_normalscore_new = np.zeros((aiming_grid_num, 61))
    prob_grid_singlescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_doublescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_triplescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_bullscore_new = np.zeros((aiming_grid_num, 2))    
    for temp_index in range(aiming_grid_num):
        prob_grid_normalscore_new[temp_index,:] = prob_grid_normalscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_singlescore_new[temp_index,:] = prob_grid_singlescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_doublescore_new[temp_index,:] = prob_grid_doublescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_triplescore_new[temp_index,:] = prob_grid_triplescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_bullscore_new[temp_index,:] = prob_grid_bullscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
    
    return [aiming_grid, prob_grid_normalscore_new, prob_grid_singlescore_new, prob_grid_doublescore_new, prob_grid_triplescore_new, prob_grid_bullscore_new]


def get_aiming_grid_v3(playername_filename, data_parameter_dir=fb.data_parameter_dir):
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = load_prob_grid(playername_filename, data_parameter_dir=data_parameter_dir)
    
    temp_num = 20000
    aiming_grid = np.zeros((temp_num,2), dtype=np.int32)
        
    ###################################################
    ## [-170,170] to [0,340]

    ## center of double bulleye is the first one 
    temp_index = 0
    aiming_grid[temp_index,0] = R
    aiming_grid[temp_index,1] = R
    
    ## square around the single bull eye. 4mm*4mm grid
    step_size = 2
    for temp_x in range(-20,20+1,step_size):
        for temp_y in range(-20,20+1,step_size):
            if (temp_x==0 and temp_y==0):
                continue
            else:
                temp_index += 1
                aiming_grid[temp_index] = [temp_x+R, temp_y+R]

    ## inside single area and outside single area
    #rgrid = [37,58,79, 120,135,150]
    rgrid = [58,135]
    theta_num = 180
    for theta in range(theta_num):
        theta = np.pi*(2.0*theta/theta_num)
        for temp_r in rgrid:
            temp_index += 1 
            temp_x = int(np.round(np.cos(theta)*temp_r))
            temp_y = int(np.round(np.sin(theta)*temp_r))
            aiming_grid[temp_index] = [temp_x+R, temp_y+R]
    
    ## double and triple area
    rgrid_triple = [94, 96, 98, 99] + [100, 103, 106] + [107, 108, 110, 112]
    rgrid_double = [157, 159, 161, 162] + [163, 166, 169] + [170]
    rgrid = rgrid_double + rgrid_triple        
    theta_num = 180
    for theta in range(theta_num):
        theta = np.pi*(2.0*theta/theta_num)
        for temp_r in rgrid:
            temp_index += 1 
            temp_x = int(np.round(np.cos(theta)*temp_r))
            temp_y = int(np.round(np.sin(theta)*temp_r))
            aiming_grid[temp_index] = [temp_x+R, temp_y+R]
    ###################################################
            
    ## points with the largest hitting probability for each score region
    ## single area
    for temp_s in range(fb.singlescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_singlescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    ## double area
    for temp_s in range(fb.doublescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_doublescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]    
    ## triple area
    for temp_s in range(fb.triplescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_triplescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]    
    ## bull    
    for temp_s in range(fb.bullscorelist_len):
        temp_index += 1
        temp_s_argmax = np.argmax(prob_grid_bullscore[:,:,temp_s])
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    
    ## the point with the largest expected score
    temp_index += 1
    e_score = prob_grid_normalscore.dot(np.arange(61)) + prob_grid_bullscore.dot(np.array([fb.score_SB, fb.score_DB]))
    max_e_score = np.max(e_score)
    temp_argmax = np.argmax(e_score)
    temp_x = temp_argmax/grid_num
    temp_y = temp_argmax%grid_num
    aiming_grid[temp_index] = [temp_x, temp_y]        
    print('max_e_score={}, max_e_score_index={}'.format(max_e_score, aiming_grid[temp_index]))

    ##[0, 340]
    aiming_grid_num = temp_index + 1
    aiming_grid = aiming_grid[:aiming_grid_num,:]
    aiming_grid = np.maximum(aiming_grid, 0)
    aiming_grid = np.minimum(aiming_grid, grid_num-1)

    ## return probability
    prob_grid_normalscore_new = np.zeros((aiming_grid_num, 61))
    prob_grid_singlescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_doublescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_triplescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_bullscore_new = np.zeros((aiming_grid_num, 2))    
    for temp_index in range(aiming_grid_num):
        prob_grid_normalscore_new[temp_index,:] = prob_grid_normalscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_singlescore_new[temp_index,:] = prob_grid_singlescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_doublescore_new[temp_index,:] = prob_grid_doublescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_triplescore_new[temp_index,:] = prob_grid_triplescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_bullscore_new[temp_index,:] = prob_grid_bullscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
    
    return [aiming_grid, prob_grid_normalscore_new, prob_grid_singlescore_new, prob_grid_doublescore_new, prob_grid_triplescore_new, prob_grid_bullscore_new]

