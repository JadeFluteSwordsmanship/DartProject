import os
import sys
import numpy as np
import time

import function_board as fb
import function_tool as ft
import function_get_aiming_grid
#import function_solve_dp
import function_solve_dp_demo as function_solve_dp

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)

#%%
result_dir = '../result'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

playerID_list = [7]

for playerID in playerID_list:
    name_pa = 'player{}'.format(playerID)
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = function_get_aiming_grid.load_aiming_grid(name_pa)
    
    
    #%%
    t1 = time.time()
    [optimal_value1, optimal_action_index1] = function_solve_dp.solve_dp_noturn_demo1(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    t2 = time.time()
    print('solve dp_noturn in {} seconds'.format(t2-t1))
    print('optimal_value1: {}'.format(optimal_value1))
    print('optimal_action_index1: {}'.format(optimal_action_index1))
    print('\n')
    
    ## show the optimal target for each score state
    for score_state in range(2,3):
        text = 'score_state = {} \n'.format(score_state) 
        text += 'expected number of thorws to zero score = {:.2f}\n'.format(optimal_value1[score_state])
        opt_target = aiming_grid[optimal_action_index1[score_state]]
        text += 'optimal target index = {}, coordinator = {}'.format(opt_target, opt_target-fb.R)
        fb.plot_dartboard(points_input=opt_target, point_marker='x', text_input=text, flag_index=True)

    #%%
    t1 = time.time()
    [optimal_value2, optimal_action_index2] = function_solve_dp.solve_dp_noturn_demo2(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    t2 = time.time()
    print('solve dp_noturn in {} seconds'.format(t2-t1))
    print('optimal_value2: {}'.format(optimal_value2))
    print('optimal_action_index2: {}'.format(optimal_action_index2))
    print('\n')

    
    t1 = time.time()
    [optimal_value, optimal_action_index] = function_solve_dp.solve_dp_noturn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    t2 = time.time()
    print('solve dp_noturn in {} seconds'.format(t2-t1))
    print('optimal_value: {}'.format(optimal_value))
    print('optimal_action_index: {}'.format(optimal_action_index))
    print('\n')
        
    
    result_filename = result_dir + '/singlegame_{}_noturn.pkl'.format(name_pa)
    result_dic_fullactionset = {'name':name_pa, 'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    ft.dump_pickle(result_filename, result_dic_fullactionset, printflag=True)        

    t1 = time.time()
    [optimal_value3, optimal_action_index3] = function_solve_dp.solve_dp_noturn_valueiteration(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    t2 = time.time()
    print('solve dp_noturn in {} seconds'.format(t2-t1))
    print('optimal_value: {}'.format(optimal_value3))
    print('optimal_action_index: {}'.format(optimal_action_index3))
    print('\n')
