# Copyright@author: Siyuan Xu 2023211457
# xu-sy19@tsinghua.org.cn
import os
import sys
import numpy as np
import time
import argparse
import logging
import function_board as fb
import function_tool as ft
import function_get_aiming_grid
from function_solve_dp_withturn import solve_dp_withturn_valueiteration, setup_logging, log_system_info

#  python HW.py --playerID_list 7 11 12 --iter_error=0.00001 --iter_limit=5000
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)
parser = argparse.ArgumentParser(description='Run value iteration for given player IDs.')
parser.add_argument('--playerID_list', nargs='+', type=int, help='List of player IDs', required=True)
parser.add_argument('--iter_limit', type=int, help='Iteration limit for the value iteration algorithm', default=1000)
parser.add_argument('--iter_error', type=float, help='Error threshold for the value iteration algorithm', default=0.0001)

args = parser.parse_args()
iter_limit = args.iter_limit
iter_error = args.iter_error
playerID_list = args.playerID_list

result_dir = '../HW_result'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

for playerID in playerID_list:
    logger = setup_logging(result_dir, playerID)
    log_system_info(logger)
    name_pa = 'player{}'.format(playerID)
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
     prob_grid_bullscore] = function_get_aiming_grid.load_aiming_grid(name_pa)

    t1 = time.time()
    V, optimal_action_index = solve_dp_withturn_valueiteration(aiming_grid, prob_grid_normalscore,
                                                               prob_grid_doublescore,
                                                               prob_grid_bullscore,iter_limit=iter_limit, iter_error=iter_error,logger=logger)
    t2 = time.time()

    print(f'\nsolve dp_withturn in {t2 - t1} seconds for player{playerID}')
    np.save(os.path.join(result_dir, f'player{playerID}_V.npy'), V)
    np.save(os.path.join(result_dir, f'player{playerID}_optimal_action_index.npy'), optimal_action_index)
    logger.info(f'Saved V matrix and optimal action index for player {playerID}')
    logger.info(f'Total time for value iteration: {t2 - t1:.4f} seconds.')
    print('\n')
