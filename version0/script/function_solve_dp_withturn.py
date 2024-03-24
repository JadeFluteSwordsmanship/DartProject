import numpy as np
import function_board as fb
import copy
import time
import os
import logging
import platform


def setup_logging(result_dir, playerID):
    logger = logging.getLogger(f'ValueIterationLogger_{playerID}')
    logger.setLevel(logging.DEBUG)

    log_file = os.path.join(result_dir, f'player{playerID}_value_iteration.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def log_system_info(logger):
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")


def solve_dp_withturn_valueiteration(aiming_grid, prob_grid_normalscore, prob_grid_doublescore=None,
                                     prob_grid_bullscore=None,iter_limit=1000, iter_error=0.0001, logger=None):
    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore_0tosmax_dic = {}
    prob_normalscore_0tosmaxsum_dic = {}
    for score_max in range(0, 61):
        score_max_plus1 = score_max + 1
        prob_normalscore_0tosmax_dic[score_max] = np.array(prob_grid_normalscore[:, 0:score_max_plus1])
        prob_normalscore_0tosmaxsum_dic[score_max] = prob_normalscore_0tosmax_dic[score_max].sum(axis=1)

    prob_doublescore_dic = {}
    for doublescore_index in range(20):
        doublescore = 2 * (doublescore_index + 1)
        prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:, doublescore_index])
    prob_doublescore_dic[fb.score_DB] = np.array(prob_grid_bullscore[:, 1])

    # Initialize the value function for all states (s, i, u)
    V = np.full((502, 3, 121), fill_value=np.nan, dtype=np.float32)
    optimal_action_index = np.full((502, 3, 121), fill_value=-1, dtype=np.int16)
    V[0, :, 0] = 0  # If the score is 0, the game is finished
    V[1, :, 0] = np.inf
    optimal_action_index[0:2, :, 0] = -999

    for s in range(2, 502):
        start_time = time.time()
        values_old = {3: np.ones(min(s - 2, 60 * 0) + 1, dtype=np.float32),
                      2: np.ones(min(s - 2, 60 * 1) + 1, dtype=np.float32),
                      1: np.ones(min(s - 2, 60 * 2) + 1, dtype=np.float32)}
        values = {3: np.ones(min(s - 2, 60 * 0) + 1, dtype=np.float32),
                  2: np.ones(min(s - 2, 60 * 1) + 1, dtype=np.float32),
                  1: np.ones(min(s - 2, 60 * 2) + 1, dtype=np.float32)}
        # values_old = np.ones((4, min(s - 2, 60 * 2) + 1))
        # values = np.ones((4, min(s - 2, 60 * 2) + 1))

        iter_index = -1
        for iter_index in range(iter_limit):
            for u in [0]:
                score_max = min(s - 2, 60)
                prob_bust = 1 - prob_normalscore_0tosmaxsum_dic.get(score_max) - prob_doublescore_dic.get(s - u,
                                                                                                          np.zeros(
                                                                                                              num_aiming_location, ))  # (984,)
                turn_to_thrownew = (prob_normalscore_0tosmax_dic.get(score_max).dot(values[2][u:u + score_max + 1])
                                    + prob_doublescore_dic.get(s, np.zeros(num_aiming_location, )) * 1
                                    + prob_bust.dot(1 + values[3][0]))
                values[3][u] = np.min(turn_to_thrownew)
                optimal_action_index[s, 2, u] = np.argmin(turn_to_thrownew)
            for u in range(0, min(s - 2, 60) + 1):
                score_max = min(s - u - 2, 60)
                prob_bust = 1 - prob_normalscore_0tosmaxsum_dic.get(score_max) - prob_doublescore_dic.get(s - u,
                                                                                                          np.zeros(
                                                                                                              num_aiming_location, ))  # (984,)
                turn_to_thrownew = (
                        prob_normalscore_0tosmax_dic.get(score_max).dot(values[1][u:u + score_max + 1])
                        + prob_doublescore_dic.get(s - u, np.zeros(num_aiming_location, )) * 1
                        + prob_bust.dot(1 + values[3][0])
                )
                values[2][u] = np.min(turn_to_thrownew)
                optimal_action_index[s, 1, u] = np.argmin(turn_to_thrownew)
            for u in range(1, min(s - 2, 120) + 1):
                score_max = min(s - u - 2, 60)
                prob_bust = 1 - prob_normalscore_0tosmaxsum_dic.get(score_max) - prob_doublescore_dic.get(s - u,
                                                                                                          np.zeros(
                                                                                                              num_aiming_location, ))  # (984,)
                turn_to_thrownew = (
                        prob_normalscore_0tosmax_dic.get(score_max).dot(1 + V[s - u:s - u - score_max - 1:-1, 2, 0])
                        + prob_doublescore_dic.get(s - u, np.zeros(num_aiming_location, )) * 1
                        + prob_bust.dot(1 + values[3][0])
                )
                values[1][u] = np.min(turn_to_thrownew)
                optimal_action_index[s, 0, u] = np.argmin(turn_to_thrownew)

            for u in [0]:
                score_max = min(s - u - 2, 60)
                prob_bust = 1 - prob_normalscore_0tosmaxsum_dic.get(score_max) - prob_doublescore_dic.get(s - u,
                                                                                                          np.zeros(
                                                                                                              num_aiming_location, ))  # (984,)
                turn_to_thrownew = (
                        prob_grid_normalscore[:, 0].dot(1 + values[3][0])
                        + np.array(prob_grid_normalscore[:, 1:score_max + 1]).dot(
                    1 + V[s - u - 1:s - u - score_max - 1:-1, 2, 0])
                        + prob_doublescore_dic.get(s - u, np.zeros(num_aiming_location, )) * 1
                        + prob_bust.dot(1 + values[3][0])
                )
                values[1][u] = np.min(turn_to_thrownew)
                optimal_action_index[s, 0, u] = np.argmin(turn_to_thrownew)

            jump_out_of_iter = True
            for remaining_throws in [1, 2, 3]:
                if not np.all(np.abs(values_old[remaining_throws] - values[remaining_throws]) < iter_error):
                    jump_out_of_iter = False
                    break
            if not jump_out_of_iter:
                values_old = copy.deepcopy(values)
            else:
                break

        V[s, 2, 0] = values[3][0]
        V[s, 1, :len(values[2])] = values[2]
        V[s, 0, :len(values[1])] = values[1]
        end_time = time.time()
        if logger:
            logger.debug(
                f"Score state {s} completed in {end_time - start_time:.4f} seconds with {iter_index} iterations.")

        print(f"\rscore_state = {s} finished.", end="")

    return V, optimal_action_index
