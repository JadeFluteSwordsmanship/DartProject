import numpy as np
import function_board as fb
import copy


def solve_dp_withturn_valueiteration(aiming_grid, prob_grid_normalscore, prob_grid_doublescore=None,
                                     prob_grid_bullscore=None):
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
    V = np.full((502, 3, 121), fill_value=np.nan)
    optimal_action_index = np.full((502, 3, 121), fill_value=-1,dtype=np.int16)
    V[0, :, 0] = 0  # If the score is 0, the game is finished
    V[1, :, 0] = np.inf
    optimal_action_index[0:2, :, 0] = -999

    iter_limit = 1000
    iter_error = 0.0001
    for s in range(2, 502):
        values_old = {3: np.ones(min(s - 2, 60 * 0) + 1, dtype=np.float64),
                      2: np.ones(min(s - 2, 60 * 1) + 1, dtype=np.float64),
                      1: np.ones(min(s - 2, 60 * 2) + 1, dtype=np.float64)}
        values = {3: np.ones(min(s - 2, 60 * 0) + 1, dtype=np.float64),
                  2: np.ones(min(s - 2, 60 * 1) + 1, dtype=np.float64),
                  1: np.ones(min(s - 2, 60 * 2) + 1, dtype=np.float64)}

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
        print(f"\rscore_state = {s} finished.",end="")

    return V,optimal_action_index
