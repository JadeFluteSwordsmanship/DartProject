import os
import sys
import numpy as np
import time

import function_board as fb
import function_tool as ft
import function_get_aiming_grid
#import function_solve_dp

import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)


#%% Understand the geometric layout of the dart board
fb.plot_dartboard(flag_index=False)
#fb.plot_dartboard(flag_index=True)

#%% Understand the skill model. 
## Assume that the landing locations of throws follow a bivariate Gaussian distribution
## with mean \mu (the intended target) and covariance matrix \Sigma.

print('Anderson (player1) skill model for the Doubles Region')
Sigma1 = np.array([[106.56, -5.25], [-5.25, 49.18]])
print(Sigma1)

print('Aspinall (player2) skill model for the Doubles Region')
Sigma2 = np.array([[51.41, -7.18], [-7.18, 102.40]])
print(Sigma2)

print('Clayton (player4) skill model for the Doubles Region')
Sigma4 = np.array([[108.08, -13.92], [-13.92, 105.53]])
print(Sigma4)

print('Van Gerwen (player7) skill model for the Doubles Region')
Sigma7 = np.array([[49.60, -0.46], [-0.46,  69.17]])
print(Sigma7)

#%%
## target is the center of D20
target = [0, 166]
plt.plot(0, 166, '+', color='red', markersize=12)

num_points = 100

text = 'Clayton (player4)'
points = np.random.multivariate_normal(target, Sigma4, num_points)
fb.plot_dartboard(points_input=points, text_input=text)
plt.plot(0, 166, '+', color='red', markersize=12)

text = 'Van Gerwen (player7)'
points = np.random.multivariate_normal(target, Sigma7, num_points)
fb.plot_dartboard(points_input=points, text_input=text)
plt.plot(0, 166, '+', color='red', markersize=12)

text = 'Anderson (player1)'
points = np.random.multivariate_normal(target, Sigma1, num_points)
fb.plot_dartboard(points_input=points, text_input=text)
plt.plot(0, 166, '+', color='red', markersize=12)

text = 'Aspinall (player2)'
points = np.random.multivariate_normal(target, Sigma2, num_points)
fb.plot_dartboard(points_input=points, text_input=text)
plt.plot(0, 166, '+', color='red', markersize=12)


#%%
## We build an (x,y)-coordinate, 
## where x-axis represents the horizontal direction and y-axis represents the vertical direction 
## by taking the center of the dartboard to be (0,0). 
## It is more convenient to work with non-negative integers for indexing purpose in python code, 
## hence we take an offset of 170 on the (x,y)-coordinate.

## compare the (x,y)-coordinate and (x,y)-index
fb.plot_dartboard(flag_index=False, text_input='(x,y)-coordinate')
fb.plot_dartboard(flag_index=True, text_input='(x,y)-index')


#%%
playerID = 7
name_pa = 'player{}'.format(playerID)
data_parameter_dir = '../../data_parameter/player_gaussin_fit'
grid_version = 'v2'
#grid_version = 'circleboard'

[aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = function_get_aiming_grid.load_aiming_grid(name_pa)

#%%
print('Each row represents an aiming location and there are totally 984 locations.') 
print(aiming_grid.shape)
print(aiming_grid)
fb.plot_dartboard(points_input=aiming_grid, point_marker='.', flag_index=True)
print()

#%%
## index aiming_grid_pa[0] = [170, 170]
## x,y coordinate = [0, 0]
## [score=50, multiplier=2] in the DB (double bull) area 
print('the fist choice of aiming locations is')
print(aiming_grid[0])
print('the corresponding score and multiplier are')
print(fb.get_score_and_multiplier_fromindex(aiming_grid[0]))
print()
plt.plot(aiming_grid[0, 0], aiming_grid[0, 1], '+', color='red', markersize=12)

print('the last choice of aiming locations is')
print(aiming_grid[-1])
print('the corresponding score and multiplier are')
print(fb.get_score_and_multiplier_fromindex(aiming_grid[-1]))
print()
plt.plot(aiming_grid[-1, 0], aiming_grid[-1, 1], '+', color='red', markersize=12)

#%%
## The following matrices provide the probability of hitting each specific score region on the dartboard 
## Each row corresponds to the aiming target at the same row of aiming_grid

## probability of hitting single regions S1(the first column),...,S20(the last column) given an aiming location
print(prob_grid_singlescore.shape)
print(prob_grid_singlescore)
print()

## probability of hitting double regions D1(the first column),...,D20(the last column) given an aiming location
print(prob_grid_doublescore.shape)
print(prob_grid_doublescore)
print()

## probability of hitting triple regions T1(the first column),...,T20(the last column) given an aiming location
print(prob_grid_triplescore.shape)
print(prob_grid_triplescore)
print()

## probability of hitting SB(the first column) and DB(the last column) given an aiming location
print(prob_grid_bullscore.shape)
print(prob_grid_bullscore)
print()

## probability of make a score of 0(first column: index 0), 1(second column: index 1), ... 1(last column: index 60)
## note that a few colums are zero: infeasible score
print(prob_grid_normalscore.shape)
print(prob_grid_normalscore)
print(prob_grid_normalscore.sum(axis=1))  ## probabilities add up 1

#%%
print('')
print('If aiming at aiming_grid_pa[-1] (the center of T20)')
target_index = -1
print('probability of hitting single areas S1,S2,...,S20 are ')
print(prob_grid_singlescore[target_index,:])
print('probability of hitting double areas D1,D2,...,D20 are ')
print(prob_grid_doublescore[target_index,:])
print('probability of hitting triple areas T1,T2,...,T20 are ')
print(prob_grid_triplescore[target_index,:])
print('probability of hitting SB (single bull) and DB (double bull) areas are ')
print(prob_grid_bullscore[target_index,:])
print('probability of making a score of 0,1,2,...,60 are ')
print(prob_grid_normalscore[target_index,:])
print('These numbers add up to {}'.format(prob_grid_normalscore[0].sum()))

#%%
## probability of making a score of 18
score_state = 18
score_index = score_state - 1
print('P(S18)={}'.format(prob_grid_singlescore[target_index,score_index]))
print('P(D9)={}'.format(prob_grid_doublescore[target_index,score_index]))
print('P(T6)={}'.format(prob_grid_triplescore[target_index,score_index]))
print('P(S18+D9+T6)={}'.format(prob_grid_singlescore[target_index,score_index]+prob_grid_doublescore[target_index,score_index]+prob_grid_triplescore[target_index,score_index]))
print('P(score=18)={}'.format(prob_grid_normalscore[target_index,score_state]))


#%%
print('Expected score of a single throw')

score = np.arange(61)
print(score)
es = np.dot(prob_grid_normalscore, score)

es_max = np.max(es)
es_argmax = np.argmax(es)
#ex_max = es[ex_argmax]

print('largest expected score is {}'.format(es_max))
print('aiming target is {}'.format(aiming_grid[es_argmax]))
print('score region is {}'.format(fb.get_score_and_multiplier_fromindex(aiming_grid[es_argmax])))

