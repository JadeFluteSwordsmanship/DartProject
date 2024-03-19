Course Project. 
Copyright@author: 王纯, 清华经管. Chun Wang, School of Economics and Management of Tsinghua University. 
wangchun@sem.tsinghua.edu.cn

1: Computation environment:
Python 3.7.4 + numpy 1.16.5 + pytorch 1.7.1 (optional for the project)

2: Code Files
2.1 All_Model_fits.mat: Provide the fitted skill model (the covariance matrix of the conditional Gaussian Model) for each players. 
2.2 function_tool.py: Provide some file operation functions, i.e., reading and saving.
2.3 function_board.py: Provide the layout of the dart board and the 1mm grid of aiming locations. 
2.4 function_get_aiming_grid.py: Provide functions to generate action set. We use an action set of 984 aiming points in the project. We can also use a large action set of 90,785 aiming points (all points in the 1mm grid over the dartboard). 
2.5 function_solve_dp_demo.py: Solve the single player dart game. 
