# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 08:31:34 2022

@author: asingh21
"""


import pandas as pd
import numpy as np

# function to remove overlap when setting 3d axes, from stackoverflow
# https://stackoverflow.com/questions/30196503/2d-plots-are-not-sitting-flush-against-3d-axis-walls-in-python-mplot3d/41779162#41779162
def get_fixed_mins_maxs(mins, maxs):
    deltas = (maxs - mins) / 12.
    mins = mins + deltas / 4.
    maxs = maxs - deltas / 4.

    return [mins, maxs]


def cellPlot_deepSpace(data,gen,radData,ROSData,N,plots_dir):
    from matplotlib import pyplot as plt
    
    minmax = get_fixed_mins_maxs(0, N)
    
    healthy = '#7FBBDF'
    damaged = '#483c6a'
    dead = '#F7AF97'

    n = len(data)

    df_data = pd.DataFrame(data, columns=['Generation', 'x', 'y', 'z', 'Health'])

    
    for g in range(gen+1):
        figName = 'fig' + str(g)
        axName = 'ax' + str(g)
        locals()[figName] = plt.figure()
        locals()[axName] = locals()[figName].add_subplot(projection='3d')
        locals()[axName].set_xlim(minmax)
        locals()[axName].set_ylim(minmax)
        locals()[axName].set_zlim(minmax)
        locals()[axName].set_xlabel('X')
        locals()[axName].set_ylabel('Y')
        locals()[axName].set_zlabel('Z')
        plt.title('g = '+str(g))

        #########################
        # Here we need to define our data as data_generation(secific g) ---> easier with pandas
        data = df_data.loc[df_data['Generation'] == g]
        ###############################
        # Now need to go back to numpy array without titles >>> <<<
        # data = data.to_numpy()

        # n = len(data)
        #
        # pos = data[:, 1:4]
        # col = data[:,5]
        # col2 = data[:,6]
        axName = 'ax' + str(g)  #### <------

        data1 = data.loc[data['Health'] == 1]
        data2 = data.loc[data['Health'] == 2]
        data3 = data.loc[data['Health'] == 3]

        data1 = data1.to_numpy()
        data2 = data2.to_numpy()
        data3 = data3.to_numpy()

        axName = 'ax' + str(g)  #### <------

        locals()[axName].scatter(data1[:, 1], data1[:, 2], data1[:, 3], c=healthy, alpha=0.15)
        locals()[axName].scatter(data2[:, 1], data2[:, 2], data2[:, 3], c=damaged, alpha=1)
        locals()[axName].scatter(data3[:, 1], data3[:, 2], data3[:, 3], c=dead, alpha=1)



    if type(ROSData) != int:
        n = len(ROSData)
        for g in range(gen + 1):
                ROSName = 'ax' + str(g)
                locals()[ROSName].scatter(ROSData[:, 0], ROSData[:, 1], ROSData[:, 2], s=5, c='#9ED9A1', alpha=1)
        
    for g in range(gen+1):
        figName = 'fig' + str(g)
        locals()[figName].savefig(plots_dir + figName)




    # n = len(data)
    # for i in range(n):
    #     currG = data[i,0]
    #     pos = data[i,1:4]
    #     health = data[i,4]
    #     if health == 1:
    #         col = healthy
    #     elif health == 2:
    #         col = damaged
    #     elif health == 3:
    #         col = dead
    #     axName = 'ax' + str(currG)
    #     locals()[axName].scatter(pos[0],pos[1],pos[2],c=col)