# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 08:31:34 2022

@author: asingh21
"""
# function to remove overlap when setting 3d axes, from stackoverflow
# https://stackoverflow.com/questions/30196503/2d-plots-are-not-sitting-flush-against-3d-axis-walls-in-python-mplot3d/41779162#41779162
def get_fixed_mins_maxs(mins, maxs):
    deltas = (maxs - mins) / 12.
    mins = mins + deltas / 4.
    maxs = maxs - deltas / 4.

    return [mins, maxs]


def cellPlot(data,gen,radData,ROSData,radGen,N,plots_dir):
    from matplotlib import pyplot as plt
    
    minmax = get_fixed_mins_maxs(0, N)
    
    healthy = 'blue'
    damaged = 'purple'
    dead = 'red'
    
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
        

    n = len(data)
    for i in range(n):
        currG = data[i,0]
        pos = data[i,1:4]
        health = data[i,4]
        if health == 1:
            col = healthy
        elif health == 2:
            col = damaged
        elif health == 3:
            col = dead
        axName = 'ax' + str(currG)
        locals()[axName].scatter(pos[0],pos[1],pos[2],c=col)
        

    radName = 'ax' + str(radGen)
    
    if type(radData) != int:
        n = len(radData)
        for i in range(n):
            locals()[radName].scatter(radData[i,0],radData[i,1],radData[i,2],s = 10,c = 'yellow')

    if type(ROSData) != int:      
        n = len(ROSData)
        for g in range(gen+1):
            if g >= radGen:
                ROSName = 'ax' + str(g)
                for i in range(n):
                    locals()[ROSName].scatter(ROSData[i,0],ROSData[i,1],ROSData[i,2],s = 5,c = 'green')
        
    
    for g in range(gen+1):
        figName = 'fig' + str(g)
        locals()[figName].savefig(plots_dir + figName)
        