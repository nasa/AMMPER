# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 07:30:15 2022

genTraverse_deepSpace function: generates omnidirectional radiation environment to mimic 0.1 month of deep space radiation 
inputs:
    N: size of simulation space (NxNxN) microns
    Gy: Gy of radiation delivered during radiation event
    protonEnergy: 
    trackNum: iterator defining which track is being built (precalculated to support Gy)
    energyThreshold: threshold of energy to be evaluated (precalculated to support Gy)

outputs:
    radData: list of radiation events
        [PosX,PosY,PosZ,energy deposition,energyType,protonEnergy]
        energyType: 2 if from ion, 1 if from electron
        protonEnergy: energy of proton that initiated radiation event
    

@author: asingh21
"""


def genTraverse_deepSpace(N,protonEnergy,trackNum,energyThreshold):
    import numpy as np
    import random as rand
    import math as math
    
    # convert proton energy to a string
    protonEnergyStr = str(protonEnergy)

    # create path to access data
    trackStr = str(trackNum)
    path = 'radiationData/'+protonEnergyStr +'/Track'+trackStr+'/'
    ElStr = 'ElEvents.dat'
    IonStr = 'IonEvents.dat'
    #PCStr = 'PCEvents.dat'
    
    # save data from file to python (electron data and ion data)
    El_data = np.genfromtxt(path+ElStr,skip_header = 1)
    Ion_data = np.genfromtxt(path+IonStr,skip_header = 1)
    #PC_data = np.genfromtxt(path+PCStr,skip_header = 1)
        
    # ID of 1 means energy from Electron
    El_ID = np.ones([len(El_data),1],dtype = int)
    # ID of 2 means energy from Ion
    Ion_ID = np.ones([len(Ion_data),1],dtype = int)*2
    
    # combine ID with data imported from files
    El_data = np.hstack([El_data,El_ID])
    Ion_data = np.hstack([Ion_data,Ion_ID])
    
    # combine electron and ion data together
    radData = np.vstack([El_data,Ion_data])
    
    #placeholder row - will be deleted after filled
    # for all of the energy depositions above the determined threshold
    radData_threshold = np.zeros([1,5])
    
    # for every entry in radData, determine whether it's above the energy threshold
    # if so, save to radData_threshold
    for i in range(len(radData)):
        if radData[i,4] > energyThreshold:
            # convert position data from angstroms o microns
            # energy data is in eV
            newDat = [radData[i,0]/1e4,radData[i,1]/1e4,radData[i,2]/1e4,radData[i,4],radData[i,6]]
            radData_threshold = np.vstack([radData_threshold,newDat])
    
    #remove placeholder row
    radData_threshold = np.delete(radData_threshold,(0),axis = 0)
    
    # placeholder values to ensure vertical concatentation 
    # position values of energy deposition by a single event
    x = [0]
    y = [0]
    z = [0]
    # value of energy deposition by a single event
    energy = [0]
    # type of energy deposition (electron/ion)
    energyType= [0]
    
    # for every energy deposition event above the threshold
    for i in range(len(radData_threshold)):
        # get the position of energy deposition
        newX = radData_threshold[i,0]
        newY = radData_threshold[i,1]
        newZ = radData_threshold[i,2]
        # get the energy of energy deposition
        newEnergy = radData_threshold[i,3]
        # get the type of energy deposition
        newType = radData_threshold[i,4]
        
        # add all of "gotten" data as new rows - not columns
        x = np.vstack([x,newX])
        y = np.vstack([y,newY])
        z = np.vstack([z,newZ])
        energy = np.vstack([energy,newEnergy])
        energyType = np.vstack([energyType,newType])
    
    #remove placeholder
    x = np.delete(x,(0),axis = 0)
    y = np.delete(y,(0),axis = 0)
    z = np.delete(z,(0),axis = 0)
    energy = np.delete(energy,(0),axis = 0)
    energyType = np.delete(energyType,(0),axis = 0)
        
    
    insideSphere = 0
    while insideSphere == 0:
        x1 = rand.uniform(0,N)
        y1 = rand.uniform(0,N)
        z1 = rand.uniform(0,N)
        x2 = rand.uniform(0,N)
        y2 = rand.uniform(0,N)
        z2 = rand.uniform(0,N)
        
        startTrav = [x1, y1, z1]                   
        endTrav = [x2, y2, z2]
        
        startLen = math.sqrt((((N/2) - x1)**2)+(((N/2) - y1)**2)+(((N/2) - z1)**2))
        endLen = math.sqrt((((N/2) - x2)**2)+(((N/2) - y2)**2)+(((N/2) - z2)**2))                   
        if startLen <= (N/2) and endLen <= (N/2):
            insideSphere = 1
            
    # equation for a 3d line: Xp = X0 + Vx*t, where [Vx,Vy,Vz] = vector of line
    
    travDir = [x2-x1,y2-y1,z2-z1]
    travNorm = np.linalg.norm(travDir)
    travUnit = travDir/travNorm #normalized vector with only direction info
    
    # calculate distance between startTrav and endTrav and sides of space to
    # find out min length to extend line to go through entire space
    dx = travUnit[0]
    dy = travUnit[1]
    dz = travUnit[2]
    
    t1_x0 = abs((0-x1)/dx);
    t1_xN = abs((N-x1)/dx);
    t1_y0 = abs((0-y1)/dy);
    t1_yN = abs((N-y1)/dy);
    t1_z0 = abs((0-z1)/dz);
    t1_zN = abs((N-z1)/dz);
    
    t2_x0 = abs((0-x2)/dx);
    t2_xN = abs((N-x2)/dx);
    t2_y0 = abs((0-y2)/dy);
    t2_yN = abs((N-y2)/dy);
    t2_z0 = abs((0-z2)/dz);
    t2_zN = abs((N-z2)/dz);
    
    t1 = min([t1_x0,t1_xN,t1_y0,t1_yN,t1_z0,t1_zN]);
    t2 = min([t2_x0,t2_xN,t2_y0,t2_yN,t2_z0,t2_zN]);
    
    startTrav_xtend = -t1*travUnit + startTrav;
    endTrav_xtend = t2*travUnit + endTrav;
    travDir = endTrav_xtend-startTrav_xtend;
    
    travNorm = np.linalg.norm(travDir)
    travUnit = travDir/travNorm
    
    # coordinate transformation part 1
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    
    # calculate rotation matrix for rotating from unit vector A to B
    # unit vector A = [0,1,0]
    # unit vector B = travUnit
    # v = A cross B
    # s = magnitude(v) = sin of angle
    # c = A dot B = cosine of angle
    # [v]_x = skew-symmetric cross-product of v = [0,-v3,v2;v3,0,-v1;-v2,v1,0]
    
    # R = Rotation Matrix = I + [v]_x + [v]_x^2 * ((1-c)/(s^2))
    # (1-c)/(s^2) = (1-c)/((1-(c^2)) = 1/(1+c)
    # R = Rotation Matrix = I + [v]_x + [v]_x^2 * (1/(1+c))
    
    # data is initially oriented along y axis
    A = np.array([0,1,0],dtype = np.float64)
    B = travUnit
    v = np.cross(A,B)
    s = np.linalg.norm(v)
    c = np.dot(A,B)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    R = np.eye(3) + vx + (np.dot(vx,vx) * (1/(1+c)))
    
    

    # number of energy deposition events above threshold
    n = len(x)
    
    
    #initialize xTrans,yTrans,zTrans - new location (x,y,z) under rotation by A
    xTrans = [0]
    yTrans = [0]
    zTrans = [0]
    
    # for every energy deposition event above threshold
    for i in range(n):
        # get initial position
        vec_x = x[i]
        vec_y = y[i]
        vec_z = z[i]
        vec = np.vstack([vec_x,vec_y,vec_z])
        # calculate position after rotation transformation
        newPoint = np.dot(R,vec)
        # translate rotated point by start of traversal 
        xTransNew = (newPoint[0]) + startTrav_xtend[0] 
        yTransNew = (newPoint[1]) + startTrav_xtend[1]
        zTransNew = (newPoint[2]) + startTrav_xtend[2] 
        
        # assign new position
        xTrans = np.vstack([xTrans,xTransNew])
        yTrans = np.vstack([yTrans,yTransNew])
        zTrans = np.vstack([zTrans,zTransNew])
    
    #remove placeholder values
    xTrans = np.delete(xTrans,(0),axis = 0)
    yTrans = np.delete(yTrans,(0),axis = 0)
    zTrans = np.delete(zTrans,(0),axis = 0)
    
    # documenting ion energy in case of multiple ions/different proton energies
    protonEnergyArr = np.ones([n,1])*protonEnergy
    
    # compiling all data together
    # 0,1,2 - energy deposition location
    # 3 - energy deposition energy
    # 4 - energy deposition type (ion = 2/electron = 1)
    # 5 - energy deposition source (proton energy level)
    radData_trans = np.hstack([xTrans,yTrans,zTrans,energy,energyType,protonEnergyArr])
    #radData_trans = np.hstack([x,y,z,energy,energyType,protonEnergyArr])
    radData = radData_trans
    
    return(radData)
    
