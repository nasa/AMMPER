# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:41:17 2022

genROS function: generates ROS environment based on radData
inputs:
    radData: list of radiation events
        [PosX,PosY,PosZ,energy deposition,energyType,protonEnergy]
        energyType: 2 if from ion, 1 if from electron
        protonEnergy: energy of proton that initiated radiation event
    cells: list of cell objects present in simulation space throughout life
outputs:
    ROSData: list of secondary radiation events
        [PosX,PosY,PosZ,yield_H2O2,yield_OH,cellHit]
        cellHit: boolean if ROS pos coincides with a cell
    

@author: asingh21
"""

def genROS(radData,cells):
    import numpy as np
    
    n = len(radData)
    #ROS yields (Plante Radiation Chemistry, from three.jsc.nasa.gov)
    # primary yields in [molecules/100 eV]
    G_H2O2 = 0.7
    G_OH = 2.5
    cellHit = 0 #boolean for whether or not cell is hit
    ROSData = np.zeros([1,6])
    for i in range(n):
        radPos = [radData[i,0],radData[i,1],radData[i,2]]
        energy = radData[i,3]
        yield_H2O2 = G_H2O2*energy
        yield_OH = G_OH*energy
        for c in cells:
            currPos = c.position
            if radPos[0] <= currPos[0]+2 and radPos[0] >= currPos[0]-2:
                if radPos[1] <= currPos[1]+2 and radPos[1] >= currPos[1]-2:
                    if radPos[2] <= currPos[2]+2 and radPos[2] >= currPos[2]-2:
                        cellHit = 1
        ROSDataEntry = [radPos[0],radPos[1],radPos[2],yield_H2O2,yield_OH,cellHit]
        ROSData = np.vstack([ROSData,ROSDataEntry])
    #remove placeholder
    ROSData = np.delete(ROSData,(0),axis = 0)
    
    
    return(ROSData)
    