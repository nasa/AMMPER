# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 27 09:42:26 2022

Contains script file that runs AMMPER simulation (asks for user input), then plots it

AMMPER function: calculates effect of radiation on microbial population

outputs:
    data: text file containing positions and healths of all cells at each generation
        [generation,PosX,PosY,PosZ,health] - outputted to results folder
    plots: .jpg images of the radiation, ROS, and cells at each generation - outputted to results folder
    simDescirption: text file containing simulation run information
    cells: list of cell objects present in simulation space throughout life
    radData: list of radiation events
        [PosX,PosY,PosZ,energy deposition,energyType,protonEnergy]
        energyType: 2 if from ion, 1 if from electron
        protonEnergy: energy of proton that initiated radiation event
    ROSData: list of secondary radiation events
        [PosX,PosY,PosZ,yield_H2O2,yield_OH,cellHit]
        cellHit: boolean if ROS pos coincides with a cell


@author: asingh21
"""

"""
@edited by Daniel Palacios 
"""

import numpy as np
import random as rand
import uuid as uuid
from cellDefinition import Cell
from genTraverse_groundTesting import genTraverse_groundTesting
from genTraverse_deepSpace import genTraverse_deepSpace
from genROS import genROS
from genROSOld import genROSOld
from cellPlot import cellPlot
from cellPlot_deepSpace import cellPlot_deepSpace
from GammaRadGen import GammaRadGen
import os
import time
import pandas as pd
import time
from sklearn.model_selection import train_test_split
start_time = time.time()

radType = input("Please enter what simulation type you would like to run:\n\ta)150MeV Proton\n\tb)NSRL GCRSim\n\tc)Deep Space\n\td)Gamma\n")
if radType == "a":
    radAmount = input("Please enter radiation dose. Options are: 0, 2.5, 5, 10, 20, 30 Gy.\n")
    Gy = float(radAmount)
    radType = "150 MeV Proton"
    gen = 15
    # Before 2 <<<- Trying higher generation to see difference between diffusion and lifetime model and old model.
    radGen = 2
    N = 64
    if Gy == 0:
        radData = 0
        ROSData = 0
elif radType == "b":
    radType = "GCRSim"
    gen = 15
    radGen = 2
    N = 64
elif radType == "c":
    radType = "Deep Space"
    gen = 15
    N = 300
    radGen = 0
    Gy = 0
    
elif radType == 'd':
    radType = "Gamma"
    gen = 15
    radGen = 10
    #radGenE = 10
    N = 64 # real 64 ? 


cellType = input("Please enter cell type:\n\ta)Wild Type\n\tb)rad51\n")
if cellType == "a":
    cellType = "wt"
elif cellType == "b":
    cellType = "rad51"

# ROS model old and new, ROS Old computes eternal and static ROS free radicals, complex ROS models diffusion and time
# mechanics.
ROSType = input("Please enter ROS Model: \n\ta)Basic ROS\n\tb)Complex ROS\n")
if ROSType == "a":
    ROSType = "Basic ROS"
if ROSType == "b":
    ROSType = "Complex ROS"



# description of simulation to be written to file
simDescription = "Cell Type: " + cellType + "\nRad Type: " + radType + "\nSim Dim: " + str(N) + "microns\nNumGen: " + str(gen) + "ROS model: " + str(ROSType)

# results folder name with the time that the simulation completed
resultsName = time.strftime('%m-%d-%y_%H-%M') + "/"
# determine path that all results will be written to
resultsFolder = "Results/"
currPath = os.path.dirname("AMMPER")
allResults_path = os.path.join(currPath,resultsFolder)
currResult_path = os.path.join(allResults_path,resultsName)
plots_path = os.path.join(currResult_path,"Plots/")

# if any of the folders do not exist, create them
if not os.path.isdir(resultsFolder):
    os.makedirs(resultsFolder)
if not os.path.isdir(currResult_path):
    os.makedirs(currResult_path)
if not os.path.isdir(plots_path):
    os.makedirs(plots_path)

# write description to file
np.savetxt(currResult_path+'simDescription.txt',[simDescription],fmt='%s')

# SIMULATION SPACE INITIALIZATION

# cubic space (0 = no cell, 1 = healthy cell, 2 = damaged cell, 3 = dead cell)
T = np.zeros((N,N,N),dtype= int)
# END OF SIMULATION SPACE INITIALIZATION
# CELL INITIALIZATION

# first cell is at center and is healthy
firstCellPos = [int(N/2),int(N/2),int(N/2)]

initCellHealth = 1

# cells = list of cells - for new cells, cells.append
#firstCellPos = initCellPos[0,:]
firstUUID = uuid.uuid4()
firstCell = Cell(firstUUID,firstCellPos,initCellHealth,0,0,0,0)
T[firstCellPos[0],firstCellPos[1],firstCellPos[2]] = firstCell.health
cells = [firstCell]
# data: [generation, cellPosition, cellHealth]
data = [0,firstCell.position[0],firstCell.position[1],firstCell.position[2],firstCell.health]    


# END OF CELL INITIALIZATION

#placeholder initialization
if radType == "Deep Space":
    radData = np.zeros([1,7],dtype = float)
    ROSData = np.zeros([1,7],dtype = float)

print("Simulation beginning.")
for g in range(1,gen+1):
    print("Generation " + str(g))
    # calculation of radiation in simulation space
    if radType == "Gamma":
        if g == radGen:
            
            dose = 1
            # radData = np.zeros([1, 6], dtype=float)
            # Dose input, radGenE stop point for gamma radiation.
            radData = GammaRadGen(dose)
            # radData = np.delete(radData, (0), axis=0)
            
            if ROSType == "Complex ROS":
                ROSData = genROS(radData, cells)
            if ROSType == "Basic ROS":
                ROSData = genROSOld(radData, cells)
        
        
        
    if radType == "150 MeV Proton":
        if g == radGen:
            protonEnergy = 150
            # these fluences are pre-calculated to deliver the dose to the volume of water
            if Gy != 0:
                if Gy == 2.5:
                    trackChoice = [1]
                    energyThreshold = 0
                elif Gy == 5:
                    trackChoice = [1,1]
                    energyThreshold = 0
                elif Gy == 10:
                    trackChoice = [1,1,1,1]
                    energyThreshold = 0
                elif Gy == 20:
                    trackChoice = [1,1,1,1,1,1,1,1]
                    energyThreshold = 0
                elif Gy == 30:
                    trackChoice = [1,1,1,1,1,1,1,1,1,1,1,1]
                    energyThreshold = 0
                
                # placeholder initialization - will hold information on all radiation energy depositions
                radData = np.zeros([1,6],dtype = float)
                # ROSData = np.zeros([1,6],dtype = float)

                for track in trackChoice:
                    trackNum = track
                    # creates a traverse for every track in trackChoice
                    radData_trans = genTraverse_groundTesting(N,protonEnergy,trackNum,energyThreshold,radType)
                    # compile all energy depositions from individual tracks together
                    radData = np.vstack([radData,radData_trans])
                
                #remove placeholder of 0s from the beginning of radData
                radData = np.delete(radData,(0),axis = 0)
                            
                # direct energy results in ROS generation - use energy depositions to calculate ROS species
                if ROSType == "Complex ROS":
                    ROSData = genROS(radData,cells)
                if ROSType == "Basic ROS":
                    ROSData = genROSOld(radData,cells)
                #
                # ROSData = np.delete(ROSData, (0), axis = 0)
                
    elif radType == "Deep Space": 
        
        # take information from text file for #tracks of each proton energy
        deepSpaceFluenceDat = np.genfromtxt('DeepSpaceFluence0.1months_data.txt')
        # NOTE: For Deep Space sim, radiation delivery is staggered over time
        for it in range(len(deepSpaceFluenceDat)):
            # get generation at which traversal will occur
            currG = int(deepSpaceFluenceDat[it,5])
            # determine how many traversals occur at this generation
            numTrav = int(deepSpaceFluenceDat[it,4])
            if g == currG:
                # determine what proton energy the traversal has
                protonEnergy = deepSpaceFluenceDat[it,0]
                # parameter that allows non-damaging energy depositions to be ignored (used to speed up simulation)
                energyThreshold = 20
                for track in range(numTrav):
                    # choose a random track out of the 8 available/proton energy
                    trackNum  = int(rand.uniform(0,7))
                    # generate traversal data for omnidirectional traversals
                    radData_trans = genTraverse_deepSpace(N,protonEnergy,trackNum,energyThreshold)
                    # generate ROS data from the traversal energy deposition
                    # ROSData_new = genROS(radData_trans,cells)
                    if ROSType == "Complex ROS":
                        ROSData_new = genROS(radData_trans, cells)
                    if ROSType == "Basic ROS":
                        ROSData_new = genROSOld(radData_trans, cells)
                    
                    # creates a column indicating what generation the ROS and radData occured at
                    genArr = np.ones([len(radData_trans),1],dtype=int)*g
                    # compile radData with the generation indicator
                    radData_trans = np.hstack((radData_trans,genArr))
                    # compile radData from this traversal with all radData
                    radData = np.vstack([radData,radData_trans])
                    
                    # compile ROSData with the generation indicator
                    ROSData_new = np.hstack((ROSData_new,genArr))
                    #compile ROSData with all ROSData
                    ROSData = np.vstack([ROSData,ROSData_new])
                    
    elif radType == "GCRSim":
        if g == radGen:
            # placeholder initialization
            radData = np.zeros([1,6],dtype = float)
            # take information from text file on traversals that will occur
            GCRSimFluenceDat = np.genfromtxt('GCRSimFluence_data.txt',skip_header = 1)
            for it in range(len(GCRSimFluenceDat)):
                # for every traversal, get the proton energy of it
                protonEnergy = int(GCRSimFluenceDat[it,0])
                # parameter that allows non-damaging energy depositions to be ignored (used to speed up simulation)
                energyThreshold = 20
                # choose a random track out of the 8 available/proton energy
                trackNum = int(rand.uniform(0,7))
                # generate traversal data for unidirectional traversals
                radData_trans = genTraverse_groundTesting(N,protonEnergy,trackNum,energyThreshold,radType)
                # compile radData from this traversal with all radData
                radData = np.vstack([radData,radData_trans])
                
                #remove placeholder from beginning
                radData = np.delete(radData,(0),axis = 0)
            # generate ROS data from all traversal energy depositions
            #ROSData = genROS(radData,cells)
            if ROSType == "Complex ROS":
                ROSData = genROS(radData, cells)
            if ROSType == "Basic ROS":
                ROSData = genROSOld(radData, cells)

    
    # initialize list of cells that have moved
    movedCells = []
    # for every existing cell, determine whether a cell moves. If it does, write it to the list
    for c in cells:
        
        initPos = c.position
        initPos = [initPos[0],initPos[1],initPos[2]]
        movedCell = c.brownianMove(T,N,g)
        newPos = movedCell.position
        # if cell has moved
        if initPos != newPos and newPos != -1:
            # document new position in simulation space, and assign old position to empty
            T[initPos[0],initPos[1],initPos[2]] = 0
            T[newPos[0],newPos[1],newPos[2]] = c.health
            
            movedCells.append(c)

    # initialize list of new cells
    newCells = []
    # for every existing cell, determine whether a cell replicates. If it does, write the new cell to the list
    for c in cells:
        
        health = c.health
        #position = c.position
        UUID = c.UUID
        
        # cell replication
        if health == 1:
            # cellRepl returns either new cell, or same cell if saturation conditions occur
            newCell = c.cellRepl(T,N,g)
            newCellPos = newCell.position
            newCellUUID = newCell.UUID
            # if newCell the same as old cell, then saturation conditions occurred, and no replication took place
            if newCellUUID != UUID and newCellPos != -1:
                # only document new cell if old cell replicated    
                # if new cell is avaialble, assign position as filled                
                T[newCellPos[0],newCellPos[1],newCellPos[2]] = 1
                newCells.append(newCell)
    
    
    # if radiation traversal has occured
    if (radType == "150 MeV Proton" and g == radGen and Gy != 0) or (radType == "Deep Space") or (radType == "GCRSim" and g == radGen) or (radType == "Gamma" and g >= radGen):
        # initialize list of cells affected by ion/electron energy depositions
        dirRadCells = []
        for c in cells:
            health = c.health
            if cellType == "wt":
                radCell = c.cellRad(g,radGen,radData,radType)
            elif cellType == "rad51":
                radCell = c.cellRad_rad51(g,radGen,radData,radType)
            if type(radCell) == Cell:
                newHealth = radCell.health
                if health != newHealth:
                    radCellPos = radCell.position
                    T[radCellPos[0],radCellPos[1],radCellPos[2]] = newHealth
                    dirRadCells.append(radCell)
                    ######################################################################
    # if ROS generation has occured (post-radiation)
    if (radType == "150 MeV Proton" and g >= radGen and Gy != 0) or (radType == "Deep Space") or (radType == "GCRSim" and g >= radGen) or (radType == "Gamma" and g >= radGen):
        # initialize list of cells affected by ROS
        ROSCells = []
        for c in cells:
            health = c.health
            if cellType == "wt":
                ROSCell = c.cellROS(g,radGen,ROSData)
            elif cellType == "rad51":
                ROSCell = c.cellROS_rad51(g,radGen,ROSData)
            newHealth = ROSCell.health
            if health != newHealth:
                ROSCellPos = ROSCell.position
                T[ROSCellPos[0],ROSCellPos[1],ROSCellPos[2]] = newHealth
                ROSCells.append(ROSCell)
    # if radiation has occured and cell type is NOT rad51 (cellType = wild type)
    if (radType == "150 MeV Proton" and g > radGen and Gy != 0 and cellType != "rad51") or (radType == "Deep Space" and cellType != "rad51") or (radType == "GCRSim" and g > radGen and cellType != "rad51") or (radType == "Gamma" and g >= radGen and cellType != "rad51"):
        # initialize list of cells that have undergone repair mechanisms
        repairedCells = []
        for c in cells:
            health = c.health
            if health == 2:
                repairedCell = c.cellRepair(g)
                newHealth = repairedCell.health
                repairedCellPos = repairedCell.position
                T[repairedCellPos[0],repairedCellPos[1],repairedCellPos[2]] = newHealth
                repairedCells.append(repairedCell)
                
    # documenting cell movement in the cells list
    for c in movedCells:
        cUUID = c.UUID
        newPos = c.position
        for c2 in cells:
            c2UUID = c2.UUID
            if cUUID == c2UUID:
                c2.position = newPos
             
    # documenting cell replication in the cells list
    cells.extend(newCells)
    
    # documenting cell damage
    # if radiation has occurred
    if (radType == "150 MeV Proton" and g >= radGen and Gy != 0) or radType == "Deep Space" or (radType == "GCRSim" and g >= radGen) or (radType == "Gamma" and g >= radGen):
        # for every cell damaged by ion or electron energy depositions
        for c in dirRadCells:
            # get information about damaged cell
            cUUID = c.UUID
            newHealth = c.health
            newSSBs = c.numSSBs
            newDSBs = c.numDSBs
            # find cell in cell list that matches damaged cell ID
            for c2 in cells:
                c2UUID = c2.UUID
                if cUUID == c2UUID:
                    # adjust information about cell in cell list to reflect damage
                    c2.health = newHealth
                    c2.numSSBs = newSSBs
                    c2.numDSBs = newDSBs
        # for every cell damaged by ROS
        for c in ROSCells:
            # get information about damaged cell
            cUUID = c.UUID
            newHealth = c.health
            newSSBs = c.numSSBs
            # find cell in cell list that matches damaged cell ID
            for c2 in cells:
                c2UUID = c2.UUID
                if cUUID == c2UUID:
                    # adjust information about cell in cell list to reflect damage
                    c2.health = newHealth
                    c2.numSSBs = newSSBs
    
    #documenting cell repair
    # if cell can repair (not rad51), and radiation has occured
    if (radType == "150 MeV Proton" and g > radGen and Gy != 0 and cellType != "rad51") or (radType == "Deep Space" and cellType != "rad51") or (radType == "GCRSim" and g > radGen and cellType != "rad51") or (radType == "Gamma" and g >= radGen and cellType != "rad51"):
        # for every cell that has undergone repair
        for c in repairedCells:
            # get information about repaired cell
            cUUID = c.UUID
            newHealth = c.health
            newSSBs = c.numSSBs
            newDSBs = c.numDSBs
            # find cell in cell list that matches repaired cell ID
            for c2 in cells:
                c2UUID = c2.UUID
                if cUUID == c2UUID:
                    # adjust information about cell in cell list to reflect repair
                    c2.health = newHealth
                    c2.numSSBs = newSSBs
                    c2.numDSBs = newDSBs
        
        
                    
    # adjust data with new generational data
    # column array to denote that new data entries are at the current generation
    genArr = np.ones([len(cells),1],dtype=int)*g
    # get cell information to store in data
    # initialize cellsHealth and cellsPos with placeholders
    cellsHealth = [0]
    cellsPos = [0,0,0]
    # for each cell, get all the associated information
    for c in cells:
        currPos = [c.position]
        currHealth = [c.health]
        # record all cell positions and healths in a list, with each cell being a new row
        cellsPos = np.vstack([cellsPos,currPos])
        cellsHealth = np.vstack([cellsHealth, currHealth])
    #remove placeholder values
    cellsPos = np.delete(cellsPos,(0),axis = 0)
    cellsHealth = cellsHealth[1:]
    # compile cell information with genArr
    newData = np.hstack([genArr,cellsPos,cellsHealth])
    
    # compile new generation data with the previous data
    data = np.vstack([data,newData])
######################################## Random decay, lifetime ROS for complex model ################################
    if ROSType == "Complex ROS":
        if g > radGen:
            ROSDatak  , ROSData_decayed = train_test_split(ROSData, train_size = 0.5)
            # half life 1 gen = .5, half life 2 gen = .707, half life 3 gen = .7937, 20 min half life = .125
            ROSData = ROSDatak

    

print("Calculations complete. Plotting and writing data to file.")

# for each simulation type, write the data to a text file titled by the radType
# for each simulation type, plot the data as 1 figure/generation
if radType == "150 MeV Proton":
    datName = str(radAmount)+'Gy'
    dat_path = currResult_path + datName + ".txt"
    np.savetxt(dat_path,data,delimiter = ',')
    # if ROSData != 0: for 0 Gy
    cellPlot(data, gen, radData,ROSData,radGen,N,plots_path)

elif radType == "Deep Space":
    datName = 'deepSpace'
    dat_path = currResult_path + datName + ".txt"
    np.savetxt(dat_path,data,delimiter = ',')
    cellPlot_deepSpace(data,gen,radData,ROSData,N,plots_path)
    
elif radType == "GCRSim":
    datName = 'GCRSim'
    dat_path = currResult_path + datName + ".txt"
    np.savetxt(dat_path,data,delimiter = ",")
    cellPlot(data,gen,radData,ROSData,radGen,N,plots_path)

elif radType == "Gamma":
    datName = 'Gamma'
    dat_path = currResult_path + datName + ".txt"
    np.savetxt(dat_path,data,delimiter = ',')
    cellPlot(data, gen, radData,ROSData,radGen,N,plots_path)



print("Plots and data written to Results folder.")

print("time elapsed: {:.2f}s".format(time.time() - start_time))
