# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:03:52 2022

Cell class: The information stored in a model cell (based on Saccharomyces cerevisiae) and the ability to move, replicate, and repair DNA damage 

__init__: cell information
    UUID: unique ID that each cell has
    position: position of cell's center in simulation space, 1x3 int
    health: 1 = no damage, 2 = damaged, 3 = dead, int
    birthGen: generation at which cell was created
    deathGen: generation at which cell died (0 if cell is alive)
    numSSBs: the number of Single Strand Breaks the cell has sustained
    numDSBs: the number of Double Strand Breaks the cell has sustained

availUnits function: for each cell, determine what units are available
inputs:
    c: current cell object being evaluated
    T: 3 dimensional space containing all cells
    N: size of simulation space
outputs:
    availableUnits: list of all free spaces around cell

brownianMove function: for each cell, attempts to move into an empty space
inputs:
    c: current cell object being evaluated
    T: 3 dimensional space containing all cells
    N: size of simulation space
    g: current generation
outputs:
    c:  cell object with adjusted parameters
 
cellRepl function: for each cell, attempts to replicate into an empty space
inputs:
    c: current cell object being evaluated
    T: 3 dimensional space containing all cells
    N: size of simulation space
    g: current generation
outputs:
    c: new cell object 

cellRad function: for each cell, evaluates whether ion/electron radiation has an effect on health
inputs:
    c: current cell object being evaluated
    g: current generation
    radGen: user-defined generation of radiation event (for NSRL type)
    radData: list of radiation events
        [PosX,PosY,PosZ,energy deposition,energyType,protonEnergy]
        energyType: 2 if from ion, 1 if from electron
        protonEnergy: energy of proton that initiated radiation event
outputs:
    c: cell object with adjusted parameters
    
cellRad_rad51 function: for each cell, evaluates whether ion/electron radiation has an effect on health. More sensitive to damage than wild type
inputs:
    c: current cell object being evaluated
    g: current generation
    radGen: user-defined generation of radiation event (for NSRL type)
    radData: list of radiation events
        [PosX,PosY,PosZ,energy deposition,energyType,protonEnergy]
        energyType: 2 if from ion, 1 if from electron
        protonEnergy: energy of proton that initiated radiation event
outputs:
    c: cell object with adjusted parameters

cellROS function: for each cell, evaluates whether indirect radiation has an effect on health
inputs:
    c: current cell object being evaluated
    g: current generation
    radGen: user-defined generation of radiation event (for NSRL type)
    ROSData: list of secondary radiation events
        [PosX,PosY,PosZ,yield_H2O2,yield_OH,cellHit]
        cellHit: boolean if ROS pos coincides with a cell
outputs:
    c: cell object with adjusted parameters
    
cellROS_rad51 function: for each cell, evaluates whether indirect radiation has an effect on health. More sensitive to damage than wild type
inputs:
    c: current cell object being evaluated
    g: current generation
    radGen: user-defined generation of radiation event (for NSRL type)
    ROSData: list of secondary radiation events
        [PosX,PosY,PosZ,yield_H2O2,yield_OH,cellHit]
        cellHit: boolean if ROS pos coincides with a cell

outputs:
    c: cell object with adjusted parameters

cellRepair function: for each cell, attempts to repair damage
inputs:
    c: current cell object being evaluated
    g: current generation
outputs:
    c: cell object with adjusted parameters
                
@author: asingh21
"""

class Cell:
    
    # define cell object attributes
    def __init__(self,UUID,position,health,birthGen,deathGen,numSSBs,numDSBs):
        self.UUID = UUID
        self.position = position
        self.health = health
        self.birthGen = birthGen
        self.deathGen = deathGen
        self.numSSBs = numSSBs
        self.numDSBs = numDSBs
     
    # determines what units directly bordering cell are empty
    def availUnits(self,T,N):
        # cell is sphere inscribed in 4x4x4 microns 
        cellDim = 4
        # creating placeholder for available units
        availableUnits = [[-1,-1,-1]]
        # getting position of cell
        cellPosX = self.position[0]
        cellPosY = self.position[1]
        cellPosZ = self.position[2]
        # for every position next to the cell (up/down, left/right, etc)
        for x in [-cellDim,0,cellDim]:
            for y in [-cellDim,0,cellDim]:
                for z in [-cellDim,0,cellDim]:
                    # get position of unit that's being evaluated
                    i = cellPosX + x
                    j = cellPosY + y
                    k = cellPosZ + z
                    # if within boundary conditions
                    if i >= 0 and i < N:
                        if j >= 0 and j < N:
                            if k >= 0 and k < N:
                                # if cell is empty
                                if T[i,j,k] == 0:
                                    # assign it to be an "open space"
                                    openSpace = [i,j,k]
                                    availableUnits.append(openSpace)
        return availableUnits
    
    # returns the new position of the cell
    def brownianMove(self,T,N,g):
        import random as rand
            
        # get all empty units around cell
        availableUnits = self.availUnits(T,N)
        # if there are available units (not saturation condition)
        if len(availableUnits) > 1:
            # availableUnits contains initial placeholder
            # remove all placeholder values
            availableUnits = availableUnits[1:]
            # randomly choose (uniform distribution) a unit from availableUnits
            newCellPos = rand.choice(availableUnits)
            # move cell c to the new unit
            self.position = newCellPos
        return self
    
    # returns a new cell created at an empty location if the cell is healthy
    # if no available locations, return self
    def cellRepl(self,T,N,g):
        import random as rand
        import uuid as uuid
            
        # get all empty units around cell
        availableUnits = self.availUnits(T,N)
        # if there are available units (not saturation condition)
        if len(availableUnits) > 1:
            # availableUnits contains initial placeholder
            # remove placeholder
            availableUnits = availableUnits[1:]
            # randomly choose (uniform distribution) a unit from availableUnits
            newCellPos = rand.choice(availableUnits)
            # determine new UUID
            newUUID = uuid.uuid4()
            # create a cell at the new unit
            return Cell(newUUID,newCellPos,1,g,0,0,0)
        else:
            # if no availableUnits - return original cell
            return self
    
    # determines SSBs and DSBs from ion and electrons
    def cellRad(self,g,radGen,radData,radType):
        import random as rand
        import numpy as np
        
        radData_all = radData
        radData = radData = np.zeros([1,6],dtype = float)
        
        # for deep space, getting only the traversals that occur at this generation
        if radType == "Deep Space":
            for radEvent in radData_all:
                radGen = int(radEvent[6])
                if radGen == g:
                    currRadData = radEvent[0:6]
                    radData = np.vstack([radData,currRadData])
        
        # direct radiation only affects the current generation
        if len(radData)>1:
            
            # get cell information
            currPos = self.position
            numSSBs = self.numSSBs
            numDSBs = self.numDSBs
            
            cellHitBool = 0
            nucleusHitBool = 0
            
            # initialization of data arrays
            SSB_map = [[0,0,0,0]]
            DSB_map = [[0,0,0,0]]
            
            # for every radiation energy deposition
            for radEvent in radData:
                radPos = radEvent[0:3]
                radEnergy = radEvent[3]
                radType = radEvent[4]
                radSource = radEvent[5]
                
                # calculate radiation dose
                #nucleus hit - determine if non-complex damage takes place
                #dose = energy/mass (J/kg)
                #assuming mass nucleus = mass water (1 m3 = 1000 kg)
                #1 m3 = 1e18 um3 -> 1 um^3 = 1000/1e18 = 1e-15 kg
                #voxels = 20 nm
                #1 m3 = 1e27 nm3 -> 1 nm^3 = 1000/1e27 = 1e-24 kg
                #20 nm3 = 20e-24kg
                #mass nucleus = 2e-15 kg
                #1 eV = 1.60218e-19 J -> eV/6.242e18 = J
                energy_J = radEnergy/6.242e18
                dose_Gy = energy_J/(20e-28) # mass of water in 20nm3 voxel
                
                # if event is within cubic location that holds cell
                if radPos[0] <= currPos[0]+2 and radPos[0] >= currPos[0]-2:
                    if radPos[1] <= currPos[1]+2 and radPos[1] >= currPos[1]-2:
                        if radPos[2] <= currPos[2]+2 and radPos[2] >= currPos[2]-2:
                            randHittingCell = rand.uniform(100,0)
                            # Volume of sphere d=4: 33.51. Volume of cube s=4: 64
                            # prob of hitting sphere within cube: 52.36%
                            if randHittingCell <= 52.36:
                                cellHitBool = 1
                                randHittingNucleus = rand.uniform(100,0)
                                # nucleus ~7.407% of size of cell
                                if randHittingNucleus <= 7.4:
                                    nucleusHitBool = 1
                
                if nucleusHitBool == 1:
                    if radType == 1: #electron
                        newSSB = [radPos[0],radPos[1],radPos[2],1]
                        SSB_map = np.vstack([SSB_map,newSSB])
                    elif radType == 2: #ion
                        # nanovolume energy deposition - Plante 25-35 DSB/cell/Gy
                        numDSB_Ion = 35*dose_Gy
                        newDSB = [radPos[0],radPos[1],radPos[2],numDSB_Ion]
                        DSB_map = np.vstack([DSB_map,newDSB])
            
                # for every damage site
                for SSB in SSB_map:
                    numSSBs = numSSBs + SSB[3]
                for DSB in DSB_map:
                    numDSBs = numDSBs + DSB[3]
            
            # adjust health to reflect damage
            if numSSBs > 0:
                self.numDSBs = numDSBs
                self.numSSBs = numSSBs
                self.health = 2
            elif numDSBs > 0:
                self.numDSBs = numDSBs
                self.numSSBs = numSSBs
                self.health = 2
            
            return self
    
    # determines SSBs and DSBs from ions and electrons for the rad51 cell type
    def cellRad_rad51(self,g,radGen,radData,radType):
        import random as rand
        import numpy as np
        
        radData_all = radData
        radData = radData = np.zeros([1,6],dtype = float)
        
        # for deep space, getting only the traversals that occur at this generation
        if radType == "Deep Space":
            for radEvent in radData_all:
                radGen = int(radEvent[6])
                if radGen == g:
                    currRadData = radEvent[0:6]
                    radData = np.vstack([radData,currRadData])
        
        
        # if there's data in radData (if there has been a radiation traversal)
        if len(radData)>1:
            # get current cell information
            currPos = self.position
            numSSBs = self.numSSBs
            numDSBs = self.numDSBs
            
            cellHitBool = 0
            nucleusHitBool = 0
            
            # initializing data arrays
            SSB_map = [[0,0,0,0]]
            DSB_map = [[0,0,0,0]]
            
            for radEvent in radData:
                radPos = radEvent[0:3]
                radEnergy = radEvent[3]
                radType = radEvent[4]
                
                # calculate radiation dose
                #nucleus hit - determine if non-complex damage takes place
                #dose = energy/mass (J/kg)
                #assuming mass nucleus = mass water (1 m3 = 1000 kg)
                #1 m3 = 1e18 um3 -> 1 um^3 = 1000/1e18 = 1e-15 kg
                #voxels = 20 nm
                #1 m3 = 1e27 nm3 -> 1 nm^3 = 1000/1e27 = 1e-24 kg
                #20 nm3 = 20e-24kg
                #mass nucleus = 2e-15 kg
                #1 eV = 1.60218e-19 J -> eV/6.242e18 = J
                energy_J = radEnergy/6.242e18
                dose_Gy = energy_J/(20e-28) # mass of water in 20nm3 voxel
                
                # if event is within cubic location that holds cell
                if radPos[0] <= currPos[0]+2 and radPos[0] >= currPos[0]-2:
                    if radPos[1] <= currPos[1]+2 and radPos[1] >= currPos[1]-2:
                        if radPos[2] <= currPos[2]+2 and radPos[2] >= currPos[2]-2:
                            randHittingCell = rand.uniform(100,0)
                            # Volume of sphere d=4: 33.51. Volume of cube s=4: 64
                            # prob of hitting sphere within cube: 52.36%
                            if randHittingCell <= 52.36:
                                cellHitBool = 1
                                randHittingNucleus = rand.uniform(100,0)
                                # nucleus ~7.407% of size of cell
                                if randHittingNucleus <= 7.4:
                                    nucleusHitBool = 1
                
                if nucleusHitBool == 1:
                    if radType == 1: #electron
                        newSSB = [radPos[0],radPos[1],radPos[2],1]
                        SSB_map = np.vstack([SSB_map,newSSB])
                    elif radType == 2: #ion
                        # nanovolume energy deposition - Plante 25/35 DSB/cell/Gy
                        numDSB_Ion = 35*dose_Gy
                        newDSB = [radPos[0],radPos[1],radPos[2],numDSB_Ion]
                        DSB_map = np.vstack([DSB_map,newDSB])
            
                   
                for SSB in SSB_map:
                    numSSBs = numSSBs + SSB[3]
                for DSB in DSB_map:
                    numDSBs = numDSBs + DSB[3]
            
            
            
            if numSSBs > 0:
                self.numDSBs = numDSBs
                self.numSSBs = numSSBs
                self.health = 3
            elif numDSBs > 0:
                self.numDSBs = numDSBs
                self.numSSBs = numSSBs
                self.health = 3
            
            return self
     
    def cellROS(self,g,radGen,ROSData):
        import random as rand
        import numpy as np
        
        
        currPos = self.position
        numSSBs = self.numSSBs
        OH = 0
        H2O2 = 0
        nucleus_OH = 0
        
        cellHitBool = 0
        nucleusHitBool = 0
        apopOccurs = 0
    
        
        for ROSEvent in ROSData:
            ROSPos = ROSEvent[0:3]
            yield_H2O2 = ROSEvent[3]
            yield_OH = ROSEvent[4]
            cellHit = ROSEvent[5]
            
            # if event is within cubic location that holds cell
            if ROSPos[0] <= currPos[0]+2 and ROSPos[0] >= currPos[0]-2:
                if ROSPos[1] <= currPos[1]+2 and ROSPos[1] >= currPos[1]-2:
                    if ROSPos[2] <= currPos[2]+2 and ROSPos[2] >= currPos[2]-2:
                        randHittingCell = rand.uniform(100,0)
                        # Volume of sphere d=4: 33.51. Volume of cube s=4: 64
                        # prob of hitting sphere within cube: 52.36%
                        if randHittingCell <= 52.36:
                            cellHitBool = 1
                            randHittingNucleus = rand.uniform(100,0)
                            # nucleus ~7.407% of size of cell
                            if randHittingNucleus <= 7.4:
                                nucleusHitBool = 1
            
            if nucleusHitBool == 1:
                nucleus_OH = nucleus_OH + yield_OH
            if cellHitBool == 1:
                OH = OH + yield_OH
                H2O2 = H2O2 + yield_H2O2
        
        damaged = 0
        
        # risk of apoptosis from H2O2
        # data from Madeo 1999
        # >5 mM -> decrease from 3mM
        # 3 mM H2O2 -> 70% apoptotic death
        # 1 mM H2O2 -> 40% apoptotic death
        # .3 mM H2O2 -> 20% apoptotic death
        
        #interpolated data based on data points
        apopDat = np.genfromtxt('Apoptosis_Data.txt',skip_header=1)
        
        Avogadro = 6.022e23
        H2O2_concentration = round(H2O2/(Avogadro*0.001*27),1)
        
        if H2O2_concentration >= 0.3:
            apopNum = (H2O2_concentration*10)-2
            apopProb = apopDat.data.Interpolation(apopNum,2)
            
            apoptosisRand = rand.uniform(100,0)
            if (apoptosisRand <= apopProb):
                apopOccurs = 1
                damaged = 1
        
        # complex DNA damage from base damage via OH
        if nucleusHitBool == 1 and OH > 1:
            damaged = 1
            numSSBs = numSSBs + OH
            self.numSSBs = numSSBs
        
        #outputs
        if damaged == 1:
            if apopOccurs == 1:
                self.health = 3
            elif nucleusHitBool == 1:
                self.health = 2
        
        return self
    
    def cellROS_rad51(self,g,radGen,ROSData):
        import random as rand
        import numpy as np
        
        currPos = self.position
        numSSBs = self.numSSBs
        OH = 0
        H2O2 = 0
        nucleus_OH = 0
        
        cellHitBool = 0
        nucleusHitBool = 0
        apopOccurs = 0
    
        
        for ROSEvent in ROSData:
            ROSPos = ROSEvent[0:3]
            yield_H2O2 = ROSEvent[3]
            yield_OH = ROSEvent[4]
            cellHit = ROSEvent[5]
            
            # if event is within cubic location that holds cell
            if ROSPos[0] <= currPos[0]+2 and ROSPos[0] >= currPos[0]-2:
                if ROSPos[1] <= currPos[1]+2 and ROSPos[1] >= currPos[1]-2:
                    if ROSPos[2] <= currPos[2]+2 and ROSPos[2] >= currPos[2]-2:
                        randHittingCell = rand.uniform(100,0)
                        # Volume of sphere d=4: 33.51. Volume of cube s=4: 64
                        # prob of hitting sphere within cube: 52.36%
                        if randHittingCell <= 52.36:
                            cellHitBool = 1
                            randHittingNucleus = rand.uniform(100,0)
                            # nucleus ~7.407% of size of cell
                            if randHittingNucleus <= 7.4:
                                nucleusHitBool = 1
            
            if nucleusHitBool == 1:
                nucleus_OH = nucleus_OH + yield_OH
            if cellHitBool == 1:
                OH = OH + yield_OH
                H2O2 = H2O2 + yield_H2O2
        
        damaged = 0
        
        # risk of apoptosis from H2O2
        # data from Madeo 1999
        # >5 mM -> decrease from 3mM
        # 3 mM H2O2 -> 70% apoptotic death
        # 1 mM H2O2 -> 40% apoptotic death
        # .3 mM H2O2 -> 20% apoptotic death
        
        #interpolated data based on data points
        apopDat = np.genfromtxt('Apoptosis_Data.txt',skip_header=1)
        
        Avogadro = 6.022e23
        H2O2_concentration = round(H2O2/(Avogadro*0.001*27),1)
        
        if H2O2_concentration >= 0.3:
            apopNum = (H2O2_concentration*10)-2
            apopProb = apopDat.data.Interpolation(apopNum,2)
            
            apoptosisRand = rand.uniform(100,0)
            if (apoptosisRand <= apopProb):
                apopOccurs = 1
                damaged = 1
        
        # complex DNA damage from base damage via OH results in death
        if OH > 1:
            damaged = 1
        
        #outputs - any form of damage results in death to rad51 cells
        if damaged == 1:
           self.health = 3
        
        return self
        
    def cellRepair(self,g):
        import random as rand
        
        # direct radiation only affects the current generation
        numSSBs = self.numSSBs
        numDSBs = self.numDSBs
        initHealth = self.health
        
        if numSSBs > 0:
            numSSBs = numSSBs - 3
            if numSSBs <= 0:
                numSSBs = 0
            else:
                health = 2
        elif numDSBs > 0:
            rand = rand.uniform(0,100)
            if rand < 50:
                numDSBs = numDSBs-1
                if numDSBs <= 0:
                    numDSBs = 0
                else: 
                    health = 2
            else:
                health = 3
    
        if numSSBs == 0 and numDSBs == 0:
            health = 1
    
        if initHealth != health or health == 2:
            self.health = health
            return self