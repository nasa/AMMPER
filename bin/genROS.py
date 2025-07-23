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
"""
We will incorporate a green function propagator in the form of a maxwell poisson distribution according to the 
radius of a sphere surrounding each radiation event.

The green function propagator expands over time i.e. the sphere should increase in radius over time following
the maxwell possion distribution. 

Note that right now ROS data refers to the position of radiation, the concentrations of H2O2 and OH, and
boolean value of cellhit.

Note inputs right now is the radiation position and energies. 

Step 1: Incorporate green function propagator without time dependence, we would need to loop over generations to generate
ROS information, then update cell health, and cycle again over the next generation to generate new ROS information, 
and new health information. 

Challenge: How to do lattice and green function propagator? Maybe easy solution is to use lattice spacing 
units for the function propagator, instead of a continuous one, a discrete one. 

> So we have the concentration quantity so far, we can take this quantity and distributed in a green function propagator 
with discrete values

Expected problem: If AMMPER resolution is not small enough, the detail of green function propagator will be bad implementation



@Daniel
"""


#
# fig, ax = plt.subplots(1, 1)
#
# x = np.linspace(maxwell.ppf(0.01),
#                 maxwell.ppf(0.99), 100)
#
# ax.plot(x, maxwell.pdf(x, loc = 0, scale = 1.2),
#        'r-', lw=1, alpha=1, label='maxwell pdf')
#
# ax.plot(x, maxwell.pdf(x, loc = 0, scale =1),
#        'r--', lw=1, alpha=1, label='maxwell pdf')


def genROS(radData, cells):
    import numpy as np
    import pandas as pd

    from scipy.stats import skewnorm
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.stats import maxwell
    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    # FOR TESTING consider 100 radiation points
    # radData = radData[:4000, :]
    # For now consider 10 lattice units

    #H2O2 few micrometers
    rxn = np.array([-4,-3,-2,-1,0,1,2,3,4])
    ryn = np.array([-4,-3,-2,-1,0,1,2,3,4])

    #OH Few 10^-9
    # rxn2 = np.array([-1, 0, 1])
    # ryn2 = np.array([-1, 0, 1])

    rx, ry = np.meshgrid(rxn, ryn)
    pos = np.dstack((rx, ry))
    #
    # rx2, ry2 = np.meshgrid(rxn2, ryn2)
    # pos2 = np.dstack((rx2, ry2))

    # Inputs for matrix controls time, and other quantities
    # Here we should change inputs for H2O2 and OH physical parameters
    rv = multivariate_normal([0, 0], [[4, -1], [-1, 4]])
    # rv2 = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    # fractional Concentration position value

    f = rv.pdf(pos)
    # f2 = rv2.pdf(pos2)
    #
    # Concentration distrubution is :::: >>>> ::::
    # maxwell.pdf(x, loc = 0, scale = 1.2)
    # Note is normalize where area under the curve = 1 = Total concentration of H2O2, OH

    # Hence we have one to one realtionship, give the position we get a probability amplitud
    # with the fraction of the total concentration >>> <<<< >>> <<<

    # The output is the concentration at each lattice point of ROS. (note we need to superpose them ,add them, in the final
    # output )

    # Is this good enough? For calculation should be good output, for figures should be bad output.

    ###################### SCALE ===== GENERATIONS (TIME) #######################################
    # Fractional (note scale is arbitrary right now, in the future it will control the time >>> <<< )
    # Note this should be different since OH has less life time as H2O2

    # f = maxwell.pdf(r, loc=0, scale=1.3)
    # f2 = maxwell.pdf(r, loc=0, scale=1.1)

    n = len(radData)
    # ROS yields (Plante Radiation Chemistry, from three.jsc.nasa.gov)
    # primary yields in [molecules/100 eV]
    G_H2O2 = 0.7
    G_OH = 2.5

    ROSData = np.empty((0,6))
    # all_data = []
    for i in range(n):
        cellHit = 0  # boolean for whether or not cell is hit
        radPos = [radData[i, 0], radData[i, 1], radData[i, 2]]
        energy = radData[i, 3]
        yield_H2O2 = G_H2O2 * energy
        yield_OH = G_OH * energy

        # Concentrations values matrix
        C_H2O2 = f * yield_H2O2
        # C_OH = f2 * yield_OH
        # Note they different sizes >>> <<<----------------------------
        for c in cells:
            currPos = c.position
            if radPos[0] <= currPos[0] + 2 and radPos[0] >= currPos[0] - 2:
                if radPos[1] <= currPos[1] + 2 and radPos[1] >= currPos[1] - 2:
                    if radPos[2] <= currPos[2] + 2 and radPos[2] >= currPos[2] - 2:
                        cellHit = 1
                        # print("cell was hit!")
                        # print(cellHit)

        # Keep initial results for now, can be deleted later

        #ROSDataEntry = [radPos[0], radPos[1], radPos[2], yield_H2O2, yield_OH , cellHit]
        ROSDataEntry = [radPos[0], radPos[1], radPos[2], 0, yield_OH, cellHit]

        #ROSDataEntry = ROSDataEntry.reshape((1,6))

        # OH data entry::::
        Posx = radPos[0]
        Posy = radPos[1] + rxn
        Posz = radPos[2] + ryn

        Px = []
        Py = []
        Pz = []
        # Cz = []
        Cz2 = []

        for i in range(len(Posy)):
            for j in range(len(Posz)):
                Pxi = Posx
                Pyi = Posy[i]
                Pzi = Posz[j]
                # Czijp_1 = C_OH[i][j]
                Czijp_2 = C_H2O2[i][j]

                Px = np.append(Px, Pxi)
                Py = np.append(Py, Pyi)
                Pz = np.append(Pz, Pzi)
                # Cz = np.append(Cz, Czijp_1)
                Cz2 = np.append(Cz2, Czijp_2)


        ######### Create matrix ######## with results
        # Posx = radPos[0] + r
        # Posy = radPos[1] + r
        # Posz = radPos[2] + r
        #
        # Posxn = radPos[0] - r
        # Posyn = radPos[1] - r
        # Poszn = radPos[2] - r
        C_H2O2 = Cz2
        # C_OH = Cz
        #cell hit -> 0
        cellHit = np.zeros((len(Px),), dtype = int)
        # Need to fix dimensions >> ><<< >>> <<< >>> <<<
        ROSDiffusionEntry = np.vstack([Px, Py, Pz, C_H2O2, cellHit, cellHit]).T
        #
        # ROSData = np.vstack([ROSData, ROSDataEntry])
        # #
        # ROSData = np.vstack([ROSData, ROSDiffusionEntry])

        # ROSDiffusionEntry = np.vstack([Px, Py, Pz, C_H2O2, cellHit, cellHit]).T
        #
        # ROSData = np.vstack([ROSData, ROSDataEntry])
        #
        # ROSData = np.concatenate((ROSData, ROSDiffusionEntry), axis = 0)

        # # Append ROSDataEntry to the all_data list
        # all_data.append(ROSDataEntry)
        #
        # # Extend the all_data list with ROSDiffusionEntry (because it's Nx6)
        # all_data.extend(ROSDiffusionEntry.tolist())  # Convert array to list for extending

        # Ensuring the shapes
        # ROSData = ROSData.reshape(1, 6)
        ROSDataEntry = np.array(ROSDataEntry).reshape(1, 6)


        # Stacking
        current_stack = np.vstack((ROSDataEntry, ROSDiffusionEntry))

        # Updating
        ROSData = np.vstack((ROSData, current_stack))


    # remove placeholder
    # ROSData = np.delete(ROSData, (0), axis=0)

    # Convert the collected data into an array
    # ROSData = np.array(all_data)

    return (ROSData)
    # ROS DATA OUTPUT FORMAT >>> <<< >>> [Posx, Posy, Posz, C_H2O2, C_OH, cellHit]
