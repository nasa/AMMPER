
"""
Created on Tuesday Jul 13 08:42:26 2022
Generating random Gamma radiation events without shielding >> <<
@author: Daniel Palacios
"""

def GammaRadGen(dose):
    import numpy as np
    import pandas as pd
    from random import randint

    # radData = pd.Dataframe(columns = ["X", "Y", "Z", "energy"])

    if dose == 0:
        print("ERROR: Gamma model does not run 0 Gy doses, run Ion model for 0 Gy instead ...")
    # N is size of all space in this case is 64
    N = 64
    min_y = 0
    min_z = 0
    max_y = 64
    max_z = 64
    k =  100 # is arbitrary for now, need physical parameters to calculate the number of events in an area 64 x 64 micrometers
    # scan k = [1,2,3,4,5] see which one matches expected population ratio.
    # for given radiation dose of gamma rays, probably this parameter must be fitted with experimental data
    # energy = 100  # See literature around 100 keV per unit length g^-1 cm^2
    # Note that AMMPEr might be using eV units instead of kev ->
    energy = 100000
    
    n_Hits = dose * k
    n_Hits = int(round(n_Hits))
    radtensor = [0,0,0,0]
    for x in range(N):
        for i in range(n_Hits):
            random_Y = randint(min_y, max_y) 
            random_Z = randint(min_z, max_z)
            radn = [x, random_Y, random_Z, energy]
            radtensor = np.vstack((radtensor, radn))
            # To incorporate shielding if necessary (see documentation to see what mean by shielding)
            # need to create a list of previously chosen randy and randz, in the next for loop,
            # we need to restrict this choice. Something like: If randY and Z in list then do nothing
    radData = radtensor

    return radData


