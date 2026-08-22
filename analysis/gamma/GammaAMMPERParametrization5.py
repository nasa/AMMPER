# -*- coding: utf-8 -*-
"""
finding right paremeters to match with experimental data >>> Gamma radiation
@Daniel
"""


from scipy.stats import skewnorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import maxwell
import matplotlib.pyplot as plt


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# Blue -> Pink
# v = d/dt([Pink])  == V_max ([Blue]/ K_M + [Blue])

# def Growthcurves(results,var, title):
#     # v = d/dt([Clear])  == V_max ([Pink]/ K_M + [Pink])
#     var = var
#
#     Healthy = results.loc[results["Health"] == 1]
#
#     Unhealthy = results.loc[results["Health"] != 1]
#
#     # Compute growth curve
#     Growth_curve = Healthy['Generation'].value_counts()
#
#     Growth_curve2 = Unhealthy['Generation'].value_counts()
#     #
#
#     Growth_curve = np.array(Growth_curve)
#
#     Growth_curve2 = np.array(Growth_curve2)
#
#     Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
#     #
#     Generations2 = np.linspace(int(Unhealthy['Generation'].min()), int(Unhealthy['Generation'].max()),
#                                num=len(Unhealthy['Generation'].unique()))
#
#     Growth_curve = np.flip(Growth_curve)
#
#     Growth_curve2 = np.flip(Growth_curve2)
#
#     # Ratios
#     n = len(Growth_curve2)
#     Growth_curve2 = Growth_curve[-n:]/(Growth_curve2+Growth_curve[-n:]) # Acces last elements only with radiation damage since other ones are 0
#     # Percentage >>> Healthy / total
#     return Growth_curve2, Generations2

def Growthcurves(results,var, title):
    var = var

    Healthy = results.loc[results["Health"] != 3] # Alive healthy 1, damaged 2, dead 3

    Growth_curve = Healthy['Generation'].value_counts()

    Growth_curve = np.array(Growth_curve)
    Growth_curve = np.flip(Growth_curve)
    Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))

    #normalize
    Growth_curve = Growth_curve/Growth_curve[-1]
    return Growth_curve, Generations

namess = ["Generation", "x", "y", "z", "Health"]

results0 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_25\07-15-23_17-07\Gamma.txt',
                          names = namess)

results1 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_25\07-15-23_17-12\Gamma.txt',
                          names = namess)

results2 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_25\07-15-23_17-17\Gamma.txt',
                          names = namess)


results3 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_250\07-15-23_17-07\Gamma.txt',
                          names = namess)

results4 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_250\07-15-23_17-12\Gamma.txt',
                          names = namess)

results5 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_250\07-15-23_17-17\Gamma.txt',
                          names = namess)

results0k = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_25k50\07-15-23_17-45\Gamma.txt',
                          names = namess)

results1k = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_25k50\07-15-23_17-46\Gamma.txt',
                          names = namess)

results2k = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPEROut_2023May\Results_Bulk_GAMMAFINAL\WT_25k50\07-15-23_17-47\Gamma.txt',
                          names = namess)


exp_data = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\gamma_expGrowth.csv')

var = "k"

Growth1, Gen1 = Growthcurves(results0, var, 'Gamma Basic WT 2.5 Gy k = 100')
Growth2, Gen2 = Growthcurves(results1, var, 'Gamma Basic WT 2.5 Gy')
Growth3, Gen3 = Growthcurves(results2, var, 'Gamma Basic WT 2.5 Gy')
n = min(len(Growth1), len(Growth2), len(Growth3))

Growth1r, Gen1r = Growthcurves(results3, var, 'Gamma Basic WT 2.5 Gy k = 1000')
Growth2r, Gen2r = Growthcurves(results4, var, 'Gamma Basic WT 2.5 Gy')
Growth3r, Gen3r = Growthcurves(results5, var, 'Gamma Basic WT 2.5 Gy')
nr = min(len(Growth1r), len(Growth2r), len(Growth3r))

Growth1k, Gen1k = Growthcurves(results0k, var, 'Gamma Basic WT 2.5 Gy k = 50')
Growth2k, Gen2k = Growthcurves(results1k, var, 'Gamma Basic WT 2.5 Gy')
Growth3k, Gen3k = Growthcurves(results2k, var, 'Gamma Basic WT 2.5 Gy')
nk = min(len(Growth1k), len(Growth2k), len(Growth3k))


plt.figure(1)
C_WT_1_mean = np.mean((Growth1[-n:], Growth2[-n:], Growth3[-n:]), axis = 0)
C_WT_1_std = np.std((Growth1[-n:], Growth2[-n:], Growth3[-n:]), axis = 0)

Cr_WT_1_mean = np.mean((Growth1r[-nr:], Growth2r[-nr:], Growth3r[-nr:]), axis = 0)
Cr_WT_1_std = np.std((Growth1r[-nr:], Growth2r[-nr:], Growth3r[-nr:]), axis = 0)

C_WT_1_meank = np.mean((Growth1k[-nk:], Growth2k[-nk:], Growth3k[-nk:]), axis = 0)
C_WT_1_stdk = np.std((Growth1k[-nk:], Growth2k[-nk:], Growth3k[-nk:]), axis = 0)

color1 = "#d31e25"
color2 = "#d7a32e"
color3 = "#369e4b"
color4 = "#5db5b7"
color5 = "#31407b"
color6 = "#d1c02b"
color7 = "#8a3f64"
color8 = "#4f2e39"

G = Gen1[-n:] * 205.6 / 60

plt.figure(1)
plt.errorbar(G, C_WT_1_mean, yerr = C_WT_1_std, xerr = None, marker = '*', color = color1, linestyle = '-.', label = 'Gamma Basic WT 2.5 Gy k = 100')

plt.errorbar(G, Cr_WT_1_mean, yerr = Cr_WT_1_std, xerr = None, marker = '*', color = color2, linestyle = '-.', label = 'Gamma Basic WT 2.5 Gy k = 1000')

plt.errorbar(G, C_WT_1_meank, yerr = C_WT_1_stdk, xerr = None, marker = '*', color = color1, linestyle = '-.', label = 'Gamma Basic WT 2.5 Gy k = 50')


plt.title("Gamma parametrization")
plt.legend()
plt.xlabel("Time [hr]")

x = np.arange(49)
y = exp_data["WT 2.5 Gy"]
print("Slope Comparison: >>>>>>>>>>>>>>>>>")
dy = (y[20] - y[5]) / (20 - 5)
print("Real data: 2.5 Gy")
print(dy)

print('Gamma Basic WT 2.5 Gy k = 100')
dy1 = (C_WT_1_mean[15] - C_WT_1_mean[9]) / (G[15] - G[9])
print(dy1)

print('Gamma Basic WT 2.5 Gy k = 1000')
dy2 = (Cr_WT_1_mean[15] - Cr_WT_1_mean[9]) / (G[15] - G[9])
print(dy2)

print('Gamma Basic WT 2.5 Gy k = 50')
dy1k = (C_WT_1_meank[15] - C_WT_1_meank[9]) / (G[15] - G[9])
print(dy1k)


plt.show()


# K = 100 hits 0.5 dose curve. at gen 10 >>> <<<< when radiation is applied.

############################
# time conversion aB curves:
# t =  t * 240 / 60