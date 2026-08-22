
"""
Created on Jun 30 08:42:26 2022

Constructing an alamarBlue metabolic dye dynamics on yeast under space radiation with Michealis Menten Kinetics.

Assume 240 min from log growth curve analysis.


0 Gy 2.5 Gy and 30 Gy average experimental values >>>> <<<

Damaged cells incorporation.
@author: Daniel Palacios
"""

import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import log
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score


def AlamarblueMechanics(results, var, title):
    # v = d/dt([Clear])  == V_max ([Pink]/ K_M + [Pink])
    Healthy = results.loc[results["Health"] == 1]
    Unhealthy = results.loc[results["Health"] == 2]
    # Compute growth curve
    Growth_curve = Healthy['Generation'].value_counts()

    Growth_curve2 = Unhealthy['Generation'].value_counts()

    Growth_curve = np.array(Growth_curve)
    e = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    g = pd.DataFrame(e, columns=['Genk'])
    g['ncelldamaged'] = np.zeros(len(e))

    for i in Growth_curve2.index:
        g.at[int(i), 'ncelldamaged'] = Growth_curve2[i]

    Growth_curve2 = g['ncelldamaged'].to_numpy()

    # Consider time steps:
    Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
    # t = Generations >> <<
    # t = np.linspace(0,1.25,1000)
    t = Generations
    # print(t)
    t = [val for val in t for _ in (0, 1)]

    for i in range(len(t)):
        if (i % 2) == 0:
            t[i] = t[i]
        else:
            t[i] = t[i] + 0.5

    Growth_curve = [val for val in Growth_curve for _ in (0, 1)]
    Growth_curve2 = [val for val in Growth_curve2 for _ in (0, 1)]

    t = np.array(t)

    # print(t)
    Growth_curve = np.array(Growth_curve)
    Growth_curve2 = np.array(Growth_curve2)
    # Initial concentrations
    Blue_0 = 10000
    Pink_0 = 100
    Clear_0 = 100

    Blue = []
    Pink = []
    Clear = []

    Blue = np.append(Blue, Blue_0)
    Pink = np.append(Pink, Pink_0)
    Clear = np.append(Clear, Clear_0)

    # Michealis Parameters

    V1_max = var[0]
    V2_max = var[1]
    K1_M = var[2]
    K2_M = var[3]
    k = var[4]
    v = [V1_max * (Blue_0 / (K1_M + Blue_0))]
    v2 = [V2_max * (Pink_0 / (K2_M + Pink_0))]

    for i in range(len(t) - 1):
        vn = V1_max * (Blue[i] / (K1_M + Blue[i]))  # <- uptake concentration rate

        ############################################## Idea
        # multiply v (rate) for each cell assume V = /sum v_i
        vn = vn * Growth_curve[i] + k * vn * Growth_curve2[i]

        ##################3333
        dt = abs(t[i] - t[i + 1])

        dPink = vn * dt

        Pinkn = Pink[i] + dPink

        dBlue = -dPink

        Bluen = Blue[i] + dBlue

        Blue = np.append(Blue, Bluen)

        v = np.append(v, vn)

        v2n = V2_max * (Pink[i] / (K2_M + Pink[i]))

        ################ Same for v2n
        v2n = v2n * Growth_curve[i] + k * v2n * Growth_curve2[i]
        ########
        v2 = np.append(v2, v2n)

        dClear = v2n * dt

        Clearn = Clear[i] + dClear

        Clear = np.append(Clear, Clearn)

        dPink = -dClear

        Pinknn = Pinkn + dPink

        Pink = np.append(Pink, Pinknn)

    # Visualize total current concentration
    T_Con = []
    for i in range(len(Blue)):
        T_C = Blue[i] + Pink[i] + Clear[i]
        T_Con = np.append(T_Con, T_C)
    # Blue / T_Con Fractionals >> <<

    Blue = Blue / T_Con
    Pink = Pink / T_Con
    Clear = Clear / T_Con

    t = t * GENCONVER / 60

    return Blue, Pink, t

def ExperimentalConencentrations(data):
    ODratio_red = 1.04
    ODratio_green = 1.06

    P_ratio = 0.06
    B_ratio = 0.7

    A690 = data['A690'].to_numpy()
    A570 = data['A570'].to_numpy()
    A600 = data['A600'].to_numpy()

    OD600 = A690 * ODratio_red
    OD570 = A690 * ODratio_green

    B_C = (A600 - OD600 - P_ratio * (A570 - OD570))/(1 - P_ratio * B_ratio)

    P_C = A570 - OD570 - B_ratio * B_C
    Time = data['Time'].to_numpy()
    
    return B_C, P_C, Time


def accuracy_ML(Experimental_B, Predicted_B, Experimental_P, Predicted_P):
    delta = sum((Experimental_B - Predicted_B) ** 2)
    delta1 = sum((Experimental_P - Predicted_P) ** 2)
    delta = delta + delta1
    # accuracy = classification_report(Experimental_B,Predicted_B).precision
    accuracy = delta
    return accuracy

namesk = ['Time', 'A570', 'A600', 'A690', 'A750']
namess = ["Generation", "x", "y", "z", "Health"]
# dir r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER
results = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueParametrization\0Gy.txt',
                          names = namess) #0 Gy
results2 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\Results\07-26-22_10-49\2.5Gy.txt',
                          names = namess) #2.5 Gy
results3 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\Results\07-26-22_11-46\30Gy.txt',
                      names = namess)

data1 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdataWTKGy.csv',
                   names = namesk) # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch

data2 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdataWT25Gy.csv',
                   names = namesk) # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch

data3 = pd.read_csv(r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdataWT30Gy.csv',
                   names = namesk) # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch


TRUNCATED = 15
GENCONVER = 198
B_C, P_C, Time = ExperimentalConencentrations(data1.head(TRUNCATED))

# assume After 30 hours sytem remains in equilibrium since model does not make anystatement of additional mechanics >><<
B_C = np.append(B_C, B_C[-1])
P_C = np.append(P_C, P_C[-1])
Time = np.append(Time, 80)

# Need to interpolate data to match generation dimensions and values *********************
B_Ct = interp1d(Time, B_C, kind = 'linear')
P_Ct = interp1d(Time, P_C, kind = 'linear')

# Data with generation time ******
Healthy = results.loc[results["Health"] == 1]
Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
# Generation to hrs ->> Conversion
Generations_t = Generations * GENCONVER / 60


# Data that matches dimensions and time  ******
B_Ci = B_Ct(Generations_t)
P_Ci = P_Ct(Generations_t)

# Normalize data so total initial concentration = 1 
B_Ci = B_Ci / (B_Ci[0] + P_Ci[0])
P_Ci = P_Ci / (B_Ci[0] + P_Ci[0])

# Best 240 min curves found after rank 4 tensor gridsearch


color1 = "#d31e25"
color2 = "#d7a32e"
color3 = "#369e4b"
color4 = "#5db5b7"
color5 = "#31407b"
color6 = "#d1c02b"
color7 = "#8a3f64"
color8 = "#4f2e39"



# 2.5 Gy
# Accuracy k = 0 -> 0.13735532784594479
# Accuracy k = 1 -> 0.13739708950797597
# accuracy k = 1/2 -> 0.13737459939217403

# 30 Gy
# accuracy k = 1/2 -> 0.1315799538193826

# print('Accuracies:')
# print(accuracy_ML(B_Ci[:8], B1[:8], P_Ci[:8], P1[:8]))

#
# plt.scatter(Generations_t, B_Ci, marker = '>', color = color1, label = 'Intrapolated Blue')
# plt.scatter(Generations_t, P_Ci, marker = '^', color = color2, label = 'Intrapolated Pink')

colorb = '#42329a'
colorp = '#e54da7'



GRIDSEARCH = False

if GRIDSEARCH:
    v1 = 0.95
    v2 = np.linspace(0.5, 2.5, 10)
    # v3 = np.linspace(100, 1000, 10)
    # v4 = np.linspace(4000, 9000, 10)
    v3 = 500
    v4 = np.linspace(1000, 10000, 10)
    v1 = np.repeat(v1, 10)
    v3 = np.repeat(v3, 10)
    acc = np.zeros((10,10,10,10))

    for i in range(len(v1)):
        for j in range(len(v2)):
            for k in range(len(v3)):
                for m in range(len(v4)):

                    B, P, t = AlamarblueMechanics(results,[v1[i], v2[j], v3[k], v4[m], 0], 'k')

                    acc[i][j][k][m] = accuracy_ML(B_Ci[:8], B[:8], P_Ci[:8], P[:8])

    print('Min error')

    MIN_LIST = np.unravel_index(np.argmin(acc), (10,10,10,10))
    print(MIN_LIST)

#MIN_LIST = [0.95,1.95,1750,10000]
MIN_LIST = [0.75,1.65,500,8000]
# print(v2[3])
# print(v4[2])
#MIN_LIST = [0.95,1.166666,500,3000]
# print(MIN_LIST)
# print(v1[MIN_LIST[0]])
# print(v2[MIN_LIST[1]])
# print(v3[MIN_LIST[2]])
# print(v4[MIN_LIST[3]])

plt.figure(1)
B1, P1, t = AlamarblueMechanics(results,[MIN_LIST[0],MIN_LIST[1],MIN_LIST[2],MIN_LIST[3], 1/2], 'k')


B_C, P_C, Time = ExperimentalConencentrations(data1.head(TRUNCATED))
B_C = B_C / (B_C[0] + P_C[0])
P_C = P_C / (B_C[0] + P_C[0])
plt.scatter(Time, B_C, marker = '*', color = colorb, label = 'Experimental Blue')
plt.scatter(Time, P_C, marker = "+", color = colorp, label = 'Experimental Pink')
plt.plot(t, B1, color='#42329a', linestyle='-.', label='Predicted Blue')
plt.plot(t, P1, color='#e54da7', linestyle='--', label='Predicted Pink')
plt.xlim([0, TRUNCATED])
plt.xlabel('Time [hrs]')
plt.title('0 Gy aB concentration fractions')
plt.legend()

# [1.4777777777777779, 1.0888888888888888, 6444.444444444444, 3427.777777777778,
#                                           1 / 2]

plt.show()
