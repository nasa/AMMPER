
"""
Created on
@author: Daniel Palacios
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import ammper_paths as P  # noqa: E402  (resolves data/ and results/ paths)

import time
import numpy as np
import os
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
from sklearn.metrics import accuracy_score, mean_squared_error

def plot_growth_curves(t, Growth_curve_healthy, Growth_curve_unhealthy):
    """
    Plots healthy and unhealthy cell counts over generation (time) 
    for a single simulation run.
    """
    
    # Use simple colors since the context of dose/strain is not passed here
    COLOR_HEALTHY = 'blue'
    COLOR_UNHEALTHY = 'red'

    plt.figure(figsize=(10, 6))
    ax = plt.gca() # Get current axes

    if t is not None:
        # Plot healthy cells (solid line)
        ax.plot(t, Growth_curve_healthy, 
                linestyle='-', color=COLOR_HEALTHY, linewidth=2.5, 
                label='Healthy Cells', alpha=0.8)
        
        # Plot unhealthy cells (dashed line)
        ax.plot(t, Growth_curve_unhealthy, 
                linestyle='--', color=COLOR_UNHEALTHY, linewidth=2.5,
                label='Unhealthy Cells', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Generation (Adjusted Time Steps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cell Count', fontsize=12, fontweight='bold')
    ax.set_title('Cell Growth Curve (Healthy vs Unhealthy)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, fontsize=9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set reasonable limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.savefig('growth_curve.png', dpi=300)

def AlamarblueMechanics(results, var, title):
    # v = d/dt([Clear])  == V_max ([Pink]/ K_M + [Pink])
    Healthy = results.loc[results["Health"] == 1]
    print(f"length of healthy {len(Healthy)}")
    Unhealthy = results.loc[results["Health"] == 2]
    print(f"length of unhealthy {len(Unhealthy)}")
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

    plot_growth_curves(t, Growth_curve, Growth_curve2)
    #plot_average_growth_curves

    # print(f"growth curve (unhealthy) {Growth_curve2}")
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
    V3_max = var[2]
    K1_M = var[3]
    K2_M = var[4]
    K3_M = var[5]
    k = var[6]
    v = [V1_max * (Blue_0 / (K1_M + Blue_0))]
    v2 = [V2_max * (Pink_0 / (K2_M + Pink_0))]
    alpha0 = Pink_0 / K2_M
    pi0 = Clear_0 / K3_M

    v2 = ((V2_max * alpha0) - (V3_max * pi0)) / (1 + alpha0 + pi0)

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

        v2 = np.append(v2, v2n)

        alpha = Pink[i] / K2_M
        pi = Clear[i] / K3_M

        v2n = ((V2_max * alpha) - (V3_max * pi)) / (1 + alpha + pi)
        v2n = v2n * Growth_curve[i] + k * v2n * Growth_curve2[i]

        dPink = v2n * dt
        dClear = v2n * dt
        Clearn = Clear[i] + dClear
        Clear = np.append(Clear, Clearn)
        # dPink = -dClear
        Pinknn = Pinkn - dPink
        Pink = np.append(Pink, Pinknn)

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
    accuracy = delta
    return accuracy


def main_DP(name, data1, data1_std):
    P_m = []
    B_m = []

    for root, dirs, files in os.walk(
            P.bulk_aB() + '/' + str(
                    name)):
        # Ensure there's at least one file before processing
        if len(files) > 1:
            namess = ["Generation", "x", "y", "z", "Health"]
            resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)
            #Bi, Pi, t = AlamarblueMechanics(resultsi, [0.75, 1.65, 500, 8000, 1 / 2, 0, 1000000], 'k') # default manual
            Bi, Pi, t = AlamarblueMechanics(resultsi, [0.7799990799515666, 1.679928455577914, 0.10002078747628415, 450, 6601, 9994, 0.5], 'k') # optimal
            B_m.append(Bi)
            P_m.append(Pi)

    # Check if B_m and P_m are empty before stacking
    if B_m:
        B_m = np.stack(B_m, axis=0)
        std_B = np.std(B_m, axis=0)
        nB = np.shape(B_m)[0]
        std_B = std_B / np.sqrt(nB)
    else:
        B_m = []
        std_B = []

    if P_m:
        P_m = np.stack(P_m, axis=0)
        std_P = np.std(P_m, axis=0)
        nP = np.shape(P_m)[0]
        std_P = std_P / np.sqrt(nP)
    else:
        P_m = []
        std_P = []

    B1 = np.mean(B_m, axis=0)
    P1 = np.mean(P_m, axis=0)

    data1p = pd.DataFrame(data1[['A570', 'A600', 'A690', 'A750']].values + data1_std[['A570', 'A600', 'A690', 'A750']].values, columns=['A570', 'A600', 'A690', 'A750'])
    data1n = pd.DataFrame(data1[['A570', 'A600', 'A690', 'A750']].values - data1_std[['A570', 'A600', 'A690', 'A750']].values, columns=['A570', 'A600', 'A690', 'A750'])
    data1p['Time'] = data1['Time']
    data1n['Time'] = data1['Time']

#
    B_C, P_C, Time = ExperimentalConencentrations(data1.head(TRUNCATED))
    B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p.head(TRUNCATED))
    B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n.head(TRUNCATED))
    # assume After 30 hours sytem remains in equilibrium since model does not make any statement of additional mechanics >><<
    B_C = np.append(B_C, B_C[-1])
    P_C = np.append(P_C, P_C[-1])
    Time = np.append(Time, 80)

    # Need to interpolate data to match generation dimensions and values *********************
    B_Ct = interp1d(Time, B_C, kind = 'linear')
    P_Ct = interp1d(Time, P_C, kind = 'linear')

    B_Cp = np.append(B_Cp, B_Cp[-1])
    P_Cp = np.append(P_Cp, P_Cp[-1])

    B_Ctp = interp1d(Time, B_Cp, kind = 'linear')
    P_Ctp = interp1d(Time, P_Cp, kind = 'linear')

    B_Cn = np.append(B_Cn, B_Cn[-1])
    P_Cn = np.append(P_Cn, P_Cn[-1])

    # Need to interpolate data to match generation dimensions and values *********************
    B_Ctn = interp1d(Time, B_Cn, kind = 'linear')
    P_Ctn = interp1d(Time, P_Cn, kind = 'linear')


    # Data with generation time ******
    Healthy = resultsi.loc[resultsi["Health"] == 1]
    Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
    # Generation to hrs ->> Conversion
    Generations_t =  Generations * GENCONVER / 60


    # Data that matches dimensions and time  ******
    B_Ci = B_Ct(Generations_t)
    P_Ci = P_Ct(Generations_t)

    # Normalize data so total initial concentration = 1
    B_Ci = B_Ci / (B_Ci[0] + P_Ci[0])
    P_Ci = P_Ci / (B_Ci[0] + P_Ci[0])

    B_Cip = B_Ctp(Generations_t)
    P_Cip = P_Ctp(Generations_t)

    # Normalize data so total initial concentration = 1
    B_Cip = B_Cip / (B_Cip[0] + P_Cip[0])
    P_Cip = P_Cip / (B_Cip[0] + P_Cip[0])

    B_Cin = B_Ctn(Generations_t)
    P_Cin = P_Ctn(Generations_t)

    # Normalize data so total initial concentration = 1
    B_Cin = B_Cin / (B_Cin[0] + P_Cin[0])
    P_Cin = P_Cin / (B_Cin[0] + P_Cin[0])

    color1 = "#d31e25"
    color2 = "#d7a32e"
    color3 = "#369e4b"
    color4 = "#5db5b7"
    color5 = "#31407b"
    color6 = "#d1c02b"
    color7 = "#8a3f64"
    color8 = "#4f2e39"

    plt.style.use('seaborn-poster')

    print('Accuracies:')
    print(accuracy_ML(B_Ci[:8], B1[:8], P_Ci[:8], P1[:8]))

    colorb = '#42329a'
    colorp = '#e54da7'

    B_C, P_C, Time = ExperimentalConencentrations(data1.head(TRUNCATED))
    B_C = B_C / (B_C[0] + P_C[0])
    P_C = P_C / (B_C[0] + P_C[0])
    B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p.head(TRUNCATED))
    B_Cp = B_Cp / (B_Cp[0] + P_Cp[0])
    P_Cp = P_Cp / (B_Cp[0] + P_Cp[0])
    B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n.head(TRUNCATED))
    B_Cn = B_Cn / (B_Cn[0] + P_Cn[0])
    P_Cn = P_Cn / (B_Cn[0] + P_Cn[0])

    plt.errorbar(Time, B_C, yerr = [np.abs(B_C - B_Cn), np.abs(B_C - B_Cp)],
                 fmt = '*',
                 capsize=5,
                 color = colorb,
                 label = 'Experimental Blue')
    plt.errorbar(Time, P_C, yerr = [np.abs(P_C - P_Cn), np.abs(P_C - P_Cp)],
                 fmt = '^',
                 capsize=5,
                 color = colorp,
                 label = 'Experimental Pink')


    plt.errorbar(t, B1, yerr = [std_B, std_B],
                 fmt = '-.',
                 capsize=5,
                 color = colorb,
                 label = 'Predicted Blue')

    plt.errorbar(t, P1, yerr = [std_P, std_P],
                 fmt = '--',
                 capsize=5,
                 color = colorp,
                 label = 'PredictedPink')

    plt.xlabel('Generation')
    plt.style.use('seaborn-poster')
    plt.xlim([0, TRUNCATED])
    plt.xlabel('Time [hrs]')
    plt.title(name + 'aB concentration fractions')
    plt.legend()

    return

TRUNCATED = 15
GENCONVER = 198 # 198 min for ion conversion data  205.62 for gamma radation
namesk = ['Time', 'A570', 'A600', 'A690', 'A750']

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plot_combined(data_list, data_std_list, names):
    plt.figure(figsize=(14, 8))

    colors = ['#d31e25', '#d7a32e', '#369e4b', '#5db5b7', '#31407b', '#d1c02b', '#8a3f64', '#4f2e39']
    markers = ['*', '^', 'o', 's', 'p', 'D', 'x', '+']
    linestyles = ['-.', '--', ':', '-', '-.', '--', ':', '-']

    for idx, (data, data_std, name) in enumerate(zip(data_list, data_std_list, names)):
        P_m = []
        B_m = []

        for root, dirs, files in os.walk(
                r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\Results_Bulk_aB" + '\\' + str(
                    name)):
            if len(files) > 1:
                namess = ["Generation", "x", "y", "z", "Health"]
                resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)
                Bi, Pi, t = AlamarblueMechanics(resultsi,
                                                [0.7799990799515666, 1.679928455577914, 0.10002078747628415, 450, 6601,
                                                 9994, 0.5], 'k')  # optimal
                B_m.append(Bi)
                P_m.append(Pi)

        if B_m:
            B_m = np.stack(B_m, axis=0)
            std_B = np.std(B_m, axis=0)
            nB = np.shape(B_m)[0]
            std_B = std_B / np.sqrt(nB)
        else:
            B_m = []
            std_B = []

        if P_m:
            P_m = np.stack(P_m, axis=0)
            std_P = np.std(P_m, axis=0)
            nP = np.shape(P_m)[0]
            std_P = std_P / np.sqrt(nP)
        else:
            P_m = []
            std_P = []

        B1 = np.mean(B_m, axis=0)
        P1 = np.mean(P_m, axis=0)

        data1p = pd.DataFrame(
            data[['A570', 'A600', 'A690', 'A750']].values + data_std[['A570', 'A600', 'A690', 'A750']].values,
            columns=['A570', 'A600', 'A690', 'A750'])
        data1n = pd.DataFrame(
            data[['A570', 'A600', 'A690', 'A750']].values - data_std[['A570', 'A600', 'A690', 'A750']].values,
            columns=['A570', 'A600', 'A690', 'A750'])
        data1p['Time'] = data['Time']
        data1n['Time'] = data['Time']

        B_C, P_C, Time = ExperimentalConencentrations(data.head(TRUNCATED))
        B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p.head(TRUNCATED))
        B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n.head(TRUNCATED))
        B_C = np.append(B_C, B_C[-1])
        P_C = np.append(P_C, P_C[-1])
        Time = np.append(Time, 80)

        B_Ct = interp1d(Time, B_C, kind='linear')
        P_Ct = interp1d(Time, P_C, kind='linear')

        B_Cp = np.append(B_Cp, B_Cp[-1])
        P_Cp = np.append(P_Cp, P_Cp[-1])

        B_Ctp = interp1d(Time, B_Cp, kind='linear')
        P_Ctp = interp1d(Time, P_Cp, kind='linear')

        B_Cn = np.append(B_Cn, B_Cn[-1])
        P_Cn = np.append(P_Cn, P_Cn[-1])

        B_Ctn = interp1d(Time, B_Cn, kind='linear')
        P_Ctn = interp1d(Time, P_Cn, kind='linear')

        Healthy = resultsi.loc[resultsi["Health"] == 1]
        Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
        Generations_t = Generations * GENCONVER / 60

        B_Ci = B_Ct(Generations_t)
        P_Ci = P_Ct(Generations_t)
        B_Ci = B_Ci / (B_Ci[0] + P_Ci[0])
        P_Ci = P_Ci / (B_Ci[0] + P_Ci[0])

        B_Cip = B_Ctp(Generations_t)
        P_Cip = P_Ctp(Generations_t)
        B_Cip = B_Cip / (B_Cip[0] + P_Cip[0])
        P_Cip = P_Cip / (B_Cip[0] + P_Cip[0])

        B_Cin = B_Ctn(Generations_t)
        P_Cin = P_Ctn(Generations_t)
        B_Cin = B_Cin / (B_Cin[0] + P_Cin[0])
        P_Cin = P_Cin / (B_Cin[0] + P_Cin[0])

        plt.errorbar(Time, B_C * (1 / max(B_C)), yerr=[np.abs(B_C - B_Cn), np.abs(B_C - B_Cp)],
                     fmt=markers[idx % len(markers)], capsize=5, color=colors[idx % len(colors)],
                     label=f'Experimental {name}')

        plt.errorbar(t, B1, yerr=[std_B, std_B], fmt=linestyles[idx % len(linestyles)], capsize=5,
                     color=colors[idx % len(colors)], label=f'Predicted {name}')

    plt.xlabel('Time [hrs]')
    plt.ylabel('Concentration')
    plt.xlim([0, TRUNCATED])
    plt.title('Combined Experimental and Predicted Concentrations')
    plt.legend()
    plt.show()



def plot_combined_pink(data_list, data_std_list, names):
    plt.figure(figsize=(14, 8))

    colors = ['#d31e25', '#d7a32e', '#369e4b', '#5db5b7', '#31407b', '#d1c02b', '#8a3f64', '#4f2e39']
    markers = ['*', '^', 'o', 's', 'p', 'D', 'x', '+']
    linestyles = ['-.', '--', ':', '-', '-.', '--', ':', '-']

    for idx, (data, data_std, name) in enumerate(zip(data_list, data_std_list, names)):
        P_m = []
        B_m = []

        for root, dirs, files in os.walk(
                P.bulk_aB() + '/' + str(
                    name)):
            if len(files) > 1:
                namess = ["Generation", "x", "y", "z", "Health"]
                resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)
                path = os.path.join(root, files[0])
                if '2.5Gy' in path or '5Gy' in path:
                    print(path)
                Bi, Pi, t = AlamarblueMechanics(resultsi,
                                                [0.7799990799515666, 1.679928455577914, 0.10002078747628415, 450, 6601,
                                                 9994, 0.5], 'k')  # optimal
                B_m.append(Bi)
                P_m.append(Pi)

        if B_m:
            B_m = np.stack(B_m, axis=0)
            std_B = np.std(B_m, axis=0)
            nB = np.shape(B_m)[0]
            std_B = std_B / np.sqrt(nB)
        else:
            B_m = []
            std_B = []

        if P_m:
            P_m = np.stack(P_m, axis=0)
            std_P = np.std(P_m, axis=0)
            nP = np.shape(P_m)[0]
            std_P = std_P / np.sqrt(nP)
        else:
            P_m = []
            std_P = []

        B1 = np.mean(B_m, axis=0)
        P1 = np.mean(P_m, axis=0)

        data1p = pd.DataFrame(
            data[['A570', 'A600', 'A690', 'A750']].values + data_std[['A570', 'A600', 'A690', 'A750']].values,
            columns=['A570', 'A600', 'A690', 'A750'])
        data1n = pd.DataFrame(
            data[['A570', 'A600', 'A690', 'A750']].values - data_std[['A570', 'A600', 'A690', 'A750']].values,
            columns=['A570', 'A600', 'A690', 'A750'])
        data1p['Time'] = data['Time']
        data1n['Time'] = data['Time']

        B_C, P_C, Time = ExperimentalConencentrations(data.head(TRUNCATED))
        B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p.head(TRUNCATED))
        B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n.head(TRUNCATED))
        B_C = np.append(B_C, B_C[-1])
        P_C = np.append(P_C, P_C[-1])
        Time = np.append(Time, 80)

        B_Ct = interp1d(Time, B_C, kind='linear')
        P_Ct = interp1d(Time, P_C, kind='linear')

        B_Cp = np.append(B_Cp, B_Cp[-1])
        P_Cp = np.append(P_Cp, P_Cp[-1])

        B_Ctp = interp1d(Time, B_Cp, kind='linear')
        P_Ctp = interp1d(Time, P_Cp, kind='linear')

        B_Cn = np.append(B_Cn, B_Cn[-1])
        P_Cn = np.append(P_Cn, P_Cn[-1])

        B_Ctn = interp1d(Time, B_Cn, kind='linear')
        P_Ctn = interp1d(Time, P_Cn, kind='linear')

        Healthy = resultsi.loc[resultsi["Health"] == 1]
        Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
        Generations_t = Generations * GENCONVER / 60

        B_Ci = B_Ct(Generations_t)
        P_Ci = P_Ct(Generations_t)
        B_Ci = B_Ci / (B_Ci[0] + P_Ci[0])
        P_Ci = P_Ci / (B_Ci[0] + P_Ci[0])

        B_Cip = B_Ctp(Generations_t)
        P_Cip = P_Ctp(Generations_t)
        B_Cip = B_Cip / (B_Cip[0] + P_Cip[0])
        P_Cip = P_Cip / (B_Cip[0] + P_Cip[0])

        B_Cin = B_Ctn(Generations_t)
        P_Cin = P_Ctn(Generations_t)
        B_Cin = B_Cin / (B_Cin[0] + P_Cin[0])
        P_Cin = P_Cin / (B_Cin[0] + P_Cin[0])

        plt.errorbar(Time, P_C, yerr=[np.abs(B_C - B_Cn), np.abs(B_C - B_Cp)],
                     fmt=markers[idx % len(markers)], capsize=5, color=colors[idx % len(colors)],
                     label=f'Experimental {name}')

        plt.errorbar(t, P1, yerr=[std_B, std_B], fmt=linestyles[idx % len(linestyles)], capsize=5,
                     color=colors[idx % len(colors)], label=f'Predicted {name}')

    plt.xlabel('Time [hrs]')
    plt.ylabel('Concentration')
    plt.xlim([0, TRUNCATED])
    plt.title('Combined Experimental and Predicted Concentrations')
    plt.legend()
    plt.show()

TRUNCATED = 15
GENCONVER = 198 # 198 min for ion conversion data  205.62 for gamma radation
namesk = ['Time', 'A570', 'A600', 'A690', 'A750']
# Now WT
dataw = pd.read_csv(P.ab_experimental("AlamarblueRawdataWTKGy.csv"), names = namesk)
datawSTD = pd.read_csv(P.ab_experimental("AlamarblueRawdataWTKGySTD.csv"), names = namesk)

dataw2 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT25Gy.csv"), names = namesk)
datawSTD2 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT25GySTD.csv"), names = namesk)

dataw3 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT10Gy.csv"), names = namesk)
datawSTD3 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT10GySTD.csv"), names = namesk)

dataw4 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT5Gy.csv"), names = namesk)
datawSTD4 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT5GySTD.csv"), names = namesk)

dataw5 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT20Gy.csv"), names = namesk)
datawSTD5 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT20GySTD.csv"), names = namesk)

dataw6 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT30Gy.csv"), names = namesk)
datawSTD6 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT30GySTD.csv"), names = namesk)
# Example usage
data_list = [dataw, dataw2, dataw3, dataw4, dataw5, dataw6]
data_std_list = [datawSTD, datawSTD2, datawSTD3, datawSTD4, datawSTD5, datawSTD6]
names = ["WT_Basic_0", "WT_Basic_25", "WT_Basic_10", "WT_Basic_50", "WT_Basic_200", "WT_Basic_300"]

# Blue
# plot_combined(data_list, data_std_list, names)
# pink
plot_combined_pink(data_list, data_std_list, names)