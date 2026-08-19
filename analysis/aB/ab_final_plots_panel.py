"""
Improved Alamarblue Concentration Fractions Panel - All Doses
@author: Daniel Palacios (modified)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import ammper_paths as P  # noqa: E402  (resolves data/ and results/ paths)

import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import log
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, mean_squared_error

def AlamarblueMechanics(results, var, title):
    Healthy = results.loc[results["Health"] == 1]
    Unhealthy = results.loc[results["Health"] == 2]
    Growth_curve = Healthy['Generation'].value_counts()
    Growth_curve2 = Unhealthy['Generation'].value_counts()

    Growth_curve = np.array(Growth_curve)
    e = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    g = pd.DataFrame(e, columns=['Genk'])
    g['ncelldamaged'] = np.zeros(len(e))

    for i in Growth_curve2.index:
        g.at[int(i), 'ncelldamaged'] = Growth_curve2[i]

    Growth_curve2 = g['ncelldamaged'].to_numpy()

    Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
    t = Generations
    t = [val for val in t for _ in (0, 1)]

    for i in range(len(t)):
        if (i % 2) == 0:
            t[i] = t[i]
        else:
            t[i] = t[i] + 0.5

    Growth_curve = [val for val in Growth_curve for _ in (0, 1)]
    Growth_curve2 = [val for val in Growth_curve2 for _ in (0, 1)]

    t = np.array(t)
    Growth_curve = np.array(Growth_curve)
    Growth_curve2 = np.array(Growth_curve2)
    
    Blue_0 = 10000
    Pink_0 = 100
    Clear_0 = 100

    Blue = []
    Pink = []
    Clear = []

    Blue = np.append(Blue, Blue_0)
    Pink = np.append(Pink, Pink_0)
    Clear = np.append(Clear, Clear_0)

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
        vn = V1_max * (Blue[i] / (K1_M + Blue[i]))
        vn = vn * Growth_curve[i] + k * vn * Growth_curve2[i]
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
        Pinknn = Pinkn - dPink
        Pink = np.append(Pink, Pinknn)

    T_Con = []
    for i in range(len(Blue)):
        T_C = Blue[i] + Pink[i] + Clear[i]
        T_Con = np.append(T_Con, T_C)

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


def main_DP_panel(name, data1, data1_std, ax, dose_label):
    """Modified function to plot on a given axis"""
    P_m = []
    B_m = []
    resultsi = None

    for root, dirs, files in os.walk(
            P.bulk_aB() + '/' + str(name)):
        print(f"Root: {root}, Dirs: {dirs}, Files: {files}")
        if len(files) > 1:
            namess = ["Generation", "x", "y", "z", "Health"]
            resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)
            print(f"Results: {resultsi}")
            Bi, Pi, t = AlamarblueMechanics(resultsi, [0.7799990799515666, 1.679928455577914, 0.10002078747628415, 450, 6601, 9994, 0.5], 'k')
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

    data1p = pd.DataFrame(data1[['A570', 'A600', 'A690', 'A750']].values + data1_std[['A570', 'A600', 'A690', 'A750']].values, 
                         columns=['A570', 'A600', 'A690', 'A750'])
    data1n = pd.DataFrame(data1[['A570', 'A600', 'A690', 'A750']].values - data1_std[['A570', 'A600', 'A690', 'A750']].values, 
                         columns=['A570', 'A600', 'A690', 'A750'])
    data1p['Time'] = data1['Time']
    data1n['Time'] = data1['Time']

    B_C, P_C, Time = ExperimentalConencentrations(data1.head(TRUNCATED))
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

    # Improved color scheme
    colorb = '#2E5EAA'  # Deep blue
    colorp = '#D64161'  # Deep pink/red
    
    B_C, P_C, Time = ExperimentalConencentrations(data1.head(TRUNCATED))
    B_C = B_C / (B_C[0] + P_C[0])
    P_C = P_C / (B_C[0] + P_C[0])
    B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p.head(TRUNCATED))
    B_Cp = B_Cp / (B_Cp[0] + P_Cp[0])
    P_Cp = P_Cp / (B_Cp[0] + P_Cp[0])
    B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n.head(TRUNCATED))
    B_Cn = B_Cn / (B_Cn[0] + P_Cn[0])
    P_Cn = P_Cn / (B_Cn[0] + P_Cn[0])

    # Plot on provided axis
    ax.errorbar(Time, B_C, yerr=[np.abs(B_C - B_Cn), np.abs(B_C - B_Cp)],
                fmt='o', markersize=5, capsize=3, linewidth=1.5, 
                color=colorb, ecolor=colorb, alpha=0.7)
    
    ax.errorbar(Time, P_C, yerr=[np.abs(P_C - P_Cn), np.abs(P_C - P_Cp)],
                fmt='s', markersize=5, capsize=3, linewidth=1.5,
                color=colorp, ecolor=colorp, alpha=0.7)

    ax.plot(t, B1, '-', linewidth=2.5, color=colorb, alpha=0.9)
    ax.fill_between(t, B1-std_B, B1+std_B, color=colorb, alpha=0.15)
    
    ax.plot(t, P1, '-', linewidth=2.5, color=colorp, alpha=0.9)
    ax.fill_between(t, P1-std_P, P1+std_P, color=colorp, alpha=0.15)

    ax.set_xlim([0, TRUNCATED])
    ax.set_ylim([-0.05, 1.05])
    ax.text(0.05, 0.95, dose_label, transform=ax.transAxes, 
            fontsize=11, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return

# Constants
TRUNCATED = 15
GENCONVER = 198
namesk = ['Time', 'A570', 'A600', 'A690', 'A750']

# Load rad51 data
datar = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad51KGy.csv"), names=namesk)
datarSTD = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad51KGySTD.csv"), names=namesk)

datar2 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad5125Gy.csv"), names=namesk)
datarSTD2 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad5125GySTD.csv"), names=namesk)

datar3 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad515Gy.csv"), names=namesk)
datarSTD3 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad515GySTD.csv"), names=namesk)

datar5 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad5110Gy.csv"), names=namesk)
datarSTD5 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad5110GySTD.csv"), names=namesk)

datar6 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad5120Gy.csv"), names=namesk)
datarSTD6 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad5120GySTD.csv"), names=namesk)

datar7 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad5130Gy.csv"), names=namesk)
datarSTD7 = pd.read_csv(P.ab_experimental("AlamarblueRawdatarad5130GySTD.csv"), names=namesk)

# Load WT data
dataw = pd.read_csv(P.ab_experimental("AlamarblueRawdataWTKGy.csv"), names=namesk)
datawSTD = pd.read_csv(P.ab_experimental("AlamarblueRawdataWTKGySTD.csv"), names=namesk)

dataw2 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT25Gy.csv"), names=namesk)
datawSTD2 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT25GySTD.csv"), names=namesk)

dataw3 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT10Gy.csv"), names=namesk)
datawSTD3 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT10GySTD.csv"), names=namesk)

dataw4 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT5Gy.csv"), names=namesk)
datawSTD4 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT5GySTD.csv"), names=namesk)

dataw5 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT20Gy.csv"), names=namesk)
datawSTD5 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT20GySTD.csv"), names=namesk)

dataw6 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT30Gy.csv"), names=namesk)
datawSTD6 = pd.read_csv(P.ab_experimental("AlamarblueRawdataWT30GySTD.csv"), names=namesk)

# Create directory for figures
if not os.path.exists('figures'):
    os.makedirs('figures')

# Create panel figure with all doses - rad51
plt.style.use('seaborn-v0_8-paper')
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Alamarblue Assay: rad51Δ Strain Response to Radiation', 
             fontsize=16, fontweight='bold', y=0.98)

# rad51 plots - all 6 doses
main_DP_panel("rad51_Basic_0", datar, datarSTD, axes1[0, 0], '0 Gy')
main_DP_panel("rad51_Basic_25", datar2, datarSTD2, axes1[0, 1], '2.5 Gy')
main_DP_panel("rad51_Basic_10", datar3, datarSTD3, axes1[0, 2], '5 Gy')
main_DP_panel("rad51_Basic_50", datar5, datarSTD5, axes1[1, 0], '10 Gy')
main_DP_panel("rad51_Basic_200", datar6, datarSTD6, axes1[1, 1], '20 Gy')
main_DP_panel("rad51_Basic_300", datar7, datarSTD7, axes1[1, 2], '30 Gy')

# Set common labels for rad51
for ax in axes1[1, :]:
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    
for ax in axes1[:, 0]:
    ax.set_ylabel('Concentration Fraction', fontsize=12, fontweight='bold')

# Create single legend
colorb = '#2E5EAA'
colorp = '#D64161'
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colorb, 
           markersize=8, label='Experimental (Blue)', markeredgecolor=colorb),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colorp, 
           markersize=8, label='Experimental (Pink)', markeredgecolor=colorp),
    Line2D([0], [0], color=colorb, linewidth=2.5, label='Predicted (Blue)'),
    Line2D([0], [0], color=colorp, linewidth=2.5, label='Predicted (Pink)')
]

fig1.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True, 
           fontsize=11, edgecolor='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/rad51_panel_all_doses.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/rad51_panel_all_doses.pdf', bbox_inches='tight')

# Create panel figure with all doses - WT
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Alamarblue Assay: Wild Type Strain Response to Radiation', 
             fontsize=16, fontweight='bold', y=0.98)

# WT plots - all 6 doses
main_DP_panel("WT_Basic_0", dataw, datawSTD, axes2[0, 0], '0 Gy')
main_DP_panel("WT_Basic_25", dataw2, datawSTD2, axes2[0, 1], '2.5 Gy')
main_DP_panel("WT_Basic_10", dataw3, datawSTD3, axes2[0, 2], '5 Gy')
main_DP_panel("WT_Basic_50", dataw4, datawSTD4, axes2[1, 0], '10 Gy')
main_DP_panel("WT_Basic_200", dataw5, datawSTD5, axes2[1, 1], '20 Gy')
main_DP_panel("WT_Basic_300", dataw6, datawSTD6, axes2[1, 2], '30 Gy')

# Set common labels for WT
for ax in axes2[1, :]:
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    
for ax in axes2[:, 0]:
    ax.set_ylabel('Concentration Fraction', fontsize=12, fontweight='bold')

# fig2.legend(handles=legend_elements, loc='upper center', 
#            bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True, 
#            fontsize=11, edgecolor='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/WT_panel_all_doses.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/WT_panel_all_doses.pdf', bbox_inches='tight')

print("Panel figures saved successfully!")
print("\nFiles created:")
print("  rad51 strain:")
print("    - figures/rad51_panel_all_doses.png")
print("    - figures/rad51_panel_all_doses.pdf")
print("\n  Wild Type strain:")
print("    - figures/WT_panel_all_doses.png")
print("    - figures/WT_panel_all_doses.pdf")