
"""
Created on
@author: Daniel Palacios
"""

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
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, precision_score

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario

def AlamarblueMechanics(results, var, title):
    # v = d/dt([Clear])  == V_max ([Pink]/ K_M + [Pink])
    Healthy = results.loc[results["Health"] == 1]
    Unhealthy = results.loc[results["Health"] == 2]
    # Compute growth curve
    Growth_curve = Healthy['Generation'].value_counts()

    Growth_curve2 = Unhealthy['Generation'].value_counts()

    Growth_curve = np.array(Growth_curve)
    e = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    g = pd.DataFrame(e, columns = ['Genk'])
    g['ncelldamaged'] = np.zeros(len(e))

    for i in Growth_curve2.index:
        g.at[int(i), 'ncelldamaged'] = Growth_curve2[i]

    Growth_curve2 = g['ncelldamaged'].to_numpy()


    # Consider time steps:
    Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
    # t = Generations >> <<
    # t = np.linspace(0,1.25,1000)
    t = Generations
    #print(t)
    t = [val for val in t for _ in (0, 1)]

    for i in range(len(t)):
        if (i % 2) == 0:
            t[i] = t[i]
        else:
            t[i] = t[i] + 0.5

            
    Growth_curve = [val for val in Growth_curve for _ in (0, 1)]
    Growth_curve2 = [val for val in Growth_curve2 for _ in (0, 1)]

    t = np.array(t)

    #print(t)
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
    V3_max = var[2]
    K1_M = var[3]
    K2_M = var[4]
    K3_M = var[5]
    k = var[6]
    v = [V1_max * ( Blue_0 / (K1_M + Blue_0))]
    v2 = [V2_max * ( Pink_0 / (K2_M + Pink_0))]
    alpha0 = Pink_0 / K2_M
    pi0 = Clear_0 / K3_M

    v2 = ((V2_max * alpha0) - (V3_max * pi0)) / (1 + alpha0 + pi0)

    for i in range(len(t) - 1):
        vn = V1_max * ( Blue[i] / (K1_M + Blue[i])) # <- uptake concentration rate

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

    t =  t * GENCONVER / 60

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
    w, n = 1, 1
    delta_blue = sum((Experimental_B - Predicted_B) ** 2)
    delta_pink = sum((Experimental_P - Predicted_P) ** 2)
    delta_diff = w * abs(delta_blue - delta_pink) ** n
    # Worth calculating the difference between the Experimental and Predicted (original function) and comparing it to the difference
    # delta2 = w * |delta - delta1| ** n difference between the two regularization term, look regularization terms in ML
    delta_total = delta_blue + delta_pink + delta_diff
    # accuracy = classification_report(Experimental_B,Predicted_B).precision
        
    return delta_total

def main_DP(name, data1, data1_std):

    #name = "WT_Basic_0"
    P_m = []
    B_m = []
    for root, dirs, files in os.walk(r"Results_Bulk_aB" + '/' + str(name)):
        if len(files) > 1:
            namess = ["Generation", "x", "y", "z", "Health"]
            resultsi = pd.read_csv(os.path.join(root, files[0]), names= namess)

            ################## DEFAULT PARAMETERS
            # Bi, Pi, t = AlamarblueMechanics(resultsi, [0.75, 1.65, 0, 500, 8000, 1000000, 1/2], 'k')

            ### Testing
            # Bi, Pi, t = AlamarblueMechanics(resultsi, [0.9418799810768097, 1.7714680707739503, 0.1977339123715871, 419, 6787, 571195, 0.5], 'k')

            Bi, Pi, t = AlamarblueMechanics(resultsi, [0.7799990799515666, 1.679928455577914, 0.10002078747628415, 450, 6601, 9994, 0.5], 'k')

            # Bi, Pi, t = AlamarblueMechanics(resultsi, [1.3558767947842083, 0.9958129817752762, 0.13882892817316034, 909, 753, 3795, 0.5], 'k')
            

            B_m.append(Bi)
            P_m.append(Pi)

    B_Ci, B1, P_Ci, P1, std_B, std_P, data1p, data1n = interpolate_data(B_m, P_m, data1, data1_std, resultsi)

    color1 = "#d31e25"
    color2 = "#d7a32e"
    color3 = "#369e4b"
    color4 = "#5db5b7"
    color5 = "#31407b"
    color6 = "#d1c02b"
    color7 = "#8a3f64"
    color8 = "#4f2e39"

    plt.style.use('classic')

    ##### B1, P1 = average(Bi), Pi, Bistd, Pistd

    loss_per_trial = []
    accuracy = accuracy_ML(B_Ci[:8], B1[:8], P_Ci[:8], P1[:8])
    loss_per_trial.append(accuracy)
    print(f'Accuracies: {accuracy}') 

    colorb = '#42329a'
    colorp = '#e54da7'

    # B_C, P_C, Time = ExperimentalConencentrations(data1.head(TRUNCATED_FINAL))
    B_C, P_C, Time = ExperimentalConencentrations(data1[TRUNCATED_START:TRUNCATED_FINAL])
    B_C = B_C / (B_C[0] + P_C[0])
    P_C = P_C / (B_C[0] + P_C[0])
    # B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p.head(TRUNCATED_FINAL))
    B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p[TRUNCATED_START:TRUNCATED_FINAL])
    B_Cp = B_Cp / (B_Cp[0] + P_Cp[0])
    P_Cp = P_Cp / (B_Cp[0] + P_Cp[0])
    # B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n.head(TRUNCATED_FINAL))
    B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n[TRUNCATED_START:TRUNCATED_FINAL])
    B_Cn = B_Cn / (B_Cn[0] + P_Cn[0])
    P_Cn = P_Cn / (B_Cn[0] + P_Cn[0])
    # plt.scatter(Time, B_C, marker = '*', color = colorb, label = 'Experimental Blue')
    # plt.scatter(Time, P_C, marker = "+", color = colorp, label = 'Experimental Pink')
    # plt.scatter(Time, B_Cp, marker='^', color=colorb, label='CI Experimental Blue')
    # plt.scatter(Time, P_Cp, marker="^", color=colorp, label='CI Experimental Pink')
    # plt.scatter(Time, B_Cn, marker='^', color=colorb, label='CI- Experimental Blue')
    # plt.scatter(Time, P_Cn, marker="^", color=colorp, label='CI- Experimental Pink')
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
    
    plt.axhline(y = 0, color = 'r', linestyle = '-')
    # print(B1)
    # print(t)
    # plt.plot(t, B1, color='#42329a', linestyle='-.', label='Predicted Blue')
    # plt.plot(t, P1, color='#e54da7', linestyle='--', label='Predicted Pink')
    # plt.plot(t, B1 + std_B, color='#42329a', linestyle='-', label='CI Blue')
    # plt.plot(t, P1 + std_P, color='#e54da7', linestyle='-', label='CI Pink')
    # plt.plot(t, B1 - std_B, color='#42329a', linestyle='-', label='CI- Blue')
    # plt.plot(t, P1 - std_P, color='#e54da7', linestyle='-', label='CI- Pink')

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
    plt.style.use('classic')
    plt.xlim([TRUNCATED_START, TRUNCATED_FINAL])
    plt.xlabel('Time [hrs]')
    plt.title(name + 'aB concentration fractions')
    plt.legend()

    return

def interpolate_data(B_m, P_m, data1, data1_std, resultsi):
    B_m = np.stack(B_m, axis = 0)
    P_m = np.stack(P_m, axis = 0)
    std_B = np.std(B_m, axis=0)
    std_P = np.std(P_m, axis=0)
    nB = np.shape(B_m)[0]
    nP = np.shape(P_m)[0]
    std_B = std_B / np.sqrt(nB)
    std_P = std_P / np.sqrt(nP)

    B1 = np.mean(B_m, axis=0)
    P1 = np.mean(P_m, axis=0)

    data1p = pd.DataFrame(data1[['A570', 'A600', 'A690', 'A750']].values + data1_std[['A570', 'A600', 'A690', 'A750']].values, columns=['A570', 'A600', 'A690', 'A750'])
    data1n = pd.DataFrame(data1[['A570', 'A600', 'A690', 'A750']].values - data1_std[['A570', 'A600', 'A690', 'A750']].values, columns=['A570', 'A600', 'A690', 'A750'])
    data1p['Time'] = data1['Time']
    data1n['Time'] = data1['Time']

#
    # B_C, P_C, Time = ExperimentalConencentrations(data1.head(TRUNCATED_FINAL))
    B_C, P_C, Time = ExperimentalConencentrations(data1[TRUNCATED_START:TRUNCATED_FINAL])
    # B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p.head(TRUNCATED_FINAL))
    B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p[TRUNCATED_START:TRUNCATED_FINAL])
    # B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n.head(TRUNCATED_FINAL))
    B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n[TRUNCATED_START:TRUNCATED_FINAL])
    #B_Cp, P_Cp, Time = ExperimentalConencentrations(data1_std.head(30))
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
    Generations = np.linspace(TRUNCATED_START, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
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

    return B_Ci, B1, P_Ci, P1, std_B, std_P, data1p, data1n

def plot_loss(iteration_values, loss_values):
    plt.plot(iteration_values, loss_values, marker='o', linestyle='-', color='b', label='Loss vs. Iteration')

    # Adding labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration Plot')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.savefig('paperfigures2024/iterationsloss.svg', format='svg')


if __name__ == "__main__":
    TRUNCATED_START = 4
    TRUNCATED_FINAL = 48 # end value
    GENCONVER = 198 # 198 min for ion conversion data  205.62 for gamma radation

    optimize_parameters = input("Would you like to optimize? [Y/N]: ")
    if optimize_parameters.capitalize() == "Y":
        num_trials = int(input("How many trials?: "))
        iteration_values = []
        cost_per_trial = []
        class MichealistMenten:
            @property
            def configspace(self) -> ConfigurationSpace:
                cs = ConfigurationSpace(seed=0)
                V1_max = Float("V1_max", (0.1, 2), default=0.7799980187037989)
                V2_max = Float("V2_max", (0.1, 2), default=1.679895485410118)
                V3_max = Float("V3_max", (0.1, 1.2), default=0.3)
                K1_M = Integer("K1_M", (1, 1000), default=450)
                K2_M = Integer("K2_M", (1, 10000) , default=6712)
                K3_M = Integer("K3_M", (100, 10000), default=5000)
                cs.add_hyperparameters([V1_max, V2_max, V3_max, K1_M, K2_M, K3_M])

                return cs
            
            def train(self, config: Configuration, seed: int = 0) -> float:
                COLUMN_NAMES = ['Time', 'A570', 'A600', 'A690', 'A750']
                blue_dye = []
                pink_dye = []
                
                for root, dirs, files in os.walk("Results_Bulk_aB/WT_Basic_0"):
                    if len(files) > 1:
                        experimental_data = pd.read_csv(os.path.join(root, files[0]), names=["Generation", "x", "y", "z", "Health"])
                        config_dict = dict(config)
                        updated_parameters = [config_dict["V1_max"], config_dict["V2_max"], config_dict["V3_max"], config_dict["K1_M"], config_dict["K2_M"], config["K3_M"], 1/2]
                        print(updated_parameters)
                        bi, pi, t = AlamarblueMechanics(experimental_data, updated_parameters, "Hypertuning")
                        blue_dye.append(bi)
                        pink_dye.append(pi)
                
                predicted_data = pd.read_csv(r'abFinalPlots/AlamarblueRawdataWTKGy.csv', names=COLUMN_NAMES)
                predicted_data_std = pd.read_csv(r'abFinalPlots/AlamarblueRawdataWTKGySTD.csv', names=COLUMN_NAMES)
                blue_concentration, blue_mean, pink_concentration, pink_mean, _, _, _, _ = interpolate_data(blue_dye, pink_dye, predicted_data, predicted_data_std, experimental_data)
                error = accuracy_ML(blue_concentration[:8], blue_mean[:8], pink_concentration[:8], pink_mean[:8])

                return error
        
        # Hyperparameter Optimization using SMAC3
        MM = MichealistMenten()
        for trials in range(1, num_trials, 50):
            iteration_values.append(trials)
            start_time = time.time()

            # Note: Check if a minimum error threshold is reached and stop optimization
            scenario = Scenario(
                MM.configspace,
                n_trials=trials,
            )

            initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

            smac = HyperparameterOptimizationFacade(
                scenario,
                MM.train,
                initial_design=initial_design,
                overwrite=True,
            )

            incumbent = smac.optimize()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")
            
            # Get cost of default configuration
            default_cost = smac.validate(MM.configspace.get_default_configuration())
            print(f"Default cost: {default_cost}")

            # Let's calculate the cost of the incumbent
            incumbent_cost = smac.validate(incumbent)
            print(f"Incumbent cost: {incumbent_cost}")

            cost_per_trial.append(incumbent_cost)

        plot_loss(iteration_values, cost_per_trial)

    else:
        # Start value truncation = 0 for WT, but rad51 not zero
        namesk = ['Time', 'A570', 'A600', 'A690', 'A750']


        # Now rad51
        datar = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad51KGy.csv", names = namesk)
        datarSTD = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad51KGySTD.csv", names = namesk)

        datar2 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad5125Gy.csv", names = namesk)
        datarSTD2 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad5125GySTD.csv", names = namesk)

        datar3 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad515Gy.csv", names = namesk)
        datarSTD3 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad515GySTD.csv", names = namesk)

        datar5 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad5110Gy.csv", names = namesk)
        datarSTD5 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad5110GySTD.csv", names = namesk)

        datar6 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad5120Gy.csv", names = namesk)
        datarSTD6 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad5120GySTD.csv", names = namesk)

        datar7 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad5130Gy.csv", names = namesk)
        datarSTD7 = pd.read_csv(r"abFinalPlots/AlamarblueRawdatarad5130GySTD.csv", names = namesk)

        # Create a directory named 'figures' if it doesn't exist
        if not os.path.exists('figures'):
            os.makedirs('figures')

        plt.figure(1)
        main_DP("rad51_Basic_0", datar, datarSTD)
        plt.savefig('figures/rad51_Basic_0.png')

        plt.figure(2)
        main_DP("rad51_Basic_25", datar2, datarSTD2)
        plt.savefig('figures/rad51_Basic_25.png')

        plt.figure(3)
        main_DP("rad51_Basic_10", datar3, datarSTD3)
        plt.savefig('figures/rad51_Basic_10.png')

        plt.figure(4)
        main_DP("rad51_Basic_50", datar5, datarSTD5)
        plt.savefig('figures/rad51_Basic_50.png')

        plt.figure(5)
        main_DP("rad51_Basic_200", datar6, datarSTD6)
        plt.savefig('figures/rad51_Basic_200.png')

        plt.figure(6)
        main_DP("rad51_Basic_300", datar7, datarSTD7)
        plt.savefig('figures/rad51_Basic_300.png')


        # Now WT
        dataw = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWTKGy.csv", names = namesk)
        datawSTD = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWTKGySTD.csv", names = namesk)

        dataw2 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT25Gy.csv", names = namesk)
        datawSTD2 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT25GySTD.csv", names = namesk)

        dataw3 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT10Gy.csv", names = namesk)
        datawSTD3 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT10GySTD.csv", names = namesk)

        dataw4 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT5Gy.csv", names = namesk)
        datawSTD4 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT5GySTD.csv", names = namesk)

        dataw5 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT20Gy.csv", names = namesk)
        datawSTD5 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT20GySTD.csv", names = namesk)

        dataw6 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT30Gy.csv", names = namesk)
        datawSTD6 = pd.read_csv(r"abFinalPlots/AlamarblueRawdataWT30GySTD.csv", names = namesk)


        plt.figure(7)
        main_DP("WT_Basic_0", dataw, datawSTD)
        plt.savefig('figures/WT_Basic_0.png')

        plt.figure(8)
        main_DP("WT_Basic_25", dataw2, datawSTD2)
        plt.savefig('figures/WT_Basic_25.png')

        plt.figure(9)
        main_DP("WT_Basic_10", dataw3, datawSTD3)
        plt.savefig('figures/WT_Basic_10.png')

        plt.figure(10)
        main_DP("WT_Basic_50", dataw4, datawSTD4)
        plt.savefig('figures/WT_Basic_50.png')

        plt.figure(11)
        main_DP("WT_Basic_200", dataw5, datawSTD5)
        plt.savefig('figures/WT_Basic_200.png')

        plt.figure(12)
        main_DP("WT_Basic_300", dataw6, datawSTD6)
        plt.savefig('figures/WT_Basic_300.png')
        plt.show()


        # Repeat errors vs n_trails for new evaluation function with regularization term
        #    6-parameter model reversible model (increase range of individual parameters)
        # Plot the rest of the figures
        # Find new time range for rad 51
        # Explore how k changes with rad51 different doses. How much damaged cells contribute to aB conversion?