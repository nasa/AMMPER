import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pandas as pd
plt.style.use('seaborn-poster')
# Function to generate the propagator distribution
def propagator(r, t):
    """ Toy function to generate a propagator distribution """
    return (r / t) * np.exp(-r**2 / (4 * t))

# Sample data generation for the first plot
r = np.linspace(0, 50, 500)
t_values = [0.2, 1, 5]
colors = ['#D62728', '#FF7F0E', '#2CA02C']
markers = ['d', '^', 'v']
labels = ['t=0.2', 't=1', 't=5']

# Create the first plot
plt.figure(figsize=(8, 6))

for t, color, marker, label in zip(t_values, colors, markers, labels):
    p_rt = propagator(r, t)
    plt.plot(r, p_rt, linestyle='-.', color=color, marker=marker, label=label)

plt.xlabel('Radius')
plt.ylabel('P(r,t)dV')
plt.legend()
plt.grid(True)
plt.xticks([])
plt.yticks([])
plt.xlim([0, 10])

# Save the first figure as a vector image for publication
plt.savefig('paperfigures2024/propagator_plot.svg', format='svg')

# EXTRAPOLATION
data = np.array([[1,58.8], [2,550.5], [3,3417.35]])
fit = np.polyfit(data[:,0], data[:,1] , deg=2) #The use of 1 signifies a linear fit.

y_int1  = np.polyval(fit, 4)
y_int2  = np.polyval(fit, 5)

# Colors for the second plot
color1 = "#d31e25"
color2 = "#d7a32e"
color3 = "#369e4b"
color4 = "#5db5b7"
color5 = "#31407b"
color6 = "#d1c02b"
color7 = "#8a3f64"
color8 = "#4f2e39"

# Data for the second plot
x = [1, 2, 3, 4, 5]
yA = [58.8, 550.5, 3417.35]
yAi = [y_int1, y_int2]
yAii = [3417.35, y_int1, y_int2]
yO = [14.86, 72.06, 235, 452, 494]

# Create the second plot
plt.figure(figsize=(8, 6))
plt.xticks(ticks=x, labels=['0', '2.5', '10', '20', '30'])
plt.xlabel("Radiation Level [Gy]")
plt.ylabel("Time (s)")
plt.scatter(x[:3], yA, marker='x', color=color1)
plt.scatter([4, 5], yAi, marker='o', color=color2)
plt.scatter(x, yO, marker='*', color=color3)

plt.legend(['Original AMMPER', 'Extrapolated: t^2', 'Optimized AMMPER'])
plt.plot(x[:3], yA, '-', color=color1)
plt.plot([3, 4, 5], yAii, '--', color=color2)
plt.plot(x, yO, '.-', color=color3)

plt.title('150 MeV Proton Radiation WildType Simulation Times')
plt.tight_layout()

# Save the second figure as a vector image for publication
plt.savefig('paperfigures2024/optimized_ammpper_plot.svg', format='svg')

######################### ROS COMPLEX BASIC ############################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, maxwell
from scipy.integrate import odeint
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

def Growthcurves(results, title):

    Growth_curve = []
    Generations = []

    for i in results["Generation"].unique():

        subdf = results.loc[results["Generation"] == i]

        healthy = subdf.loc[subdf["Health"] == 1]
        unhealthy = subdf.loc[subdf["Health"] != 1]

        Healthy_counts = healthy.shape[0]
        Unhealthy_counts = unhealthy.shape[0]

        if healthy.empty:
            Healthy_counts = 0

        if unhealthy.empty:
            Unhealthy_counts = 0

        try:
            Gi = Unhealthy_counts/Healthy_counts
        except:
            # there are 0 healthy cells
            Gi = 0

        # Growth_curve ratio
        Growth_curve = np.append(Growth_curve, Gi)

        Generations = np.append(Generations, i)

    return Growth_curve, Generations

def Means_and_variances(name):

    name = name

    Growth_m = []

    for root, dirs, files in os.walk(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\Results_Bulk_half" + '\\' + str(name)):
         #for file in files:
             if len(files) > 1:

                 namess = ["Generation", "x", "y", "z", "Health"]

                 resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)

                 Growthi, Geni = Growthcurves(resultsi, 'Complex')

                 Growth_m.append(Growthi)

    Growth_m = np.stack(Growth_m, axis = 0)
    mean =  np.mean(Growth_m, axis=0)
    std = np.std(Growth_m, axis=0)
    n = np.shape(Growth_m)[0]
    std = std / np.sqrt(n)
    Growth_m = Growth_m[:,-4:]

    gf_m = pd.DataFrame(Growth_m, columns = ['t12', 't13', 't14','t15'])
    gf_m = pd.melt(gf_m, value_vars = ['t12', 't13', 't14','t15'])
    gf_m.to_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\StatisticsBulk_half" + "\\" + str(name) + "pivoted", sep = '\t')
    np.savetxt(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\StatisticsBulk_half" + "\\" + str(name), Growth_m, delimiter = ',')

    return mean, std, n


mean, std, n = Means_and_variances("half_life_1")

mean2, std2, n2 = Means_and_variances("half_life_2")

mean3, std3, n3 = Means_and_variances("half_life_3")

mean4, std4, n4 = Means_and_variances("half_life_20min")


# Color palletee
color1 = "#d31e25"
color2 = "#d7a32e"
color3 = "#369e4b"
color4 = "#5db5b7"
color5 = "#31407b"
color6 = "#d1c02b"
color7 = "#8a3f64"
color8 = "#4f2e39"

plt.style.use('seaborn-poster')

plt.figure(3)

plt.errorbar(np.arange(len(mean)), mean, yerr = std, xerr = None, marker = '*', color = color1, linestyle = '--', label = '1 generation, n=' + str(n), elinewidth = 1)
plt.errorbar(np.arange(len(mean2)), mean2, yerr = std2, xerr = None, marker = '^', color = color2, linestyle = '-.', label = '2 generation, n=' + str(n2), elinewidth = 1)
plt.errorbar(np.arange(len(mean3)), mean3, yerr = std3, xerr = None, marker = 's', color = color3, linestyle = '--', label = '3 generation, n=' + str(n3), elinewidth = 1)
plt.errorbar(np.arange(len(mean4)), mean4, yerr = std4, xerr = None, marker = 'v', color = color4, linestyle = '-.', label = '20 min, n=' + str(n4), elinewidth = 1)
plt.title("rad51 Unhealthy-Healthy cell ratios")
plt.legend()
plt.xlabel("Generation")
plt.xticks(np.arange(12, 16))  # Set xticks to only 12, 13, 14, 15

plt.xlim(11.8,15.2)
plt.savefig('paperfigures2024/half_life_ratios.svg', format='svg')

### BASIC VS COMPLEX
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import maxwell, skewnorm
from scipy.integrate import odeint
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

def Growthcurves(results, title):

    Growth_curve = []
    Generations = []

    for i in results["Generation"].unique():

        subdf = results.loc[results["Generation"] == i]

        healthy = subdf.loc[subdf["Health"] == 1]
        unhealthy = subdf.loc[subdf["Health"] != 1]

        Healthy_counts = healthy.shape[0]
        Unhealthy_counts = unhealthy.shape[0]

        if healthy.empty:
            Healthy_counts = 0

        if unhealthy.empty:
            Unhealthy_counts = 0

        try:
            Gi = Unhealthy_counts/Healthy_counts
        except:
            # there are 0 healthy cells
            Gi = 0

        # Growth_curve ratio
        Growth_curve = np.append(Growth_curve, Gi)

        Generations = np.append(Generations, i)

    return Growth_curve, Generations

def Means_and_variances(name):

    name = name

    Growth_m = []

    for root, dirs, files in os.walk(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\Results_Bulk" + '\\' + str(name)):
         #for file in files:
             if len(files) > 1:

                 namess = ["Generation", "x", "y", "z", "Health"]

                 resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)

                 Growthi, Geni = Growthcurves(resultsi, 'Basic rad51 2.5 Gy')

                 Growth_m.append(Growthi)

    Growth_m = np.stack(Growth_m, axis = 0)
    mean =  np.mean(Growth_m, axis=0)
    std = np.std(Growth_m, axis=0)
    n = np.shape(Growth_m)[0]

    std = std / np.sqrt(n)
    Growth_m = Growth_m[:,-4:]

    gf_m = pd.DataFrame(Growth_m, columns = ['t12', 't13', 't14','t15'])
    gf_m = pd.melt(gf_m, value_vars = ['t12', 't13', 't14','t15'])
    gf_m.to_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\StatisticsBulk" + "\\" + str(name) + "pivoted", sep = '\t')
    np.savetxt(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\StatisticsBulk" + "\\" + str(name), Growth_m, delimiter = ',')

    return mean, std, n


mean, std, n = Means_and_variances("rad_Basic_1")

mean2, std2, n2 = Means_and_variances("rad_Basic_2")

mean3, std3, n3 = Means_and_variances("rad_Complex_1")

mean4, std4, n4 = Means_and_variances("rad_Complex_2")

mean5, std5, n5 = Means_and_variances("WT_Basic_1")

mean6, std6, n6 = Means_and_variances("WT_Basic_2")

mean7, std7, n7 = Means_and_variances("WT_Complex_1")

mean8, std8, n8 = Means_and_variances("WT_Complex_2")

# Color palletee
color1 = "#d31e25"
color2 = "#d7a32e"
color3 = "#369e4b"
color4 = "#5db5b7"
color5 = "#31407b"
color6 = "#d1c02b"
color7 = "#8a3f64"
color8 = "#4f2e39"


plt.style.use('seaborn-poster')

plt.figure(4)



plt.errorbar(np.arange(len(mean5)), mean5, yerr = std5, xerr = None, marker = '*', color = color3, linestyle = '--', label = 'Basic WT 2.5 Gy, n=' + str(n5), elinewidth = 1)
plt.errorbar(np.arange(len(mean6)), mean6, yerr = std6, xerr = None, marker = '*', color = color4, linestyle = '--', label = 'Basic WT 5 Gy, n=' + str(n6),elinewidth = 1)
plt.errorbar(np.arange(len(mean7)), mean7, yerr = std7, xerr = None, marker = '^', color = color3, linestyle = '-.', label = 'Complex WT 2.5 Gy, n=' + str(n7),elinewidth = 1)
plt.errorbar(np.arange(len(mean8)), mean8, yerr = std8, xerr = None, marker = '^', color = color4, linestyle = '-.', label = 'Complex WT 5 Gy, n=' + str(n8),elinewidth = 1)
plt.xlim(11.8,15.2)
plt.title("WT Unhealthy-Healthy cell ratios")
plt.legend()
plt.xticks(np.arange(12, 16))  # Set xticks to only 12, 13, 14, 15
plt.xlabel("Generation")
plt.savefig('paperfigures2024/complexvsbasic_1.svg', format='svg')

plt.figure(5)
plt.errorbar(np.arange(len(mean)), mean, yerr = std, xerr = None, marker = '*', color = color3, linestyle = '--', label = 'Basic rad51 2.5 Gy, n=' + str(n),elinewidth = 1)
plt.errorbar(np.arange(len(mean2)), mean2, yerr = std2, xerr = None, marker = '*', color = color4, linestyle = '--', label = 'Basic rad51 5 Gy, n=' + str(n2),elinewidth = 1)
plt.errorbar(np.arange(len(mean3)), mean3, yerr = std3, xerr = None, marker = '^', color = color3, linestyle = '-.', label = 'Complex rad51 2.5 Gy, n=' + str(n3),elinewidth = 1)
plt.errorbar(np.arange(len(mean4)), mean4, yerr = std4, xerr = None, marker = '^', color = color4, linestyle = '-.', label = 'Complex rad51 5 Gy, n=' + str(n4),elinewidth = 1)
plt.title("rad51 Unhealthy-Healthy cell ratios")
plt.legend()
plt.xticks(np.arange(12, 16))  # Set xticks to only 12, 13, 14, 15
plt.xlabel("Generation")

plt.xlim(11.8,15.2)
plt.savefig('paperfigures2024/complexvsbasic_2.svg', format='svg')
################# NON PARAM DISTRIBUTIONS ##########

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import maxwell, skewnorm
from scipy.integrate import odeint
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

def Growthcurves(results, title):

    Growth_curve = []
    Generations = []

    for i in results["Generation"].unique():

        subdf = results.loc[results["Generation"] == i]

        healthy = subdf.loc[subdf["Health"] == 1]
        unhealthy = subdf.loc[subdf["Health"] != 1]

        Healthy_counts = healthy.shape[0]
        Unhealthy_counts = unhealthy.shape[0]

        if healthy.empty:
            Healthy_counts = 0

        if unhealthy.empty:
            Unhealthy_counts = 0

        try:
            Gi = Unhealthy_counts/Healthy_counts
        except:
            # there are 0 healthy cells
            Gi = 0

        # Growth_curve ratio
        Growth_curve = np.append(Growth_curve, Gi)

        Generations = np.append(Generations, i)

    return Growth_curve, Generations

def Means_and_variances(name):

    name = name

    Growth_m = []

    for root, dirs, files in os.walk(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\Results_Bulk" + '\\' + str(name)):
         #for file in files:
             if len(files) > 1:

                 namess = ["Generation", "x", "y", "z", "Health"]

                 resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)

                 Growthi, Geni = Growthcurves(resultsi, 'Basic rad51 2.5 Gy')

                 Growth_m.append(Growthi)

    Growth_m = np.stack(Growth_m, axis = 0)
    mean =  np.mean(Growth_m, axis=0)
    std = np.std(Growth_m, axis=0)
    n = np.shape(Growth_m)[0]

    std = std / np.sqrt(n)
    Growth_m = Growth_m[:,-4:]

    gf_m = pd.DataFrame(Growth_m, columns = ['t12', 't13', 't14','t15'])
    gf_m = pd.melt(gf_m, value_vars = ['t12', 't13', 't14','t15'])
    gf_m.to_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\StatisticsBulk" + "\\" + str(name) + "pivoted", sep = '\t')
    np.savetxt(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\StatisticsBulk" + "\\" + str(name), Growth_m, delimiter = ',')

    return mean, std, n, Growth_m


mean, std, n, Growth_m = Means_and_variances("WT_Basic_1")

mean2, std2, n2, Growth_m2  = Means_and_variances("WT_Basic_2")

mean3, std3, n3, Growth_m3  = Means_and_variances("WT_Complex_1")

mean4, std4, n4, Growth_m4 = Means_and_variances("WT_Complex_2")

# Trim arrays to have the same shape (24, 4)
Growth_m = Growth_m[:24, :]
Growth_m2 = Growth_m2[:24, :]
Growth_m3 = Growth_m3[:24, :]
Growth_m4 = Growth_m4[:24, :]

# Now all arrays should have shape (24, 4)
column1_Growth_m = Growth_m[:, 2]
column1_Growth_m2 = Growth_m2[:, 2]
column1_Growth_m3 = Growth_m3[:, 2]
column1_Growth_m4 = Growth_m4[:, 2]
color1 = "#d31e25"
color2 = "#d7a32e"
color3 = "#369e4b"
color4 = "#5db5b7"
color5 = "#31407b"
color6 = "#d1c02b"
color7 = "#8a3f64"
color8 = "#4f2e39"

# Plotting histograms with different colors
plt.figure(40)
plt.hist(column1_Growth_m, bins=20, alpha=0.5, label='WT Basic 2.5Gy', color=color1)
plt.hist(column1_Growth_m2, bins=20, alpha=0.5, label='WT Basic 5Gy', color=color2)
plt.hist(column1_Growth_m3, bins=20, alpha=0.5, label='WT Complex 2.5Gy', color=color3)
plt.hist(column1_Growth_m4, bins=20, alpha=0.5, label='WT Complex 5Gy', color=color4)

# Adding labels and title
plt.xlabel('Unhealthy-Healthy cell ratios')
plt.ylabel('Frequency')
plt.title('Unhealthy-Healthy cell ratios nonparametric distributions')
plt.legend()
plt.savefig('paperfigures2024/nonparametric_distributions.svg', format='svg')

##################### ERROR ANALYSIS ##############
import matplotlib.pyplot as plt
import numpy as np

# Provided errors
manual_errors = [0.6373249026708974, 0.6955181692388223, 0.5471709792283441, 0.5957490859964685, 0.5725566036371199, 0.4696130438287181]
bo_errors = [0.576673099854368, 0.6262511595988866, 0.4820317214608146, 0.5312940527466247, 0.5057455993190662, 0.410623842402046]
doses = [0, 2.5, 10, 5, 20, 30]
manual_errors = [0.6373249026708974, 0.6955181692388223, 0.5957490859964685,0.5471709792283441, 0.5725566036371199, 0.4696130438287181]
bo_errors = [0.576673099854368, 0.6262511595988866, 0.5312940527466247,0.4820317214608146, 0.5057455993190662, 0.410623842402046]
doses = [0, 2.5, 5, 10, 20, 30]

# Create the plot
plt.figure(6)

# Plot manual errors
plt.plot(doses, manual_errors, 'o-', label='Manual Errors')
# Plot BO errors
plt.plot(doses, bo_errors, 's-', label='BO Errors')

# Adding labels and title
plt.xlabel('Dose (Gy)')
plt.ylabel('Error Magnitude')
plt.title('Comparison of Manual and BO Errors by Dose')
plt.legend()

# Showing the plot
plt.grid(True)
plt.savefig('paperfigures2024/error_analysis.svg', format='svg')

########################### Loss BO ##### This is toy simulation, real simulation is get by
# running SMAC3 code but it is hard to get because smac 3 does not work in our python version. I would show both
import matplotlib.pyplot as plt

# Data for the plot
iterations = list(range(0, 1050, 50))
loss = [0.86] + [0.58] + [0.57] * 19
loss = loss + np.random.normal(0, 0.002, len(loss))

# Create the plot
plt.figure(7)
plt.plot(iterations, loss, 'bo-', label='Loss vs. Iteration')

# Add title and labels
plt.title('Loss vs. Iteration Plot')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# Add legend
plt.legend()
plt.savefig('paperfigures2024/simulated_loss.svg', format='svg')



#################################### aB figures ##################################################################

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
from sklearn.metrics import accuracy_score, mean_squared_error


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
            r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\Results_Bulk_aB" + '\\' + str(
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


# Now rad51

datar = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad51KGy.csv", names = namesk)
datarSTD = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad51KGySTD.csv", names = namesk)

datar2 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad5125Gy.csv", names = namesk)
datarSTD2 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad5125GySTD.csv", names = namesk)

datar3 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad515Gy.csv", names = namesk)
datarSTD3 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad515GySTD.csv", names = namesk)

datar5 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad5110Gy.csv", names = namesk)
datarSTD5 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad5110GySTD.csv", names = namesk)

datar6 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad5120Gy.csv", names = namesk)
datarSTD6 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad5120GySTD.csv", names = namesk)

datar7 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad5130Gy.csv", names = namesk)
datarSTD7 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdatarad5130GySTD.csv", names = namesk)

# Create a directory named 'figures' if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

plt.figure(8)
main_DP("rad51_Basic_0", datar, datarSTD)
plt.savefig('paperfigures2024/rad51_Basic_0.svg', format='svg')

plt.figure(9)
main_DP("rad51_Basic_25", datar2, datarSTD2)
plt.savefig('paperfigures2024/rad51_Basic_25.svg', format='svg')

plt.figure(10)
main_DP("rad51_Basic_10", datar3, datarSTD3)
plt.savefig('paperfigures2024/rad51_Basic_10.svg', format='svg')

plt.figure(11)
main_DP("rad51_Basic_50", datar5, datarSTD5)
plt.savefig('paperfigures2024/rad51_Basic_50.svg', format='svg')

plt.figure(12)
main_DP("rad51_Basic_200", datar6, datarSTD6)
plt.savefig('paperfigures2024/rad51_Basic_200.svg', format='svg')

plt.figure(13)
main_DP("rad51_Basic_300", datar7, datarSTD7)
plt.savefig('paperfigures2024/rad51_Basic_300.svg', format='svg')


# Now WT
dataw = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWTKGy.csv", names = namesk)
datawSTD = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWTKGySTD.csv", names = namesk)

dataw2 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT25Gy.csv", names = namesk)
datawSTD2 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT25GySTD.csv", names = namesk)

dataw3 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT10Gy.csv", names = namesk)
datawSTD3 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT10GySTD.csv", names = namesk)

dataw4 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT5Gy.csv", names = namesk)
datawSTD4 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT5GySTD.csv", names = namesk)

dataw5 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT20Gy.csv", names = namesk)
datawSTD5 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT20GySTD.csv", names = namesk)

dataw6 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT30Gy.csv", names = namesk)
datawSTD6 = pd.read_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPERunofficial\abFinalPlots\AlamarblueRawdataWT30GySTD.csv", names = namesk)

plt.figure(14)
main_DP("WT_Basic_0", dataw, datawSTD)
plt.savefig('paperfigures2024/WT_Basic_0.svg', format='svg')

plt.figure(15)
main_DP("WT_Basic_25", dataw2, datawSTD2)
plt.savefig('paperfigures2024/WT_Basic_25.svg', format='svg')

plt.figure(16)
main_DP("WT_Basic_10", dataw3, datawSTD3)
plt.savefig('paperfigures2024/WT_Basic_10.svg', format='svg')

plt.figure(17)
main_DP("WT_Basic_50", dataw4, datawSTD4)
plt.savefig('paperfigures2024/WT_Basic_50.svg', format='svg')

plt.figure(18)
main_DP("WT_Basic_200", dataw5, datawSTD5)
plt.savefig('paperfigures2024/WT_Basic_200.svg', format='svg')

plt.figure(19)
main_DP("WT_Basic_300", dataw6, datawSTD6)
plt.savefig('paperfigures2024/WT_Basic_300.svg', format='svg')


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

    B_C = (A600 - OD600 - P_ratio * (A570 - OD570)) / (1 - P_ratio * B_ratio)

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


def main_DP(name, data1, data1_std):
    # name = "WT_Basic_0"
    P_m = []
    B_m = []
    for root, dirs, files in os.walk(
            r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\Results_Bulk_GAMMAFINAL" + '\\' + str(
                    name)):
        if len(files) > 1:
            namess = ["Generation", "x", "y", "z", "Health"]
            resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)
            Bi, Pi, t = AlamarblueMechanics(resultsi,
                                            [0.75, 1.65, 500, 8000, 1 / 2], 'k')
            # Bi, Pi, t = AlamarblueMechanics(resultsi,
            #                                 [1.4777777777777779, 2, 6444.444444444444,
            #                                  3427.777777777778,
            #                                  1 / 2], 'k')
            # [0.5,1.35,500,8000, 1/2]
            # [1.4777777777777779, 2, 6444.444444444444, 3427.777777777778, 1 / 2]
            B_m.append(Bi)
            P_m.append(Pi)

    B_m = np.stack(B_m, axis=0)
    P_m = np.stack(P_m, axis=0)
    std_B = np.std(B_m, axis=0)
    std_P = np.std(P_m, axis=0)
    nB = np.shape(B_m)[0]
    nP = np.shape(P_m)[0]
    std_B = std_B / np.sqrt(nB)
    std_P = std_P / np.sqrt(nP)

    B1 = np.mean(B_m, axis=0)
    P1 = np.mean(P_m, axis=0)

    data1p = pd.DataFrame(
        data1[['A570', 'A600', 'A690', 'A750']].values + data1_std[['A570', 'A600', 'A690', 'A750']].values,
        columns=['A570', 'A600', 'A690', 'A750'])
    data1n = pd.DataFrame(
        data1[['A570', 'A600', 'A690', 'A750']].values - data1_std[['A570', 'A600', 'A690', 'A750']].values,
        columns=['A570', 'A600', 'A690', 'A750'])
    data1p['Time'] = data1['Time']
    data1n['Time'] = data1['Time']

    #
    data1 = data1.head(TRUNCATED2)
    data1 = data1.tail(-TRUNCATED1)

    data1['Time'] = range(len(data1.index))

    data1p = data1p.head(TRUNCATED2)
    data1p = data1p.tail(-TRUNCATED1)

    data1p['Time'] = range(len(data1p.index))

    data1n = data1n.head(TRUNCATED2)
    data1n = data1n.tail(-TRUNCATED1)

    data1n['Time'] = range(len(data1n.index))

    B_C, P_C, Time = ExperimentalConencentrations(data1)
    B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p)
    B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n)
    # B_Cp, P_Cp, Time = ExperimentalConencentrations(data1_std.head(30))
    # assume After 30 hours sytem remains in equilibrium since model does not make any statement of additional mechanics >><<
    B_C = np.append(B_C, B_C[-1])
    P_C = np.append(P_C, P_C[-1])
    Time = np.append(Time, 80)

    # Need to interpolate data to match generation dimensions and values *********************
    B_Ct = interp1d(Time, B_C, kind='linear')
    P_Ct = interp1d(Time, P_C, kind='linear')

    B_Cp = np.append(B_Cp, B_Cp[-1])
    P_Cp = np.append(P_Cp, P_Cp[-1])

    B_Ctp = interp1d(Time, B_Cp, kind='linear')
    P_Ctp = interp1d(Time, P_Cp, kind='linear')

    B_Cn = np.append(B_Cn, B_Cn[-1])
    P_Cn = np.append(P_Cn, P_Cn[-1])

    # Need to interpolate data to match generation dimensions and values *********************
    B_Ctn = interp1d(Time, B_Cn, kind='linear')
    P_Ctn = interp1d(Time, P_Cn, kind='linear')

    # Data with generation time ******
    Healthy = resultsi.loc[resultsi["Health"] == 1]
    Generations = np.linspace(0, int(Healthy['Generation'].max()), num=len(Healthy['Generation'].unique()))
    # Generation to hrs ->> Conversion
    Generations_t = Generations * GENCONVER / 60

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

    ##### B1, P1 = average(Bi), Pi, Bistd, Pistd

    print('Accuracies:')
    print(accuracy_ML(B_Ci[:8], B1[:8], P_Ci[:8], P1[:8]))

    colorb = '#42329a'
    colorp = '#e54da7'

    B_C, P_C, Time = ExperimentalConencentrations(data1)
    B_C = B_C / (B_C[0] + P_C[0])
    P_C = P_C / (B_C[0] + P_C[0])
    B_Cp, P_Cp, Time = ExperimentalConencentrations(data1p)
    B_Cp = B_Cp / (B_Cp[0] + P_Cp[0])
    P_Cp = P_Cp / (B_Cp[0] + P_Cp[0])
    B_Cn, P_Cn, Time = ExperimentalConencentrations(data1n)
    B_Cn = B_Cn / (B_Cn[0] + P_Cn[0])
    P_Cn = P_Cn / (B_Cn[0] + P_Cn[0])
    # plt.scatter(Time, B_C, marker = '*', color = colorb, label = 'Experimental Blue')
    # plt.scatter(Time, P_C, marker = "+", color = colorp, label = 'Experimental Pink')
    # plt.scatter(Time, B_Cp, marker='^', color=colorb, label='CI Experimental Blue')
    # plt.scatter(Time, P_Cp, marker="^", color=colorp, label='CI Experimental Pink')
    # plt.scatter(Time, B_Cn, marker='^', color=colorb, label='CI- Experimental Blue')
    # plt.scatter(Time, P_Cn, marker="^", color=colorp, label='CI- Experimental Pink')
    plt.errorbar(Time, B_C, yerr=[np.abs(B_C - B_Cn), np.abs(B_C - B_Cp)],
                 fmt='*',
                 capsize=5,
                 color=colorb,
                 label='Experimental Blue')
    plt.errorbar(Time, P_C, yerr=[np.abs(P_C - P_Cn), np.abs(P_C - P_Cp)],
                 fmt='^',
                 capsize=5,
                 color=colorp,
                 label='Experimental Pink')
    # print(B1)
    # print(t)
    # plt.plot(t, B1, color='#42329a', linestyle='-.', label='Predicted Blue')
    # plt.plot(t, P1, color='#e54da7', linestyle='--', label='Predicted Pink')
    # plt.plot(t, B1 + std_B, color='#42329a', linestyle='-', label='CI Blue')
    # plt.plot(t, P1 + std_P, color='#e54da7', linestyle='-', label='CI Pink')
    # plt.plot(t, B1 - std_B, color='#42329a', linestyle='-', label='CI- Blue')
    # plt.plot(t, P1 - std_P, color='#e54da7', linestyle='-', label='CI- Pink')

    plt.errorbar(t, B1, yerr=[std_B, std_B],
                 fmt='-.',
                 capsize=5,
                 color=colorb,
                 label='Predicted Blue')

    plt.errorbar(t, P1, yerr=[std_P, std_P],
                 fmt='--',
                 capsize=5,
                 color=colorp,
                 label='PredictedPink')

    plt.xlabel('Generation')
    plt.style.use('seaborn-poster')
    plt.xlim([0, TRUNCATED2 - TRUNCATED1])
    plt.xlabel('Time [hrs]')
    plt.title(name + 'aB concentration fractions')
    plt.legend()

    return


TRUNCATED1 = 5
TRUNCATED2 = 20
GENCONVER = 205.62  # 198 min for ion conversion data  205.62 for gamma radation
namesk = ['Time', 'A570', 'A600', 'A690', 'A750']

##############
data1 = pd.read_csv(
    r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdatarad51KGy.csv',
    names=namesk)  # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch

data2 = pd.read_csv(
    r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdatarad5125Gy.csv',
    names=namesk)  # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch

data3 = pd.read_csv(
    r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdatarad5130Gy.csv',
    names=namesk)  # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch

data1STD = pd.read_csv(
    r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdatarad51KGySTD.csv',
    names=namesk)  # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch
# "

data2STD = pd.read_csv(
    r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdatarad5125GySTD.csv',
    names=namesk)  # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch

data3STD = pd.read_csv(
    r'C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\AlamarblueRawdatarad5130GySTD.csv',
    names=namesk)  # Average 0 Gy, use Average data instead of single data >>>>>>> <<<<<<<<<<<< Re run 4 rank tensor gridsearch

plt.figure(20)
# Takes name of folder with AMMPER runs, and takes corresponding experimental data, computes aB plots with error intervals
main_DP("rad51_0", data1, data1STD)
plt.savefig('paperfigures2024/gamma_0.svg', format='svg')
plt.figure(21)
main_DP("rad51_25", data2, data2STD)
plt.savefig('paperfigures2024/gamma_25.svg', format='svg')
plt.figure(22)
main_DP("rad51_300", data3, data3STD)
plt.savefig('paperfigures2024/gamma_300.svg', format='svg')


