import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define the error arrays
# ALL ERRORS >>>>
# manual_errors = [
#     0.2297801840594883,
#     0.17607586551210525,
#     0.9257636591371455,
#     1.3603402355544394,
#     1.5401297491626245,
#     0.24533173794450217,
#     0.6373249026708974,
#     0.6955181692388223,
#     0.5471709792283441,
#     0.5957490859964685,
#     0.5725566036371199,
#     0.4696130438287181
# ]
#
# bo_errors = [
#     0.2043229996932675,
#     0.15903740773895092,
#     1.0672698556308264,
#     1.5366130411135495,
#     1.7193207697139143,
#     0.2991058557270414,
#     0.576673099854368,
#     0.6262511595988866,
#     0.4820317214608146,
#     0.5312940527466247,
#     0.5057455993190662,
#     0.410623842402046
# ]
# Only WT errors >>>>
import numpy as np
from scipy.stats import ttest_rel, mannwhitneyu

# Given errors
manual_errors = [
    0.6373249026708974,
    0.6955181692388223,
    0.5471709792283441,
    0.5957490859964685,
    0.5725566036371199,
    0.4696130438287181
]

bo_errors = [
    0.576673099854368,
    0.6262511595988866,
    0.4820317214608146,
    0.5312940527466247,
    0.5057455993190662,
    0.410623842402046
]

# Calculate mean errors
mean_manual_error = np.mean(manual_errors)
mean_bo_error = np.mean(bo_errors)

# Perform paired t-test
t_stat, t_p_value = ttest_rel(manual_errors, bo_errors)

# Perform Mann-Whitney U test
u_stat, u_p_value = mannwhitneyu(manual_errors, bo_errors, alternative='two-sided')

print(mean_manual_error, mean_bo_error, t_p_value, u_p_value)

