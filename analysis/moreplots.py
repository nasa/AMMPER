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
plt.figure(figsize=(10, 6))

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
plt.show()
