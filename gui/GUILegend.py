
"""
Created on Tuesday Jun 14 08:42:26 2022
Mideterm presentation growth curves from ammper
@author: Daniel Palacios
"""

import cobra
from cobra.io import read_sbml_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import log

healthy = '#91bfdb'
damaged = '#ffffbf'
dead = '#fc8d59'

plt.figure()
plt.scatter(1, 1, s = 100, c=healthy, alpha=0.75, marker='o')
plt.scatter(1, 2, s = 100, c=damaged, alpha=1, marker='^')
plt.scatter(2, 1, s= 100, c=dead, alpha=1, marker='*')
plt.scatter(1, 3, s=100, c='#9ED9A1', alpha=1)
plt.legend(['Healthy', 'Damaged', 'Dead', 'ROS'], prop={'size': 20})
plt.show()

