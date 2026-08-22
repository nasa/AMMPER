
"""
Created on Tuesday Jun 14 08:42:26 2022

Try to create skew gauss function to replicate ROS level as a function of time in several biological systems.
Note perfect symmetric looks like acute stress while asymmetric contributions yield in chronic stress.


@author: Daniel Palacios
"""

from scipy.stats import skewnorm
import numpy as np
import matplotlib.pyplot as plt

#Idea create a certain number of ROS molecules, choose a probability for them to decay, and try to simulate a skew gaussian.
# 
# x = np.linspace(0,10,100)
# 
# y = skewnorm.pdf(x, 1, 3, 5)
# 
# plt.plot(x,y)
# # Create an array of 0 and 1, 0 represents alive ROS, 1 represents dead ROS.
# ROS = np.zeros(1000)
# 
# # Consider 15 Generations
# Generations = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# 
# # 
# 
# for i in range(len(Generations)):
#     ROSu = np.random.choice([0,1], size = 1000, p = [0.9, 0.1])
#     for j in range(len(ROS)):
#         
#     
# plt.show()
# Consider Asymmetric Gauss Function

# Refering to the paper of ROS lifetime and difussion we can use ::: >>>> :::: >>>>
# 1.1, and 1.2
##########################################>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# r (distance) = \sqrt{6 * D (diffussion constant) * t_exp }
# t_exp = 1/k_exp = 1 / (k_0 + k_qS * [S]) 0 unimolecular reaction, qS scavenger, [S] abundance of scavengers. 

# How to incorporated in ammper?? 1.- create sphere around each individual ROS molecules, things inside that sphere are affected
# 2.- Give each ROS molecule brownian motion limited with the r distance. 
#
# k_o = 1
# k_qS = 1
# S = 1
# D = 10 ** (-5)
#
# t_exp = 1 / (k_o + k_qS * S)
#
# r = np.sqrt(6 * D * t_exp)

# assume constant r for simplicity (sphere method)

# Define propagator

# # 2d
# P = (2 * r * dr / rms ** 2) * np.exp(-(r ** 2) / rms ** 2 )

# 3D
#
# d = 3
#
# rms = np.linspace(0.1,50,1000)
# t = 1
#
#
# D = rms / (2 * d * t)
# P = (1 / (4 * np.pi * D * t) ** (d/2)) * np.exp(-np.sqrt(rms) **2 / (4 * D * t))
#
# t2 = 2
# D2 = rms / (2 * d * t2)
# P2 = (1 / (4 * np.pi * D2 * t2) ** (d/2)) * np.exp(-np.sqrt(rms) **2 / (4 * D2 * t2))
#
# t3 = 3
# D3 = rms / (2 * d * t3)
# P3 = (1 / (4 * np.pi * D3 * t3) ** (d/2)) * np.exp(-np.sqrt(rms)**2 / (4 * D3 * t3))
#
# plt.plot(rms, P)
# plt.plot(rms, P2)
# plt.plot(rms, P3)
#
# plt.show()
#
# def MB_speed(v,m,T):
#     """ Maxwell-Boltzmann speed distribution for speeds """
#     kB = 1.38e-23
#     return (m/(2*np.pi*kB*T))**1.5 * 4*np.pi * v**2 * np.exp(-m*v**2/(2*kB*T))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# v = np.arange(0,800,1)
# amu = 1.66e-27
# mass = 85*amu
#
# for T in [100,200,300,400]:
#     fv = MB_speed(v,mass,T)
#     ax.plot(v,fv,label='T='+str(T)+' K')
#
# ax.legend(loc=0)
# ax.set_xlabel('$v$ (m/s)')
# ax.set_ylabel('PDF, $f_v(v)$')
# plt.plot()

from scipy.stats import maxwell
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import skewnorm
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import maxwell
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# fig, ax = plt.subplots(1, 1)
#
# x = np.linspace(maxwell.ppf(0.01),
#                 maxwell.ppf(0.99), 100)
#
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# ax.plot(x, maxwell.pdf(x, loc = 0, scale = 1.2),
#        'r-', lw=1, alpha=1, label='maxwell pdf')
#
# ax.plot(x, maxwell.pdf(x, loc = 0, scale =1),
#        'r--', lw=1, alpha=1, label='maxwell pdf')

# # First approach consider 3 independent directions: 2d case looks like:
#
# y = np.linspace(maxwell.ppf(0.01),
#                 maxwell.ppf(0.99), 100)
#
# z = np.linspace(maxwell.ppf(0.01),
#                 maxwell.ppf(0.99), 100)
# Instead considering a 3d poisson Maxwell distribution to locate the ROS particle or distribute its concentration.
# We use spherical coordinates to use a 1d poisson Maxwell equation based on the radius of an imaginary sphere.
# Hence sphere grows in size with radius defined above respect to time. NOW need to transform ROS AMMPER data to
# ROS Diffusion lifetime AMMPER concentration data. <<< >>>> <<<< >>>> <<<< >>>
#
# a = [1,2 ,3 ,4 ,5 ]
# b = [1,2,3,4,4]
# c = [1, 0 , 2 ,3 ,4]
#
# M = [[3, 3, 3],[3,2,3], [4,3,2]]
#
# Tat = np.vstack([a,b,c]).T
# print(Tat)
#
# print(np.vstack([Tat, M]))




########################## Testing bivariate normal distributions

rxn = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
ryn = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])

rx, ry = np.meshgrid(rxn, ryn)
pos = np.dstack((rx, ry))

# Inputs for matrix controls time, and other quantities
# Here we should change inputs for H2O2 and OH physical parameters
rv = multivariate_normal([0, 0], [[4, -2], [-2, 4]])
rv2 = multivariate_normal([0, 0], [[4, -1], [-1, 4]])
# fractional Concentration position value

f = rv.pdf(pos)
f2 = rv2.pdf(pos)

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(rx,ry,f)
plt.figure(2)
ax2 = plt.axes(projection='3d')
ax2.plot_surface(rx,ry,f2)
#
plt.show()