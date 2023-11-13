#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import math
from math import exp
import langevin_dynamics as ld
import random
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
get_ipython().run_line_magic('matplotlib', 'inline')


#plot potential energy surface contours
N = 100
x_vec = np.linspace(-3.5, 3.5, N)
y_vec = np.linspace(-3.5, 3.5, N)
energy = np.zeros((N, N))
for i in range(len(x_vec)):
    for j in range(len(y_vec)):
        energy[i][j] = ld.potential(x_vec[i],y_vec[j])
plt.contour(x_vec,y_vec,energy,[-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3])
plt.xlabel('x')
plt.ylabel('y')
plt.show

