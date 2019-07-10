import math
import sys
from math import exp
import numpy as np
import argparse

def potential(x,y):
    return 0.02*(x**4+y**4) - 4*exp(-((x+2)**2 + (y+2)**2)) - 4*exp(-((x-2)**2 + (y-2)**2)) + 0.3*(x-y)**2 + 0.0026

def kinetic_energy(px,py):
    return 0.5*(px*px+py*py) 

def force(x,y,px,py,dt,beta,gamma):
    std_dev = math.sqrt(2.0*gamma/(beta*dt))
    dV_dx = 0.08*x**3 + 8*(x-2)*exp(-(x-2)**2-(y-2)**2) + 8*(x+2)*exp(-(x+2)**2-(y+2)**2) + 0.6*(x-y)
    dV_dy = -0.6*(x-y) + 8*(y-2)*exp(-(x-2)**2-(y-2)**2) + 8*(y+2)*exp(-(x+2)**2-(y+2)**2) + 0.08*y**3
    fx = -dV_dx - gamma*px + np.random.normal(0,std_dev)
    fy = -dV_dy - gamma*py + np.random.normal(0,std_dev)
    return fx,fy

def vv_step(phasepoint,dt,beta,gamma):
    x = phasepoint[0]
    y = phasepoint[1]
    px = phasepoint[2]
    py = phasepoint[3]
    fx = phasepoint[4]
    fy = phasepoint[5]
    px = px + (1/2)*dt*fx
    py = py + (1/2)*dt*fy
    x = x + dt*px
    y = y + dt*py
    fx,fy = force(x,y,px,py,dt,beta,gamma)
    px = px + (1/2)*dt*fx
    py = py + (1/2)*dt*fy
    return np.asarray([x,y,px,py,fx,fy])

def calc_op(op_type,x,y):
    if op_type == 1:
        return x
    elif op_type == 2:
        return y
    elif op_type == 3:
        return x + y
    else:
        sys.exit("Invalid choice of OP")
