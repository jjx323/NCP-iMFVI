#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:52:40 2022

@author: jjx323
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt

import fenics as fe
import dolfin as dl

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
# os.chdir("/home/ishihara/Desktop/SIMIP202002/aa")
from core.model import Domain2D
from core.probability import GaussianElliptic2
from core.misc import save_expre, generate_points

from NCP_MFVI.common_Darcy import EquSolver



DATA_DIR = './NCP_MFVI/Darcy/DATA/'
RESULT_DIR = './NCP_MFVI/Darcy/RESULTS/'
result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"

## domain for solving PDE
equ_nx = 300
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)

alpha = 1.0
a_fun = 1.0
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=alpha, a_fun=a_fun, theta=1, boundary='Neumann'
    )

## generate a sample, set this sample as the ground truth
truth_fun = fe.Function(domain_equ.function_space)
exp1 = "exp(-20.0*pow(x[0]-0.3, 2) - 20.0*pow(x[1]-0.4, 2)) + "
exp2 = "exp(-20.0*pow(x[0]-0.7, 2) - 20.0*pow(x[1]-0.6, 2))"
mtruthexp = fe.Expression(exp1 + exp2, degree=5)
truth_fun = fe.interpolate(mtruthexp, domain_equ.function_space)
# truth_fun.vector()[:] = prior_measure.generate_sample()
#fe.plot(truth_fun)
## save the truth
os.makedirs(DATA_DIR, exist_ok=True)
np.save(DATA_DIR + 'truth_vec', truth_fun.vector()[:])
file1 = fe.File(DATA_DIR + "truth_fun.xml")
file1 << truth_fun
file2 = fe.File(DATA_DIR + 'saved_mesh_truth.xml')
file2 << domain_equ.function_space.mesh()

## load the ground truth
truth_fun = fe.Function(domain_equ.function_space)
truth_fun.vector()[:] = np.load(DATA_DIR + 'truth_vec.npy')

## specify the measurement points
num_x, num_y = 20, 20
x = np.linspace(0.01, 0.99, num_x)
y = np.linspace(0.01, 0.99, num_y)
coordinates = generate_points(x, y)
np.save(DATA_DIR + "coordinates_2D", coordinates)

## construct a solver to generate data
f_expre = "sin(x[0])*cos(x[1])"
# f_expre = "10 * sin(x[0])*cos(x[1])"
## If we change f to be f_expre = "sin(a*pi*x[0])*sin(a*pi*x[1])" with a == 10,
## the nonlinear behavior may increase. And all of the optimization methods will
## not work well. 
f = fe.Expression(f_expre, degree=5)
save_expre(DATA_DIR + 'f_2D.txt', f_expre)

equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=coordinates)

sol = fe.Function(domain_equ.function_space)
sol.vector()[:] = equ_solver.forward_solver()
clean_data = [sol(point) for point in coordinates]
np.save(DATA_DIR + 'measurement_points_2D', coordinates)
np.save(DATA_DIR + 'measurement_clean_2D', clean_data)
data_max = max(clean_data)
## add noise to the clean data
noise_levels = [0.01, 0.05, 0.1]
for noise_level in noise_levels:
    np.random.seed(0)
    data = clean_data + noise_level*data_max*np.random.normal(0, 1, (len(clean_data),))
    path = DATA_DIR + 'measurement_noise_2D' + '_' + str(noise_level)
    np.save(path, data)

    











