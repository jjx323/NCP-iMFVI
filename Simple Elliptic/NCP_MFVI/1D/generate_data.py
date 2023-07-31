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
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.misc import save_expre

from NCP_MFVI.common_simple_elliptic import EquSolver


DATA_DIR = './DATA/'

## domain for solving PDE
equ_nx = 1000
domain = Domain1D(low_point=0, high_point=1, n=equ_nx, mesh_type='P', mesh_order=1)

truth_fun = fe.interpolate(
    fe.Expression('10*(cos(4*pi*x[0])+1)', degree=5), domain.function_space
    )
# sss = 0.1
# prior_measure = GaussianElliptic2(
#     domain=domain, alpha=sss, a_fun=sss, theta=1, boundary="Neumann"
#     )
# truth_fun = fe.Function(domain.function_space)
# truth_fun.vector()[:] = prior_measure.generate_sample()
#fe.plot(truth_fun)
## save the truth
os.makedirs(DATA_DIR, exist_ok=True)
np.save(DATA_DIR + 'truth_vec', truth_fun.vector()[:])
file1 = fe.File(DATA_DIR + "truth_fun.xml")
file1 << truth_fun
file2 = fe.File(DATA_DIR + 'saved_mesh_truth.xml')
file2 << domain.function_space.mesh()

## load the ground truth
truth_fun = fe.Function(domain.function_space)
truth_fun.vector()[:] = np.load(DATA_DIR + 'truth_vec.npy')

## specify the measurement points
coordinates = np.linspace(0, 1, 50)

## construct a solver to generate data
equ_solver = EquSolver(
    domain=domain, points=np.array([coordinates]).T, 
    m=truth_fun, alpha=0.05
    )
sol = fe.Function(domain.function_space)
sol.vector()[:] = np.array(equ_solver.forward_solver(truth_fun.vector()[:]))
clean_data = [sol(point) for point in coordinates]
np.save(DATA_DIR + 'measurement_points_1D', coordinates)
np.save(DATA_DIR + 'measurement_clean_1D', clean_data)
data_max = max(np.abs(clean_data))
## add noise to the clean data
noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
for noise_level in noise_levels:
    np.random.seed(0)
    data = clean_data + noise_level*data_max*np.random.normal(0, 1, (len(clean_data),))
    path = DATA_DIR + 'measurement_noise_1D' + '_' + str(noise_level)
    np.save(path, data)

    
# u_ = fe.TrialFunction(domain.function_space) 
# v_ = fe.TestFunction(domain.function_space)
# M = fe.assemble(fe.inner(u_, v_)*fe.dx)
# FF = fe.assemble(
#     fe.inner(u_,v_)*fe.dx + fe.Constant(0.05)*fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx
#     )
# b = fe.assemble(fe.inner(truth_fun, v_)*fe.dx)
# def boundary(x, on_boundary):
#     return on_boundary
# bc = fe.DirichletBC(domain.function_space, fe.Constant('0.0'), boundary)
# bc.apply(FF, b)
# ff = fe.Function(domain.function_space)
# fe.solve(FF, ff.vector(), b)










