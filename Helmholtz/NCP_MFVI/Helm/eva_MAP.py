#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:26:54 2023

@author: ishihara
"""

import numpy as np
import fenics as fe
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.misc import my_project, trans2spnumpy, construct_measurement_matrix, \
    make_symmetrize
from NCP_MFVI.common_Helm import EquSolver, ModelHelmNCP, Domain2DPML

DATA_DIR = './DATA/'
RESULT_DIR = './RESULT/MAP/'
noise_level = 0.05

## domain for solve PDE
equ_nx = 40
domainPML = Domain2DPML(nx=equ_nx, ny=equ_nx, dPML=0.1, xx=1.0, yy=1.0)

mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
P1 = fe.FiniteElement('P', fe.triangle, 1)
element = fe.MixedElement([P1, P1])
V_truth = fe.FunctionSpace(mesh_truth, element)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')

truth_fun = fe.interpolate(truth_fun, domainPML.function_space)

## setting the prior measure
prior_measure = GaussianElliptic2(
    domain=domainPML, alpha=0.001, a_fun=0.01, theta=1, boundary="Neumann", img=True
    )


freqs = np.load(DATA_DIR + "freqs" + ".npy")

equ_solver = EquSolver(
    domainPML=domainPML, m=None, freq=freqs[0], num_points=20
    )


## load the measurement data
d = np.load(DATA_DIR + "dn" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "dc" + ".npy")

## setting the noise
len_points  = equ_solver.Shybird.shape[0]

noise_level_ = noise_level*max(d_clean)
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)












