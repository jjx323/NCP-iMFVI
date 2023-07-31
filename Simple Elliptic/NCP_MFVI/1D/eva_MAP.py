#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:27:41 2022

@author: jjx323
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

from NCP_MFVI.common import EquSolver, ModelSimpleElliptic


## set data and result dir
DATA_DIR = './DATA/'
RESULT_DIR = './RESULT/MAP/'
noise_level = 0.1

## domain for solve PDE
equ_nx = 100
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)

## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain.function_space)

## setting the prior measure
prior_measure = GaussianElliptic2(
    domain=domain, alpha=0.001, a_fun=0.01, theta=1, boundary="Neumann"
    )

## loading coordinates of the measurement points
measurement_points = np.load(DATA_DIR + "measurement_points_1D.npy")

## setting the forward problem
equ_solver = EquSolver(
    domain=domain, points=np.array([measurement_points]).T,
    m=None, alpha=0.05
    )

## load the measurement data
d = np.load(DATA_DIR + "measurement_noise_1D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_1D.npy")

## setting the noise
noise_level_ = noise_level*max(d_clean)
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)

## setting the Model
model = ModelSimpleElliptic(
    d=d, domain=domain, prior=prior_measure, noise=noise, 
    equ_solver=equ_solver
    )

## set optimizer NewtonCG
newton_cg = NewtonCG(model=model)
max_iter = 100
loss_pre, _, _ = model.loss()
for itr in range(max_iter):
    newton_cg.descent_direction(cg_max=500, method='cg_my')
    # newton_cg.descent_direction(cg_max=500, method='bicgstab')
    newton_cg.step(method='armijo', show_step=False)
    loss, _, _ = model.loss()
    print("iter = %d/%d, loss = %.4f" % (itr+1, max_iter, loss))
    if newton_cg.converged == False:
        break
    if np.abs(loss - loss_pre) < 1e-3*loss:
        print("Iteration stoped at iter = %d" % itr)
        break 
    loss_pre = loss

m_newton_cg = fe.Function(domain.function_space)
m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())
# m_newton_cg.vector()[:] = np.array(init_mk.vector()[:].copy())
model.update_m(m_newton_cg.vector()[:], update_sol=True)
d_est = model.S@model.p.vector()[:]

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
fe.plot(m_newton_cg, label='estimate')
fe.plot(truth_fun, label='truth')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(d_est, label='d_est')
plt.plot(d, label='d')
plt.legend()
plt.title('NewtonCG')


# ## set optimizer GradientDescent
# model.update_m(
#     fe.interpolate(fe.Constant(0.0), domain.function_space).vector()[:], update_sol=True
#     )
# gradient_descent = GradientDescent(model=model)
# max_iter = 2000
# loss_pre, _, _ = model.loss()
# for itr in range(max_iter):
#     gradient_descent.descent_direction()
#     gradient_descent.step(method='armijo', show_step=False)
#     if gradient_descent.converged == False:
#         break
#     loss, _, _ = model.loss()
#     if itr % 100 == 0:
#         print("iter = %d/%d, loss = %.4f" % (itr+1, max_iter, loss))
#     if np.abs(loss - loss_pre) < 1e-5*loss:
#         print("Iteration stoped at iter = %d" % itr)
#         break 
#     loss_pre = loss

# m_gradient_descent = fe.Function(domain.function_space)
# m_gradient_descent.vector()[:] = np.array(gradient_descent.mk.copy())
# model.update_m(m_gradient_descent.vector()[:], update_sol=True)
# d_est = model.S@model.p.vector()[:]

# plt.figure(figsize=(13, 5))
# plt.subplot(1, 2, 1)
# fe.plot(m_gradient_descent, label='estimate')
# fe.plot(truth_fun, label='truth')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(d_est, label='d_est')
# plt.plot(d, label='d')
# plt.legend()
# plt.title('Gradient Descent')
# plt.show()




















