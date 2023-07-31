#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 10:58:54 2023

@author: ubuntu
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
from core.approximate_sample import LaplaceApproximate

from NCP_MFVI.common_simple_elliptic import EquSolver, ModelSimpleEllipticNCP, \
            GaussianLam, PosteriorOfV
from NCP_MFVI.common_simple_elliptic import relative_error


## set data and result dir
DATA_DIR = './DATA/'
RESULT_DIR = './RESULTS/'
result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"
noise_level = 0.05

## domain for solve PDE
equ_nx = 100
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)

## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain.function_space)

## setting the prior measure
alpha = 1
a_fun = 1
ss = 0.5
# ss = 1e-4
alpha = ss*alpha
a_fun = ss*a_fun
prior_measure = GaussianElliptic2(
    domain=domain, alpha=alpha, a_fun=a_fun, theta=ss, boundary="Neumann"
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
noise_level_ = noise_level*max(abs(d_clean))
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)

lam_dis = GaussianLam(mean=1, cov=10000)

## setting the Model
model = ModelSimpleEllipticNCP(
    d=d, domain=domain, prior=prior_measure, noise=noise, 
    equ_solver=equ_solver, lam_dis=lam_dis
    )

lams = []
errors = []
step_errs = []
IterNum = np.int64(5e3)
for idx in range(IterNum):
    lams.append(model.lam_dis.mean)
    
    posterior_v = PosteriorOfV(model=model)
    posterior_v.eval_mean_iterative(iter_num=500, cg_max=1000, method="bicgstab")
    posterior_v.calculate_eigensystem_lam(num_eigval=80, method="double_pass", cut_val=0.0)

    model.update_lam_dis(posterior_v.mean, posterior_v.eigval)
    model.update_lam()
    
    u_fun = fe.Function(domain.function_space)
    u_fun.vector()[:] = np.array(model.lam*posterior_v.mean)
    v_fun = fe.Function(domain.function_space)
    v_fun.vector()[:] = np.array(posterior_v.mean)
    
    if idx % 1 == 0:
        errors.append(relative_error(u_fun, truth_fun, domain))
        print("lam_mean, lam_cov, IterNum = ", model.lam_dis.mean, model.lam_dis.cov, idx)
   
    if idx > 2:
    
        err_now = errors[-1]
        err_pre = errors[-2]
        step_err = (err_now - err_pre)**2 / err_now**2
        step_errs.append(step_err)
        
        lam_now = lams[-1]
        lam_pre = lams[-2]
        step_lam = (lam_now - lam_pre)**2 / lam_now**2
    
        if step_err < 1e-4 and step_lam < 2 * 1e-13:
            print("----------------------------------------")
            print('The covgerence progress is done, and the iteration number is ', idx)
            break

np.save(result_figs_dir + 'lam0.5.npy', [model.lam_dis.mean, model.lam_dis.cov])
########################################################## Plt Estimate and Truth


import matplotlib as mpl
from matplotlib import cm
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter



from scipy.stats import norm
import math
lam_ncp = np.load(result_figs_dir + 'lam.npy')
lam_ncphalf = np.load(result_figs_dir + 'lam0.5.npy')


with plt.style.context('seaborn-ticks','ggplot'):
    plt.figure()
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    
    x_axis = np.arange(300, 325, 0.01) 
    y = norm.pdf(x_axis, lam_ncp[0], np.sqrt(lam_ncp[1]))
    yy = norm.pdf(lam_ncp[0], lam_ncp[0], np.sqrt(lam_ncp[1]))
    plt.figure(figsize=(6, 5))
    plt.plot(x_axis, y, color='green', label = 'NCP-iMFVI')
    plt.fill_between(x_axis, y, 0, alpha=0.3, color='g')
    plt.vlines(lam_ncp[0], 0, yy, colors='black', linestyles='dashed')
    plt.title('Lambda Density')
    plt.legend()
    plt.savefig(result_figs_dir + 'lam.png', dpi=400)

    
    x_axis = np.arange(168, 185, 0.01) 
    y = norm.pdf(x_axis, lam_ncphalf[0], np.sqrt(lam_ncphalf[1]))
    yy = norm.pdf(lam_ncphalf[0], lam_ncphalf[0], np.sqrt(lam_ncphalf[1]))
    plt.figure(figsize=(6, 5))
    plt.plot(x_axis, y, color='green', label = 'NCP-iMFVI')
    plt.fill_between(x_axis, y, 0, alpha=0.3, color='g')
    plt.vlines(lam_ncphalf[0], 0, yy, colors='black', linestyles='dashed')
    plt.title('Lambda Density')
    plt.legend()
    plt.savefig(result_figs_dir + 'lamhalf.png', dpi=400)






