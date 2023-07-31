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
ss = 1
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
    # posterior_v.eval_mean(cg_max=1000, method='cg_my')
    # posterior_v.eval_mean(cg_max=5000, method='bicgstab')
    # posterior_v.eval_mean(cg_max=5000, method='cgs')
    # posterior_v.eval_mean_iterative(iter_num=100, cg_max=2000, method="cg_my")
    posterior_v.eval_mean_iterative(iter_num=500, cg_max=1000, method="bicgstab")
    # posterior_v.eval_mean_iterative(iter_num=500, cg_max=1000, method="cg")
    # posterior_v.calculate_eigensystem(num_eigval=40, method="scipy_eigsh", cut_val=0.0)
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

posterior_v.calculate_eigensystem_lam(num_eigval=80, method="scipy_eigsh", cut_val=0.0)
posterior_v.mean = v_fun.vector()[:]
np.save(result_figs_dir + 'lam.npy', [model.lam_dis.mean, model.lam_dis.cov])
np.save(result_figs_dir + 'errors'+ "_" + str(equ_nx) + '.npy', errors)
np.save(result_figs_dir + 'steperrors'+ "_" + str(equ_nx) + '.npy', step_errs)
########################################################## Plt Estimate and Truth
  

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

########################################################## Calculate covariance
xx = np.linspace(0, 1, 101)
xx = np.array([xx]).T
cov = posterior_v.pointwise_variance_field_lam(xx, xx)
cov_prior = prior_measure.pointwise_variance_field(xx, xx)
np.save(result_figs_dir + 'cov.npy', cov*model.lam*model.lam)


import matplotlib as mpl
from matplotlib import cm
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter

fig = plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

with plt.style.context('seaborn-ticks','ggplot'):
    
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    ax = plt.axes(projection='3d')
    xplt = np.linspace(0, 1, cov.shape[0])
    yplt = np.linspace(0, 1, cov.shape[1])
    Xplt, Yplt = np.meshgrid(xplt, yplt)
    
    ax.plot_surface(Xplt, Yplt, np.mat(cov)*model.lam*model.lam, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.set_zlim3d(zmin = -0.0010, zmax = 0.0015)
    ax.grid(False)
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.zaxis.set_major_formatter(formatter)
    # ax.set_title('MCMC Covariance')
    plt.savefig(result_figs_dir + 'Covarianceu.png', dpi=400)
    plt.close()
    # plt.show()


fig = plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

with plt.style.context('seaborn-ticks','ggplot'):
    
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    ax = plt.axes(projection='3d')
    xplt = np.linspace(0, 1, cov.shape[0])
    yplt = np.linspace(0, 1, cov.shape[1])
    Xplt, Yplt = np.meshgrid(xplt, yplt)
    ax.plot_surface(Xplt, Yplt, np.mat(cov), rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
    
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.set_zlim3d(zmin = -0.0010, zmax = 0.0015)
    ax.grid(False)
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.zaxis.set_major_formatter(formatter)
    # ax.set_title('MCMC Covariance')
    plt.savefig(result_figs_dir + 'Covariance.png', dpi=400)
    plt.close()

v_std = np.sqrt(cov.diagonal()) * model.lam
prior_std = np.sqrt(cov_prior.diagonal())
xx = np.linspace(0, 1, 101)
true_val = []
est_val = []
for idx, x in enumerate(xx):
    true_val.append(truth_fun(x))
    est_val.append(u_fun(x))
true_val = np.array(true_val)
est_val = np.array(est_val)


def to_percent(temp, position):
    return '%.f'%(100 * temp) + '%'

result_figs_dir = RESULT_DIR + "Fig/GibbsSampler/"
estg_val = np.load(result_figs_dir + 'u_vec.npy')
estg_val = estg_val[::-1]
result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"

plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

with plt.style.context('seaborn-ticks','ggplot'):
    
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5

    plt.figure()
    xx = np.linspace(0, 1, 101)
    xxx = np.linspace(0, 100, 101)
    plt.plot(est_val, color="blue", label='NCP-iMFVI')
    plt.plot(estg_val, color="orange", linestyle='dashed', label='Gibbs')
    labels = [0, 20, 40, 60, 80, 100]
    labelss = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(labels, labelss)
    plt.legend(frameon='True', loc='upper right')
    plt.title("Estimate")
    plt.savefig(result_figs_dir + 'Estimateg.png', dpi=400)
    plt.close()


########################################################## Plt Credibility region
    plt.figure()
    xx = np.linspace(0, 1, 101)
    xxx = np.linspace(0, 100, 101)
    plt.plot(est_val, color="black", label='NCP-iMFVI')
    plt.plot(estg_val, color="orange", linestyle='dashed', label='Gibbs')
    plt.plot(true_val, color="red", label='Truth')
    plt.plot(est_val + 2*v_std, color="green", alpha=0.5)
    plt.plot(est_val - 2*v_std, color="green", alpha=0.5)
    plt.fill_between(xxx, est_val + 2*v_std, est_val - 2*v_std, 
                      facecolor='green', alpha=0.1)

    labels = [0, 20, 40, 60, 80, 100]
    labelss = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(labels, labelss)
    plt.legend(frameon='True', loc='upper right')
    plt.title("Estimate")
    plt.savefig(result_figs_dir + 'Estimatee.png', dpi=400)
    plt.close()

########################################################## Plt Lambdas    
    length = len(lams)
    xx = list(range(length))
    xx = np.array([xx[i]+1 for i in range(length)])
    y = lams[:length]
    yy = []
    for i in range(len(xx)):
        yy.append('%.1f' % y[i])
    
    plt.plot(xx, y)
    for a, b, c in zip(xx, y, yy):
        if a == 1:
            plt.text(a, b, c, ha='center', va='top', fontsize=10)
        if a % 500 == 0:
            plt.text(a, b, c, ha='center', va='bottom', fontsize=10)
    plt.title("Lambda")
    plt.xlabel('Iteration Number')
    plt.savefig(result_figs_dir + 'lams.png', dpi=400)
    plt.close()

########################################################## Plt Relative errors
    plt.figure()
    length = len(errors)
    length = 100
    xx = list(range(length))
    xx = np.array([xx[i]+1 for i in range(length)])
    y = errors[:length]
    
    yy = []
    for i in range(len(xx)):
        yy.append(format(errors[i], '.1%'))
    
    plt.plot(xx, y) 
    for a, b, c in zip(xx, y, yy):
        if a == 1:
            plt.text(a, b, c, ha='center', va='bottom', fontsize=10)
        if a % 20 == 0:
            plt.text(a, b, c, ha='center', va='bottom', fontsize=10)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.title('Relative Errors')
    plt.xlabel('Iteration Number')
    plt.savefig(result_figs_dir + 'Relative.png', dpi=400)
    plt.close()




