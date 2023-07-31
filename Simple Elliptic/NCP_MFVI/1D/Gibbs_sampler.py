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

from NCP_MFVI.common_simple_elliptic import EquSolver, ModelSimpleEllipticLam, \
            GaussianLam, PosteriorOfV
from NCP_MFVI.common_simple_elliptic import relative_error



## set data and result dir
DATA_DIR = './DATA/'
RESULT_DIR = './RESULTS/'
result_figs_dir = RESULT_DIR + "Fig/GibbsSampler/"
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
alpha = ss*alpha
a_fun = ss*a_fun
prior_measure = GaussianElliptic2(
    domain=domain, alpha=alpha, a_fun=a_fun, theta=1, boundary="Neumann"
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

lam0 = 313  ## accelerate mixing
lam0std = 10000
lam_dis = GaussianLam(mean=lam0, cov=lam0std)

## setting the Model
model = ModelSimpleEllipticLam(
    d=d, domain=domain, prior=prior_measure, noise=noise, 
    equ_solver=equ_solver, lam_dis=lam_dis
    )


errors = []
lams = []
vks = []
uks = []
IterNum = np.int64(1e3)
vk = fe.interpolate(truth_fun, domain.function_space).vector()[:]/lam0  ## accelerate mixing
sample_fun = fe.Function(domain.function_space)
vks.append(vk.copy())
AccNumV = 0
AccNumLam = 0
for idx in range(IterNum):
    vks = []
    
    lams.append(model.lam)
        
    posterior_v = PosteriorOfV(model=model)
    
    posterior_v.eval_mean_iterative(iter_num=500, cg_max=1000, method="bicgstab")
    # posterior_v.calculate_eigensystem(num_eigval=40, method="double_pass", cut_val=0.0)
    posterior_v.calculate_eigensystem_lam(num_eigval=80, method="scipy_eigsh", cut_val=0.0)

    # calculate the posterior variance
    posterior_v.set_mean(posterior_v.mean)
    
    ## sampling from the posterior measure 

    vk = posterior_v.generate_sample_lam()
    vks.append(vk.copy())
    uks.append(vk.copy()*model.lam)
    model.p.vector()[:] = model.equ_solver.forward_solver(vk)


    
    lam_hat = np.random.normal(model.lam, 0.01)
    
    tmp1 = model.lam*(model.S@model.p.vector()[:])
    tmp1 = (tmp1- model.noise.mean - model.d)
    tmp1 = tmp1@model.noise.precision@tmp1
    tmp1 = 0.5*tmp1
    
    tmp2 = lam_hat*(model.S@model.p.vector()[:])
    tmp2 = (tmp2 - model.noise.mean - model.d)
    tmp2 = tmp2@model.noise.precision@tmp2
    tmp2 = 0.5*tmp2
    
    tmp11 = 0.5*((model.lam - lam0)**2)/lam0std
    tmp22 = 0.5*((lam_hat - lam0)**2)/lam0std
    
    aa = np.exp(min(0, tmp1 + tmp11 - tmp2 - tmp22))
    tmp_num = np.random.uniform(0, 1)
    if tmp_num <= aa:
        model.lam = lam_hat 
        AccNumLam += 1
    else:
        model.lam = model.lam
        
        
    u_fun = fe.Function(domain.function_space)
    # u_fun.vector()[:] = np.array(model.lam*posterior_v.mean)
    u_fun.vector()[:] = np.array(model.lam*sample_fun.vector()[:])[::-1]


    if idx % 1 == 0:
        errors.append(relative_error(u_fun, truth_fun, domain))
        print("lam_mean, lam_cov = ", model.lam, model.lam_dis.cov)
        print("relative error = ", errors[-1])
        print("IterNum = ", idx)
        print("AccNumV = ", AccNumV/(idx + 1))
        print("AccNumLam = ", AccNumLam/(idx + 1))
    



lam_mean = np.mean(lams)

vks = np.array(vks)
v_mean = np.mean(vks, axis=0)
v_std = np.std(vks, axis=0)

uks = np.array(uks)
u_mean = np.mean(uks, axis=0)
np.save(result_figs_dir + 'u_vec.npy', u_mean)
u_std = np.std(uks, axis=0)

def cal_covariance(uchain, lamchain, meanlist):
    cov_matrix = np.eye(len(uchain[0]))
    for i in range(len(uchain[0])):
        for j in range(len(uchain[0])):
            mean_i = meanlist[i]
            mean_j = meanlist[j]
            sum = (uchain[:, i] - mean_i) @ (uchain[:, j] - mean_j)
            cov = sum / (len(lamchain) - 1)
            cov_matrix[i, j] = cov
    return cov_matrix

cov_v = cal_covariance(vks, lams, v_mean)
cov_u = cal_covariance(uks, lams, u_mean)


xx = np.linspace(0, 1, 101)
xx = np.array([xx]).T
cov = posterior_v.pointwise_variance_field_lam(xx, xx)
cov_prior = prior_measure.pointwise_variance_field(xx, xx)

# ########################################################## Plt Covariance   


result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"
covn = np.load(result_figs_dir + 'cov.npy')

result_figs_dir = RESULT_DIR + "Fig/GibbsSampler/"
# covg = cov * model.lam * model.lam
covg = cov_u



import matplotlib as mpl
from matplotlib import cm
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter

fig = plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

with plt.style.context('seaborn-ticks','ggplot'):
    plt.figure()
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    ax = plt.axes(projection='3d')
    xplt = np.linspace(0, 1, cov.shape[0])
    yplt = np.linspace(0, 1, cov.shape[1])
    Xplt, Yplt = np.meshgrid(xplt, yplt)
    
    # ax.plot_surface(Xplt, Yplt, np.mat(cov_v)*lam_mean*lam_mean, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
    ax.plot_surface(Xplt, Yplt, np.mat(covg), rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
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



fig = plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

with plt.style.context('seaborn-ticks','ggplot'):
    plt.figure()
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    ax = plt.axes(projection='3d')
    xplt = np.linspace(0, 1, cov.shape[0])
    yplt = np.linspace(0, 1, cov.shape[1])
    Xplt, Yplt = np.meshgrid(xplt, yplt)
    ax.plot_surface(Xplt, Yplt, np.mat(covn-covg), rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
    
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
    plt.savefig(result_figs_dir + 'Covariancediff.png', dpi=400)
    plt.close()

########################################################## Plt Variance



fig = plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

with plt.style.context('seaborn-ticks','ggplot'):
    plt.figure()
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    
    
    k = 0
    covn_sub = np.diagonal(covn, offset=k)
    covg_sub = np.diagonal(covg, offset=k)
    plt.figure()
    xx = np.linspace(0, 100-k, 101-k)
    plt.plot(covn_sub, color="blue", label='NCP-iMFVI')
    plt.plot(covg_sub, color="orange", linestyle='dashed', label='Gibbs')
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    labels = [0, 20, 40, 60, 80, 100]
    labelss = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(labels, labelss)
    # plt.plot()
    plt.savefig(result_figs_dir + 'Varianceu.png', dpi=400)
    plt.close()
    
    
    
    k = 20
    covn_sub = np.diagonal(covn, offset=k)
    covg_sub = np.diagonal(covg, offset=k)
    plt.figure()
    xx = np.linspace(0, 100-k, 101-k)
    plt.plot(covn_sub, color="blue", label='NCP-iMFVI')
    plt.plot(covg_sub, color="orange", linestyle='dashed', label='Gibbs')
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    labels = [0, 20, 40, 60, 80]
    labelss = [0, 0.2, 0.4, 0.6, 0.8]
    plt.xticks(labels, labelss)
    # plt.plot()
    plt.savefig(result_figs_dir + 'Varianceu20.png', dpi=400)
    plt.close()
    
    k = 40
    covn_sub = np.diagonal(covn, offset=k)
    covg_sub = np.diagonal(covg, offset=k)
    plt.figure()
    xx = np.linspace(0, 100-k, 101-k)
    plt.plot(covn_sub, color="blue", label='NCP-iMFVI')
    plt.plot(covg_sub, color="orange", linestyle='dashed', label='Gibbs')
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    labels = [0, 10, 20, 30, 40, 50, 60]
    labelss = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    plt.xticks(labels, labelss)
    # plt.plot()
    plt.savefig(result_figs_dir + 'Varianceu40.png', dpi=400)
    plt.close()

# ##########################################################  Plot Lambda Comparison

from scipy.stats import norm
def cal_cov_lam(mean):
    p = equ_solver.forward_solver(mean)
    Sp = model.S@p
    temp = Sp@spsl.spsolve(noise.covariance, Sp)
    
    
    temp1 = temp + 1/lam0std
    # rho = self.lam_dis.cov + self.lam*self.lam
    cov = 1/temp1
    
    return cov

lam_mean = np.mean(lams)
lam_cov = cal_cov_lam(v_mean)

## load NCP-iMFVI lambda
result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"
lam_ncp = np.load(result_figs_dir + 'lam.npy')

result_figs_dir = RESULT_DIR + "Fig/GibbsSampler/"
with plt.style.context('seaborn-ticks','ggplot'):
    plt.figure()
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5

    plt.figure(figsize=(6, 5))
    x_axis = np.arange(280, 340, 0.01) 
    plt.plot(x_axis, norm.pdf(x_axis, lam_ncp[0], np.sqrt(lam_ncp[1])), color='red', linestyle='dashed', label = 'NCP-iMFVI')
    plt.plot(x_axis, norm.pdf(x_axis, lam_mean, np.sqrt(lam_cov)), color='blue', label = 'Gibbs')
    plt.title('Lambda Density')
    plt.legend()
    plt.savefig(result_figs_dir + 'lams.png', dpi=400)
    
    
def cal_KL(lamG, lamN, CG, CN):
    KL = np.log(np.sqrt(CG/CN)) + 0.5 * (CN - CG) / CG + 0.5 * (lamG - lamN)**2 / CG
    return KL

KL = cal_KL(lam_mean, lam_ncp[0], lam_cov, lam_ncp[1])

