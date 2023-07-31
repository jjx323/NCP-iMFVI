#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:05:26 2023

@author: ubuntu
"""

import numpy as np
import fenics as fe
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
# os.chdir("/home/ishihara/Desktop/SIMIP202002/aa")
from core.model import Domain2D
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.approximate_sample import LaplaceApproximate
from core.misc import load_expre, smoothing, generate_points


from NCP_MFVI.common_Darcy import EquSolver_linear, ModelDarcyFlow_linear, \
            GaussianLam, PosteriorOfV, EquSolver
from NCP_MFVI.common_Darcy import relative_error


## set data and result dir
DATA_DIR = './NCP_MFVI/Darcy/DATA/'
RESULT_DIR = './NCP_MFVI/Darcy/RESULTS/'
result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"

## domain for solve PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)

## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain.function_space)

## setting the prior
ss = 40
prior_measure = GaussianElliptic2(
    domain=domain, alpha=ss, a_fun=ss, theta=0.1, boundary='Neumann'
    )

## load m_Map
mMap = np.load(DATA_DIR + "m_Map.npy")
mMap = np.array(mMap)

## loading coordinates of the measurement points
measurement_points = np.load(DATA_DIR + "coordinates_2D.npy")

## construct a solver to generate data
f_expre = "sin(x[0])*cos(x[1])"
# f_expre = "10 * sin(x[0])*cos(x[1])"
## If we change f to be f_expre = "sin(a*pi*x[0])*sin(a*pi*x[1])" with a == 10,
## the nonlinear behavior may increase. And all of the optimization methods will
## not work well. 
f = fe.Expression(f_expre, degree=5)

equ_solver_ = EquSolver(domain_equ=domain, m=truth_fun, f=f, points=measurement_points)

## load p0
p0 = np.load(DATA_DIR + "p0.npy")

equ_solver = EquSolver_linear(
    domain_equ=domain, mMap=mMap, points=measurement_points,
    p0=p0
    )

##### test non-linear
# d_clean = equ_solver.S @ equ_solver_.forward_solver(truth_fun.vector()[:])
# dd = np.array(equ_solver.S @ (equ_solver_.forward_solver(truth_fun.vector()[:])-p0)).squeeze()


d_clean = np.load(DATA_DIR + "measurement_clean_2D.npy")
d_linear = np.load(DATA_DIR + "measurement_2D_linear.npy")
d_linear_clean = np.load(DATA_DIR + "measurement_2D_linear_clean.npy")

## setting the noise

data_max = abs(max(d_clean))
noise_level = 0.05
# noise_level = 0.1
# dn = dd + noise_level*data_max*np.random.normal(0, 1, (len(dd),))
dn = d_linear

noise_level_ = noise_level*max(abs(d_clean))
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)


## setting lambda dis
mean0 = 1
cov0 = 1e2
lam_dis = GaussianLam(mean=mean0, cov=cov0)

## setting the Model
model = ModelDarcyFlow_linear(
    d=dn, domain_equ=domain, prior=prior_measure, 
    noise=noise, equ_solver=equ_solver, lam_dis=lam_dis
    )

## set optimizer NewtonCG
newton_cg = NewtonCG(model=model)
# posterior_v = PosteriorOfV(model=model, newton_cg=newton_cg)
m_newton_cg = fe.Function(domain.function_space)
max_iter = 1

relative_errors = []
lams = []
covs = []
for i in range(2000):
    
    
    posterior_v = PosteriorOfV(model=model, newton_cg=newton_cg)
    ## Without a good initial value, it seems hard for us to obtain a good solution
    # init_fun = smoothing(truth_fun, alpha=0.1)
    # newton_cg.re_init(init_fun.vector()[:])
    # loss_pre = model.loss()[0]
    # for itr in range(max_iter):
    #     newton_cg.descent_direction(cg_max=30, method='cg_my')
    #     # newton_cg.descent_direction(cg_max=30, method='bicgstab')
    #     # print(newton_cg.hessian_terminate_info)
    #     newton_cg.step(method='armijo', show_step=False)
    #     # if newton_cg.converged == False:
    #     #     break
    #     loss = model.loss()[0]
    #     print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
    #     if np.abs(loss - loss_pre) < 1e-3*loss:
    #         print("Iteration stoped at iter = %d" % itr)
    #         break 
    #     loss_pre = loss
    
    
    # rho = lam_dis.mean*lam_dis.mean + lam_dis.cov

    # m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy()) * lam_dis.mean / np.sqrt(rho)
    # # m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())
    
    # relative_errors.append(relative_error(m_newton_cg, truth_fun, domain))
    # print("error = ", relative_errors[-1])
    
    # rho = lam_dis.mean*lam_dis.mean + lam_dis.cov**2
    # vk = posterior_v.mean
    # vk = m_newton_cg.vector()[:] * lam_dis.mean / (rho)
    # vk = m_newton_cg.vector()[:] / np.sqrt(rho)
    
    # m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())
    # d_est = model.equ_solver.S@model.equ_solver.forward_solver(m_newton_cg.vector()[:])
    
    
    # posterior_v.eval_mean_iterative(iter_num=200, cg_max=30, method="cg_my")
    posterior_v.eval_mean_iterative(iter_num=1, cg_max=50, method="cg_my")
    posterior_v.calculate_eigensystem_lam(num_eigval=100, method="double_pass", cut_val=0.0)
    m_newton_cg.vector()[:] = np.array(posterior_v.mean)
    
    relative_errors.append(relative_error(m_newton_cg, truth_fun, domain))
    print("error = ", relative_errors[-1])
    
    # Hvk = equ_solver.S @ np.array(equ_solver.forward_solver(vk)).squeeze()
    # Hvk2 = Hvk@spsl.spsolve(noise.covariance, Hvk)
    # temp1 = Hvk2 + 1/cov0
    
    # eigval = posterior_v.eigval_lam / rho
    # temp2 = np.sum(eigval/(rho*eigval + 1))
    # # temp2 = 0
    # lam_dis.cov = 1/(temp1 + temp2)
    
    # tmp1 = dn @spsl.spsolve(noise.covariance, Hvk)
    # lam_dis.mean = lam_dis.cov*(tmp1 + mean0/cov0)
    
    # print('*********', tmp1, Hvk2)
    model.update_lam_dis_(m_newton_cg.vector()[:], posterior_v.eigval_lam)
    print('lam_mean, lam_cov, idx = ', lam_dis.mean, lam_dis.cov, i)
    lams.append(lam_dis.mean)
    covs.append(lam_dis.cov)
    
    model.update_lam()
    
    if i > 2:
    
        err_now = relative_errors[-1]
        err_pre = relative_errors[-2]
        step_err = (err_now - err_pre)**2 / err_now**2
        
        lam_now = lams[-1]
        lam_pre = lams[-2]
        step_lam = (lam_now - lam_pre)**2 / lam_now**2
    
        if step_err < 1e-4 and step_lam < 1e-5:
            print("----------------------------------------")
            print('The covgerence progress is done, and the iteration number is ', i)
            break



posterior_v.calculate_eigensystem_lam(num_eigval=100, method="scipy_eigsh", cut_val=0.0)
posterior_v.mean = m_newton_cg.vector()[:]
tmp_fun = fe.Function(domain.function_space)
tmps = []
for i in range(150):
    tmp = posterior_v.generate_sample_lam()
    tmps.append(tmp)
tmps = np.array(tmps)


tmp_var = np.var(tmps, axis=0)
tmp_fun.vector()[:] = np.sqrt(tmp_var*model.lam)
samp_var = tmp_fun.compute_vertex_values()
samp_var = (np.array(samp_var).reshape((51, 51)))
v_std = samp_var.diagonal()




fun = fe.Function(domain.function_space)

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
fig = fe.plot(m_newton_cg, label='estimate')
plt.colorbar(fig)
plt.legend()
plt.title("Estimated Function")
plt.subplot(1, 3, 2)
fig = fe.plot(truth_fun, label='truth')
plt.colorbar(fig)
plt.title("Truth")
plt.legend()
plt.subplot(1, 3, 3)
# fig = plt.imshow(samp_var, origin='lower')
fig = fe.plot(tmp_fun)
plt.colorbar(fig)
plt.title('Sample')
plt.show()


########################################################## Plt Estimate and Truth
  
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.5

plt.figure()
fig = fe.plot(m_newton_cg)
plt.colorbar(fig)
plt.title('Estimate')
plt.savefig(result_figs_dir + 'Estimate.png', dpi=400)
plt.close()

plt.figure()
fig = fe.plot(truth_fun)
plt.colorbar(fig)
plt.title("Truth")
plt.savefig(result_figs_dir + 'Truth.png', dpi=400)
plt.close()

plt.figure()
fig = fe.plot(tmp_fun)
plt.colorbar(fig)
plt.title('Point-wise Variance Field')
plt.savefig(result_figs_dir + 'Pointwise.png', dpi=400)
plt.close()

########################################################## 

xx = np.linspace(0, 1, 51)
xx = np.array([xx]).T

coords = domain.mesh.coordinates()
coord_diag = []

for coord in coords:
    if coord[0] == coord[1]:
        coord_diag.append(coord)
    
true_val = []
est_val = []
for idx, x in enumerate(coord_diag):
    true_val.append(truth_fun(x))
    est_val.append(m_newton_cg(x))
true_val = np.array(true_val)
est_val = np.array(est_val)

truth_mat = truth_fun.compute_vertex_values()
truth_mat = (np.array(truth_mat).reshape((51, 51)))[::-1]

est_mat = m_newton_cg.compute_vertex_values()
est_mat = (np.array(est_mat).reshape((51, 51)))[::-1]


def to_percent(temp, position):
    return '%.f'%(100 * temp) + '%'

plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

with plt.style.context('seaborn-ticks','ggplot'):
    
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    

    
    
########################################################## Plt Credibility region
    plt.figure()
    xx = np.linspace(0, 50, 51)
    plt.plot(est_val, color="black", label='Estimate')
    plt.plot(true_val, color="red", label='Truth')
    plt.plot(est_val + 2*v_std, color="green", alpha=0.5)
    plt.plot(est_val - 2*v_std, color="green", alpha=0.5)
    plt.fill_between(xx, est_val + 2*v_std, est_val - 2*v_std, 
                      facecolor='green', alpha=0.1)
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    plt.savefig(result_figs_dir + 'Credibility region.png', dpi=400)
    plt.close()
 
########################################################## Plt Sub-Credibility region
    
    k = 5
    true_sub = np.diagonal(truth_mat, offset=k)
    est_sub = np.diagonal(est_mat, offset=k)
    covsub = np.diagonal(samp_var, offset=k)
    
    
    plt.figure()
    xx = np.linspace(0, 50-k, 51-k)
    plt.plot(est_sub, color="black", label='Estimate')
    plt.plot(true_sub, color="red", label='Truth')
    plt.plot(est_sub + 2*covsub, color="green", alpha=0.5)
    plt.plot(est_sub - 2*covsub, color="green", alpha=0.5)
    plt.fill_between(xx, est_sub + 2*covsub, est_sub - 2*covsub, 
                      facecolor='green', alpha=0.1)
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    labels = [0, 10, 20, 30, 40]
    labelss = [0, 0.25, 0.5, 0.75, 1.0]
    plt.xticks(labels, labelss)
    # plt.plot()
    plt.savefig(result_figs_dir + 'Credibility regionsub1.png', dpi=400)
    plt.close()
    
    
    
    k = 10
    true_sub = np.diagonal(truth_mat, offset=k)
    est_sub = np.diagonal(est_mat, offset=k)
    covsub = np.diagonal(samp_var, offset=k)
    
    plt.figure()
    xx = np.linspace(0, 50-k, 51-k)
    plt.plot(est_sub, color="black", label='Estimate')
    plt.plot(true_sub, color="red", label='Truth')
    plt.plot(est_sub + 2*covsub, color="green", alpha=0.5)
    plt.plot(est_sub - 2*covsub, color="green", alpha=0.5)
    plt.fill_between(xx, est_sub + 2*covsub, est_sub - 2*covsub, 
                      facecolor='green', alpha=0.1)
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    labels = [0, 10, 20, 30, 40]
    labelss = [0, 0.25, 0.5, 0.75, 1.0]
    plt.xticks(labels, labelss)
    # plt.plot()
    plt.savefig(result_figs_dir + 'Credibility regionsub2.png', dpi=400)
    plt.close()

########################################################## Plt Lambdas    
    plt.figure()
    length = len(lams)
    xx = list(range(length))
    xx = np.array([xx[i]+1 for i in range(length)])
    y = lams[:length]
    yy = []
    for i in range(len(xx)):
        yy.append('%.3f' % y[i])
    
    plt.plot(xx, y, marker='*')
    for a, b, c in zip(xx, y, yy):
        if a % 15 == 0:
            plt.text(a, b, c, ha='right', va='top', fontsize=10)
    plt.title("Lambda")
    plt.xlabel('Iteration Number')
    plt.savefig(result_figs_dir + 'lams.png', dpi=400)
    plt.close()


########################################################## Plt Relative errors
    plt.figure()
    length = len(relative_errors)
    length = 40
    xx = list(range(length))
    xx = np.array([xx[i]+1 for i in range(length)])
    y = relative_errors[:length]
    
    yy = []
    for i in range(len(xx)):
        yy.append(format(relative_errors[i], '.1%'))
    
    plt.plot(xx, y, marker='*') 
    for a, b, c in zip(xx, y, yy):
        if a == 1:
            plt.text(a, b, c, ha='center', va='bottom', fontsize=10)
        if a % 8 == 0:
            plt.text(a, b, c, ha='center', va='bottom', fontsize=10)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.title('Relative Errors')
    plt.xlabel('Iteration Number')
    plt.savefig(result_figs_dir + 'Relative.png', dpi=400)
    plt.close()








