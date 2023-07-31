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
########################################################## Plt Inde
import matplotlib as mpl
from matplotlib import cm
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter

# norm1 = np.load(result_figs_dir + 'steperrors_100' + '.npy')
# norm2 = np.load(result_figs_dir + 'steperrors_300' + '.npy')
# norm3 = np.load(result_figs_dir + 'steperrors_500' + '.npy')
# norm4 = np.load(result_figs_dir + 'steperrors_700' + '.npy')
# norm5 = np.load(result_figs_dir + 'steperrors_900' + '.npy')



# def plot_(i ,j, mode='log'):
    


#         plt.figure(figsize=(6, 5))
#         plt.style.use('ggplot')
        
#         with plt.style.context('seaborn-ticks','ggplot'):
            
#             mpl.rcParams['font.family'] = 'Times New Roman'
#             plt.rcParams['font.size'] = 14
#             plt.rcParams['axes.linewidth'] = 0.5
            
#             # plt.plot(norm1[i:j], linewidth=3,
#             #           label="d=" + str(100), marker='.')   
#             # plt.plot(norm2[i:j], linewidth=3,
#             #          label="d=" + str(300), marker='+')   
#             # plt.plot(norm3[i:j], linewidth=3,
#             #          label="d=" + str(500), marker='*')   
#             # plt.plot(norm4[i:j], linewidth=3,
#             #          label="d=" + str(700), marker='v')   
#             # plt.plot(norm5[i:j], linewidth=3,
#             #          label="d=" + str(900), marker='s')   
            
         
            
#             plt.plot(np.log(norm1)[i:j], linewidth=3,
#                       label="d=" + str(100), marker='.')   
#             plt.plot(np.log(norm2)[i:j], linewidth=3,
#                       label="d=" + str(300), marker='+')   
#             plt.plot(np.log(norm3)[i:j], linewidth=3,
#                       label="d=" + str(500), marker='*')   
#             plt.plot(np.log(norm4)[i:j], linewidth=3,
#                       label="d=" + str(700), marker='v')   
#             plt.plot(np.log(norm5)[i:j], linewidth=3,
#                       label="d=" + str(900), marker='s')   
    
#             plt.legend()
#             # plt.title("log(step norm)")
#             plt.xlabel("Iterative numbers")
#             plt.ylabel("log(Step norm)")
#             plt.ticklabel_format(style='sci', scilimits=(0, 100))
#             plt.grid(False)
#             plt.tight_layout(pad=0.3, w_pad=2, h_pad=2)
#             plt.savefig(result_figs_dir + 'inde.png', dpi=400)
#             # plt.show()


# plot_(1, 51)





