#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:18:47 2023

@author: ishihara
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
from core.misc import save_expre, generate_points, trans2spnumpy, smoothing
from core.noise import NoiseGaussianIID
from NCP_MFVI.common_Darcy import relative_error
from core.optimizer import GradientDescent, NewtonCG
from NCP_MFVI.common_Helm import EquSolver, Domain2DPML, ModelHelmNCP, GaussianLam
from NCP_MFVI.common_Helm import ModelHelmNCP_Mult, PosteriorOfV

DATA_DIR = './NCP_MFVI/Helm/DATA/'
RESULT_DIR = './NCP_MFVI/Helm/RESULTS/'
result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"

noise_level = 0.05

## setting the PML domain
equ_nx = 75
dPML = 0.1
domain_equ = Domain2DPML(nx=equ_nx, ny=equ_nx, dPML=dPML, xx=1.0, yy=1.0)
VR, VI = domain_equ.function_space.sub(0).collapse(), domain_equ.function_space.sub(1).collapse()


## setting prior and PDE domain
domain = Domain2D(
    low_point=[-dPML, -dPML], high_point=[1+dPML, 1+dPML], \
        nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1
    )
V=domain.function_space
ss = 10
prior_measure = GaussianElliptic2(
    domain, alpha=ss, a_fun=ss, theta=0.1, boundary='Neumann'
    )


## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain.function_space)




freqs = np.load(DATA_DIR + "freqs" + ".npy")
f = truth_fun
equ_solver = EquSolver(
    domainPML=domain_equ, domain=domain, m=f, freq=freqs[0], num_points=20
    )

## load the measurement data
# tau = 1
taus = np.load(DATA_DIR + "tau" + "_" + str(noise_level) + ".npy")
ds = np.load(DATA_DIR + "dn" + "_" + str(noise_level) + ".npy")
ds_clean = np.load(DATA_DIR + "dc.npy")
## setting the noise
noise_level_ = taus[0]
noise = NoiseGaussianIID(dim=np.int32(2*len(equ_solver.points)))
noise.set_parameters(variance=noise_level_**2)
# noise.set_parameters(variance=1e-10)


# setting lambda distribution
mean0 = 1
# mean0 = 614
# cov0 = 1e-6
cov0 = 1e2
lam_dis = GaussianLam(mean=mean0, cov=cov0)
relative_errors = []
lams = []
dmn_fun = fe.interpolate(fe.Constant('0.0'), V)
dmn = dmn_fun.vector()[:]



model = ModelHelmNCP_Mult(
    ds=ds, domain=domain, prior=prior_measure , noise=noise, 
    equ_solver=equ_solver, lam_dis=lam_dis, freqs=freqs, taus=taus
    )
step_errs = []
uks = []
## test NewtonCG
newton_cg = NewtonCG(model=model)
newton_cg.re_init(dmn_fun.vector()[:])
m_newton_cg = fe.Function(domain.function_space)
uk_old = fe.Function(domain.function_space)
uk_new = fe.Function(domain.function_space)
# model.end = np.int32(len(freqs) / 2)
max_iter=1
for i in range(20):
    
    posterior_v = PosteriorOfV(model=model, newton_cg=newton_cg)
    ## Without a good initial value, it seems hard for us to obtain a good solution
    # init_fun = smoothing(truth_fun, alpha=0.1)
    # newton_cg.re_init(init_fun.vector()[:])
    # newton_cg.re_init(newton_cg.mk.copy())
    # loss_pre = model.loss()[0]
    # for itr in range(max_iter):
        
        
    #     # NewtonCG
    #     # newton_cg.descent_direction(cg_max=50, method='cgs')
    #     newton_cg.descent_direction(cg_max=50, method='cg_my')
    #     # print(newton_cg.hessian_terminate_info)
    #     # gradient_decent.lr = 1e-6
    #     # gradient_decent.step(method='fixed')
    #     newton_cg.step(method='armijo', show_step=False)
    #     # if newton_cg.converged == False:
    #     #     newton_cg.descent_direction(cg_max=0, method='cg_my')
    #     #     newton_cg.step(method='armijo', show_step=False)
    #     # if newton_cg.converged == False:
    #     #     break
    #     loss = model.loss()[0]
    #     # print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
    #     # print("loss_res, loss_reg = ", model.loss()[1], model.loss()[2])
    #     # if np.abs(loss - loss_pre) < 1e-3*loss:
    #     #     print("Iteration stoped at iter = %d" % itr)
    #     #     break 
    #     loss_pre = loss
    
    posterior_v.eval_mean_iterative(iter_num=5, cg_max=50, method="cg_my")
    dmn_fun.vector()[:] = np.array(newton_cg.mk.copy()) * lam_dis.mean
    m_newton_cg.vector()[:] = np.array(posterior_v.mean)
    relative_errors.append(relative_error(dmn_fun, truth_fun, domain))
    print("error = ", relative_errors[-1])
    uks.append(dmn_fun.vector()[:])
    # rho = lam_dis.mean*lam_dis.mean + lam_dis.cov
    # vk = newton_cg.mk * lam_dis.mean / np.sqrt(rho)
    
    
    
    # # tmp = lam_dis.cov + lam_dis.mean * lam_dis.mean
    # # eigval = eigval/tmp
    # eig_val = 0
    # # update covariance
    
    # Hvk2 = 0
    # dHvk = 0
    
    # for ii, freq in list(enumerate(freqs))[model.start: model.end]:
        
    #     equ_solver.update_freq(freq)
    #     model.update_noise(model.taus[ii])
        
        
    #     m = vk
    #     p = equ_solver.forward_solver(m)
    #     Sp = model.S@p
        
        
    #     if model.noise.precision is None:
    #         tmp1 = Sp@spsl.spsolve(model.noise.covariance, Sp)
    #         tmp2 = model.ds[ii] @ spsl.spsolve(model.noise.covariance, Sp)
    #     else:
    #         tmp1 = Sp@model.noise.precision@Sp
    #         tmp2 = model.ds[ii] @ model.noise.precision @ (Sp)
            
            
    #     Hvk2 += tmp1
    #     dHvk += tmp2
            
    # # temp1 = temp + 1/self.lam_cov0
    # temp1 = Hvk2 + 1/model.lam_cov0
    # rho = model.lam_dis.cov + model.lam*model.lam
    # temp2 = 0
    # # temp2 = np.sum(eigval/(rho*eigval + 1))
    # lam_dis.cov = 1/(temp1 + temp2)
    
    # print("-----", Hvk2, dHvk)
    if i >= 1 :
        # err_now = relative_errors[-1]
        # err_pre = relative_errors[-2]
        # step_err = (err_now - err_pre)**2 / err_now**2    
        # step_errs.append(step_err)
        
        uk_new.vector()[:] = uks[-1]
        uk_old.vector()[:] = uks[-2]
        step_err = relative_error(uk_new, uk_old, domain)
        step_errs.append(step_err)
        
    # # update mean
    # lam_dis.mean = lam_dis.cov*(dHvk + model.lam_mean0/model.lam_cov0)
    # print("+++++", self.lam_mean0/self.lam_cov0, tmp/temp)
    
    
    posterior_v.calculate_eigensystem_lam(num_eigval=100, cut_val=0.0)
    # eigval = 0
    model.update_lam_dis(newton_cg.mk, posterior_v.eigval)
    # print('lam_mean, lam_cov, idx = ', lam_dis.mean, lam_dis.cov, i)
    model.update_lam()
    lams.append(model.lam)
    print('index, mean, cov = ', i+1, model.lam, model.lam_dis.cov)
    

np.save(result_figs_dir + 'steperrors'+ "_" + str(equ_nx) + '.npy', step_errs[:20])

############################################################### Plot mesh_inde

# norm1 = np.load(result_figs_dir + 'steperrors_58' + '.npy')
# norm2 = np.load(result_figs_dir + 'steperrors_59' + '.npy')
# norm3 = np.load(result_figs_dir + 'steperrors_60' + '.npy')
# norm4 = np.load(result_figs_dir + 'steperrors_61' + '.npy')
# norm5 = np.load(result_figs_dir + 'steperrors_62' + '.npy')



# import matplotlib as mpl
# from matplotlib.ticker import FuncFormatter

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
            
         
            
#             plt.plot((norm1)[i:j], linewidth=3,
#                       label="d=" + str(1600), marker='.')   
#             plt.plot((norm2)[i:j], linewidth=3,
#                       label="d=" + str(2025), marker='+')   
#             plt.plot((norm3)[i:j], linewidth=3,
#                       label="d=" + str(2500), marker='*')   
#             plt.plot((norm4)[i:j], linewidth=3,
#                       label="d=" + str(3025), marker='v')   
#             plt.plot((norm5)[i:j], linewidth=3,
#                       label="d=" + str(3600), marker='s')   
    
#             plt.legend()
#             # plt.title("log(step norm)")
#             plt.xlabel("Iterative numbers")
#             plt.ylabel("log(Step norm)")
#             plt.ticklabel_format(style='sci', scilimits=(0, 100))
#             plt.grid(False)
#             plt.tight_layout(pad=0.3, w_pad=2, h_pad=2)
#             # plt.savefig(result_figs_dir + 'inde.png', dpi=400)
#             plt.show()


# plot_(0, 15)


