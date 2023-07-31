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
for i in range(50):
    
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
    


posterior_v.calculate_eigensystem_lam(num_eigval=100, method="scipy_eigsh", cut_val=0.0)
# posterior_v.mean = m_newton_cg.vector()[:]
posterior_v.mean = dmn_fun.vector()[:]
tmp_fun = fe.Function(domain.function_space)
tmps = []
for i in range(50):
    tmp = posterior_v.generate_sample_lam()
    tmps.append(tmp)
tmps = np.array(tmps)

tmp_var = np.std(tmps, axis=0)
tmp_fun.vector()[:] = tmp_var
samp_var = tmp_fun.compute_vertex_values()
samp_var = (np.array(samp_var).reshape((61, 61)))
v_std = samp_var.diagonal()

fun = fe.Function(domain.function_space)

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
fig = fe.plot(dmn_fun, label='estimate')
plt.colorbar(fig)
plt.legend()
plt.title("Estimated Function")
plt.subplot(1, 3, 2)
fig = fe.plot(truth_fun, label='truth')
plt.colorbar(fig)
plt.title("Truth")
plt.legend()
plt.subplot(1, 3, 3)
fig = fe.plot(tmp_fun)
plt.colorbar(fig)
plt.title('Sample')
plt.show()

# ########################################################## Plt Estimate and Truth
  
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


plt.figure(figsize=(6, 5))
plt.style.use('ggplot')

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.5

plt.figure()
fig = fe.plot(dmn_fun)
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
plt.title('Pointwise Variance Field')
plt.savefig(result_figs_dir + 'Pointwise.png', dpi=400)
plt.close()


########################################################## 
coords = domain.mesh.coordinates()
coord_diag = []
deltax = 1.2 / 60
for coord in coords:
    if coord[0] == coord[1]:
        coord_diag.append(coord)


true_val = []
est_val = []
for idx, x in enumerate(coord_diag):
    true_val.append(truth_fun(x))
    est_val.append(dmn_fun(x))
true_val = np.array(true_val)
est_val = np.array(est_val)


truth_mat = truth_fun.compute_vertex_values()
truth_mat = (np.array(truth_mat).reshape((61, 61)))[::-1]

est_mat = dmn_fun.compute_vertex_values()
est_mat = (np.array(est_mat).reshape((61, 61)))[::-1]



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
    xx = np.linspace(0, 60, 61)
    plt.plot(est_val, color="black", label='Estimate')
    plt.plot(true_val, color="red", label='Truth')
    plt.plot(est_val + 2*v_std, color="green", alpha=0.5)
    plt.plot(est_val - 2*v_std, color="green", alpha=0.5)
    plt.fill_between(xx, est_val + 2*v_std, est_val - 2*v_std, 
                      facecolor='green', alpha=0.1)
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    labels = [0, 10, 20, 30, 40, 50, 60]
    labelss = [round(label*deltax-0.1, 1) for label in labels]
    plt.xticks(labels, labelss)
    # plt.plot()
    plt.savefig(result_figs_dir + 'Credibility region.png', dpi=400)
    plt.close()
    
########################################################## Plt Sub-Credibility region
    
    k = 5
    true_sub = np.diagonal(truth_mat, offset=k)
    est_sub = np.diagonal(est_mat, offset=k)
    covsub = np.diagonal(samp_var, offset=k)
    
    
    plt.figure()
    xx = np.linspace(0, 60-k, 61-k)
    plt.plot(est_sub, color="black", label='Estimate')
    plt.plot(true_sub, color="red", label='Truth')
    plt.plot(est_sub + 2*covsub, color="green", alpha=0.5)
    plt.plot(est_sub - 2*covsub, color="green", alpha=0.5)
    plt.fill_between(xx, est_sub + 2*covsub, est_sub - 2*covsub, 
                      facecolor='green', alpha=0.1)
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    labels = [0, 10, 20, 30, 40, 50]
    labelss = [round(label*deltax-0.1, 1) for label in labels]
    plt.xticks(labels, labelss)
    # plt.plot()
    plt.savefig(result_figs_dir + 'Credibility regionsub1.png', dpi=400)
    plt.close()
    
    
    
    k = 10
    true_sub = np.diagonal(truth_mat, offset=k)
    est_sub = np.diagonal(est_mat, offset=k)
    covsub = np.diagonal(samp_var, offset=k)
    
    plt.figure()
    xx = np.linspace(0, 60-k, 61-k)
    plt.plot(est_sub, color="black", label='Estimate')
    plt.plot(true_sub, color="red", label='Truth')
    plt.plot(est_sub + 2*covsub, color="green", alpha=0.5)
    plt.plot(est_sub - 2*covsub, color="green", alpha=0.5)
    plt.fill_between(xx, est_sub + 2*covsub, est_sub - 2*covsub, 
                      facecolor='green', alpha=0.1)
    plt.legend(frameon='True', loc='upper right')
    plt.title('Credibility Region')
    labels = [0, 10, 20, 30, 40, 50]
    labelss = [round(label*deltax-0.1, 1) for label in labels]
    plt.xticks(labels, labelss)
    # plt.plot()
    plt.savefig(result_figs_dir + 'Credibility regionsub2.png', dpi=400)
    plt.close()
    
# ########################################################## Plt Lambdas    
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
        if a == 1:
            plt.text(a, b, c, ha='center', va='bottom', fontsize=10)
        if a % 12 == 0:
            plt.text(a, b, c, ha='center', va='bottom', fontsize=10)
    plt.title("Lambda")
    plt.xlabel('Iteration Number')
    # plt.plot()
    plt.savefig(result_figs_dir + 'lams.png', dpi=400)
    plt.close()
    


# ########################################################## Plt Relative errors
    plt.figure()
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
            plt.text(a, b, c, ha='center', va='top', fontsize=10)
        if a % 8 == 0:
            plt.text(a, b, c, ha='center', va='bottom', fontsize=10)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.title('Relative Errors')
    plt.xlabel('Iteration Number')
    # plt.plot()
    plt.savefig(result_figs_dir + 'Relative.png', dpi=400)
    plt.close()



