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


DATA_DIR = './NCP_MFVI/Darcy/DATA/'
RESULT_DIR = './NCP_MFVI/Darcy/RESULTS/'
result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"

## domain for solve PDE
equ_nx = 50
# domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)

mMap = fe.interpolate(fe.Constant(0.0), domain.function_space).vector()[:]

## specify the measurement points
num_x, num_y = 20, 20
x = np.linspace(0.01, 0.99, num_x)
y = np.linspace(0.01, 0.99, num_y)
coordinates = generate_points(x, y)
measurement_points = coordinates

## construct a solver to generate data
f_expre = "sin(x[0])*cos(x[1])"
# f_expre = "10 * sin(x[0])*cos(x[1])"
## If we change f to be f_expre = "sin(a*pi*x[0])*sin(a*pi*x[1])" with a == 10,
## the nonlinear behavior may increase. And all of the optimization methods will
## not work well. 
f = fe.Expression(f_expre, degree=5)

## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain.function_space)

equ_solver_ = EquSolver(domain_equ=domain, m=truth_fun, f=f, points=coordinates)

p0 = equ_solver_.forward_solver(mMap)

equ_solver = EquSolver_linear(
    domain_equ=domain, mMap=mMap, points=coordinates,
    p0=p0
    )

dm = truth_fun.vector()[:] - mMap

# d = equ_solver.S@equ_solver.forward_solver(dm)

tmp1 = equ_solver_.forward_solver(truth_fun.vector()[:])
tmp2 = equ_solver_.forward_solver(mMap)
d = equ_solver.S@(tmp1 - tmp2)

F = equ_solver.F.todense()
bc_idx = equ_solver.bc_idx
F[bc_idx, :] = 0.0
K = equ_solver.K.todense()
S = equ_solver.S
M = equ_solver.M.todense()
Minv = np.linalg.inv(M)

HH = S@np.linalg.inv(K)@F

Id = np.identity(HH.shape[-1])
HessianM = HH.T@HH + Id
InvHM = np.linalg.inv(HessianM)


##### test linear
# dd = np.array(HH@dm).squeeze()
# d_clean = dd

##### test non-linear
d_clean = S@equ_solver_.forward_solver(truth_fun.vector()[:])
dd = np.array(S@(equ_solver_.forward_solver(truth_fun.vector()[:])-p0)).squeeze()

data_max = abs(max(d_clean))
noise_level = 0.05
# noise_level = 0.01
dn = dd + noise_level*data_max*np.random.normal(0, 1, (len(dd),))


dmn = fe.interpolate(fe.Constant(0.0), domain.function_space).vector()[:]

def lossf(dmn):
    tmp = np.array(HH@dmn).squeeze() - dd
    return np.sum(0.5*tmp*tmp)

e = 1e-1
lossall = []
relative_errors = []
dmn_fun = fe.Function(domain.function_space)
dm_fun = fe.Function(domain.function_space)


## enable prior
ss = 20
prior_measure = GaussianElliptic2(
    domain=domain, alpha=ss, a_fun=ss, theta=1, boundary='Neumann'
    )

## setting the noise
# noise_level_ = 1
noise_level_ = noise_level*max(abs(d_clean))
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)


## enable lambda
mean0 = 1
cov0 = 1e2
lam_dis = GaussianLam(mean=mean0, cov=cov0)

## setting the Model_noise
model = ModelDarcyFlow_linear(
    d=dn, domain_equ=domain, prior=prior_measure, 
    noise=noise, equ_solver=equ_solver, lam_dis=lam_dis
    )




step_errs = []
## set optimizer NewtonCG
newton_cg = NewtonCG(model=model)
# posterior_v = PosteriorOfV(model=model, newton_cg=newton_cg)
m_newton_cg = fe.Function(domain.function_space)
max_iter = 1

for i in range(50):
    

    posterior_v = PosteriorOfV(model=model, newton_cg=newton_cg)
    ## Without a good initial value, it seems hard for us to obtain a good solution
    # init_fun = smoothing(truth_fun, alpha=0.1)
    # newton_cg.re_init(init_fun.vector()[:])
    loss_pre = model.loss()[0]
    for itr in range(max_iter):
        newton_cg.descent_direction(cg_max=50, method='cg_my')
        # newton_cg.descent_direction(cg_max=30, method='bicgstab')
        # print(newton_cg.hessian_terminate_info)
        newton_cg.step(method='armijo', show_step=False)
        # if newton_cg.converged == False:
        #     break
        loss = model.loss()[0]
        print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
        # if np.abs(loss - loss_pre) < 1e-3*loss:
        #     print("Iteration stoped at iter = %d" % itr)
        #     break 
        loss_pre = loss
    
    
    rho = lam_dis.mean*lam_dis.mean + lam_dis.cov

    m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())
    
    relative_errors.append(relative_error(m_newton_cg, dm_fun, domain))
    print("error = ", relative_errors[-1])
    

    vk = m_newton_cg.vector()[:] * lam_dis.mean / (rho)
    posterior_v.calculate_eigensystem_lam(num_eigval=100, method="double_pass", cut_val=0.0)


    err_now = relative_errors[-1]
    err_pre = relative_errors[-2]
    step_err = (err_now - err_pre)**2 / err_now**2    
    step_errs.append(step_err)

    # vk = posterior_v.mean

    
    
    Hvk = np.array(HH @ vk).squeeze()
    Hvk = equ_solver.S @ np.array(equ_solver.forward_solver(vk)).squeeze()
    Hvk2 = Hvk@spsl.spsolve(noise.covariance, Hvk)
    temp1 = Hvk2 + 1/cov0
    
    eigval = posterior_v.eigval_lam / rho
    temp2 = np.sum(eigval/(rho*eigval + 1))
    # temp2 = 0
    lam_dis.cov = 1/(temp1 + temp2)
    
    tmp1 = dn @spsl.spsolve(noise.covariance, Hvk)
    lam_dis.mean = lam_dis.cov*(tmp1 + mean0/cov0)
    
    # print('*********', tmp1, Hvk2)
    print('lam_mean, lam_cov, idx = ', lam_dis.mean, lam_dis.cov, i)
    
    model.update_lam()


np.save(result_figs_dir + 'steperrors'+ "_" + str(equ_nx) + '.npy', step_errs)


# norm1 = np.load(result_figs_dir + 'relative_40' + '.npy')
# norm2 = np.load(result_figs_dir + 'relative_45' + '.npy')
# norm3 = np.load(result_figs_dir + 'relative_50' + '.npy')
# norm4 = np.load(result_figs_dir + 'relative_55' + '.npy')
# norm5 = np.load(result_figs_dir + 'relative_60' + '.npy')



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
#             #           label="d=" + str(1600), marker='.')   
#             # plt.plot(norm2[i:j], linewidth=3,
#             #           label="d=" + str(2025), marker='+')   
#             # plt.plot(norm3[i:j], linewidth=3,
#             #           label="d=" + str(2500), marker='*')   
#             # plt.plot(norm4[i:j], linewidth=3,
#             #           label="d=" + str(3025), marker='v')   
#             # plt.plot(norm5[i:j], linewidth=3,
#             #           label="d=" + str(3600), marker='s')   
            
         
            
#             plt.plot(np.log(norm1)[i:j], linewidth=3,
#                       label="d=" + str(1600), marker='.')   
#             plt.plot(np.log(norm2)[i:j], linewidth=3,
#                       label="d=" + str(2025), marker='+')   
#             plt.plot(np.log(norm3)[i:j], linewidth=3,
#                       label="d=" + str(2500), marker='*')   
#             plt.plot(np.log(norm4)[i:j], linewidth=3,
#                       label="d=" + str(3015), marker='v')   
#             plt.plot(np.log(norm5)[i:j], linewidth=3,
#                       label="d=" + str(3600), marker='s')   
    
#             plt.legend()
#             # plt.title("log(step norm)")
#             plt.xlabel("Iterative numbers")
#             plt.ylabel("log(Step norm)")
#             plt.ticklabel_format(style='sci', scilimits=(0, 100))
#             plt.grid(False)
#             plt.tight_layout(pad=0.3, w_pad=6, h_pad=5)
#             plt.savefig(result_figs_dir + 'relacom.png', dpi=400)
#             # plt.show()


# plot_(2, 19)







