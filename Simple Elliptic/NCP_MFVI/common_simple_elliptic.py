#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:49:25 2023

@author: ubuntu
"""

import numpy as np
# import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
# import dolfin as dl 

import sys, os
sys.path.append(os.pardir)
from core.model import ModelBase
from core.misc import construct_measurement_matrix, trans2spnumpy
from core.linear_eq_solver import cg_my
from core.approximate_sample import LaplaceApproximate
from core.optimizer import NewtonCG


###########################################################################
class EquSolver(object):
    def __init__(self, domain, points, m=None, alpha=0.05):
        '''
        model: "vec" or "matrix"

        '''
        self.domain = domain
        if m is None:
            self.m = fe.interpolate(fe.Constant("0.0"), self.domain.function_space)
        else:
            self.m = fe.interpolate(m, self.domain.function_space)
        self.points = points
        self.alpha = fe.Constant(alpha)
        u_ = fe.TrialFunction(self.domain.function_space)
        v_ = fe.TestFunction(self.domain.function_space)
        self.M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.F_ = fe.assemble(
            fe.inner(u_, v_)*fe.dx + self.alpha*fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx
            )
        self.S = construct_measurement_matrix(self.points, self.domain.function_space)

        def boundary(x, on_boundary):
            return on_boundary
        bc = fe.DirichletBC(self.domain.function_space, fe.Constant('0.0'), boundary)
        bc.apply(self.F_)
        bc.apply(self.M_)
        
        ## specify which element will be set zero for force term 
        ## Here we ignore the internal mechnisms of the FEniCS software.
        ## If we know the details of the FEM software, there should be more clear way to specify self.bc_idx
        temp1 = fe.assemble(fe.inner(fe.Constant("1.0"), v_)*fe.dx)
        temp2 = temp1[:].copy()
        bc.apply(temp1)
        self.bc_idx = (temp2 != temp1)
        
        self.M = trans2spnumpy(self.M_)
        self.F = trans2spnumpy(self.F_) 
        self.len_vec = self.M.shape[0]

    def update_m(self, m_vec):
        assert len(self.m.vector()[:]) == len(m_vec) 
        self.m.vector()[:] = np.array(m_vec[:])
    
    def update_points(self, points):
        self.points = points
        self.S = construct_measurement_matrix(self.points, self.domain.function_space)

    def forward_solver(self, m_vec=None):
        if m_vec is not None:
            self.update_m(m_vec)
                
        rhs = self.M@self.m.vector()[:]
        rhs[self.bc_idx] = 0.0 
            
        self.forward_sol = spsl.spsolve(self.F, rhs)
        self.forward_sol = np.array(self.forward_sol)
        
        return self.forward_sol
    
    def incremental_forward_solver(self, m_hat=None):
        ## we need this function can accept matrix input
        ## For linear problems, the incremental forward == forward
        rhs = self.M@m_hat
        rhs[self.bc_idx] = 0.0 
            
        self.inc_forward_sol = spsl.spsolve(self.F, rhs)
        self.inc_forward_sol = np.array(self.inc_forward_sol)
        
        return self.inc_forward_sol

    def adjoint_solver(self, res_vec):
        ## res_vec = Sv - d
        rhs = self.S.T @ res_vec
        rhs[self.bc_idx] = 0.0 
        self.adjoint_sol = spsl.spsolve(self.F, rhs)
        self.adjoint_sol = np.array(self.adjoint_sol) 
        
        return self.adjoint_sol

    def incremental_adjoint_solver(self, vec, m_hat=None):        
        self.inc_adjoint_sol = self.adjoint_solver(vec)
        self.inc_adjoint_sol = np.array(self.inc_adjoint_sol)
        
        return self.inc_adjoint_sol
        
    def construct_fun(self, f_vec):
        f = fe.Function(self.domain.function_space)
        assert len(f.vector()[:]) == len(f_vec)
        f.vector()[:] = np.array(f_vec)
        
        return f


###########################################################################
class ModelSimpleElliptic(ModelBase):
    def __init__(self, d, domain, prior, noise, equ_solver):
        super().__init__(d, domain, prior, noise, equ_solver)

    def update_m(self, m_vec, update_sol=True):
        # print(np.array(m_vec).shape, self.m.vector()[:].shape)
        self.m.vector()[:] = np.array(m_vec)
        self.equ_solver.update_m(self.m.vector())
        if update_sol is True:
            self.p.vector()[:] = self.equ_solver.forward_solver()

    def loss_residual(self):
        temp = (self.S@self.p.vector()[:] - self.noise.mean - self.d)
        if self.noise.precision is None:
            temp = temp@spsl.spsolve(self.noise.covariance, temp)
        else:
            temp = temp@self.noise.precision@temp
        return 0.5*temp

    def loss_residual_L2(self):
        temp = (self.S@self.p.vector()[:] - self.d)
        temp = temp@temp
        return 0.5*temp

    def eval_grad_residual(self, m_vec):
        pass
        # # self.equ_solver.update_m(m_vec)
        # self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        # res_vec = spsl.spsolve(self.noise.covariance, self.S@(self.p.vector()[:]) - self.d)
        # # print(res_vec)
        # self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        # g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        # g = g_.vector()[:]
        # return np.array(g)

    def eval_hessian_res_vec(self, dm):
        pass
        # # self.equ_solver.update_m(dm)
        # self.p.vector()[:] = self.equ_solver.forward_solver(dm)
        # res_vec = spsl.spsolve(self.noise.covariance, self.S@(self.p.vector()[:]))
        # self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        # g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        # HM = g_.vector()[:]
        # return np.array(HM)


class GaussianLam(object):
    def __init__(self, mean, cov):
        self.mean, self.cov = mean, cov
        
    def generate_sample(self):
        return np.random.normal(self.mean, np.sqrt(self.cov))


class ModelSimpleEllipticLam(ModelSimpleElliptic):
    def __init__(self, d, domain, prior, noise, equ_solver, lam_dis):
        super().__init__(d, domain, prior, noise, equ_solver)
        self.lam_dis = lam_dis
        self.lam_mean0 = self.lam_dis.mean
        self.lam_cov0 = self.lam_dis.cov
        self.lam = self.lam_dis.mean
        # self.lam_dis.cov = 0.0
        
    def update_lam(self, lam=None):
        if lam is not None:
            self.lam = lam
        else:
            self.lam = self.lam_dis.generate_sample()
   
    def loss_residual(self, m=None):
        if m is not None:
            self.p.vector()[:] = self.equ_solver.forward_solver(m)
        ## scale lam
        cc = np.sqrt(self.lam*self.lam)
        temp = cc*(self.S@self.p.vector()[:])
        temp = (temp - self.noise.mean - self.d)
        if self.noise.precision is None:
            temp = temp@spsl.spsolve(self.noise.covariance, temp)
        else:
            temp = temp@self.noise.precision@temp
        return 0.5*temp
    
    def loss_residual_L2(self):
        ## scale lam
        cc = np.sqrt(self.lam*self.lam)
        temp = cc*(self.S@self.p.vector()[:])
        temp = (temp - self.d)
        temp = temp@temp
        return 0.5*temp
    
    def eval_HAdjointData(self):
        res_vec = spsl.spsolve(self.noise.covariance, self.d)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        ## scale lam
        g = self.lam*g_.vector()[:]
        return np.array(g)

    
    def eval_grad_residual(self, m_vec):
        cc = np.sqrt(self.lam*self.lam)
        self.equ_solver.update_m(m_vec)
        self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        ## scale lam
        measure_scale = cc*self.S@(self.p.vector()[:])
        res_vec = spsl.spsolve(self.noise.covariance, measure_scale - self.d)
        # print(res_vec)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        ## scale lam
        g = cc*g_.vector()[:]
        return np.array(g)

    
    def eval_hessian_res_vec(self, dm):
        self.equ_solver.update_m(dm)
        self.p.vector()[:] = self.equ_solver.forward_solver(dm)
        measure = (self.S@(self.p.vector()[:]))
        res_vec = spsl.spsolve(self.noise.covariance, measure)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        HM = g_.vector()[:]
        tmp = self.lam*self.lam
        return np.array(HM)*tmp
 
    
    def update_lam_dis(self, m):
        tmp = self.lam*self.lam
        # update covariance
        self.p.vector()[:] = self.equ_solver.forward_solver(m)
        p = self.p.vector()[:]
        Sp = self.S@p
        if self.noise.precision is None:
            temp = Sp@spsl.spsolve(self.noise.covariance, Sp)
        else:
            temp = Sp@self.noise.precision@Sp
        
        temp1 = temp + 1/self.lam_cov0
        # rho = self.lam_dis.cov + self.lam*self.lam
        self.lam_dis.cov = 1/temp1
        # print("-----", temp1, temp2)
        
        # update mean
        if self.noise.precision is None:
            tmp = self.d@spsl.spsolve(self.noise.covariance, Sp)
        else:
            tmp = self.d@self.noise.precision@(Sp)
        # print("*****", tmp, temp)
            
        self.lam_dis.mean = self.lam_dis.cov*(tmp + self.lam_mean0/self.lam_cov0)
        # print("+++++", self.lam_mean0/self.lam_cov0, tmp/temp)
        

class ModelSimpleEllipticNCP(ModelSimpleElliptic):
    def __init__(self, d, domain, prior, noise, equ_solver, lam_dis):
        super().__init__(d, domain, prior, noise, equ_solver)
        self.lam_dis = lam_dis
        self.lam_mean0 = self.lam_dis.mean
        self.lam_cov0 = self.lam_dis.cov
        self.lam = self.lam_dis.mean
        # self.lam_dis.cov = 0.0
        
    def update_lam(self, lam=None):
        if lam is not None:
            self.lam = lam
        else:
            self.lam = self.lam_dis.mean
   
    def loss_residual(self):
        ## scale lam
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        temp = cc*(self.S@self.p.vector()[:])
        temp = (temp - self.noise.mean - self.d)
        if self.noise.precision is None:
            temp = temp@spsl.spsolve(self.noise.covariance, temp)
        else:
            temp = temp@self.noise.precision@temp
        return 0.5*temp
    
    def loss_residual_L2(self):
        ## scale lam
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        temp = cc*(self.S@self.p.vector()[:])
        temp = (temp - self.d)
        temp = temp@temp
        return 0.5*temp
    
    def eval_HAdjointData(self):
        res_vec = spsl.spsolve(self.noise.covariance, self.d)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        ## scale lam
        g = self.lam*g_.vector()[:]
        return np.array(g)
    
    # def eval_grad_residual(self, m_vec):
    #     self.equ_solver.update_m(m_vec)
    #     self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
    #     ## scale lam
    #     measure_scale = self.lam*self.S@(self.p.vector()[:])
    #     res_vec = spsl.spsolve(self.noise.covariance, measure_scale - self.d)
    #     # print(res_vec)
    #     self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
    #     g_ = fe.interpolate(self.q, self.domain_equ.function_space)
    #     ## scale lam
    #     g = self.lam*g_.vector()[:]
    #     return np.array(g)
    
    def eval_grad_residual(self, m_vec):
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        self.equ_solver.update_m(m_vec)
        self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        ## scale lam
        measure_scale = cc*self.S@(self.p.vector()[:])
        res_vec = spsl.spsolve(self.noise.covariance, measure_scale - self.d)
        # print(res_vec)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        ## scale lam
        g = cc*g_.vector()[:]
        return np.array(g)

    # def eval_hessian_res_vec_1(self, dm):
    #     self.equ_solver.update_m(dm)
    #     self.p.vector()[:] = self.equ_solver.forward_solver(dm)
    #     measure = (self.S@(self.p.vector()[:]))
    #     res_vec = spsl.spsolve(self.noise.covariance, measure)
    #     self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
    #     g_ = fe.interpolate(self.q, self.domain_equ.function_space)
    #     HM = g_.vector()[:]
    #     tmp = self.lam*self.lam + self.lam_dis.cov
    #     return np.array(HM)*tmp
    
    # def hessian_1(self, m_vec):
    #     hessian_res = self.eval_hessian_res_vec_1(m_vec)
    #     hessian_prior = self.eval_hessian_prior_vec(m_vec)
    #     return hessian_res + hessian_prior
        
    # def hessian_linear_operator_1(self):
    #     leng = self.M.shape[0]
    #     linear_ope = spsl.LinearOperator((leng, leng), matvec=self.hessian_1)
    #     return linear_ope
    
    # def MxHessian_1(self, m_vec):
    #     ## Usually, algorithms need a symmetric matrix.
    #     ## Here, we calculate MxHessian to make a symmetric matrix. 
    #     return self.M@self.hessian_1(m_vec)
    
    # def MxHessian_linear_operator_1(self):
    #     leng = self.M.shape[0]
    #     linear_op = spsl.LinearOperator((leng, leng), matvec=self.MxHessian_1)
    #     return linear_op
    
    def eval_hessian_res_vec(self, dm):
        self.equ_solver.update_m(dm)
        self.p.vector()[:] = self.equ_solver.forward_solver(dm)
        measure = (self.S@(self.p.vector()[:]))
        res_vec = spsl.spsolve(self.noise.covariance, measure)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        HM = g_.vector()[:]
        tmp = self.lam*self.lam + self.lam_dis.cov
        return np.array(HM)*tmp
    
    # def eval_hessian_res_vec(self, dm):
    #     self.equ_solver.update_m(dm)
    #     self.p.vector()[:] = self.equ_solver.forward_solver(dm)
    #     measure = (self.S@(self.p.vector()[:]))
    #     res_vec = spsl.spsolve(self.noise.covariance, measure)
    #     self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
    #     g_ = fe.interpolate(self.q, self.domain_equ.function_space)
    #     HM = g_.vector()[:]
    #     return np.array(HM)
    
    def update_lam_dis(self, m, eigval):
        tmp = self.lam_dis.cov + self.lam*self.lam
        eigval = eigval/tmp
        # update covariance
        self.p.vector()[:] = self.equ_solver.forward_solver(m)
        p = self.p.vector()[:]
        Sp = self.S@p
        if self.noise.precision is None:
            temp = Sp@spsl.spsolve(self.noise.covariance, Sp)
        else:
            temp = Sp@self.noise.precision@Sp
        
        temp1 = temp + 1/self.lam_cov0
        rho = self.lam_dis.cov + self.lam*self.lam
        temp2 = np.sum(eigval/(rho*eigval + 1))
        self.lam_dis.cov = 1/(temp1 + temp2)
        # print("-----", temp1, temp2)
        
        # update mean
        if self.noise.precision is None:
            tmp = self.d@spsl.spsolve(self.noise.covariance, Sp)
        else:
            tmp = self.d@self.noise.precision@(Sp)
        # print("*****", tmp, temp)
            
        self.lam_dis.mean = self.lam_dis.cov*(tmp + self.lam_mean0/self.lam_cov0)
        # print("+++++", self.lam_mean0/self.lam_cov0, tmp/temp)
        

class PosteriorOfV(LaplaceApproximate):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.hessian_operator = self.model.MxHessian_linear_operator()
        
    def calculate_eigensystem_lam(self, num_eigval, method='double_pass', 
                              oversampling_factor=20, cut_val=0.9, **kwargs):
        ## the calculate_eigensystem calculate the eigensystem of 
        ## the operator C_0^{1/2}H_{misfit}C_0^{1/2} 
        self.calculate_eigensystem(num_eigval, method, oversampling_factor, 
                                   cut_val, **kwargs)
        self.eigval_lam = self.eigval*(self.model.lam*self.model.lam + self.model.lam_dis.cov)
     
    def pointwise_variance_field_lam(self, xx, yy):
        assert hasattr(self, "eigval") and hasattr(self, "eigvec")
        
        SN = construct_measurement_matrix(np.array(xx), self.prior.domain.function_space)
        SM = construct_measurement_matrix(np.array(xx), self.prior.domain.function_space)
        SM = SM.T
        
        val = spsl.spsolve(self.prior.K, SM)
        dr = self.eigval_lam/(self.eigval_lam + 1.0)
        Dr = sps.csc_matrix(sps.diags(dr))
        val1 = self.eigvec@Dr@self.eigvec.T@self.M@val
        val = self.M@(val - val1)
        val = spsl.spsolve(self.prior.K, val)
        val = SN@val     
        
        if type(val) == type(self.M):
            val = val.todense()
        
        return np.array(val)  
    
    def generate_sample_lam(self):
        assert hasattr(self, "mean") and hasattr(self, "eigval") and hasattr(self, "eigvec")
        n = np.random.normal(0, 1, (self.fun_dim,))
        val1 = self.Minv_lamped_half@n
        pr = 1.0/np.sqrt(self.eigval_lam+1.0) - 1.0
        Pr = sps.csc_matrix(sps.diags(pr))
        val2 = self.eigvec@Pr@self.eigvec.T@self.M@val1
        val = self.M@(val1 + val2)
        val = spsl.spsolve(self.prior.K, val)
        val = self.mean + val
        return np.array(val)
     
    def eval_mean_iterative(self, iter_num=50, cg_max=1000, method="cg_my"):
        newton_cg = NewtonCG(model=self.model)
        
        ## calculate the posterior mean 
        max_iter = iter_num
        loss_pre, _, _ = self.model.loss()
        for itr in range(max_iter):
            newton_cg.descent_direction(cg_max=cg_max, method=method)
            newton_cg.step(method='armijo', show_step=False)
            loss, _, _ = self.model.loss()
            # print("iter = %d/%d, loss = %.4f" % (itr+1, max_iter, loss))
            if newton_cg.converged == False:
                break
            if np.abs(loss - loss_pre) < 1e-5*loss:
                # print("Iteration stoped at iter = %d" % itr)
                break 
            loss_pre = loss
        
        tmp = self.model.lam/np.sqrt(self.model.lam*self.model.lam + self.model.lam_dis.cov)
        self.mean = tmp*newton_cg.mk
        
    def eval_mean(self, cg_tol=None, cg_max=1000, method='cg_my', curvature_detector=False):
        self.g = self.model.eval_HAdjointData()
        pre_cond = self.model.precondition_linear_operator()
            
        if cg_tol is None:
            ## if cg_tol is None, specify cg_tol according to the rule in the following paper:
            ## Learning physics-based models from data: perspective from inverse problems
            ## and model reduction, Acta Numerica, 2021. Page: 465-466
            norm_grad = np.sqrt(self.g@self.M@self.g)
            cg_tol = min(0.5, np.sqrt(norm_grad))
        atol = 0.1
        ## cg iteration will terminate when norm(residual) <= min(atol, cg_tol*|self.grad|)
             
        if method == 'cg_my':
            ## Note that the M and Minv option of cg_my are designed different to the methods 
            ## in scipy, e.g., Minv should be M in scipy
            ## Here, in order to keep the matrix to be symmetric and positive definite,
            ## we actually need to solve M H g = M (-grad) instead of H g = -grad
            self.mean, info, k = cg_my(
                self.hessian_operator, self.M@self.g, Minv=pre_cond,
                tol=cg_tol, atol=atol, maxiter=cg_max, curvature_detector=True
                )
            # if k == 1:  
            #     ## If the curvature is negative for the first iterate, use grad as the 
            #     ## search direction.
            #     self.g = -self.grad
            # # print(info)
        elif method == 'bicgstab':
            ## It seems hardly to incoporate the curvature terminate condition 
            ## into the cg type methods in scipy. However, we keep the implementation.
            ## Since there may be some ways to add curvature terminate condition, and 
            ## cg type methods in scipy can be used as a basedline for testing linear problems.
            self.mean, info = spsl.bicgstab(
                self.hessian_operator, self.M@self.g, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )
        elif method == 'cg':
            self.mean, info = spsl.cg(
                self.hessian_operator, self.M@self.g, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )
        elif method == 'cgs':
            self.mean, info = spsl.cgs(
                self.hessian_operator, self.M@self.g, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )       
        else:
            assert False, "method should be cg, cgs, bicgstab"
        
        self.hessian_terminate_info = info
        
        

def relative_error(u, v, domain):
    u = fe.interpolate(u, domain.function_space)
    v = fe.interpolate(v, domain.function_space)
    fenzi = fe.assemble(fe.inner(u-v, u-v)*fe.dx)
    fenmu = fe.assemble(fe.inner(v, v)*fe.dx)
    return fenzi/fenmu





